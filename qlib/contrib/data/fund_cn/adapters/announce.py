"""Disclosure / announcement calendar adapters (cninfo-oriented SPI)."""

from __future__ import annotations

import logging
from datetime import date, datetime, time, timezone
from typing import Callable, Sequence

import pandas as pd

from .base import AnnounceAdapter, AnnounceRow, a_code_to_instrument, instrument_to_a_code

_log = logging.getLogger(__name__)

AnnounceFetcher = Callable[[Sequence[str], date | None, date | None], list[AnnounceRow]]


# Date-only disclosures may be released after the close (evening); treat them as
# visible from Beijing D+1 00:00 (= D 16:00 UTC) so the D close cannot use them.
_DATE_ONLY_VISIBLE_UTC = time(16, 0)


def as_utc(value: datetime | date | str) -> datetime:
    """Normalize to tz-aware UTC.

    Pure dates and naive midnight datetimes are treated as date-only disclosures.
    """
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if isinstance(value, date) and not isinstance(value, datetime):
        return datetime.combine(value, _DATE_ONLY_VISIBLE_UTC, tzinfo=timezone.utc)
    if value.tzinfo is None:
        if value.time() == time(0, 0):
            return datetime.combine(value.date(), _DATE_ONLY_VISIBLE_UTC, tzinfo=timezone.utc)
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class StaticAnnounceAdapter:
    """Injected disclosure calendar — preferred for tests and curated CNINFO dumps."""

    def __init__(self, rows: Sequence[AnnounceRow] | None = None) -> None:
        self._rows = list(rows or [])

    def fetch_announcements(
        self,
        instrument_keys: Sequence[str],
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[AnnounceRow]:
        keys = set(instrument_keys)
        out = [r for r in self._rows if r.instrument_key in keys]
        if start is not None:
            out = [r for r in out if r.report_period_end >= start]
        if end is not None:
            out = [r for r in out if r.report_period_end <= end]
        return out


class AkshareAnnounceAdapter:
    """Best-effort AKShare earnings calendar; missing dates must fail PIT gate."""

    def __init__(self, *, fetcher: Callable[[], pd.DataFrame | None] | None = None) -> None:
        self._fetcher = fetcher or self._default_yjbb

    @staticmethod
    def _default_yjbb() -> pd.DataFrame | None:
        try:
            import akshare as ak  # type: ignore

            # Latest board-style report; historical coverage varies by AKShare version.
            return ak.stock_yjbb_em(date=datetime.now(timezone.utc).strftime("%Y%m%d"))
        except Exception as exc:
            _log.warning("akshare announce fetch failed: %s", exc)
            return None

    def fetch_announcements(
        self,
        instrument_keys: Sequence[str],
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[AnnounceRow]:
        frame = self._fetcher()
        if frame is None or frame.empty:
            return []
        wanted = {instrument_to_a_code(k): (k if ":" in k else a_code_to_instrument(k)) for k in instrument_keys}
        code_col = next((c for c in ("股票代码", "证券代码", "code", "CODE") if c in frame.columns), None)
        period_col = next(
            (c for c in ("报告期", "REPORT_DATE", "截止日期") if c in frame.columns),
            None,
        )
        announce_col = next(
            (c for c in ("公告日期", "NOTICE_DATE", "披露日期") if c in frame.columns),
            None,
        )
        if not code_col or not period_col or not announce_col:
            _log.warning("announce frame missing required columns: %s", list(frame.columns)[:20])
            return []
        rows: list[AnnounceRow] = []
        for _, raw in frame.iterrows():
            code = str(raw[code_col]).zfill(6)
            if code not in wanted:
                continue
            period_end = pd.to_datetime(raw[period_col], errors="coerce")
            announce = pd.to_datetime(raw[announce_col], errors="coerce")
            if pd.isna(period_end) or pd.isna(announce):
                continue
            pe = period_end.date()
            at = as_utc(announce.to_pydatetime())
            if start is not None and pe < start:
                continue
            if end is not None and pe > end:
                continue
            rows.append(
                AnnounceRow(
                    instrument_key=wanted[code],
                    report_period_end=pe,
                    publish_time=at,
                    available_time=at,
                    revision=0,
                    is_restatement=False,
                    source="akshare_yjbb",
                    source_record_id=f"{wanted[code]}:{pe.isoformat()}:r0",
                )
            )
        return rows


class CompositeAnnounceAdapter:
    """Prefer curated/static rows; fill gaps from secondary adapter."""

    def __init__(self, primary: AnnounceAdapter, secondary: AnnounceAdapter | None = None) -> None:
        self.primary = primary
        self.secondary = secondary

    def fetch_announcements(
        self,
        instrument_keys: Sequence[str],
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[AnnounceRow]:
        primary = self.primary.fetch_announcements(instrument_keys, start=start, end=end)
        seen = {(r.instrument_key, r.report_period_end, r.revision) for r in primary}
        if self.secondary is None:
            return primary
        secondary = self.secondary.fetch_announcements(instrument_keys, start=start, end=end)
        for row in secondary:
            key = (row.instrument_key, row.report_period_end, row.revision)
            if key not in seen:
                primary.append(row)
                seen.add(key)
        return primary


_: AnnounceAdapter = StaticAnnounceAdapter([])
