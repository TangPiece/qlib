"""Total share capital change events (cninfo) for daily market cap.

``ak.stock_share_change_cninfo`` (verified akshare 1.19) returns one row per
change with ``变动日期`` (effective), ``公告日期`` (announce) and ``总股本`` in
**万股**. Events cover 送转 / 增发 / 回购注销 / 定期报告 snapshots, so the
piecewise-constant series stays current between financial reports.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Callable, Sequence

import pandas as pd

from .announce import as_utc
from .base import ShareCapitalRow, a_code_to_instrument, instrument_to_a_code

_log = logging.getLogger(__name__)

ShareFetcher = Callable[[str, date | None, date | None], pd.DataFrame | None]

_TEN_THOUSAND = 10_000.0


class StaticShareCapitalAdapter:
    """Injected events — tests and curated dumps."""

    def __init__(self, rows: Sequence[ShareCapitalRow] | None = None) -> None:
        self._rows = list(rows or [])

    def fetch_share_capital(
        self,
        instrument_keys: Sequence[str],
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[ShareCapitalRow]:
        keys = set(instrument_keys)
        out = [r for r in self._rows if r.instrument_key in keys]
        if start is not None:
            out = [r for r in out if r.effective_date >= start]
        if end is not None:
            out = [r for r in out if r.effective_date <= end]
        return out


def dataframe_to_share_rows(
    frame: pd.DataFrame,
    *,
    instrument_key: str,
    source: str = "cninfo_share_change",
) -> list[ShareCapitalRow]:
    if frame is None or frame.empty:
        return []
    needed = ("变动日期", "公告日期", "总股本")
    if any(col not in frame.columns for col in needed):
        _log.warning("share change frame missing columns: %s", list(frame.columns)[:20])
        return []
    out: list[ShareCapitalRow] = []
    for _, raw in frame.iterrows():
        effective = pd.to_datetime(raw["变动日期"], errors="coerce")
        announced = pd.to_datetime(raw["公告日期"], errors="coerce")
        total = pd.to_numeric(raw["总股本"], errors="coerce")
        if pd.isna(effective) or pd.isna(announced) or pd.isna(total) or float(total) <= 0:
            continue
        eff = effective.date()
        out.append(
            ShareCapitalRow(
                instrument_key=instrument_key,
                effective_date=eff,
                announce_time=as_utc(announced.date()),
                total_shares=float(total) * _TEN_THOUSAND,
                source=source,
                source_record_id=f"{instrument_key}:{eff.isoformat()}:{announced.date().isoformat()}",
            )
        )
    return out


class CninfoShareCapitalAdapter:
    def __init__(self, *, fetcher: ShareFetcher | None = None) -> None:
        self._fetcher = fetcher or self._default_fetch

    @staticmethod
    def _default_fetch(code: str, start: date | None, end: date | None) -> pd.DataFrame | None:
        try:
            import akshare as ak  # type: ignore

            return ak.stock_share_change_cninfo(
                symbol=code,
                start_date=(start or date(1990, 1, 1)).strftime("%Y%m%d"),
                end_date=(end or datetime.now().date()).strftime("%Y%m%d"),
            )
        except Exception as exc:
            _log.warning("cninfo share change failed %s: %s", code, exc)
            return None

    def fetch_share_capital(
        self,
        instrument_keys: Sequence[str],
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[ShareCapitalRow]:
        rows: list[ShareCapitalRow] = []
        for key in instrument_keys:
            code = instrument_to_a_code(key)
            instrument = key if ":" in key else a_code_to_instrument(code)
            rows.extend(
                dataframe_to_share_rows(self._fetcher(code, start, end), instrument_key=instrument)
            )
        return rows
