# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""东财 NOTICE_DATE 公告日历补洞（非科目主源）。

只产出 AnnounceRow，不把 EM 三表科目写入 staging。
与国泰安对齐时：
- revision=0 一律挂 NOTICE_DATE（覆盖 CSMAR 主报表行）
- 若 UPDATE_DATE > NOTICE，再额外挂 revision=1（供差错更正行）
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from typing import Sequence

import pandas as pd

from .akshare_adapter import (
    AkshareStatementAdapter,
    _NOTICE_COLS,
    _PERIOD_COLS,
    _UPDATE_COLS,
    _first_col,
    _parse_period,
)
from .announce import CompositeAnnounceAdapter, StaticAnnounceAdapter, as_utc
from .base import AnnounceAdapter, AnnounceRow, a_code_to_instrument, instrument_to_a_code

_log = logging.getLogger(__name__)

# 全市场公告拉取默认并发；可用 FUND_CN_EM_WORKERS 覆盖
_DEFAULT_EM_WORKERS = 8
# 单票失败重试次数 / 间隔（秒）
_DEFAULT_EM_RETRIES = 3
_DEFAULT_EM_RETRY_SLEEP = 1.5


def _dual_revision_announces_from_sheet(
    frame: pd.DataFrame,
    *,
    instrument_key: str,
) -> dict[tuple[date, int], AnnounceRow]:
    """从单张报表提取 NOTICE→r0 / UPDATE→r1 公告。"""
    if frame is None or frame.empty:
        return {}
    period_col = _first_col(frame, _PERIOD_COLS)
    notice_col = _first_col(frame, _NOTICE_COLS)
    if period_col is None or notice_col is None:
        return {}
    update_col = _first_col(frame, _UPDATE_COLS)
    out: dict[tuple[date, int], AnnounceRow] = {}
    for _, raw in frame.iterrows():
        pe = _parse_period(raw[period_col])
        notice = _parse_period(raw.get(notice_col))
        if pe is None or notice is None:
            continue
        notice_at = as_utc(notice)
        out[(pe, 0)] = AnnounceRow(
            instrument_key=instrument_key,
            report_period_end=pe,
            publish_time=notice_at,
            available_time=notice_at,
            revision=0,
            is_restatement=False,
            source="em_notice",
            source_record_id=f"{instrument_key}:{pe.isoformat()}:r0",
        )
        if update_col is not None:
            update = _parse_period(raw.get(update_col))
            if update is not None and update > notice:
                update_at = as_utc(update)
                out[(pe, 1)] = AnnounceRow(
                    instrument_key=instrument_key,
                    report_period_end=pe,
                    publish_time=update_at,
                    available_time=update_at,
                    revision=1,
                    is_restatement=True,
                    source="em_notice",
                    source_record_id=f"{instrument_key}:{pe.isoformat()}:r1",
                )
    return out


class EmNoticeAnnounceAdapter:
    """东财三表 NOTICE/UPDATE → 双 revision 公告日历；不写科目。"""

    def __init__(
        self,
        *,
        inner: AkshareStatementAdapter | None = None,
        retries: int | None = None,
        retry_sleep: float | None = None,
    ) -> None:
        self._inner = inner or AkshareStatementAdapter(source="akshare_em_notice_only")
        self._retries = max(
            1,
            int(retries if retries is not None else os.getenv("FUND_CN_EM_RETRIES", _DEFAULT_EM_RETRIES)),
        )
        self._retry_sleep = float(
            retry_sleep
            if retry_sleep is not None
            else os.getenv("FUND_CN_EM_RETRY_SLEEP", _DEFAULT_EM_RETRY_SLEEP)
        )

    def _fetch_one(self, key: str) -> list[AnnounceRow]:
        """单票拉三表 NOTICE/UPDATE（线程内独立 adapter；失败重试）。"""
        code = instrument_to_a_code(key)
        instrument = key if ":" in key else a_code_to_instrument(code)
        last_exc: Exception | None = None
        for attempt in range(1, self._retries + 1):
            try:
                local = AkshareStatementAdapter(source=self._inner.source)
                rows: list[AnnounceRow] = []
                for statement_type in ("income", "balance", "cashflow"):
                    sheet = local._sheet(code, statement_type)
                    rows.extend(
                        _dual_revision_announces_from_sheet(sheet, instrument_key=instrument).values()
                    )
                return rows
            except Exception as exc:
                last_exc = exc
                _log.warning(
                    "em_notice attempt %s/%s failed %s: %s",
                    attempt,
                    self._retries,
                    key,
                    exc,
                )
                if attempt < self._retries:
                    time.sleep(self._retry_sleep * attempt)
        if last_exc is not None:
            raise last_exc
        return []

    def fetch_announcements(
        self,
        instrument_keys: Sequence[str],
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[AnnounceRow]:
        keys = list(instrument_keys)
        merged: dict[tuple[str, date, int], AnnounceRow] = {}
        failed_keys: list[str] = []
        workers = max(1, int(os.getenv("FUND_CN_EM_WORKERS", str(_DEFAULT_EM_WORKERS))))
        # 小样本串行，避免线程开销；全市场并行拉 NOTICE
        if len(keys) <= 2 or workers == 1:
            batches = []
            for k in keys:
                try:
                    batches.append(self._fetch_one(k))
                except Exception as exc:
                    _log.warning("em_notice failed %s: %s", k, exc)
                    failed_keys.append(k)
                    batches.append([])
        else:
            batches = []
            done = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(self._fetch_one, k): k for k in keys}
                for fut in as_completed(futures):
                    done += 1
                    if done % 100 == 0 or done == len(keys):
                        _log.info("em_notice progress %s/%s", done, len(keys))
                    try:
                        batches.append(fut.result())
                    except Exception as exc:
                        key = futures[fut]
                        _log.warning("em_notice failed %s: %s", key, exc)
                        failed_keys.append(key)
        for rows in batches:
            for row in rows:
                merged[(row.instrument_key, row.report_period_end, row.revision)] = row
        out = list(merged.values())
        if start is not None:
            out = [r for r in out if r.report_period_end >= start]
        if end is not None:
            out = [r for r in out if r.report_period_end <= end]
        _log.info(
            "em_notice announces=%s instruments=%s failed=%s",
            len(out),
            len(keys),
            len(failed_keys),
        )
        if failed_keys:
            _log.warning("em_notice failed_keys sample=%s", failed_keys[:20])
        return out


def build_csmar_announce_calendar(
    *,
    instrument_keys: Sequence[str],
    csmar_correct_rows: Sequence[AnnounceRow] | None = None,
    em_adapter: AnnounceAdapter | None = None,
    start: date | None = None,
    end: date | None = None,
) -> list[AnnounceRow]:
    """CSMAR 差错更正日优先，其余 period 用 EM NOTICE(r0)/UPDATE(r1) 补洞。"""
    primary = StaticAnnounceAdapter(list(csmar_correct_rows or []))
    secondary = em_adapter or EmNoticeAnnounceAdapter()
    return CompositeAnnounceAdapter(primary, secondary).fetch_announcements(
        instrument_keys, start=start, end=end
    )


_: AnnounceAdapter = EmNoticeAnnounceAdapter()
