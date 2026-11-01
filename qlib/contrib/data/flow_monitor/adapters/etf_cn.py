# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A 股代表性 ETF：行情优先新浪（东财 push 常断连），份额仍走交易所。"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from ..catalog import SERIES_DEFINITIONS
from .base import (
    FlowRow,
    finite_number,
    first_col,
    in_range,
    lookup,
    parse_day,
    row_from_def,
    safe_call,
    yyyymmdd,
)

_log = logging.getLogger(__name__)


def _etf_codes() -> list[str]:
    codes: list[str] = []
    for spec in SERIES_DEFINITIONS:
        if spec.series_id.startswith("etf_") and spec.series_id.endswith("_px"):
            codes.append(spec.series_id.split("_")[1])
    return codes


def _sina_symbol(code: str) -> str:
    """上交所 5 开头，深交所 1 开头。"""
    return f"sh{code}" if code.startswith("5") else f"sz{code}"


class EtfCnAdapter:
    """先拉日行情，再补交易所份额；份额缺失时只保留价格/成交额。"""

    name = "etf_cn"

    def fetch(self, *, start: date | None = None, end: date | None = None) -> list[FlowRow]:
        rows: list[FlowRow] = []
        for code in _etf_codes():
            rows.extend(self._fetch_hist(code, start, end))
        rows.extend(self._fetch_sse_shares(start, end))
        rows.extend(self._fetch_szse_shares(start, end))
        return rows

    def _fetch_hist(self, code: str, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        sina_sym = _sina_symbol(code)
        frame = safe_call(f"fund_etf_hist_sina:{sina_sym}", ak.fund_etf_hist_sina, symbol=sina_sym)
        if frame.empty:
            # 兜底东财；多数网络下会失败，但保留路径
            start_text = yyyymmdd(start) or "20180101"
            end_text = yyyymmdd(end) or "20500101"
            frame = safe_call(
                f"fund_etf_hist_em:{code}",
                ak.fund_etf_hist_em,
                symbol=code,
                period="daily",
                start_date=start_text,
                end_date=end_text,
                adjust="",
            )
        if frame.empty:
            return []
        date_col = first_col(frame, ("date", "日期"))
        close_col = first_col(frame, ("close", "收盘"))
        amount_col = first_col(frame, ("amount", "成交额"))
        if date_col is None:
            return []
        px = lookup(f"etf_{code}_px")
        amount_spec = lookup(f"etf_{code}_amount")
        rows: list[FlowRow] = []
        for idx in range(len(frame)):
            day = parse_day(frame.iloc[idx][date_col])
            if not in_range(day, start, end):
                continue
            if close_col is not None:
                close = finite_number(frame.iloc[idx][close_col])
                if close is not None:
                    rows.append(row_from_def(px, day, close))
            if amount_col is not None:
                amount = finite_number(frame.iloc[idx][amount_col])
                if amount is not None:
                    rows.append(row_from_def(amount_spec, day, amount))
        return rows

    def _fetch_sse_shares(self, start: date | None, end: date | None) -> list[FlowRow]:
        """上交所 ETF 份额按日接口；只抽最近若干交易日，避免逐日打爆。"""
        import akshare as ak

        days = _sample_days(start, end, limit=8)
        rows: list[FlowRow] = []
        wanted = {code for code in _etf_codes() if code.startswith("5")}
        for day in days:
            frame = safe_call(f"fund_etf_scale_sse:{day}", ak.fund_etf_scale_sse, date=yyyymmdd(day))
            rows.extend(_shares_from_scale_frame(frame, wanted, fallback_day=day))
        return rows

    def _fetch_szse_shares(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        rows: list[FlowRow] = []
        wanted = {code for code in _etf_codes() if not code.startswith("5")}
        if start is not None and end is not None:
            frame = safe_call(
                "fund_scale_daily_szse",
                ak.fund_scale_daily_szse,
                start_date=yyyymmdd(start),
                end_date=yyyymmdd(end),
                symbol="ETF",
            )
            rows.extend(_shares_from_scale_frame(frame, wanted, fallback_day=end))
        snap = safe_call("fund_etf_scale_szse", ak.fund_etf_scale_szse)
        rows.extend(_shares_from_scale_frame(snap, wanted, fallback_day=end or date.today()))
        return rows


def _sample_days(start: date | None, end: date | None, *, limit: int) -> list[date]:
    """从区间末端往前抽工作日。周末/假日上交所份额接口会返回空表。"""
    last = end or date.today()
    first = start or (last - timedelta(days=365))
    days: list[date] = []
    cursor = last
    while cursor >= first and len(days) < limit:
        if cursor.weekday() < 5:
            days.append(cursor)
        cursor -= timedelta(days=1)
    return days


def _shares_from_scale_frame(frame, wanted: set[str], *, fallback_day: date) -> list[FlowRow]:
    if frame is None or getattr(frame, "empty", True):
        return []
    code_col = first_col(frame, ("基金代码", "代码", "SEC_CODE"))
    share_col = first_col(frame, ("基金份额", "份额", "TOT_VOL", "规模"))
    date_col = first_col(frame, ("统计日期", "日期", "STAT_DATE"))
    if code_col is None or share_col is None:
        return []
    rows: list[FlowRow] = []
    for idx in range(len(frame)):
        raw_code = str(frame.iloc[idx][code_col]).split(".")[0].zfill(6)
        if raw_code not in wanted:
            continue
        series_id = f"etf_{raw_code}_shares"
        try:
            spec = lookup(series_id)
        except KeyError:
            continue
        value = finite_number(frame.iloc[idx][share_col])
        if value is None:
            continue
        day = parse_day(frame.iloc[idx][date_col]) if date_col else fallback_day
        if day is None:
            day = fallback_day
        rows.append(row_from_def(spec, day, value, asof=day))
    return rows
