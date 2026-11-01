# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ETF 行情 / NAV / 份额。行情走新浪；NAV 走天天基金；份额走交易所。"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from ..mappings import load_etf_master
from ..schema import finite_number, first_col, in_range, parse_day, yyyymmdd
from .base import lookup, row_of, safe_call, sample_weekdays

_log = logging.getLogger(__name__)


def _sina_symbol(code: str) -> str:
    return f"sh{code}" if code.startswith("5") else f"sz{code}"


def _listed_etf_codes() -> list[str]:
    master = load_etf_master()
    listed = master[master["exchange"].isin({"SH", "SZ"})]
    return [str(code) for code in listed["etf_code"]]


class EtfAdapter:
    """每个交易日尽量齐：close / amount / nav / shares；缺列保持 NA。"""

    name = "etf"

    def fetch(self, *, start: date | None = None, end: date | None = None) -> list:
        rows = []
        for code in _listed_etf_codes():
            rows.extend(self._fetch_hist(code, start, end))
            rows.extend(self._fetch_nav(code, start, end))
        rows.extend(self._fetch_sse_shares(start, end))
        rows.extend(self._fetch_szse_shares(start, end))
        return rows

    def _fetch_hist(self, code: str, start: date | None, end: date | None) -> list:
        import akshare as ak

        spec = lookup("etf_bar")
        sina_sym = _sina_symbol(code)
        frame = safe_call(f"fund_etf_hist_sina:{sina_sym}", ak.fund_etf_hist_sina, symbol=sina_sym)
        if frame.empty:
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
        rows = []
        for idx in range(len(frame)):
            day = parse_day(frame.iloc[idx][date_col])
            if not in_range(day, start, end):
                continue
            if close_col is not None:
                close = finite_number(frame.iloc[idx][close_col])
                if close is not None:
                    rows.append(row_of(spec, code, day, "close", close, unit="price", evidence_class="proxy"))
            if amount_col is not None:
                amount = finite_number(frame.iloc[idx][amount_col])
                if amount is not None:
                    rows.append(row_of(spec, code, day, "amount", amount, unit="cny", evidence_class="proxy"))
        return rows

    def _fetch_nav(self, code: str, start: date | None, end: date | None) -> list:
        """天天基金历史净值；失败则跳过，不把收盘价冒充 NAV。"""
        import akshare as ak

        spec = lookup("etf_nav")
        start_text = yyyymmdd(start) or "20180101"
        end_text = yyyymmdd(end) or date.today().strftime("%Y%m%d")
        # akshare 1.19.1 参数名是 fund，不是 symbol
        frame = safe_call(
            f"fund_etf_fund_info_em:{code}",
            ak.fund_etf_fund_info_em,
            fund=code,
            start_date=start_text,
            end_date=end_text,
        )
        if frame.empty:
            return []
        date_col = first_col(frame, ("净值日期", "日期", "date"))
        nav_col = first_col(frame, ("单位净值", "nav", "净值"))
        if date_col is None or nav_col is None:
            return []
        rows = []
        for raw_day, raw_value in zip(frame[date_col], frame[nav_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            # 净值通常 T+1 才可观测
            rows.append(
                row_of(
                    spec,
                    code,
                    day,
                    "nav",
                    value,
                    available_at=day + timedelta(days=1),
                    unit="nav",
                    evidence_class="direct",
                )
            )
        return rows

    def _fetch_sse_shares(self, start: date | None, end: date | None) -> list:
        import akshare as ak

        spec = lookup("etf_shares")
        days = sample_weekdays(start, end, limit=8)
        wanted = {code for code in _listed_etf_codes() if code.startswith("5")}
        rows = []
        for day in days:
            frame = safe_call(f"fund_etf_scale_sse:{day}", ak.fund_etf_scale_sse, date=yyyymmdd(day))
            rows.extend(_shares_from_scale(frame, wanted, spec, fallback_day=day))
        return rows

    def _fetch_szse_shares(self, start: date | None, end: date | None) -> list:
        import akshare as ak

        spec = lookup("etf_shares")
        wanted = {code for code in _listed_etf_codes() if not code.startswith("5")}
        rows = []
        if start is not None and end is not None:
            frame = safe_call(
                "fund_scale_daily_szse",
                ak.fund_scale_daily_szse,
                start_date=yyyymmdd(start),
                end_date=yyyymmdd(end),
                symbol="ETF",
            )
            rows.extend(_shares_from_scale(frame, wanted, spec, fallback_day=end))
        snap = safe_call("fund_etf_scale_szse", ak.fund_etf_scale_szse)
        rows.extend(_shares_from_scale(snap, wanted, spec, fallback_day=end or date.today()))
        return rows


def _shares_from_scale(frame, wanted: set[str], spec, *, fallback_day: date) -> list:
    if frame is None or getattr(frame, "empty", True):
        return []
    code_col = first_col(frame, ("基金代码", "代码", "SEC_CODE"))
    share_col = first_col(frame, ("基金份额", "份额", "TOT_VOL", "规模"))
    date_col = first_col(frame, ("统计日期", "日期", "STAT_DATE"))
    if code_col is None or share_col is None:
        return []
    rows = []
    for idx in range(len(frame)):
        raw_code = str(frame.iloc[idx][code_col]).split(".")[0].zfill(6)
        if raw_code not in wanted:
            continue
        value = finite_number(frame.iloc[idx][share_col])
        if value is None:
            continue
        day = parse_day(frame.iloc[idx][date_col]) if date_col else fallback_day
        if day is None:
            day = fallback_day
        rows.append(
            row_of(
                spec,
                raw_code,
                day,
                "shares",
                value,
                available_at=day + timedelta(days=1),
                unit="share",
                evidence_class="direct",
            )
        )
    return rows
