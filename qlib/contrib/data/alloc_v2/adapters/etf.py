# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ETF 行情 / NAV / 份额。行情走新浪；NAV 走天天基金；份额走交易所。"""

from __future__ import annotations

import logging
from datetime import date, timedelta

from ..mappings import load_etf_master
from ..schema import finite_number, first_col, in_range, parse_day, yyyymmdd
from .base import lookup, row_of, safe_call

_log = logging.getLogger(__name__)

# 上交所份额接口按日查询：近端日频 + 更早周频，兼顾 window=20 与全量耗时
_SSE_DAILY_LOOKBACK_DAYS = 180
_SZSE_CHUNK_DAYS = 90


def _sina_symbol(code: str) -> str:
    return f"sh{code}" if code.startswith("5") else f"sz{code}"


def _listed_etf_codes() -> list[str]:
    master = load_etf_master()
    listed = master[master["exchange"].isin({"SH", "SZ"})]
    return [str(code) for code in listed["etf_code"]]


def _sse_share_sample_days(start: date | None, end: date | None) -> list[date]:
    """上交所份额采样日：最近约半年每个工作日，更早仅取周五。"""
    last = end or date.today()
    first = start or (last - timedelta(days=365))
    if first > last:
        first, last = last, first
    recent_cut = last - timedelta(days=_SSE_DAILY_LOOKBACK_DAYS)
    days: list[date] = []
    cursor = last
    while cursor >= first:
        if cursor.weekday() < 5:
            if cursor >= recent_cut or cursor.weekday() == 4:
                days.append(cursor)
        cursor -= timedelta(days=1)
    return days


def _month_chunks(start: date, end: date, *, chunk_days: int = _SZSE_CHUNK_DAYS) -> list[tuple[date, date]]:
    """深交所区间接口按块拉取，避免单次跨度过大。"""
    chunks: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + timedelta(days=chunk_days - 1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + timedelta(days=1)
    return chunks


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
        """上交所按日接口：近端日频、远端周频。

        远端周采样会写入 quality_flag=sparse_sample；全样本流因子早期区间勿当完整日频。
        """
        import akshare as ak

        spec = lookup("etf_shares")
        days = _sse_share_sample_days(start, end)
        wanted = {code for code in _listed_etf_codes() if code.startswith("5")}
        last = end or date.today()
        recent_cut = last - timedelta(days=_SSE_DAILY_LOOKBACK_DAYS)
        rows = []
        _log.info("etf_shares SSE sample days=%s (daily after %s)", len(days), recent_cut)
        for day in days:
            # 近 180 日之外仅周五点 → sparse_sample，供因子/文档降权
            qflag = "ok" if day >= recent_cut else "sparse_sample"
            frame = safe_call(f"fund_etf_scale_sse:{day}", ak.fund_etf_scale_sse, date=yyyymmdd(day))
            rows.extend(_shares_from_scale(frame, wanted, spec, fallback_day=day, quality_flag=qflag))
        return rows

    def _fetch_szse_shares(self, start: date | None, end: date | None) -> list:
        """深交所区间日频份额：按块拉取全区间，再补一张快照。"""
        import akshare as ak

        spec = lookup("etf_shares")
        wanted = {code for code in _listed_etf_codes() if not code.startswith("5")}
        rows = []
        last = end or date.today()
        first = start or (last - timedelta(days=365))
        for chunk_start, chunk_end in _month_chunks(first, last):
            frame = safe_call(
                f"fund_scale_daily_szse:{yyyymmdd(chunk_start)}_{yyyymmdd(chunk_end)}",
                ak.fund_scale_daily_szse,
                start_date=yyyymmdd(chunk_start),
                end_date=yyyymmdd(chunk_end),
                symbol="ETF",
            )
            rows.extend(_shares_from_scale(frame, wanted, spec, fallback_day=chunk_end, quality_flag="ok"))
        snap = safe_call("fund_etf_scale_szse", ak.fund_etf_scale_szse)
        rows.extend(_shares_from_scale(snap, wanted, spec, fallback_day=last, quality_flag="ok"))
        return rows


def _shares_from_scale(
    frame,
    wanted: set[str],
    spec,
    *,
    fallback_day: date,
    quality_flag: str = "ok",
) -> list:
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
                quality_flag=quality_flag,
            )
        )
    return rows