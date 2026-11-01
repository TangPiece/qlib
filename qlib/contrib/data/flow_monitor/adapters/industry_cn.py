# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A 股行业与市值风格：风格走新浪指数；行业走申万一级（东财板块 push 常断）。"""

from __future__ import annotations

from datetime import date

from .base import (
    FlowRow,
    finite_number,
    first_col,
    in_range,
    lookup,
    parse_day,
    row_from_def,
    safe_call,
)

# 申万一级指数代码 → catalog slug（近似映射，不是东财板块同名）
_SW_BOARDS = {
    "801080": "electronics",  # 电子
    "801050": "nmetal",  # 有色金属
    "801770": "telecom",  # 通信
    "801730": "grid",  # 电力设备（含电网设备）
    "801880": "auto",  # 汽车
    "801890": "machinery",  # 机械设备
    "801150": "pharma",  # 医药生物（近似化学制药）
    "801780": "bank",  # 银行
    "801790": "broker",  # 非银金融（近似证券）
    "801120": "food",  # 食品饮料
    "801180": "property",  # 房地产
}

# 半导体无独立申万一级；用电子二级近似会更合适，本期先缺省，避免误标
_STYLE = {
    "style_sz50_px": "sh000016",
    "style_csi300_px": "sh000300",
    "style_csi500_px": "sh000905",
    "style_csi1000_px": "sh000852",
    "style_chinext_px": "sz399006",
    "style_star50_px": "sh000688",
    "style_dividend_px": "sh000015",
}


class IndustryCnAdapter:
    """行业与风格只有价格/成交额，不能据此写行业净流入金额。"""

    name = "industry_cn"

    def fetch(self, *, start: date | None = None, end: date | None = None) -> list[FlowRow]:
        rows: list[FlowRow] = []
        rows.extend(self._fetch_boards(start, end))
        rows.extend(self._fetch_style(start, end))
        return rows

    def _fetch_boards(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        rows: list[FlowRow] = []
        for code, slug in _SW_BOARDS.items():
            frame = safe_call(f"index_hist_sw:{code}", ak.index_hist_sw, symbol=code, period="day")
            if frame.empty:
                continue
            date_col = first_col(frame, ("日期", "date"))
            close_col = first_col(frame, ("收盘", "close"))
            amount_col = first_col(frame, ("成交额", "amount"))
            if date_col is None:
                continue
            px = lookup(f"ind_{slug}_px")
            amount_spec = lookup(f"ind_{slug}_amount")
            for idx in range(len(frame)):
                day = parse_day(frame.iloc[idx][date_col])
                if not in_range(day, start, end):
                    continue
                if close_col is not None:
                    close = finite_number(frame.iloc[idx][close_col])
                    if close is not None:
                        rows.append(row_from_def(px, day, close, source="akshare.index_hist_sw"))
                if amount_col is not None:
                    amount = finite_number(frame.iloc[idx][amount_col])
                    if amount is not None:
                        rows.append(row_from_def(amount_spec, day, amount, source="akshare.index_hist_sw"))
        return rows

    def _fetch_style(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        rows: list[FlowRow] = []
        for series_id, symbol in _STYLE.items():
            spec = lookup(series_id)
            frame = safe_call(f"stock_zh_index_daily:{symbol}", ak.stock_zh_index_daily, symbol=symbol)
            if frame.empty:
                continue
            date_col = first_col(frame, ("date", "日期"))
            close_col = first_col(frame, ("close", "收盘"))
            if date_col is None or close_col is None:
                continue
            for raw_day, raw_value in zip(frame[date_col], frame[close_col]):
                day = parse_day(raw_day)
                value = finite_number(raw_value)
                if value is None or not in_range(day, start, end):
                    continue
                rows.append(row_from_def(spec, day, value, source="akshare.stock_zh_index_daily"))
        return rows
