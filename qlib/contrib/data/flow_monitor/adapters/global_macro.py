# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""全球价格、利率、汇率与黄金 ETF 持仓。

东财 push2his 在部分网络会 RemoteDisconnected，因此股指优先走新浪；
收益率/汇率/上海金仍用可用源。价格一律登记为 L3。
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Callable

import pandas as pd

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

# series_id → (fetcher_kind, symbol)
# us_sina: index_us_stock_sina；cn_sina: stock_zh_index_daily；
# hk_sina: stock_hk_index_daily_sina；global_sina: index_global_hist_sina
_INDEX_SOURCES: dict[str, tuple[str, str]] = {
    "px_spx": ("us_sina", ".INX"),
    "px_ndx": ("us_sina", ".IXIC"),  # 纳斯达克综合；东财 NDX 不通时的 L3 代理
    "px_djia": ("us_sina", ".DJI"),
    "px_csi300": ("cn_sina", "sh000300"),
    "px_sse": ("cn_sina", "sh000001"),
    "px_hsi": ("hk_sina", "HSI"),
    "px_n225": ("global_sina", "日经225指数"),
    "px_sx5e": ("global_sina", "欧洲Stoxx50指数"),
    "px_sensex": ("global_sina", "印度孟买SENSEX指数"),
}

_YIELD_COLS = {
    "yld_us_2y": ("美国国债收益率2年",),
    "yld_us_10y": ("美国国债收益率10年",),
    "yld_cn_2y": ("中国国债收益率2年",),
    "yld_cn_10y": ("中国国债收益率10年",),
}


class GlobalMacroAdapter:
    """全球宏观：新浪股指 + 东财国债收益率 + 汇率/黄金。"""

    name = "global_macro"

    def fetch(self, *, start: date | None = None, end: date | None = None) -> list[FlowRow]:
        rows: list[FlowRow] = []
        rows.extend(self._fetch_indexes(start, end))
        rows.extend(self._fetch_yields(start, end))
        rows.extend(self._fetch_usdcny(start, end))
        rows.extend(self._fetch_sge_gold(start, end))
        rows.extend(self._fetch_gld(start, end))
        return rows

    def _fetch_indexes(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        fetchers: dict[str, Callable[..., pd.DataFrame]] = {
            "us_sina": ak.index_us_stock_sina,
            "cn_sina": ak.stock_zh_index_daily,
            "hk_sina": ak.stock_hk_index_daily_sina,
            "global_sina": ak.index_global_hist_sina,
        }
        rows: list[FlowRow] = []
        for series_id, (kind, symbol) in _INDEX_SOURCES.items():
            spec = lookup(series_id)
            fn = fetchers[kind]
            frame = safe_call(f"{kind}:{symbol}", fn, symbol=symbol)
            rows.extend(_close_rows(frame, spec, start, end))
        # 美元指数 / CRB：东财推送不可用时跳过，catalog 仍保留定义
        return rows

    def _fetch_yields(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        start_text = yyyymmdd(start) or "20180101"
        frame = safe_call("bond_zh_us_rate", ak.bond_zh_us_rate, start_date=start_text)
        if frame.empty:
            return []
        date_col = first_col(frame, ("日期", "SOLAR_DATE"))
        if date_col is None:
            return []
        rows: list[FlowRow] = []
        for series_id, candidates in _YIELD_COLS.items():
            spec = lookup(series_id)
            value_col = first_col(frame, candidates)
            if value_col is None:
                continue
            for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
                day = parse_day(raw_day)
                value = finite_number(raw_value)
                if value is None or not in_range(day, start, end):
                    continue
                rows.append(row_from_def(spec, day, value))
        return rows

    def _fetch_usdcny(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        frame = safe_call("currency_boc_safe", ak.currency_boc_safe)
        if frame.empty:
            return []
        date_col = first_col(frame, ("日期", "date"))
        value_col = first_col(frame, ("美元", "美元兑人民币", "USD"))
        if date_col is None or value_col is None:
            return []
        spec = lookup("fx_usdcny")
        rows: list[FlowRow] = []
        for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            rows.append(row_from_def(spec, day, value))
        return rows

    def _fetch_sge_gold(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        frame = safe_call("spot_golden_benchmark_sge", ak.spot_golden_benchmark_sge)
        if frame.empty:
            return []
        date_col = first_col(frame, ("交易时间", "交易日期", "日期", "date"))
        value_col = first_col(frame, ("晚盘价", "早盘价", "收盘价", "价格"))
        if date_col is None or value_col is None:
            return []
        spec = lookup("px_gold_sge")
        rows: list[FlowRow] = []
        for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            rows.append(row_from_def(spec, day, value))
        return rows

    def _fetch_gld(self, start: date | None, end: date | None) -> list[FlowRow]:
        """金十分页较慢，失败则跳过，不影响其他序列。"""
        import akshare as ak

        _log.info("拉取 SPDR 黄金 ETF 持仓（可能需数十秒）…")
        frame = safe_call("macro_cons_gold", ak.macro_cons_gold)
        if frame.empty:
            return []
        date_col = first_col(frame, ("日期",))
        stock_col = first_col(frame, ("总库存",))
        change_col = first_col(frame, ("增持/减持", "增减持"))
        if date_col is None:
            return []
        stock_spec = lookup("gold_gld_tonnes")
        change_spec = lookup("gold_gld_change")
        rows: list[FlowRow] = []
        for idx in range(len(frame)):
            day = parse_day(frame.iloc[idx][date_col])
            if not in_range(day, start, end):
                continue
            if stock_col is not None:
                stock = finite_number(frame.iloc[idx][stock_col])
                if stock is not None:
                    rows.append(row_from_def(stock_spec, day, stock))
            if change_col is not None:
                change = finite_number(frame.iloc[idx][change_col])
                if change is not None:
                    rows.append(row_from_def(change_spec, day, change))
        return rows


def _close_rows(frame: pd.DataFrame, spec, start: date | None, end: date | None) -> list[FlowRow]:
    if frame is None or frame.empty:
        return []
    date_col = first_col(frame, ("date", "日期", "交易时间"))
    value_col = first_col(frame, ("close", "收盘", "最新价"))
    if date_col is None or value_col is None:
        return []
    rows: list[FlowRow] = []
    for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
        day = parse_day(raw_day)
        value = finite_number(raw_value)
        if value is None or not in_range(day, start, end):
            continue
        rows.append(row_from_def(spec, day, value, source=f"akshare/{spec.adapter}"))
    return rows
