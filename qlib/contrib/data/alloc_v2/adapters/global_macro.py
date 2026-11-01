# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""B 层全球与中国跨资产：指数、收益率、汇率、黄金。价格一律 proxy。"""

from __future__ import annotations

import logging
from datetime import date
from typing import Callable

import pandas as pd

from ..schema import finite_number, first_col, in_range, parse_day, yyyymmdd
from .base import close_rows, lookup, row_of, safe_call

_log = logging.getLogger(__name__)

# entity_id → (fetcher_kind, symbol)
_INDEX_SOURCES: dict[str, tuple[str, str]] = {
    "SPX": ("us_sina", ".INX"),
    "IXIC": ("us_sina", ".IXIC"),
    "DJI": ("us_sina", ".DJI"),
    "CSI300": ("cn_sina", "sh000300"),
    "SSE": ("cn_sina", "sh000001"),
    "HSI": ("hk_sina", "HSI"),
    "N225": ("global_sina", "日经225指数"),
    "SX5E": ("global_sina", "欧洲Stoxx50指数"),
    "SENSEX": ("global_sina", "印度孟买SENSEX指数"),
}

_YIELD_COLS = {
    "US2Y": ("美国国债收益率2年",),
    "US10Y": ("美国国债收益率10年",),
    "CN2Y": ("中国国债收益率2年",),
    "CN10Y": ("中国国债收益率10年",),
}


class GlobalMacroAdapter:
    name = "global_macro"

    def fetch(self, *, start: date | None = None, end: date | None = None) -> list:
        rows = []
        rows.extend(self._fetch_indexes(start, end))
        rows.extend(self._fetch_yields(start, end))
        rows.extend(self._fetch_usdcny(start, end))
        rows.extend(self._fetch_sge_gold(start, end))
        rows.extend(self._fetch_gld(start, end))
        return rows

    def _fetch_indexes(self, start: date | None, end: date | None) -> list:
        import akshare as ak

        spec = lookup("idx_global")
        fetchers: dict[str, Callable[..., pd.DataFrame]] = {
            "us_sina": ak.index_us_stock_sina,
            "cn_sina": ak.stock_zh_index_daily,
            "hk_sina": ak.stock_hk_index_daily_sina,
            "global_sina": ak.index_global_hist_sina,
        }
        rows = []
        for entity_id, (kind, symbol) in _INDEX_SOURCES.items():
            fn = fetchers[kind]
            frame = safe_call(f"{kind}:{symbol}", fn, symbol=symbol)
            rows.extend(close_rows(frame, spec, entity_id, start, end))
        return rows

    def _fetch_yields(self, start: date | None, end: date | None) -> list:
        import akshare as ak

        spec = lookup("yield_cn_us")
        start_text = yyyymmdd(start) or "20180101"
        frame = safe_call("bond_zh_us_rate", ak.bond_zh_us_rate, start_date=start_text)
        if frame.empty:
            return []
        date_col = first_col(frame, ("日期", "SOLAR_DATE"))
        if date_col is None:
            return []
        rows = []
        for entity_id, candidates in _YIELD_COLS.items():
            value_col = first_col(frame, candidates)
            if value_col is None:
                continue
            for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
                day = parse_day(raw_day)
                value = finite_number(raw_value)
                if value is None or not in_range(day, start, end):
                    continue
                rows.append(row_of(spec, entity_id, day, "yield", value, unit="pct"))
        return rows

    def _fetch_usdcny(self, start: date | None, end: date | None) -> list:
        import akshare as ak

        spec = lookup("fx_usdcny")
        frame = safe_call("currency_boc_safe", ak.currency_boc_safe)
        if frame.empty:
            return []
        date_col = first_col(frame, ("日期", "date"))
        value_col = first_col(frame, ("美元", "美元兑人民币", "USD"))
        if date_col is None or value_col is None:
            return []
        rows = []
        for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            rows.append(row_of(spec, "USDCNY", day, "fx", value, unit="cny"))
        return rows

    def _fetch_sge_gold(self, start: date | None, end: date | None) -> list:
        import akshare as ak

        spec = lookup("gold_sge")
        frame = safe_call("spot_golden_benchmark_sge", ak.spot_golden_benchmark_sge)
        if frame.empty:
            return []
        date_col = first_col(frame, ("交易时间", "交易日期", "日期", "date"))
        value_col = first_col(frame, ("晚盘价", "早盘价", "收盘价", "价格"))
        if date_col is None or value_col is None:
            return []
        rows = []
        for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            rows.append(row_of(spec, "SGE_GOLD", day, "close", value, unit="cny/g"))
        return rows

    def _fetch_gld(self, start: date | None, end: date | None) -> list:
        import akshare as ak

        _log.info("拉取 SPDR 黄金 ETF 持仓（可能需数十秒）…")
        spec = lookup("gold_gld")
        frame = safe_call("macro_cons_gold", ak.macro_cons_gold)
        if frame.empty:
            return []
        date_col = first_col(frame, ("日期",))
        stock_col = first_col(frame, ("总库存",))
        change_col = first_col(frame, ("增持/减持", "增减持"))
        if date_col is None:
            return []
        rows = []
        for idx in range(len(frame)):
            day = parse_day(frame.iloc[idx][date_col])
            if not in_range(day, start, end):
                continue
            if stock_col is not None:
                stock = finite_number(frame.iloc[idx][stock_col])
                if stock is not None:
                    rows.append(row_of(spec, "GLD", day, "tonnes", stock, evidence_class="direct", unit="tonne"))
            if change_col is not None:
                change = finite_number(frame.iloc[idx][change_col])
                if change is not None:
                    rows.append(row_of(spec, "GLD", day, "change", change, evidence_class="direct", unit="tonne"))
        return rows
