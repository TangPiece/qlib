# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""中国市场结构：北向、融资、开户、保险、央行储备、公募规模快照。"""

from __future__ import annotations

import logging
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

_log = logging.getLogger(__name__)


class ChinaFlowAdapter:
    """跨境与机构数据频率不一；月/季序列保持原日期，不伪装成日频。"""

    name = "china_flow"

    def fetch(self, *, start: date | None = None, end: date | None = None) -> list[FlowRow]:
        rows: list[FlowRow] = []
        rows.extend(self._fetch_hsgt(start, end))
        rows.extend(self._fetch_margin(start, end))
        rows.extend(self._fetch_accounts(start, end))
        rows.extend(self._fetch_insurance(start, end))
        rows.extend(self._fetch_pboc_gold(start, end))
        rows.extend(self._fetch_money(start, end))
        rows.extend(self._fetch_lpr(start, end))
        rows.extend(self._fetch_fund_scale_snapshots(end))
        rows.extend(self._fetch_institute_snapshot(end))
        return rows

    def _fetch_hsgt(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        rows: list[FlowRow] = []
        mapping = (
            ("北向资金", "hsgt_north_net", "hsgt_north_hold_mcap"),
            ("南向资金", "hsgt_south_net", ""),
        )
        for symbol, net_id, hold_id in mapping:
            frame = safe_call(f"stock_hsgt_hist_em:{symbol}", ak.stock_hsgt_hist_em, symbol=symbol)
            if frame.empty:
                continue
            date_col = first_col(frame, ("日期",))
            net_col = first_col(frame, ("当日成交净买额", "当日资金流入"))
            hold_col = first_col(frame, ("持股市值",))
            if date_col is None:
                continue
            net_spec = lookup(net_id)
            hold_spec = lookup(hold_id) if hold_id else None
            for idx in range(len(frame)):
                day = parse_day(frame.iloc[idx][date_col])
                if not in_range(day, start, end):
                    continue
                if net_col is not None:
                    net = finite_number(frame.iloc[idx][net_col])
                    if net is not None:
                        rows.append(row_from_def(net_spec, day, net))
                if hold_spec is not None and hold_col is not None:
                    hold = finite_number(frame.iloc[idx][hold_col])
                    if hold is not None:
                        rows.append(row_from_def(hold_spec, day, hold))
        return rows

    def _fetch_margin(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        rows: list[FlowRow] = []
        for series_id, fn_name in (
            ("margin_sh_balance", "macro_china_market_margin_sh"),
            ("margin_sz_balance", "macro_china_market_margin_sz"),
        ):
            fn = getattr(ak, fn_name)
            frame = safe_call(fn_name, fn)
            if frame.empty:
                continue
            spec = lookup(series_id)
            date_col = first_col(frame, ("日期", "信用交易日期"))
            value_col = first_col(frame, ("融资余额",))
            if date_col is None or value_col is None:
                continue
            for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
                day = parse_day(raw_day)
                value = finite_number(raw_value)
                if value is None or not in_range(day, start, end):
                    continue
                rows.append(row_from_def(spec, day, value))
        return rows

    def _fetch_accounts(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        frame = safe_call("stock_account_statistics_em", ak.stock_account_statistics_em)
        if frame.empty:
            return []
        date_col = first_col(frame, ("数据日期", "日期"))
        value_col = first_col(frame, ("新增投资者-数量", "新增投资者", "新增投资者-户"))
        if date_col is None or value_col is None:
            return []
        spec = lookup("account_new")
        rows: list[FlowRow] = []
        for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            rows.append(row_from_def(spec, day, value))
        return rows

    def _fetch_insurance(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        frame = safe_call("macro_china_insurance_income", ak.macro_china_insurance_income)
        if frame.empty:
            return []
        date_col = first_col(frame, ("月份", "日期", "统计时间"))
        value_col = first_col(frame, ("原保险保费收入", "当月", "数值"))
        if date_col is None or value_col is None:
            return []
        spec = lookup("insurance_income")
        rows: list[FlowRow] = []
        for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            rows.append(row_from_def(spec, day, value))
        return rows

    def _fetch_pboc_gold(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        frame = safe_call("macro_china_foreign_exchange_gold", ak.macro_china_foreign_exchange_gold)
        if frame.empty:
            return []
        date_col = first_col(frame, ("日期", "月份"))
        value_col = first_col(frame, ("黄金储备", "黄金", "外汇储备"))
        if date_col is None or value_col is None:
            return []
        spec = lookup("pboc_gold_fx")
        rows: list[FlowRow] = []
        for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            rows.append(row_from_def(spec, day, value))
        return rows

    def _fetch_money(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        frame = safe_call("macro_china_money_supply", ak.macro_china_money_supply)
        if frame.empty:
            return []
        date_col = first_col(frame, ("月份", "日期"))
        value_col = first_col(frame, ("货币和准货币(M2)-数量", "M2", "货币和准货币"))
        if date_col is None or value_col is None:
            return []
        spec = lookup("cn_m2")
        rows: list[FlowRow] = []
        for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            rows.append(row_from_def(spec, day, value))
        return rows

    def _fetch_lpr(self, start: date | None, end: date | None) -> list[FlowRow]:
        import akshare as ak

        frame = safe_call("macro_china_lpr", ak.macro_china_lpr)
        if frame.empty:
            return []
        date_col = first_col(frame, ("TRADE_DATE", "日期"))
        value_col = first_col(frame, ("LPR1Y", "1年期LPR", "LPR_1Y"))
        if date_col is None or value_col is None:
            return []
        spec = lookup("cn_lpr_1y")
        rows: list[FlowRow] = []
        for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            rows.append(row_from_def(spec, day, value))
        return rows

    def _fetch_fund_scale_snapshots(self, end: date | None) -> list[FlowRow]:
        """新浪开放式基金规模是截面快照，只写观测日一条，不当历史日频。"""
        import akshare as ak

        asof = end or date.today()
        mapping = (
            ("股票型基金", "fund_scale_equity_snapshot"),
            ("债券型基金", "fund_scale_bond_snapshot"),
            ("货币型基金", "fund_scale_money_snapshot"),
        )
        rows: list[FlowRow] = []
        for symbol, series_id in mapping:
            frame = safe_call(f"fund_scale_open_sina:{symbol}", ak.fund_scale_open_sina, symbol=symbol)
            if frame.empty:
                continue
            scale_col = first_col(frame, ("最新规模", "规模", "zmjgm", "基金规模"))
            if scale_col is None:
                # 常见列名是中文「最新规模」或数值列 zmjgm
                for col in frame.columns:
                    if "规模" in str(col) or str(col).lower() in {"zmjgm", "ljjz"}:
                        scale_col = col
                        break
            if scale_col is None:
                continue
            total = 0.0
            count = 0
            for raw in frame[scale_col]:
                value = finite_number(raw)
                if value is None:
                    continue
                total += value
                count += 1
            if count == 0:
                continue
            spec = lookup(series_id)
            rows.append(row_from_def(spec, asof, total, asof=asof))
        return rows

    def _fetch_institute_snapshot(self, end: date | None) -> list[FlowRow]:
        import akshare as ak

        # symbol 形如 20051 = 2020 年 1 季；用最近可得年季尝试几次
        asof = end or date.today()
        quarter = (asof.month - 1) // 3 + 1
        candidates = []
        year, q = asof.year, quarter
        for _ in range(6):
            candidates.append(f"{year}{q}")
            q -= 1
            if q <= 0:
                q = 4
                year -= 1
        spec = lookup("institute_hold_snapshot")
        for symbol in candidates:
            frame = safe_call(f"stock_institute_hold:{symbol}", ak.stock_institute_hold, symbol=symbol)
            if frame.empty:
                continue
            return [row_from_def(spec, asof, float(len(frame)), asof=asof, source=f"{spec.source}:{symbol}")]
        return []
