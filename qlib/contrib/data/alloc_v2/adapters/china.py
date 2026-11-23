# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""B/D 中国市场：北向拆成交与持股，融资与宏观。净买额只作对照。"""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, timedelta

from ..schema import finite_number, first_col, in_fetch_window, in_range, parse_day
from .base import lookup, row_of, safe_call


def parse_cn_month(value) -> date | None:
    """解析「2026年09月份」或「201501」为该月最后一天；失败再退回 parse_day。"""
    if value is None:
        return None
    text = str(value).strip()
    match = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return date(year, month, monthrange(year, month)[1])
    match = re.fullmatch(r"(\d{4})(\d{2})", text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return date(year, month, monthrange(year, month)[1])
    return parse_day(value)


def _macro_available_at(event_day: date, *, lag_days: int = 15) -> date:
    """月频宏观：保守按观察月末 + lag 才可进入回测（非精确发布时间）。"""
    return event_day + timedelta(days=lag_days)


class ChinaAdapter:
    name = "china"

    def fetch(self, *, start: date | None = None, end: date | None = None) -> list:
        rows = []
        rows.extend(self._fetch_hsgt(start, end))
        rows.extend(self._fetch_margin(start, end))
        rows.extend(self._fetch_macro(start, end))
        return rows

    def _fetch_hsgt(self, start: date | None, end: date | None) -> list:
        """北向/南向：核心写成交额与持股，净买额标 reference_only。"""
        import akshare as ak

        turn_spec = lookup("hsgt_turnover")
        hold_spec = lookup("hsgt_hold")
        net_spec = lookup("hsgt_net")
        rows = []
        mapping = (("北向资金", "NORTH"), ("南向资金", "SOUTH"))
        for symbol, entity in mapping:
            frame = safe_call(f"stock_hsgt_hist_em:{symbol}", ak.stock_hsgt_hist_em, symbol=symbol)
            if frame.empty:
                continue
            date_col = first_col(frame, ("日期",))
            amount_col = first_col(frame, ("当日成交额", "成交额"))
            hold_col = first_col(frame, ("持股市值",))
            net_col = first_col(frame, ("当日成交净买额", "当日资金流入"))
            if date_col is None:
                continue
            for idx in range(len(frame)):
                day = parse_day(frame.iloc[idx][date_col])
                if not in_range(day, start, end):
                    continue
                if amount_col is not None:
                    amount = finite_number(frame.iloc[idx][amount_col])
                    if amount is not None:
                        rows.append(row_of(turn_spec, entity, day, "amount", amount, unit="cny", evidence_class="proxy"))
                if hold_col is not None:
                    hold = finite_number(frame.iloc[idx][hold_col])
                    if hold is not None:
                        rows.append(row_of(hold_spec, entity, day, "hold_mcap", hold, unit="cny", evidence_class="estimate"))
                # 净买额入库但 role=reference_only，因子层不会当核心
                if net_col is not None:
                    net = finite_number(frame.iloc[idx][net_col])
                    if net is not None:
                        rows.append(
                            row_of(
                                net_spec,
                                entity,
                                day,
                                "net_buy",
                                net,
                                unit="cny_yi",
                                evidence_class="estimate",
                                quality_flag="ok",
                            )
                        )
        return rows

    def _fetch_margin(self, start: date | None, end: date | None) -> list:
        import akshare as ak

        spec = lookup("margin")
        rows = []
        for entity, fn_name in (("SH", "macro_china_market_margin_sh"), ("SZ", "macro_china_market_margin_sz")):
            fn = getattr(ak, fn_name)
            frame = safe_call(fn_name, fn)
            if frame.empty:
                continue
            date_col = first_col(frame, ("日期", "信用交易日期"))
            value_col = first_col(frame, ("融资余额",))
            if date_col is None or value_col is None:
                continue
            for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
                day = parse_day(raw_day)
                value = finite_number(raw_value)
                if value is None or not in_range(day, start, end):
                    continue
                rows.append(row_of(spec, entity, day, "balance", value, unit="cny"))
        return rows

    def _fetch_macro(self, start: date | None, end: date | None) -> list:
        """货币供给 / 社融增量 / PMI / LPR。月频用月末 event_date + 保守 available_at。"""
        import akshare as ak

        spec = lookup("cn_macro")
        rows = []
        money = safe_call("macro_china_money_supply", ak.macro_china_money_supply)
        if not money.empty:
            date_col = first_col(money, ("月份", "日期"))
            field_map = (
                ("M2", "m2", ("货币和准货币(M2)-数量", "M2", "货币和准货币"), "cny"),
                ("M2", "m2_yoy", ("货币和准货币(M2)-同比",), "pct"),
                ("M1", "m1", ("货币(M1)-数量", "M1"), "cny"),
                ("M1", "m1_yoy", ("货币(M1)-同比",), "pct"),
            )
            if date_col is not None:
                for entity, field, candidates, unit in field_map:
                    value_col = first_col(money, candidates)
                    if value_col is None:
                        continue
                    for raw_day, raw_value in zip(money[date_col], money[value_col]):
                        day = parse_cn_month(raw_day)
                        value = finite_number(raw_value)
                        # 月频按发布日或月末落窗，避免增量 7 日重叠漏掉上月数
                        avail = _macro_available_at(day) if day is not None else None
                        if value is None or not in_fetch_window(
                            day, available_at=avail, freq="month", start=start, end=end
                        ):
                            continue
                        rows.append(
                            row_of(
                                spec,
                                entity,
                                day,
                                field,
                                value,
                                unit=unit,
                                available_at=avail,
                            )
                        )

        # 社融规模增量（非存量）——字段与实体名显式标明 NEW
        socfin = safe_call("macro_china_shrzgm", ak.macro_china_shrzgm)
        if not socfin.empty:
            date_col = first_col(socfin, ("月份", "日期"))
            value_col = first_col(socfin, ("社会融资规模增量", "社融增量"))
            if date_col is not None and value_col is not None:
                for raw_day, raw_value in zip(socfin[date_col], socfin[value_col]):
                    day = parse_cn_month(raw_day)
                    value = finite_number(raw_value)
                    avail = _macro_available_at(day) if day is not None else None
                    if value is None or not in_fetch_window(
                        day, available_at=avail, freq="month", start=start, end=end
                    ):
                        continue
                    rows.append(
                        row_of(
                            spec,
                            "SOCFIN_NEW",
                            day,
                            "amount",
                            value,
                            unit="cny_yi",
                            available_at=avail,
                        )
                    )

        pmi = safe_call("macro_china_pmi", ak.macro_china_pmi)
        if not pmi.empty:
            date_col = first_col(pmi, ("月份", "日期"))
            manu_col = first_col(pmi, ("制造业-指数", "制造业"))
            if date_col is not None and manu_col is not None:
                for raw_day, raw_value in zip(pmi[date_col], pmi[manu_col]):
                    day = parse_cn_month(raw_day)
                    value = finite_number(raw_value)
                    # PMI 通常月初公布上月值，lag 取 3 日保守估计
                    avail = _macro_available_at(day, lag_days=3) if day is not None else None
                    if value is None or not in_fetch_window(
                        day, available_at=avail, freq="month", start=start, end=end
                    ):
                        continue
                    rows.append(
                        row_of(
                            spec,
                            "PMI_MANU",
                            day,
                            "pmi",
                            value,
                            unit="index",
                            available_at=avail,
                        )
                    )

        lpr = safe_call("macro_china_lpr", ak.macro_china_lpr)
        if not lpr.empty:
            date_col = first_col(lpr, ("TRADE_DATE", "日期"))
            value_col = first_col(lpr, ("LPR1Y", "1年期LPR", "LPR_1Y"))
            if date_col is not None and value_col is not None:
                for raw_day, raw_value in zip(lpr[date_col], lpr[value_col]):
                    day = parse_day(raw_day)
                    value = finite_number(raw_value)
                    # LPR 挂在月频 cn_macro 上，增量同样按月窗口保留最近公布
                    if value is None or not in_fetch_window(
                        day, available_at=day, freq="month", start=start, end=end
                    ):
                        continue
                    rows.append(row_of(spec, "LPR1Y", day, "lpr", value, unit="pct"))
        return rows
