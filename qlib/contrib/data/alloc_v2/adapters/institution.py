# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""D 层：公募持仓、规模快照、龙虎榜、大宗、增减持。无数据保持 NA。"""

from __future__ import annotations

from calendar import monthrange
from datetime import date

from ..schema import finite_number, first_col, in_fetch_window, in_range, parse_day
from .base import lookup, row_of, safe_call


def quarter_period_end(year: int, quarter: int) -> date:
    """季度最后一个自然日。一季度是 3 月 31 日，不是 3 月 1 日。"""
    month = int(quarter) * 3
    return date(int(year), month, monthrange(int(year), month)[1])


def quarter_available_at(event_date: date) -> date:
    """披露日不早于季末次月：取次月 15 日，避免把季末当月写成可观测。

    公募季报大约在季末后 15 个工作日，次月 15 日仍偏早时由 quality_flag 禁止 PIT。
    """
    if event_date.month == 12:
        return date(event_date.year + 1, 1, 15)
    return date(event_date.year, event_date.month + 1, 15)


class InstitutionAdapter:
    name = "institution"

    def fetch(self, *, start: date | None = None, end: date | None = None) -> list:
        rows = []
        rows.extend(self._fetch_fund_scale(end))
        rows.extend(self._fetch_lhb(start, end))
        rows.extend(self._fetch_block(start, end))
        rows.extend(self._fetch_hold_change(start, end))
        rows.extend(self._fetch_fund_holding(start, end))
        return rows

    def _fetch_fund_scale(self, end: date | None) -> list:
        import akshare as ak

        spec = lookup("fund_scale_snapshot")
        asof = end or date.today()
        mapping = (("股票型基金", "equity"), ("债券型基金", "bond"), ("货币型基金", "money"))
        rows = []
        for symbol, entity in mapping:
            frame = safe_call(f"fund_scale_open_sina:{symbol}", ak.fund_scale_open_sina, symbol=symbol)
            if frame.empty:
                continue
            scale_col = first_col(frame, ("最新规模", "规模", "zmjgm", "基金规模"))
            if scale_col is None:
                for col in frame.columns:
                    if "规模" in str(col):
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
            rows.append(row_of(spec, entity, asof, "aum_snapshot", total, unit="cny_yi", available_at=asof))
        return rows

    def _fetch_lhb(self, start: date | None, end: date | None) -> list:
        """龙虎榜只在披露日写值；无表则不写 0。"""
        import akshare as ak

        spec = lookup("lhb")
        last = end or date.today()
        first = start or last
        frame = safe_call(
            "stock_lhb_detail_em",
            ak.stock_lhb_detail_em,
            start_date=first.strftime("%Y%m%d"),
            end_date=last.strftime("%Y%m%d"),
        )
        if frame.empty:
            return []
        date_col = first_col(frame, ("上榜日", "日期", "交易日期"))
        code_col = first_col(frame, ("代码", "股票代码"))
        inst_col = first_col(frame, ("机构买入合计", "买方机构", "机构专用"))
        amount_col = first_col(frame, ("龙虎榜成交额", "成交额", "买入额"))
        if date_col is None or code_col is None:
            return []
        rows = []
        for idx in range(len(frame)):
            day = parse_day(frame.iloc[idx][date_col])
            if not in_range(day, start, end):
                continue
            code = str(frame.iloc[idx][code_col]).split(".")[0].zfill(6)
            inst = finite_number(frame.iloc[idx][inst_col]) if inst_col else None
            amount = finite_number(frame.iloc[idx][amount_col]) if amount_col else None
            # 缺机构金额保持 NA，不填 0
            if inst is not None and amount not in (None, 0.0):
                rows.append(row_of(spec, code, day, "inst_buy", inst, unit="cny", evidence_class="proxy"))
                rows.append(row_of(spec, code, day, "lhb_amount", amount, unit="cny", evidence_class="proxy"))
            elif inst is not None:
                rows.append(row_of(spec, code, day, "inst_buy", inst, unit="cny", evidence_class="proxy"))
        return rows

    def _fetch_block(self, start: date | None, end: date | None) -> list:
        import akshare as ak

        spec = lookup("block_trade")
        last = end or date.today()
        first = start or last
        frame = safe_call(
            "stock_dzjy_mrmx",
            ak.stock_dzjy_mrmx,
            start_date=first.strftime("%Y%m%d"),
            end_date=last.strftime("%Y%m%d"),
        )
        if frame.empty:
            return []
        date_col = first_col(frame, ("交易日期", "日期"))
        code_col = first_col(frame, ("证券代码", "代码"))
        prem_col = first_col(frame, ("折溢价率", "溢价率"))
        amt_col = first_col(frame, ("成交额", "成交金额"))
        if date_col is None or code_col is None:
            return []
        rows = []
        for idx in range(len(frame)):
            day = parse_day(frame.iloc[idx][date_col])
            if not in_range(day, start, end):
                continue
            code = str(frame.iloc[idx][code_col]).split(".")[0].zfill(6)
            if prem_col is not None:
                prem = finite_number(frame.iloc[idx][prem_col])
                if prem is not None:
                    rows.append(row_of(spec, code, day, "premium", prem, unit="pct", evidence_class="proxy"))
            if amt_col is not None:
                amt = finite_number(frame.iloc[idx][amt_col])
                if amt is not None:
                    rows.append(row_of(spec, code, day, "amount", amt, unit="cny", evidence_class="proxy"))
        return rows

    def _fetch_hold_change(self, start: date | None, end: date | None) -> list:
        """增减持/回购公告骨架：能解析多少算多少。"""
        import akshare as ak

        spec = lookup("holder_event")
        frame = safe_call("stock_hold_control_cninfo", getattr(ak, "stock_hold_control_cninfo", lambda **_: None))
        if frame is None or getattr(frame, "empty", True):
            return []
        date_col = first_col(frame, ("公告日期", "日期"))
        code_col = first_col(frame, ("代码", "证券代码"))
        if date_col is None or code_col is None:
            return []
        rows = []
        for idx in range(len(frame)):
            day = parse_day(frame.iloc[idx][date_col])
            if not in_range(day, start, end):
                continue
            code = str(frame.iloc[idx][code_col]).split(".")[0].zfill(6)
            rows.append(row_of(spec, code, day, "event", 1.0, unit="flag", available_at=day, evidence_class="direct"))
        return rows

    def _fetch_fund_holding(self, start: date | None, end: date | None) -> list:
        """公募重仓：尝试最近几个季报截面，写 asof 与 available_at。"""
        import akshare as ak

        spec = lookup("fund_holding")
        asof = end or date.today()
        quarter = (asof.month - 1) // 3 + 1
        year, q = asof.year, quarter
        candidates = []
        for _ in range(6):
            candidates.append(f"{year}{q}")
            q -= 1
            if q <= 0:
                q = 4
                year -= 1
        fn = getattr(ak, "stock_institute_hold", None)
        if fn is None:
            return []
        rows = []
        for symbol in candidates:
            year, quarter = int(symbol[:4]), int(symbol[4])
            event = quarter_period_end(year, quarter)
            # 行数不是持仓明细：可观测日放到季末次月，并标 unavailable，禁止当 PIT 持仓
            avail = quarter_available_at(event)
            if not in_fetch_window(event, available_at=avail, freq="quarter", start=start, end=end):
                continue
            frame = safe_call(f"stock_institute_hold:{symbol}", fn, symbol=symbol)
            if frame.empty:
                continue
            rows.append(
                row_of(
                    spec,
                    "ALL",
                    event,
                    "n_names",
                    float(len(frame)),
                    available_at=avail,
                    announcement_date=avail,
                    unit="count",
                    quality_flag="unavailable",
                    source=f"{spec.source}:{symbol}:n_names_not_holdings",
                )
            )
        return rows
