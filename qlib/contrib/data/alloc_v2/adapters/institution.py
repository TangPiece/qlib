# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""D 层：公募持仓、规模快照、龙虎榜、大宗、增减持。无数据保持 NA。"""

from __future__ import annotations

from datetime import date, timedelta

from ..schema import finite_number, first_col, in_range, parse_day
from .base import lookup, row_of, safe_call


def _quarter_available_at(event_date: date) -> date:
    """季报最早按披露日后约 30 日可观测；缺公告日时用季末 + 30。"""
    return event_date + timedelta(days=30)


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
        for symbol in candidates:
            frame = safe_call(f"stock_institute_hold:{symbol}", fn, symbol=symbol)
            if frame.empty:
                continue
            event = date(int(symbol[:4]), int(symbol[4]) * 3, 1)
            return [
                row_of(
                    spec,
                    "ALL",
                    event,
                    "n_names",
                    float(len(frame)),
                    available_at=_quarter_available_at(event),
                    announcement_date=_quarter_available_at(event),
                    unit="count",
                    source=f"{spec.source}:{symbol}",
                )
            ]
        return []
