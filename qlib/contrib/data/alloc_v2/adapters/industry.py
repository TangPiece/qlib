# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""申万一级行情：指数涨跌 ≠ 行业净流入。"""

from __future__ import annotations

from datetime import date

from ..schema import finite_number, first_col, in_range, parse_day
from .base import lookup, row_of, safe_call

# 申万一级指数代码
SW_L1 = {
    "801010": "农林牧渔",
    "801030": "化工",
    "801040": "钢铁",
    "801050": "有色金属",
    "801080": "电子",
    "801110": "家用电器",
    "801120": "食品饮料",
    "801150": "医药生物",
    "801160": "公用事业",
    "801170": "交通运输",
    "801180": "房地产",
    "801200": "商贸零售",
    "801210": "社会服务",
    "801230": "综合",
    "801710": "建筑材料",
    "801720": "建筑装饰",
    "801730": "电力设备",
    "801740": "国防军工",
    "801750": "计算机",
    "801760": "传媒",
    "801770": "通信",
    "801780": "银行",
    "801790": "非银金融",
    "801880": "汽车",
    "801890": "机械设备",
    "801950": "煤炭",
    "801960": "石油石化",
    "801970": "环保",
    "801980": "美容护理",
}


class IndustryAdapter:
    name = "industry"

    def fetch(self, *, start: date | None = None, end: date | None = None) -> list:
        import akshare as ak

        spec = lookup("sw_industry")
        rows = []
        for code, name in SW_L1.items():
            frame = safe_call(f"index_hist_sw:{code}", ak.index_hist_sw, symbol=code, period="day")
            if frame.empty:
                continue
            date_col = first_col(frame, ("日期", "date"))
            close_col = first_col(frame, ("收盘", "close"))
            amount_col = first_col(frame, ("成交额", "amount"))
            if date_col is None:
                continue
            for idx in range(len(frame)):
                day = parse_day(frame.iloc[idx][date_col])
                if not in_range(day, start, end):
                    continue
                if close_col is not None:
                    close = finite_number(frame.iloc[idx][close_col])
                    if close is not None:
                        rows.append(row_of(spec, code, day, "close", close, unit="index", source="akshare.index_hist_sw"))
                if amount_col is not None:
                    amount = finite_number(frame.iloc[idx][amount_col])
                    if amount is not None:
                        rows.append(row_of(spec, code, day, "amount", amount, unit="cny", source="akshare.index_hist_sw"))
            _ = name
        return rows
