"""L2 计算用到的价格尺度、市场判断和时间列整理。"""
from __future__ import annotations

import pandas as pd

# 大单：按委托号聚合成交额，达到该金额（元）才计入大单。
BIG_ORDER_AMOUNT = 1_000_000.0
# 原始价格是「元 × 10000」的整数。
PRICE_SCALE = 10_000.0

CONTINUOUS_START = "092500000"
OPEN_AUCTION_END = "093000000"
CONTINUOUS_QUOTE_START = "093000000"
CONTINUOUS_QUOTE_END = "145700000"
OPEN_30_END = "100000000"
TAIL_30_START = "143000000"


def market_of(code: str) -> str:
    """6/9 开头为沪市（含科创板），其余深市。"""
    text = str(code).split(".")[0]
    return "SH" if text[:1] in "69" else "SZ"


def is_equity(code: str) -> bool:
    """A 股：深市 0/3、沪市 6。基金和可转债不进入因子。"""
    text = str(code).split(".")[0]
    return len(text) == 6 and text[:1] in ("0", "3", "6")


def as_time(series: pd.Series) -> pd.Series:
    """补齐到 9 位。上午小时经常丢前导 0。"""
    text = series.astype(str).str.replace("\x00", "", regex=False).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    return text.str.zfill(9)


def flag(series: pd.Series) -> pd.Series:
    """去掉空字符和空白。"""
    return series.astype(str).str.replace("\x00", "", regex=False).str.strip()
