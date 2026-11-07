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


def canonical_symbol(code: str) -> str:
    """统一成 600000.SH / 000001.SZ，避免万得目录大小写把同一只股票拆成两行。"""
    num = str(code).split(".")[0]
    return f"{num}.{market_of(num)}"


def is_equity(code: str) -> bool:
    """A 股：深市 0/3、沪市 6。基金和可转债不进入因子。"""
    text = str(code).split(".")[0]
    return len(text) == 6 and text[:1] in ("0", "3", "6")


def as_time(series: pd.Series) -> pd.Series:
    """归一成 9 位 HHMMSSmmm。

    深市行情多为 8–9 位（已含毫秒），左补零即可。
    沪市部分日只给 HHMMSS（≤6 位）；整数读入后再 zfill(9) 会变成 000091500，
    连续竞价字符串窗 093000000 会把全日滤光。
    规则：≤6 位，或已是 000HHMMSS 左填充形态 → 按 HHMMSS 右补毫秒 000。
    """
    text = series.astype(str).str.replace("\x00", "", regex=False).str.strip()
    text = text.str.replace(r"\.0$", "", regex=True)
    # 只留数字，避免 '91500.0' / 杂字符干扰长度判断
    digits = text.str.replace(r"\D", "", regex=True)
    short = digits.str.len().le(6)
    # 读入后已是 9 位、但实为 HHMMSS 左补三零（如 000091500）
    left_padded = digits.str.match(r"^000\d{6}$", na=False)
    out = digits.str.zfill(9)
    fix = short | left_padded
    # 短戳先补到 6 位再接毫秒；左填充形态取末 6 位
    hhmmss = digits.where(~left_padded, digits.str[-6:]).where(~short, digits.str.zfill(6))
    out.loc[fix] = hhmmss.loc[fix] + "000"
    return out


def flag(series: pd.Series) -> pd.Series:
    """去掉空字符和空白。"""
    return series.astype(str).str.replace("\x00", "", regex=False).str.strip()
