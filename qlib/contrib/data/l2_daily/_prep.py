"""成交、委托、撤单的统一预处理。各因子组只从这里取明细。

价格是「元 × 10000」的整数，金额（元）= 价格 × 数量 / 10000。
集合竞价成交的 BS 标志没有主动方向含义（沪市收盘竞价全部标 S），方向类统计只用连续竞价。
沪市逐笔委托不含即时全部成交的主动单，部分成交的主动单只记剩余量，需要用成交补回。
深市撤单在逐笔成交里（成交代码 C），价格为 0，金额要按委托号回查委托价格。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .util import PRICE_SCALE, as_time, flag, market_of

from ._math import clock_seconds

CONT_START = "093000000"
CONT_END = "145700000"
CONT_START_SEC = 9 * 3600 + 30 * 60
CONT_END_SEC = 14 * 3600 + 57 * 60
AFTERNOON_START_SEC = 13 * 3600
# 09:30–11:30 加 13:00–14:57。
CONT_SECONDS = 2 * 3600 + (CONT_END_SEC - 13 * 3600)

TRADE_COLUMNS = ["clock", "sec", "price", "qty", "amount", "side", "buy_id", "sell_id"]
ORDER_COLUMNS = ["order_id", "side", "qty", "amount", "sec"]

_INVALID_IDS = {"", "0", "nan", "None", "NaN"}


def _ids(series: pd.Series) -> pd.Series:
    text = series.astype(str).str.replace("\x00", "", regex=False).str.strip()
    return text.str.replace(r"\.0$", "", regex=True)


def _valid_id(series: pd.Series) -> pd.Series:
    return ~series.isin(_INVALID_IDS)


def _column(frame: pd.DataFrame, name: str, default: str = "") -> pd.Series:
    if name in frame.columns:
        return frame[name]
    return pd.Series(default, index=frame.index)


def _numeric(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return pd.to_numeric(frame[name], errors="coerce")


def prepare_trades(trades: pd.DataFrame, code: str) -> pd.DataFrame:
    """全天有效成交：价格和数量为正；深市只留成交代码 0。按时间稳定排序。"""
    if trades is None or len(trades) == 0 or "时间" not in trades.columns:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    clock = as_time(trades["时间"])
    price_raw = _numeric(trades, "成交价格")
    qty = _numeric(trades, "成交数量")
    keep = (price_raw > 0) & (qty > 0)
    if market_of(code) == "SZ" and "成交代码" in trades.columns:
        keep = keep & flag(trades["成交代码"]).eq("0")
    frame = pd.DataFrame(
        {
            "clock": clock,
            "sec": clock_seconds(clock),
            "price": price_raw / PRICE_SCALE,
            "qty": qty,
            "amount": price_raw * qty / PRICE_SCALE,
            "side": flag(_column(trades, "BS标志")),
            "buy_id": _ids(_column(trades, "叫买序号")),
            "sell_id": _ids(_column(trades, "叫卖序号")),
        }
    )
    frame = frame.loc[keep]
    return frame.sort_values("sec", kind="mergesort").reset_index(drop=True)


def continuous(frame: pd.DataFrame) -> pd.DataFrame:
    """只留 09:30–14:57 的连续竞价成交。"""
    if frame is None or len(frame) == 0:
        return frame
    mask = (frame["clock"] >= CONT_START) & (frame["clock"] < CONT_END)
    return frame.loc[mask].reset_index(drop=True)


def signed(frame: pd.DataFrame) -> pd.Series:
    """主动买 +1，主动卖 -1，其余 0。"""
    return frame["side"].map({"B": 1.0, "S": -1.0}).fillna(0.0)


def minute_bars(frame: pd.DataFrame) -> pd.DataFrame:
    """按交易时段和分钟聚合，避免午休前后价格被当作一分钟收益。"""
    columns = ["session", "minute", "close", "amount", "signed_qty", "signed_amount"]
    if frame is None or len(frame) == 0:
        return pd.DataFrame(columns=columns)
    sign = signed(frame)
    work = pd.DataFrame(
        {
            "session": np.where(frame["sec"] < AFTERNOON_START_SEC, "am", "pm"),
            "minute": (frame["sec"] // 60),
            "price": frame["price"],
            "amount": frame["amount"],
            "signed_qty": sign * frame["qty"],
            "signed_amount": sign * frame["amount"],
        }
    ).dropna(subset=["minute"])
    bars = work.groupby(["session", "minute"], sort=False).agg(
        close=("price", "last"),
        amount=("amount", "sum"),
        signed_qty=("signed_qty", "sum"),
        signed_amount=("signed_amount", "sum"),
    )
    return bars.reset_index()[columns]


def _order_side(series: pd.Series) -> pd.Series:
    text = flag(series)
    return text.map({"B": "B", "1": "B", "S": "S", "2": "S"})


def _empty_orders() -> pd.DataFrame:
    return pd.DataFrame(columns=ORDER_COLUMNS)


def reconstruct_orders(orders: pd.DataFrame, trades: pd.DataFrame, code: str) -> pd.DataFrame:
    """每笔委托的原始方向、数量、金额与首次出现时间。

    深市逐笔委托完整，直接使用。沪市原始量 = A 记录量 + 该委托作为主动方的连续竞价成交量；
    委托流里没有的主动单从成交补出。竞价期间的委托都会挂入委托流，不需要补。
    """
    has_orders = isinstance(orders, pd.DataFrame) and len(orders) > 0
    if has_orders and "交易所委托号" in orders.columns:
        kind = flag(_column(orders, "委托类型"))
        side = _order_side(_column(orders, "委托代码"))
        qty = _numeric(orders, "委托数量")
        price = _numeric(orders, "委托价格")
        adds = pd.DataFrame(
            {
                "order_id": _ids(orders["交易所委托号"]),
                "side": side,
                "qty": qty,
                "amount": price * qty / PRICE_SCALE,
                "sec": clock_seconds(as_time(_column(orders, "时间", "0"))),
            }
        )
        keep = kind.ne("D") & side.notna() & (qty > 0)
        if market_of(code) == "SH":
            keep = keep & kind.eq("A")
        adds = adds.loc[keep & _valid_id(adds["order_id"])]
    else:
        adds = _empty_orders()

    fills = _aggressor_fills(trades, code)
    if market_of(code) == "SZ":
        book = adds
        if len(book) and len(fills):
            # 市价单委托价格为 0，金额用实际成交额补。
            filled = book["order_id"].map(fills.set_index("order_id")["amount"])
            book = book.assign(amount=book["amount"].where(book["amount"] > 0, filled))
        return book.reset_index(drop=True)[ORDER_COLUMNS]

    if len(fills) == 0:
        return adds.reset_index(drop=True)[ORDER_COLUMNS]
    if len(adds) == 0:
        return fills.reset_index(drop=True)[ORDER_COLUMNS]
    merged = adds.merge(fills, on="order_id", how="outer", suffixes=("", "_fill"))
    merged["side"] = merged["side"].fillna(merged["side_fill"])
    merged["qty"] = merged["qty"].fillna(0.0) + merged["qty_fill"].fillna(0.0)
    merged["amount"] = merged["amount"].fillna(0.0) + merged["amount_fill"].fillna(0.0)
    merged["sec"] = merged[["sec", "sec_fill"]].min(axis=1)
    return merged.reset_index(drop=True)[ORDER_COLUMNS]


def _aggressor_fills(trades: pd.DataFrame, code: str) -> pd.DataFrame:
    """连续竞价成交按主动方委托号汇总。"""
    frame = continuous(prepare_trades(trades, code))
    if frame is None or len(frame) == 0:
        return _empty_orders()
    buy = frame["side"].eq("B")
    sell = frame["side"].eq("S")
    order_id = pd.Series(np.where(buy, frame["buy_id"], np.where(sell, frame["sell_id"], "")), index=frame.index)
    work = frame.assign(order_id=order_id).loc[(buy | sell) & _valid_id(order_id)]
    if len(work) == 0:
        return _empty_orders()
    grouped = work.groupby("order_id", sort=False).agg(
        side=("side", "first"),
        qty=("qty", "sum"),
        amount=("amount", "sum"),
        sec=("sec", "min"),
    )
    return grouped.reset_index()[ORDER_COLUMNS]


def cancel_records(orders: pd.DataFrame, trades: pd.DataFrame, code: str) -> pd.DataFrame:
    """撤单明细。沪市取委托流 D 记录；深市取成交里的 C/D 行，价格按委托号回查。"""
    if market_of(code) == "SH":
        return _sh_cancels(orders)
    return _sz_cancels(orders, trades)


def _sh_cancels(orders: pd.DataFrame) -> pd.DataFrame:
    if orders is None or len(orders) == 0 or "委托类型" not in orders.columns:
        return _empty_orders()
    rows = orders.loc[flag(orders["委托类型"]).eq("D")]
    if len(rows) == 0:
        return _empty_orders()
    qty = _numeric(rows, "委托数量")
    price = _numeric(rows, "委托价格")
    frame = pd.DataFrame(
        {
            "order_id": _ids(_column(rows, "交易所委托号")),
            "side": _order_side(_column(rows, "委托代码")),
            "qty": qty,
            "amount": price * qty / PRICE_SCALE,
            "sec": clock_seconds(as_time(rows["时间"])),
        }
    )
    bad = ~(price > 0)  # 覆盖 NaN 与 <=0
    if bad.any():
        lookup = _order_prices(orders)
        fallback = frame["order_id"].map(lookup) * qty / PRICE_SCALE
        frame["amount"] = frame["amount"].where(~bad, fallback)
    return frame.loc[frame["side"].notna() & (frame["qty"] > 0)].reset_index(drop=True)


def _sz_cancels(orders: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    if trades is None or len(trades) == 0 or "成交代码" not in trades.columns:
        return _empty_orders()
    rows = trades.loc[flag(trades["成交代码"]).isin(["C", "D"])]
    if len(rows) == 0:
        return _empty_orders()
    buy_id = _ids(_column(rows, "叫买序号"))
    sell_id = _ids(_column(rows, "叫卖序号"))
    is_buy = _valid_id(buy_id)
    order_id = buy_id.where(is_buy, sell_id)
    side = pd.Series(np.where(is_buy, "B", np.where(_valid_id(sell_id), "S", None)), index=rows.index)
    qty = _numeric(rows, "成交数量")
    price = _order_prices(orders)
    frame = pd.DataFrame(
        {
            "order_id": order_id,
            "side": side,
            "qty": qty,
            "amount": order_id.map(price) * qty / PRICE_SCALE,
            "sec": clock_seconds(as_time(rows["时间"])),
        }
    )
    return frame.loc[frame["side"].notna() & (frame["qty"] > 0)].reset_index(drop=True)


def _order_prices(orders: pd.DataFrame) -> pd.Series:
    """委托号 → 委托价格（原始整数）。只看新增委托，价格为正。"""
    if orders is None or len(orders) == 0 or "交易所委托号" not in orders.columns:
        return pd.Series(dtype=float)
    kind = flag(_column(orders, "委托类型"))
    price = _numeric(orders, "委托价格")
    frame = pd.DataFrame({"order_id": _ids(orders["交易所委托号"]), "price": price})
    frame = frame.loc[kind.ne("D") & (price > 0)]
    return frame.drop_duplicates("order_id", keep="first").set_index("order_id")["price"]


def in_continuous_sec(frame: pd.DataFrame) -> pd.DataFrame:
    """按秒数筛 09:30–14:57。用于委托与撤单。"""
    if frame is None or len(frame) == 0:
        return frame
    mask = (frame["sec"] >= CONT_START_SEC) & (frame["sec"] < CONT_END_SEC)
    return frame.loc[mask].reset_index(drop=True)
