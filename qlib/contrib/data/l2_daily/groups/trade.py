"""成交组：一次扫描产出基础统计、方向、大单（因子 1–42）。

基础统计、大单规模与集中度用全天成交（含开盘、收盘集合竞价）。
方向类与大单方向只用连续竞价成交，集合竞价的 BS 标志不代表主动方。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..util import BIG_ORDER_AMOUNT

from .._math import empty_nan, finite, imbalance_ratio, safe_div, series_stat
from .._prep import continuous, prepare_trades
from ..catalog import factor_ids_for_group

_KEYS = factor_ids_for_group("trade")
_INVALID_IDS = {"", "0", "nan", "None", "NaN"}


def calc_trade_factors(trades: pd.DataFrame, code: str) -> dict[str, float]:
    out = empty_nan(_KEYS)
    frame = prepare_trades(trades, code)
    if len(frame) == 0:
        return out

    _basic(frame, out)
    cont = continuous(frame)
    if len(cont):
        _direction(cont, out)
    _large(frame, cont, out)
    _concentration(frame, out)
    return out


def _basic(frame: pd.DataFrame, out: dict[str, float]) -> None:
    qty = frame["qty"]
    price = frame["price"]
    amount = frame["amount"]
    total_vol = float(qty.sum())
    total_amt = float(amount.sum())
    out["l2_trade_count"] = float(len(frame))
    out["l2_trade_volume"] = finite(total_vol)
    out["l2_trade_amount"] = finite(total_amt)
    out["l2_trade_volume_mean"] = series_stat(qty, "mean")
    out["l2_trade_volume_std"] = series_stat(qty, "std")
    out["l2_trade_volume_max"] = series_stat(qty, "max")
    out["l2_trade_amount_mean"] = series_stat(amount, "mean")
    out["l2_trade_amount_std"] = series_stat(amount, "std")
    out["l2_trade_amount_max"] = series_stat(amount, "max")
    out["l2_trade_amount_median"] = series_stat(amount, "median")
    out["l2_trade_amount_skew"] = series_stat(amount, "skew")
    out["l2_trade_amount_kurt"] = series_stat(amount, "kurt")
    out["l2_trade_price_mean"] = series_stat(price, "mean")
    out["l2_trade_price_std"] = series_stat(price, "std")
    vwap = safe_div(total_amt, total_vol)
    out["l2_vwap"] = vwap
    out["l2_vwap_deviation"] = safe_div(float(price.iloc[-1]) - vwap, vwap)


def _direction(cont: pd.DataFrame, out: dict[str, float]) -> None:
    # 缺 BS 时不能把主动占比静默写成 0，整组方向因子保持 NaN。
    if not cont["side"].isin(["B", "S"]).any():
        return
    qty = cont["qty"]
    amount = cont["amount"]
    buy = cont["side"].eq("B")
    sell = cont["side"].eq("S")
    total_amt = float(amount.sum())
    buy_vol = float(qty.loc[buy].sum())
    sell_vol = float(qty.loc[sell].sum())
    buy_amt = float(amount.loc[buy].sum())
    sell_amt = float(amount.loc[sell].sum())
    buy_n = float(buy.sum())
    sell_n = float(sell.sum())
    out["l2_active_buy_volume"] = finite(buy_vol)
    out["l2_active_sell_volume"] = finite(sell_vol)
    out["l2_active_buy_amount"] = finite(buy_amt)
    out["l2_active_sell_amount"] = finite(sell_amt)
    out["l2_active_buy_count"] = buy_n
    out["l2_active_sell_count"] = sell_n
    out["l2_active_buy_amount_ratio"] = safe_div(buy_amt, total_amt)
    out["l2_active_sell_amount_ratio"] = safe_div(sell_amt, total_amt)
    out["l2_trade_amount_imbalance"] = finite(buy_amt - sell_amt)
    out["l2_trade_volume_imbalance"] = finite(buy_vol - sell_vol)
    out["l2_trade_amount_imbalance_ratio"] = imbalance_ratio(buy_amt, sell_amt)
    out["l2_trade_volume_imbalance_ratio"] = imbalance_ratio(buy_vol, sell_vol)
    out["l2_trade_count_imbalance"] = finite(buy_n - sell_n)
    out["l2_trade_count_imbalance_ratio"] = imbalance_ratio(buy_n, sell_n)


def _order_totals(frame: pd.DataFrame, column: str) -> pd.Series:
    ids = frame[column]
    valid = ~ids.isin(_INVALID_IDS)
    return frame.loc[valid].groupby(column)["amount"].sum()


def _large_flags(frame: pd.DataFrame, buy_totals: pd.Series, sell_totals: pd.Series) -> tuple[pd.Series, pd.Series]:
    """按委托累计成交额标记每笔成交的买方、卖方是否为大单。"""
    large_buy = frame["buy_id"].map(buy_totals).fillna(0.0) >= BIG_ORDER_AMOUNT
    large_sell = frame["sell_id"].map(sell_totals).fillna(0.0) >= BIG_ORDER_AMOUNT
    return large_buy, large_sell


def _large(frame: pd.DataFrame, cont: pd.DataFrame, out: dict[str, float]) -> None:
    buy_totals = _order_totals(frame, "buy_id")
    sell_totals = _order_totals(frame, "sell_id")
    large_buy, large_sell = _large_flags(frame, buy_totals, sell_totals)
    large = large_buy | large_sell
    total_amt = float(frame["amount"].sum())
    total_vol = float(frame["qty"].sum())
    large_amt = float(frame.loc[large, "amount"].sum())
    large_vol = float(frame.loc[large, "qty"].sum())
    out["l2_large_trade_count"] = float(large.sum())
    out["l2_large_trade_volume"] = finite(large_vol)
    out["l2_large_trade_amount"] = finite(large_amt)
    out["l2_large_trade_amount_ratio"] = safe_div(large_amt, total_amt)
    out["l2_large_trade_volume_ratio"] = safe_div(large_vol, total_vol)

    if len(cont) == 0:
        return
    cont_buy, cont_sell = _large_flags(cont, buy_totals, sell_totals)
    active_buy = float(cont.loc[cont["side"].eq("B") & cont_buy, "amount"].sum())
    active_sell = float(cont.loc[cont["side"].eq("S") & cont_sell, "amount"].sum())
    net = active_buy - active_sell
    out["l2_large_active_buy_amount"] = finite(active_buy)
    out["l2_large_active_sell_amount"] = finite(active_sell)
    out["l2_large_net_buy_amount"] = finite(net)
    out["l2_large_net_buy_amount_ratio"] = safe_div(net, float(cont["amount"].sum()))


def _concentration(frame: pd.DataFrame, out: dict[str, float]) -> None:
    amount = frame["amount"]
    total_amt = float(amount.sum())
    out["l2_max_trade_amount_ratio"] = safe_div(float(amount.max()), total_amt)
    out["l2_top5_trade_amount_ratio"] = safe_div(float(amount.nlargest(5).sum()), total_amt)
    if total_amt > 0:
        shares = (amount / total_amt).to_numpy(dtype=float)
        out["l2_trade_amount_hhi"] = finite(float(np.sum(shares * shares)))
