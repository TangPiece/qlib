"""委托与撤单（因子 65–78）。只统计 09:30–14:57 进入的委托和撤单。

沪市委托量用成交补回即时成交的主动单，深市撤单金额按委托号回查委托价格，
两市口径才可比。撤单率 = 撤单量 / 委托量（委托量已包含后来被撤的单）。
"""
from __future__ import annotations

import pandas as pd

from .._math import empty_nan, finite, imbalance_ratio, safe_div
from .._prep import cancel_records, in_continuous_sec, reconstruct_orders
from ..catalog import factor_ids_for_group

_KEYS = factor_ids_for_group("order")


def _side_sums(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    if frame is None or len(frame) == 0:
        return 0.0, 0.0, 0.0, 0.0
    buy = frame["side"].eq("B")
    sell = frame["side"].eq("S")
    amount = pd.to_numeric(frame["amount"], errors="coerce")
    qty = pd.to_numeric(frame["qty"], errors="coerce").fillna(0.0)

    def amount_sum(mask: pd.Series) -> float:
        selected = amount.loc[mask]
        if len(selected) == 0:
            return 0.0
        if selected.isna().any():
            return float("nan")
        return float(selected.sum())

    return (
        float(qty.loc[buy].sum()),
        float(qty.loc[sell].sum()),
        amount_sum(buy),
        amount_sum(sell),
    )


def calc_order_factors(
    orders: pd.DataFrame,
    trades: pd.DataFrame,
    code: str,
) -> dict[str, float]:
    out = empty_nan(_KEYS)
    orders = orders if isinstance(orders, pd.DataFrame) else pd.DataFrame()
    trades = trades if isinstance(trades, pd.DataFrame) else pd.DataFrame()
    if len(orders) == 0 and len(trades) == 0:
        return out

    book = in_continuous_sec(reconstruct_orders(orders, trades, code))
    cancels = in_continuous_sec(cancel_records(orders, trades, code))
    if len(cancels):
        # 分子和分母必须来自同一批连续竞价委托；竞价时挂入、盘中撤掉的单不混入。
        cancels = cancels.loc[cancels["order_id"].isin(book["order_id"])].reset_index(drop=True)
    if len(book) == 0 and len(cancels) == 0:
        return out

    bid_vol, ask_vol, bid_amt, ask_amt = _side_sums(book)
    out["l2_bid_order_volume"] = finite(bid_vol)
    out["l2_ask_order_volume"] = finite(ask_vol)
    out["l2_bid_order_amount"] = finite(bid_amt)
    out["l2_ask_order_amount"] = finite(ask_amt)
    out["l2_order_volume_imbalance_ratio"] = imbalance_ratio(bid_vol, ask_vol)
    out["l2_order_amount_imbalance_ratio"] = imbalance_ratio(bid_amt, ask_amt)

    bid_c_vol, ask_c_vol, bid_c_amt, ask_c_amt = _side_sums(cancels)
    out["l2_bid_cancel_volume"] = finite(bid_c_vol)
    out["l2_ask_cancel_volume"] = finite(ask_c_vol)
    out["l2_bid_cancel_amount"] = finite(bid_c_amt)
    out["l2_ask_cancel_amount"] = finite(ask_c_amt)
    out["l2_bid_cancel_rate"] = safe_div(bid_c_vol, bid_vol)
    out["l2_ask_cancel_rate"] = safe_div(ask_c_vol, ask_vol)
    out["l2_cancel_volume_imbalance_ratio"] = imbalance_ratio(bid_c_vol, ask_c_vol)
    out["l2_cancel_amount_imbalance_ratio"] = imbalance_ratio(bid_c_amt, ask_c_amt)
    return out
