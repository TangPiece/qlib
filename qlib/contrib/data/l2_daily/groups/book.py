"""盘口与 OBI（因子 43–64）。"""
from __future__ import annotations

import pandas as pd

from ..util import CONTINUOUS_QUOTE_END, CONTINUOUS_QUOTE_START, PRICE_SCALE, as_time

from .._math import empty_nan, finite, series_stat
from .._prep import _numeric
from ..catalog import factor_ids_for_group

_KEYS = factor_ids_for_group("book")


def _level_sum(frame: pd.DataFrame, prefix: str, levels: int) -> pd.Series:
    total = pd.Series(0.0, index=frame.index)
    seen = False
    for level in range(1, levels + 1):
        column = f"{prefix}{level}"
        if column not in frame.columns:
            continue
        seen = True
        total = total + pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    if not seen:
        return pd.Series(float("nan"), index=frame.index)
    return total


def calc_book_factors(snapshot: pd.DataFrame) -> dict[str, float]:
    out = empty_nan(_KEYS)
    if snapshot is None or len(snapshot) == 0 or "时间" not in snapshot.columns:
        return out
    if "申买价1" not in snapshot.columns or "申卖价1" not in snapshot.columns:
        return out
    if "申买量1" not in snapshot.columns or "申卖量1" not in snapshot.columns:
        return out

    frame = snapshot.copy()
    clock = as_time(frame["时间"])
    frame = frame.loc[(clock >= CONTINUOUS_QUOTE_START) & (clock < CONTINUOUS_QUOTE_END)]
    if len(frame) == 0:
        return out

    bid = pd.to_numeric(frame["申买价1"], errors="coerce") / PRICE_SCALE
    ask = pd.to_numeric(frame["申卖价1"], errors="coerce") / PRICE_SCALE
    mid = (bid + ask) / 2.0
    valid = (bid > 0) & (ask > 0) & (mid > 0)
    spread = (ask - bid).where(valid)
    rel = ((ask - bid) / mid).where(valid)

    bid1_qty = _numeric(frame, "申买量1")
    ask1_qty = _numeric(frame, "申卖量1")
    bid5 = _level_sum(frame, "申买量", 5)
    ask5 = _level_sum(frame, "申卖量", 5)
    bid10 = _level_sum(frame, "申买量", 10)
    ask10 = _level_sum(frame, "申卖量", 10)
    depth5_sum = bid5 + ask5
    depth_ok = valid & (depth5_sum > 0)
    obi5 = ((bid5 - ask5) / depth5_sum).where(depth_ok)
    depth10_sum = bid10 + ask10
    obi10 = ((bid10 - ask10) / depth10_sum).where(valid & (depth10_sum > 0))

    # microprice = (ask*bid_qty + bid*ask_qty) / (bid_qty+ask_qty)
    denom = bid1_qty.fillna(0) + ask1_qty.fillna(0)
    micro = ((ask * bid1_qty.fillna(0) + bid * ask1_qty.fillna(0)) / denom).where(denom > 0)
    micro_dev = ((micro - mid) / mid).where(valid & micro.notna())

    out["l2_mid_price_mean"] = series_stat(mid.where(valid), "mean")
    out["l2_mid_price_std"] = series_stat(mid.where(valid), "std")
    out["l2_spread_mean"] = series_stat(spread, "mean")
    out["l2_spread_std"] = series_stat(spread, "std")
    out["l2_relative_spread_mean"] = series_stat(rel, "mean")
    out["l2_relative_spread_std"] = series_stat(rel, "std")
    out["l2_bid1_volume"] = series_stat(bid1_qty.where(valid), "mean")
    out["l2_ask1_volume"] = series_stat(ask1_qty.where(valid), "mean")
    out["l2_bid_depth5_mean"] = series_stat(bid5.where(valid), "mean")
    out["l2_ask_depth5_mean"] = series_stat(ask5.where(valid), "mean")
    out["l2_bid_depth5_std"] = series_stat(bid5.where(valid), "std")
    out["l2_ask_depth5_std"] = series_stat(ask5.where(valid), "std")
    out["l2_depth5_diff"] = series_stat((bid5 - ask5).where(valid), "mean")
    # 买侧深度占比，避免与 l2_obi5_mean = (bid-ask)/(bid+ask) 重复
    out["l2_depth5_ratio"] = series_stat((bid5 / depth5_sum).where(depth_ok), "mean")
    out["l2_obi5_mean"] = series_stat(obi5, "mean")
    out["l2_obi5_std"] = series_stat(obi5, "std")
    out["l2_obi5_max"] = series_stat(obi5, "max")
    out["l2_obi5_min"] = series_stat(obi5, "min")
    out["l2_obi5_close"] = series_stat(obi5.dropna(), "last") if obi5.notna().any() else float("nan")
    out["l2_obi10_mean"] = series_stat(obi10, "mean")
    out["l2_microprice_deviation_mean"] = series_stat(micro_dev, "mean")
    out["l2_microprice_deviation_std"] = series_stat(micro_dev, "std")
    # 保证有限
    for key in _KEYS:
        out[key] = finite(out[key]) if out[key] == out[key] else float("nan")
    return out
