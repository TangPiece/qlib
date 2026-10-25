"""流动性、日内结构与订单流联合因子（79–100）。

成交类只用 09:30–14:57 连续竞价。Amihud、收益标准差、Kyle Lambda 用 1 分钟线，
避免沪市同一主动单拆成多笔、买卖价跳动把逐笔估计变成噪声。
OFI 除以对应档位平均深度，横截面可比。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..util import (
    CONTINUOUS_QUOTE_END,
    CONTINUOUS_QUOTE_START,
    OPEN_30_END,
    PRICE_SCALE,
    TAIL_30_START,
    as_time,
)

from .._math import clock_seconds, corr, empty_nan, finite, imbalance_ratio, safe_div, series_stat
from .._prep import CONT_SECONDS, continuous, minute_bars, prepare_trades, signed
from ..catalog import factor_ids_for_group

_KEYS = factor_ids_for_group("microstructure")
_NAN = float("nan")
_OFI_MAX_GAP = 60.0
_MAX_TRADE_GAP = 3600.0


def calc_microstructure_factors(
    snapshot: pd.DataFrame,
    trades: pd.DataFrame,
    orders: pd.DataFrame,
    code: str,
) -> dict[str, float]:
    del orders
    out = empty_nan(_KEYS)
    cont = continuous(prepare_trades(trades, code))
    quotes = _continuous_quotes(snapshot)

    trade_part = _from_trades(cont)
    book_part = _from_book(quotes)
    for source in (trade_part, book_part):
        for key, value in source.items():
            if key in out:
                out[key] = value

    total_amt = trade_part.get("_total_amt", _NAN)
    out["l2_depth_amount_ratio"] = safe_div(book_part.get("l2_book_depth_mean", _NAN), total_amt)
    trade_imb = trade_part.get("_trade_imb", _NAN)
    obi5 = book_part.get("_obi5_mean", _NAN)
    if np.isfinite(trade_imb) and np.isfinite(obi5):
        out["l2_trade_book_imbalance_diff"] = finite(trade_imb - obi5)
    out["l2_effective_spread"] = _effective_spread(cont, quotes)

    for key in _KEYS:
        value = out[key]
        out[key] = finite(value) if value == value else _NAN
    return out


def _from_trades(cont: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if cont is None or len(cont) == 0:
        return out

    amount = cont["amount"]
    total_amt = float(amount.sum())
    out["_total_amt"] = finite(total_amt)
    buy = cont["side"].eq("B")
    sell = cont["side"].eq("S")
    out["_trade_imb"] = imbalance_ratio(float(amount.loc[buy].sum()), float(amount.loc[sell].sum()))

    bars = minute_bars(cont)
    ret = np.log(bars["close"].astype(float)).groupby(bars["session"], sort=False).diff()
    bar_amount = bars["amount"].astype(float)
    illiq = (ret.abs() / bar_amount).where(bar_amount > 0)
    out["l2_amihud_illiquidity"] = series_stat(illiq, "mean")
    out["l2_intraday_return_std"] = series_stat(ret, "std")
    out["l2_kyle_lambda"] = _ols_slope(bars["signed_amount"].to_numpy(dtype=float), ret.to_numpy(dtype=float))
    out["l2_intraday_amount_cv"] = safe_div(series_stat(bar_amount, "std"), series_stat(bar_amount, "mean"))

    events = np.unique(cont["sec"].dropna().to_numpy(dtype=float))
    gaps = pd.Series(np.diff(events)) if len(events) > 1 else pd.Series(dtype=float)
    gaps = gaps.where((gaps > 0) & (gaps < _MAX_TRADE_GAP))
    out["l2_trade_interval_mean"] = series_stat(gaps, "mean")
    out["l2_trade_interval_std"] = series_stat(gaps, "std")
    out["l2_trade_arrival_rate"] = safe_div(float(len(cont)), float(CONT_SECONDS))

    price = cont["price"].to_numpy(dtype=float)
    peak = np.maximum.accumulate(price)
    out["l2_intraday_max_drawdown"] = finite(float(np.min((price - peak) / peak)))

    clock = cont["clock"]
    out["l2_open_amount_ratio"] = safe_div(float(amount.loc[clock < OPEN_30_END].sum()), total_amt)
    tail = clock >= TAIL_30_START
    out["l2_close_amount_ratio"] = safe_div(float(amount.loc[tail].sum()), total_amt)
    if tail.any():
        out["l2_close_imbalance"] = imbalance_ratio(
            float(amount.loc[tail & buy].sum()),
            float(amount.loc[tail & sell].sum()),
        )

    sign = signed(cont)
    dprice = cont["price"].diff()
    valid = (sign != 0) & dprice.notna() & (dprice != 0)
    if valid.any():
        agree = np.sign(dprice.loc[valid]) == sign.loc[valid]
        out["l2_trade_price_direction_agreement"] = finite(float(agree.mean()))
    return out


def _ols_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 5:
        return _NAN
    x = x[mask]
    y = y[mask]
    var_x = float(np.var(x, ddof=1))
    if var_x <= 0:
        return _NAN
    return safe_div(float(np.cov(x, y, ddof=1)[0, 1]), var_x)


def _continuous_quotes(snapshot: pd.DataFrame) -> pd.DataFrame:
    """09:30–14:57、买一卖一都为正的快照，按时间排序并带上秒数。"""
    if snapshot is None or len(snapshot) == 0 or "时间" not in snapshot.columns:
        return pd.DataFrame()
    if "申买价1" not in snapshot.columns or "申卖价1" not in snapshot.columns:
        return pd.DataFrame()
    clock = as_time(snapshot["时间"])
    bid = pd.to_numeric(snapshot["申买价1"], errors="coerce")
    ask = pd.to_numeric(snapshot["申卖价1"], errors="coerce")
    mask = (clock >= CONTINUOUS_QUOTE_START) & (clock < CONTINUOUS_QUOTE_END) & (bid > 0) & (ask > 0)
    frame = snapshot.loc[mask].copy()
    frame["_clock"] = clock.loc[mask]
    frame["_sec"] = clock_seconds(frame["_clock"])
    frame = frame.loc[frame["_sec"].notna()]
    return frame.sort_values("_sec", kind="mergesort").reset_index(drop=True)


def _depth(frame: pd.DataFrame, prefix: str, levels: int) -> pd.Series:
    total = pd.Series(0.0, index=frame.index)
    for level in range(1, levels + 1):
        column = f"{prefix}{level}"
        if column in frame.columns:
            total = total + pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return total


def _obi(frame: pd.DataFrame, levels: int) -> pd.Series:
    bid = _depth(frame, "申买量", levels)
    ask = _depth(frame, "申卖量", levels)
    total = bid + ask
    return ((bid - ask) / total).where(total > 0)


def _from_book(quotes: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    if quotes is None or len(quotes) == 0:
        return out

    depth10 = _depth(quotes, "申买量", 10) + _depth(quotes, "申卖量", 10)
    depth_mean = series_stat(depth10, "mean")
    out["l2_book_depth_mean"] = depth_mean
    out["l2_book_depth_volatility"] = safe_div(series_stat(depth10, "std"), depth_mean)
    out["_obi5_mean"] = series_stat(_obi(quotes, 5), "mean")
    tail = quotes["_clock"] >= TAIL_30_START
    if tail.any():
        out["l2_close_obi"] = series_stat(_obi(quotes.loc[tail], 5), "mean")

    ofi1, ofi5, mid = _ofi_series(quotes, levels=5)
    depth1 = (_depth(quotes, "申买量", 1) + _depth(quotes, "申卖量", 1)).to_numpy(dtype=float)
    depth5 = (_depth(quotes, "申买量", 5) + _depth(quotes, "申卖量", 5)).to_numpy(dtype=float)
    scale1 = _mean_where(depth1, np.isfinite(ofi1))
    scale5 = _mean_where(depth5, np.isfinite(ofi5))
    if np.isfinite(ofi1).any():
        out["l2_ofi1"] = safe_div(float(np.nansum(ofi1)), scale1)
    if np.isfinite(ofi5).any():
        out["l2_ofi5"] = safe_div(float(np.nansum(ofi5)), scale5)
    if np.isfinite(ofi1).sum() > 1 and np.isfinite(scale1) and scale1 > 0:
        out["l2_ofi_std"] = finite(float(np.nanstd(ofi1 / scale1, ddof=1)))
    dmid = np.full_like(mid, np.nan)
    dmid[1:] = mid[1:] - mid[:-1]
    out["l2_ofi_price_corr"] = corr(ofi1, dmid)
    return out


def _mean_where(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask & np.isfinite(values)]
    if len(selected) == 0:
        return _NAN
    return finite(float(np.mean(selected)))


def _ofi_series(quotes: pd.DataFrame, levels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """相邻快照的一档、五档 OFI（股数）。间隔超过 60 秒的快照不相减。"""
    sec = quotes["_sec"].to_numpy(dtype=float)
    bid_p = pd.to_numeric(quotes["申买价1"], errors="coerce").to_numpy(dtype=float)
    ask_p = pd.to_numeric(quotes["申卖价1"], errors="coerce").to_numpy(dtype=float)
    mid = (bid_p + ask_p) / 2.0
    n = len(quotes)
    ofi1 = np.full(n, np.nan)
    ofi5 = np.full(n, np.nan)
    if n < 2:
        return ofi1, ofi5, mid

    gap = np.full(n, np.nan)
    gap[1:] = sec[1:] - sec[:-1]
    pair_ok = (gap > 0) & (gap <= _OFI_MAX_GAP)
    total = np.zeros(n)
    for level in range(1, levels + 1):
        bid_d = _queue_delta(_numeric(quotes, f"申买价{level}"), _numeric(quotes, f"申买量{level}"), up_positive=True)
        ask_d = -_queue_delta(_numeric(quotes, f"申卖价{level}"), _numeric(quotes, f"申卖量{level}"), up_positive=False)
        total = total + bid_d + ask_d
        if level == 1:
            ofi1 = np.where(pair_ok, total, np.nan)
    ofi5 = np.where(pair_ok, total, np.nan)
    return ofi1, ofi5, mid


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _queue_delta(price: np.ndarray, size: np.ndarray, *, up_positive: bool) -> np.ndarray:
    prev_p = np.full_like(price, np.nan)
    prev_s = np.full_like(size, np.nan)
    prev_p[1:] = price[:-1]
    prev_s[1:] = size[:-1]
    q = np.where(np.isfinite(size), size, 0.0)
    pq = np.where(np.isfinite(prev_s), prev_s, 0.0)
    if up_positive:
        delta = np.where(price >= prev_p, q, 0.0) - np.where(price <= prev_p, pq, 0.0)
    else:
        delta = np.where(price <= prev_p, q, 0.0) - np.where(price >= prev_p, pq, 0.0)
    valid = (price > 0) & (prev_p > 0) & np.isfinite(price) & np.isfinite(prev_p)
    return np.where(valid, delta, 0.0)


def _effective_spread(cont: pd.DataFrame, quotes: pd.DataFrame) -> float:
    """2|成交价 − 中间价| / 中间价，中间价取严格早于成交时间的最近一帧双边报价。"""
    if cont is None or len(cont) == 0 or quotes is None or len(quotes) == 0:
        return _NAN
    snap_sec = quotes["_sec"].to_numpy(dtype=float)
    mid = (
        (pd.to_numeric(quotes["申买价1"], errors="coerce") + pd.to_numeric(quotes["申卖价1"], errors="coerce"))
        / (2.0 * PRICE_SCALE)
    ).to_numpy(dtype=float)
    trade_sec = cont["sec"].to_numpy(dtype=float)
    trade_px = cont["price"].to_numpy(dtype=float)
    idx = np.searchsorted(snap_sec, trade_sec, side="left") - 1
    ok = (idx >= 0) & np.isfinite(trade_px) & np.isfinite(trade_sec)
    if not ok.any():
        return _NAN
    mids = mid[idx[ok]]
    px = trade_px[ok]
    good = np.isfinite(mids) & (mids > 0)
    if not good.any():
        return _NAN
    return finite(float(np.mean(2.0 * np.abs(px[good] - mids[good]) / mids[good])))
