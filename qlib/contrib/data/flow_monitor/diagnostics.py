# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""价格调整净流量、涨跌与流量背离、并发配置变化矩阵。

公开数据只能给出配置变化证据，不能证明「卖股票的钱买了黄金」。
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from .catalog import SERIES_DEFINITIONS, SeriesDef, series_by_id

ASSET_ORDER = ("equity", "bond", "gold", "cash")


def price_adjusted_flow(aum: pd.Series, price: pd.Series) -> pd.Series:
    """用市值与价格估计净流入：flow_est = ΔAUM - AUM_lag * r。

    只有市值没有成交净额时使用。价格上涨造成的账面增大会被扣掉。
    """
    aligned = pd.concat({"aum": aum, "price": price}, axis=1).sort_index()
    aligned = aligned.dropna()
    if aligned.empty:
        return pd.Series(dtype="float64")
    ret = aligned["price"].pct_change()
    flow = aligned["aum"] - aligned["aum"].shift(1) * (1.0 + ret)
    flow.name = "flow_est"
    return flow


def share_to_flow_value(shares: pd.Series, price: pd.Series) -> pd.Series:
    """份额变化 × 当期价格 ≈ 净申购金额（仍可能含现金申购）。"""
    aligned = pd.concat({"shares": shares, "price": price}, axis=1).sort_index().dropna()
    if aligned.empty:
        return pd.Series(dtype="float64")
    delta = aligned["shares"].diff()
    return (delta * aligned["price"]).rename("flow_est")


def classify_price_vs_flow(
    *,
    price_change: float | None,
    flow: float | None,
    amount_change: float | None = None,
    has_l1: bool,
) -> str:
    """把一条观察标成证据类型，避免把涨幅写成流入。"""
    if has_l1 and flow is not None and price_change is not None:
        if price_change > 0 and flow <= 0:
            return "PRICE_DRIVEN"
        if price_change < 0 and flow >= 0:
            return "FLOW_DIVERGE"
        if price_change == 0 and flow == 0:
            return "FLAT"
        return "CONSISTENT"
    if not has_l1 and amount_change is not None and amount_change > 0:
        return "ACTIVITY_ONLY"
    if price_change is not None and flow is None:
        return "PRICE_ONLY"
    return "INSUFFICIENT"


def window_change(series: pd.Series, days: int) -> float | None:
    """窗口首尾变化；点数不足则 None，不做前填冒充。"""
    clean = series.dropna()
    if len(clean) < 2:
        return None
    end = clean.iloc[-1]
    start_idx = clean.index[-1] - pd.Timedelta(int(days), unit="D")
    head = clean[clean.index <= start_idx]
    start = head.iloc[-1] if len(head) else clean.iloc[0]
    return float(end - start)


def window_return(series: pd.Series, days: int) -> float | None:
    clean = series.dropna()
    if len(clean) < 2:
        return None
    end = clean.iloc[-1]
    start_idx = clean.index[-1] - pd.Timedelta(int(days), unit="D")
    head = clean[clean.index <= start_idx]
    start = head.iloc[-1] if len(head) else clean.iloc[0]
    if start == 0:
        return None
    return float(end / start - 1.0)


def allocation_evidence_matrix(class_scores: dict[str, float]) -> pd.DataFrame:
    """由各大类净变化构造并发证据矩阵，不是已证实的资金迁移。

    仅当 A 为净减少且 B 为净增加时，矩阵[A,B] = min(-score_A, score_B)。
    """
    frame = pd.DataFrame(0.0, index=list(ASSET_ORDER), columns=list(ASSET_ORDER))
    for src in ASSET_ORDER:
        src_score = float(class_scores.get(src, 0.0) or 0.0)
        if src_score >= 0:
            continue
        for dst in ASSET_ORDER:
            if src == dst:
                continue
            dst_score = float(class_scores.get(dst, 0.0) or 0.0)
            if dst_score <= 0:
                continue
            frame.loc[src, dst] = min(-src_score, dst_score)
    return frame


def class_flow_scores(long_df: pd.DataFrame, *, days: int = 20) -> dict[str, float]:
    """按资产大类汇总窗口内 L1 / 价格调整后 L2 的变化，L3 不计入分数。"""
    lookup = series_by_id()
    scores = {name: 0.0 for name in ASSET_ORDER}
    if long_df.empty:
        return scores
    work = long_df.copy()
    work["date"] = pd.to_datetime(work["date"])
    for series_id, part in work.groupby("series_id"):
        spec = lookup.get(str(series_id))
        if spec is None or spec.asset_class not in scores:
            continue
        if spec.evidence_grade not in {"L1", "L2"}:
            continue
        series = part.set_index("date")["value"].sort_index()
        change = window_change(series, days)
        if change is None:
            continue
        # 不同单位不可比，先按自身波动标准化后再加总
        vol = float(series.diff().std() or 0.0)
        if vol <= 0:
            continue
        scores[spec.asset_class] += float(change) / vol
    return scores


def specs_by_layer(layer: str) -> list[SeriesDef]:
    return [row for row in SERIES_DEFINITIONS if row.layer == layer]


def latest_asof(long_df: pd.DataFrame, series_ids: Iterable[str]) -> pd.Timestamp | None:
    if long_df.empty:
        return None
    part = long_df[long_df["series_id"].isin(set(series_ids))]
    if part.empty:
        return None
    return pd.to_datetime(part["asof"]).max()
