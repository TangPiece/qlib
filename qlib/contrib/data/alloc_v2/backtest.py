# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""IC / RankIC / 分组 / 持有期回测记录。失败结果也留档，不只留最优。"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema import SCHEMA_VERSION

HOLD_DAYS = (5, 10, 20, 40)
COST_BPS = 10  # 单边 10bp，可配置


def _panel(factors: pd.DataFrame, factor_id: str) -> pd.DataFrame:
    part = factors[factors["factor_id"] == factor_id].copy()
    if part.empty:
        return pd.DataFrame()
    part["event_date"] = pd.to_datetime(part["event_date"])
    return part.pivot_table(index="event_date", columns="entity_id", values="value", aggfunc="last")


def forward_returns(price: pd.DataFrame, horizons: Iterable[int] = HOLD_DAYS) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for h in horizons:
        out[int(h)] = price.pct_change(int(h)).shift(-int(h))
    return out


def ic_series(factor: pd.DataFrame, forward: pd.DataFrame) -> pd.Series:
    """逐日截面 Spearman ≈ Pearson on ranks。"""
    aligned_f, aligned_y = factor.align(forward, join="inner")
    ics = []
    index = []
    for ts in aligned_f.index:
        x = aligned_f.loc[ts]
        y = aligned_y.loc[ts]
        mask = x.notna() & y.notna()
        if mask.sum() < 5:
            continue
        xr = x[mask].rank()
        yr = y[mask].rank()
        ics.append(float(xr.corr(yr)))
        index.append(ts)
    return pd.Series(ics, index=index, name="rank_ic")


def group_returns(factor: pd.DataFrame, forward: pd.DataFrame, *, n_groups: int = 5) -> pd.DataFrame:
    rows = []
    aligned_f, aligned_y = factor.align(forward, join="inner")
    for ts in aligned_f.index:
        x = aligned_f.loc[ts].dropna()
        y = aligned_y.loc[ts].reindex(x.index).dropna()
        common = x.index.intersection(y.index)
        if len(common) < n_groups * 2:
            continue
        x = x.loc[common]
        y = y.loc[common]
        try:
            buckets = pd.qcut(x, n_groups, labels=False, duplicates="drop")
        except ValueError:
            continue
        for g, part in y.groupby(buckets):
            rows.append({"date": ts, "group": int(g), "ret": float(part.mean()), "n": int(len(part))})
    return pd.DataFrame(rows)


def apply_cost(gross: pd.Series, *, cost_bps: float = COST_BPS) -> pd.Series:
    """粗成本：每次换仓扣双边。"""
    return gross - 2.0 * (cost_bps / 10000.0)


def evaluate_factor(
    factor: pd.DataFrame,
    price: pd.DataFrame,
    *,
    factor_id: str,
    cost_bps: float = COST_BPS,
) -> dict[str, Any]:
    """对单个因子写 IC/分组/持有期结果；样本不足则记 failed。"""
    result: dict[str, Any] = {
        "factor_id": factor_id,
        "version": SCHEMA_VERSION,
        "status": "ok",
        "cost_bps": cost_bps,
        "metrics": {},
        "failures": [],
    }
    if factor.empty or price.empty:
        result["status"] = "failed"
        result["failures"].append("empty_input")
        return result

    forwards = forward_returns(price)
    for h, fwd in forwards.items():
        ics = ic_series(factor, fwd)
        if ics.empty:
            result["failures"].append(f"ic_empty_h{h}")
            continue
        result["metrics"][f"rank_ic_mean_h{h}"] = float(ics.mean())
        result["metrics"][f"rank_ic_ir_h{h}"] = float(ics.mean() / ics.std()) if float(ics.std() or 0) > 0 else np.nan
        groups = group_returns(factor, fwd)
        if groups.empty:
            result["failures"].append(f"group_empty_h{h}")
            continue
        top = groups[groups["group"] == groups["group"].max()].set_index("date")["ret"]
        bottom = groups[groups["group"] == groups["group"].min()].set_index("date")["ret"]
        spread = (top - bottom).dropna()
        net = apply_cost(spread, cost_bps=cost_bps)
        result["metrics"][f"long_short_mean_h{h}"] = float(spread.mean()) if len(spread) else np.nan
        result["metrics"][f"long_short_net_h{h}"] = float(net.mean()) if len(net) else np.nan
    if not result["metrics"]:
        result["status"] = "failed"
    elif result["failures"]:
        result["status"] = "partial"
    return result


def run_factor_backtests(factors: pd.DataFrame, standard: pd.DataFrame) -> pd.DataFrame:
    """批量评测核心因子；失败也写入。"""
    price = standard[(standard["dataset"] == "sw_industry") & (standard["field"] == "close")].copy()
    if price.empty:
        price = standard[(standard["dataset"] == "etf_bar") & (standard["field"] == "close")].copy()
    if price.empty:
        return pd.DataFrame([{"factor_id": "ALL", "status": "failed", "failures": "no_price"}])
    price["event_date"] = pd.to_datetime(price["event_date"])
    px = price.pivot_table(index="event_date", columns="entity_id", values="value", aggfunc="last")

    wanted = [
        "IndustryAllocationScore",
        "IndustryETFFlow20",
        "Industry_Turnover_Ratio",
        "Industry_RelRet20",
        "LHB_Institution_Ratio",
    ]
    rows = []
    for factor_id in wanted:
        panel = _panel(factors, factor_id)
        # 行业价与 ETF 实体不对齐时，评测可能失败——必须记录
        if panel.empty:
            rows.append({"factor_id": factor_id, "status": "failed", "failures": "factor_missing", "version": SCHEMA_VERSION})
            continue
        # 只在共同实体上评
        common_cols = [c for c in panel.columns if c in px.columns]
        if not common_cols:
            rows.append(
                {
                    "factor_id": factor_id,
                    "status": "failed",
                    "failures": "entity_mismatch",
                    "n_factor_entities": int(panel.shape[1]),
                    "n_price_entities": int(px.shape[1]),
                    "version": SCHEMA_VERSION,
                }
            )
            continue
        ev = evaluate_factor(panel[common_cols], px[common_cols], factor_id=factor_id)
        flat = {"factor_id": factor_id, "status": ev["status"], "failures": "|".join(ev["failures"]), "version": SCHEMA_VERSION}
        flat.update(ev["metrics"])
        rows.append(flat)
    return pd.DataFrame(rows)
