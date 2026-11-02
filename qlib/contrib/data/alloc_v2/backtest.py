# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""IC / RankIC / 分组 / 持有期回测记录。失败结果也留档，不只留最优。

区分：factor_missing（未建成）vs 有结果但 IC≈0（无稳定预测力）。
全样本 / 近252日 / 分年诊断一并写出，禁止只报好看窗口。
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from .schema import SCHEMA_VERSION

HOLD_DAYS = (5, 10, 20, 40)
COST_BPS = 10  # 单边 10bp，可配置
LAST_N_IC = 252

# 核心评测清单：拥挤水平单独留作「反向」对照
WANTED_FACTORS = (
    "IndustryAllocationScore",
    "IndustryETFFlow20",
    "Industry_Turnover_Ratio",
    "Industry_Turnover_Delta20",
    "Industry_RelRet20",
    "Industry_ResidualRet20",
    "LHB_Institution_Ratio",
)


def _panel(factors: pd.DataFrame, factor_id: str) -> pd.DataFrame:
    part = factors[factors["factor_id"] == factor_id].copy()
    if part.empty:
        return pd.DataFrame()
    part["event_date"] = pd.to_datetime(part["event_date"])
    return part.pivot_table(index="event_date", columns="entity_id", values="value", aggfunc="last")


def forward_returns(price: pd.DataFrame, horizons: Iterable[int] = HOLD_DAYS) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for h in horizons:
        # 不 pad：避免停牌/缺失被前向填充污染前向收益
        out[int(h)] = price.pct_change(int(h), fill_method=None).shift(-int(h))
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


def group_returns(
    factor: pd.DataFrame,
    forward: pd.DataFrame,
    *,
    n_groups: int = 5,
) -> tuple[pd.DataFrame, int]:
    """返回 (分组收益表, qcut 跳过天数)。跳过 ≠ 因子无效，常因截面并列过多。"""
    rows = []
    skipped = 0
    aligned_f, aligned_y = factor.align(forward, join="inner")
    for ts in aligned_f.index:
        x = aligned_f.loc[ts].dropna()
        y = aligned_y.loc[ts].reindex(x.index).dropna()
        common = x.index.intersection(y.index)
        if len(common) < n_groups * 2:
            skipped += 1
            continue
        x = x.loc[common]
        y = y.loc[common]
        try:
            buckets = pd.qcut(x, n_groups, labels=False, duplicates="drop")
        except ValueError:
            skipped += 1
            continue
        for g, part in y.groupby(buckets):
            rows.append({"date": ts, "group": int(g), "ret": float(part.mean()), "n": int(len(part))})
    return pd.DataFrame(rows), skipped


def apply_cost(gross: pd.Series, *, cost_bps: float = COST_BPS, hold_days: int = 1) -> pd.Series:
    """持有期摊销成本：双边换仓成本按 hold_days 摊到每个信号日的前向收益上。

    避免对 h=20 的收益按「每日满仓换仓」扣满 20bp 导致系统性偏悲观。
    """
    h = max(int(hold_days), 1)
    return gross - 2.0 * (cost_bps / 10000.0) / h


def _spread_series(
    factor: pd.DataFrame,
    forward: pd.DataFrame,
    *,
    cost_bps: float,
    hold_days: int,
) -> tuple[pd.Series, pd.Series, int]:
    groups, skipped = group_returns(factor, forward)
    if groups.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), skipped
    top = groups[groups["group"] == groups["group"].max()].set_index("date")["ret"]
    bottom = groups[groups["group"] == groups["group"].min()].set_index("date")["ret"]
    spread = (top - bottom).dropna()
    net = apply_cost(spread, cost_bps=cost_bps, hold_days=hold_days)
    return spread, net, skipped


def _aligned_tail(ics: pd.Series, spread: pd.Series, net: pd.Series, *, n: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """近 n 日：优先 IC∩多空共同日历；若多空为空则仅截断 IC（分组失败时仍可报近端 IC）。"""
    if ics.empty:
        return ics, spread, net
    if spread.empty:
        ics_w = ics.tail(n)
        return ics_w, spread, net
    common = ics.index.intersection(spread.index)
    if len(common) == 0:
        # 无交集时各自 tail，并在 metrics 里靠 n_days 暴露差异
        return ics.tail(n), spread.tail(n), net.tail(n)
    window = common[-n:] if len(common) > n else common
    return ics.reindex(window).dropna(), spread.reindex(window).dropna(), net.reindex(window).dropna()


def _window_metrics(ics: pd.Series, spread: pd.Series, net: pd.Series, *, suffix: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not ics.empty:
        out[f"rank_ic_mean_{suffix}"] = float(ics.mean())
        out[f"rank_ic_ir_{suffix}"] = float(ics.mean() / ics.std()) if float(ics.std() or 0) > 0 else np.nan
        out[f"ic_n_days_{suffix}"] = int(len(ics))
        out[f"ic_start_{suffix}"] = str(ics.index.min().date())
        out[f"ic_end_{suffix}"] = str(ics.index.max().date())
    if len(spread):
        out[f"long_short_mean_{suffix}"] = float(spread.mean())
    if len(net):
        out[f"long_short_net_{suffix}"] = float(net.mean())
    return out


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
        "ic_by_year": [],
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
        spread, net, qcut_skipped = _spread_series(factor, fwd, cost_bps=cost_bps, hold_days=int(h))
        # 全样本（兼容旧列名）
        result["metrics"][f"rank_ic_mean_h{h}"] = float(ics.mean())
        result["metrics"][f"rank_ic_ir_h{h}"] = float(ics.mean() / ics.std()) if float(ics.std() or 0) > 0 else np.nan
        result["metrics"][f"ic_n_days_h{h}"] = int(len(ics))
        result["metrics"][f"ic_start_h{h}"] = str(ics.index.min().date())
        result["metrics"][f"ic_end_h{h}"] = str(ics.index.max().date())
        result["metrics"][f"cost_bps_h{h}"] = float(cost_bps)
        result["metrics"][f"cost_hold_days_h{h}"] = int(h)
        if len(spread):
            result["metrics"][f"long_short_mean_h{h}"] = float(spread.mean())
            result["metrics"][f"long_short_net_h{h}"] = float(net.mean())
        else:
            result["failures"].append(f"group_empty_h{h}")
        if qcut_skipped:
            # 记入 metrics，不抬 failures：并列过多导致跳过 ≠ 因子缺失
            result["metrics"][f"qcut_skipped_h{h}"] = int(qcut_skipped)

        # 近 252 日：共同日历截断
        ics_252, spread_252, net_252 = _aligned_tail(ics, spread, net, n=LAST_N_IC)
        result["metrics"].update(_window_metrics(ics_252, spread_252, net_252, suffix=f"h{h}_last252"))

        # 分年（主要用于 h20 诊断表）
        if int(h) == 20:
            for year, part in ics.groupby(ics.index.year):
                result["ic_by_year"].append(
                    {
                        "factor_id": factor_id,
                        "horizon": int(h),
                        "year": int(year),
                        "rank_ic_mean": float(part.mean()),
                        "n_days": int(len(part)),
                        "version": SCHEMA_VERSION,
                    }
                )

    if not result["metrics"]:
        result["status"] = "failed"
    elif result["failures"]:
        result["status"] = "partial"
    return result


def pairwise_factor_corr(factors: pd.DataFrame, factor_ids: Iterable[str]) -> pd.DataFrame:
    """逐日截面相关再取均值，用于检查综合分是否再次贴住成交占比。"""
    panels = {fid: _panel(factors, fid) for fid in factor_ids}
    ids = [fid for fid, panel in panels.items() if not panel.empty]
    rows: list[dict[str, Any]] = []
    for i, left_id in enumerate(ids):
        for right_id in ids[i:]:
            left, right = panels[left_id].align(panels[right_id], join="inner")
            day_corrs: list[float] = []
            for ts in left.index:
                x = left.loc[ts]
                y = right.loc[ts]
                mask = x.notna() & y.notna()
                if mask.sum() < 5:
                    continue
                day_corrs.append(float(x[mask].corr(y[mask])))
            finite = [c for c in day_corrs if np.isfinite(c)]
            rows.append(
                {
                    "factor_a": left_id,
                    "factor_b": right_id,
                    "mean_cs_corr": float(np.mean(finite)) if finite else np.nan,
                    "n_days": int(len(finite)),
                    "version": SCHEMA_VERSION,
                    "note": "score_vs_turnover_should_not_be_near_1",
                }
            )
    return pd.DataFrame(rows)


def run_factor_backtests(factors: pd.DataFrame, standard: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """批量评测核心因子；返回 summary / by_year / corr 三张表。"""
    price = standard[(standard["dataset"] == "sw_industry") & (standard["field"] == "close")].copy()
    if price.empty:
        price = standard[(standard["dataset"] == "etf_bar") & (standard["field"] == "close")].copy()
    if price.empty:
        empty = pd.DataFrame([{"factor_id": "ALL", "status": "failed", "failures": "no_price", "version": SCHEMA_VERSION}])
        return {"summary": empty, "by_year": pd.DataFrame(), "corr": pd.DataFrame()}

    price["event_date"] = pd.to_datetime(price["event_date"])
    px = price.pivot_table(index="event_date", columns="entity_id", values="value", aggfunc="last")

    rows = []
    by_year_rows: list[dict[str, Any]] = []
    for factor_id in WANTED_FACTORS:
        panel = _panel(factors, factor_id)
        if panel.empty:
            rows.append(
                {
                    "factor_id": factor_id,
                    "status": "failed",
                    "failures": "factor_missing",
                    "result_kind": "missing",
                    "version": SCHEMA_VERSION,
                }
            )
            continue
        common_cols = [c for c in panel.columns if c in px.columns]
        if not common_cols:
            rows.append(
                {
                    "factor_id": factor_id,
                    "status": "failed",
                    "failures": "entity_mismatch",
                    "result_kind": "missing",
                    "n_factor_entities": int(panel.shape[1]),
                    "n_price_entities": int(px.shape[1]),
                    "version": SCHEMA_VERSION,
                }
            )
            continue
        ev = evaluate_factor(panel[common_cols], px[common_cols], factor_id=factor_id)
        flat = {
            "factor_id": factor_id,
            "status": ev["status"],
            "failures": "|".join(ev["failures"]),
            "result_kind": "evaluated",
            "version": SCHEMA_VERSION,
        }
        flat.update(ev["metrics"])
        rows.append(flat)
        by_year_rows.extend(ev.get("ic_by_year") or [])

    summary = pd.DataFrame(rows)
    by_year = pd.DataFrame(by_year_rows)
    corr = pairwise_factor_corr(
        factors,
        [
            "IndustryAllocationScore",
            "Industry_Turnover_Ratio",
            "Industry_Turnover_Delta20",
            "Industry_ResidualRet20",
            "IndustryETFFlow20",
        ],
    )
    return {"summary": summary, "by_year": by_year, "corr": corr}
