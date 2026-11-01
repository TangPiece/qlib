# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""C/D 因子与 IndustryAllocationScore。

滚动分位只用当时已有窗口；权重先均等，样本外再估。
北向净买额不进核心因子。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .etf_metrics import build_etf_factor_panel, isolate_universe
from .mappings import domestic_etf_codes, load_etf_master
from .registry import dataset_by_id
from .schema import SCHEMA_VERSION

FORMULA_VERSION = "alloc_factors@v2"


def _asof_filter(frame: pd.DataFrame, asof: pd.Timestamp | None) -> pd.DataFrame:
    """PIT：只用 available_at ≤ T。"""
    if frame.empty or asof is None:
        return frame
    work = frame.copy()
    work["available_at"] = pd.to_datetime(work["available_at"])
    return work[work["available_at"] <= asof]


def _pivot(standard: pd.DataFrame, dataset: str, field: str) -> pd.DataFrame:
    part = standard[(standard["dataset"] == dataset) & (standard["field"] == field)].copy()
    if part.empty:
        return pd.DataFrame()
    part["event_date"] = pd.to_datetime(part["event_date"])
    return part.pivot_table(index="event_date", columns="entity_id", values="value", aggfunc="last")


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    """滚动分位：只用窗口内历史，不含未来。"""

    def _pct(arr: np.ndarray) -> float:
        if len(arr) < 2 or np.isnan(arr[-1]):
            return np.nan
        hist = arr[:-1]
        hist = hist[~np.isnan(hist)]
        if len(hist) == 0:
            return np.nan
        return float((hist < arr[-1]).mean())

    return series.rolling(window, min_periods=max(5, window // 5)).apply(_pct, raw=True)


def industry_turnover_ratio(amount: pd.DataFrame) -> pd.DataFrame:
    """行业成交额占全市场比例。"""
    if amount.empty:
        return amount
    total = amount.sum(axis=1).replace(0, np.nan)
    return amount.div(total, axis=0)


def industry_breadth(up_ratio: pd.DataFrame) -> pd.DataFrame:
    """成分股上涨比例；缺成分时上游应写 unavailable，此处不填 0。"""
    return up_ratio


def industry_etf_flow20(estimated: pd.DataFrame, aum: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """20 日估算配置 / 20 日均 AUM。跨境与联接已在上游隔离。"""
    flow = estimated.rolling(window, min_periods=max(5, window // 2)).sum()
    avg_aum = aum.rolling(window, min_periods=max(5, window // 2)).mean().replace(0, np.nan)
    return flow / avg_aum


def industry_allocation_score(
    components: dict[str, pd.DataFrame],
    *,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """综合评分：权重默认均等；禁止用全样本事后调权宣称有效。"""
    if not components:
        return pd.DataFrame()
    names = list(components.keys())
    if weights is None:
        weights = {name: 1.0 / len(names) for name in names}
    # 截面 z-score 再加权；缺失不按 0 填
    aligned = None
    score = None
    for name in names:
        frame = components[name]
        if frame.empty:
            continue
        z = frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1).replace(0, np.nan), axis=0)
        part = z * float(weights.get(name, 0.0))
        score = part if score is None else score.add(part, fill_value=0.0)
        aligned = frame
    _ = aligned
    return score if score is not None else pd.DataFrame()


def lhb_institution_ratio(inst_buy: pd.DataFrame, lhb_amount: pd.DataFrame) -> pd.DataFrame:
    """龙虎榜机构买入占比；无披露日保持 NaN，不填 0。"""
    if inst_buy.empty or lhb_amount.empty:
        return pd.DataFrame()
    left, right = inst_buy.align(lhb_amount, join="outer")
    return left / right.replace(0, np.nan)


def lhb_decay(ratio: pd.DataFrame, *, days: int = 5) -> pd.DataFrame:
    """披露日后线性衰减；仅作可测窗口之一。"""
    if ratio.empty:
        return ratio
    out = ratio.copy().astype(float)
    for entity in out.columns:
        series = out[entity]
        decayed = series.copy()
        last_val = np.nan
        last_idx = None
        for i, (ts, val) in enumerate(series.items()):
            if pd.notna(val):
                last_val = float(val)
                last_idx = i
                decayed.iloc[i] = last_val
            elif last_idx is not None and i - last_idx < days:
                decayed.iloc[i] = last_val * (1.0 - (i - last_idx) / days)
            else:
                decayed.iloc[i] = np.nan
        out[entity] = decayed
    return out


def _long_factor_rows(
    panel: pd.DataFrame,
    *,
    factor_id: str,
    evidence_class: str,
    unit: str,
    window: int,
    layer: str = "C",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entity in panel.columns:
        for ts, value in panel[entity].items():
            if pd.isna(value):
                continue
            day = ts.date() if hasattr(ts, "date") else ts
            rows.append(
                {
                    "dataset": "factor",
                    "entity_id": entity,
                    "event_date": day,
                    "announcement_date": day,
                    "available_at": day,
                    "field": factor_id,
                    "value": float(value),
                    "unit": unit,
                    "freq": "day",
                    "evidence_class": evidence_class,
                    "quality_flag": "ok",
                    "version": SCHEMA_VERSION,
                    "source": "computed",
                    "layer": layer,
                    "factor_id": factor_id,
                    "window": window,
                    "formula_version": FORMULA_VERSION,
                }
            )
    return rows


def assert_no_north_net_in_core(factor_frame: pd.DataFrame) -> None:
    """硬约束：北向净买额不得进核心因子表。"""
    if factor_frame.empty:
        return
    banned = dataset_by_id()["hsgt_net"]
    if "source" in factor_frame.columns and banned.dataset in set(factor_frame["source"].astype(str)):
        raise AssertionError("hsgt_net must not enter core factors")
    if "factor_id" in factor_frame.columns:
        bad = factor_frame["factor_id"].astype(str).str.contains("north_net|hsgt_net", case=False, na=False)
        if bool(bad.any()):
            raise AssertionError("north net buy must not enter core factors")


def build_all_factors(standard: pd.DataFrame, *, asof: pd.Timestamp | None = None) -> pd.DataFrame:
    """组装 C/D 因子；ETF 三类指标 + 行业流 + 龙虎榜。"""
    work = _asof_filter(standard, asof)
    rows: list[dict[str, Any]] = []

    etf_panel = build_etf_factor_panel(work)
    if not etf_panel.empty:
        rows.extend(etf_panel.to_dict("records"))

    master = load_etf_master()
    domestics = set(domestic_etf_codes(master))
    etf_est = work[(work["dataset"] == "etf_shares") | (work["field"].isin({"shares", "nav", "close", "amount"}))]
    # 行业 ETF 流：用已计算的 estimated_allocation
    if not etf_panel.empty:
        est = etf_panel[etf_panel["factor_id"] == "Estimated_Allocation"].copy()
        shares = etf_panel[etf_panel["factor_id"] == "RawShares"].copy()
        nav = etf_panel[etf_panel["factor_id"] == "NAV"].copy()
        if not est.empty and not shares.empty:
            est_p = est.pivot_table(index=pd.to_datetime(est["event_date"]), columns="entity_id", values="value", aggfunc="last")
            sh_p = shares.pivot_table(index=pd.to_datetime(shares["event_date"]), columns="entity_id", values="value", aggfunc="last")
            if not nav.empty:
                nav_p = nav.pivot_table(index=pd.to_datetime(nav["event_date"]), columns="entity_id", values="value", aggfunc="last")
            else:
                close = etf_panel[etf_panel["factor_id"] == "Close"]
                nav_p = close.pivot_table(index=pd.to_datetime(close["event_date"]), columns="entity_id", values="value", aggfunc="last")
            keep = [c for c in est_p.columns if c in domestics]
            est_p = est_p[keep] if keep else est_p.iloc[:, 0:0]
            sh_p = sh_p.reindex(columns=keep)
            nav_p = nav_p.reindex(columns=keep)
            aum = sh_p * nav_p
            flow20 = industry_etf_flow20(est_p, aum, window=20)
            rows.extend(_long_factor_rows(flow20, factor_id="IndustryETFFlow20", evidence_class="estimate", unit="ratio", window=20))
            for win in (120, 250):
                for entity in flow20.columns:
                    pct = rolling_percentile(flow20[entity], win)
                    tmp = pct.to_frame(name=entity)
                    rows.extend(
                        _long_factor_rows(
                            tmp,
                            factor_id=f"IndustryETFFlow20_pct{win}",
                            evidence_class="estimate",
                            unit="pctile",
                            window=win,
                        )
                    )

    amount = _pivot(work, "sw_industry", "amount")
    if not amount.empty:
        turn = industry_turnover_ratio(amount)
        rows.extend(_long_factor_rows(turn, factor_id="Industry_Turnover_Ratio", evidence_class="proxy", unit="ratio", window=1))
        for win in (20, 60):
            rel = turn / turn.rolling(win, min_periods=max(5, win // 2)).mean()
            rows.extend(_long_factor_rows(rel, factor_id=f"Industry_Turnover_Rel{win}", evidence_class="proxy", unit="ratio", window=win))
        close = _pivot(work, "sw_industry", "close")
        if not close.empty:
            ret20 = close.pct_change(20)
            rows.extend(_long_factor_rows(ret20, factor_id="Industry_RelRet20", evidence_class="proxy", unit="ret", window=20))

        # IndustryAllocationScore：均等权重，样本外再估
        comps = {
            "turnover": turn,
        }
        if not etf_panel.empty:
            flow_rows = [r for r in rows if r.get("factor_id") == "IndustryETFFlow20"]
            if flow_rows:
                flow_df = pd.DataFrame(flow_rows)
                flow_p = flow_df.pivot_table(
                    index=pd.to_datetime(flow_df["event_date"]),
                    columns="entity_id",
                    values="value",
                    aggfunc="last",
                )
                # ETF 流与行业指数不同主键时，评分只在各自实体上算；这里用行业成交 + 相对收益
                comps["relret"] = ret20 if not close.empty else turn
        score = industry_allocation_score(comps)
        rows.extend(_long_factor_rows(score, factor_id="IndustryAllocationScore", evidence_class="estimate", unit="z", window=1))

    inst = _pivot(work, "lhb", "inst_buy")
    lhb_amt = _pivot(work, "lhb", "lhb_amount")
    ratio = lhb_institution_ratio(inst, lhb_amt)
    if not ratio.empty:
        rows.extend(_long_factor_rows(ratio, factor_id="LHB_Institution_Ratio", evidence_class="proxy", unit="ratio", window=1, layer="D"))
        decayed = lhb_decay(ratio, days=5)
        rows.extend(_long_factor_rows(decayed, factor_id="LHB_Institution_Ratio_d5", evidence_class="proxy", unit="ratio", window=5, layer="D"))

    # 北向成交与持股可作 proxy/estimate，但净买额禁止
    hold = _pivot(work, "hsgt_hold", "hold_mcap")
    if not hold.empty:
        rows.extend(_long_factor_rows(hold.pct_change(), factor_id="HSGT_HoldMcapChg", evidence_class="estimate", unit="ret", window=1, layer="B"))
    turn_nb = _pivot(work, "hsgt_turnover", "amount")
    if not turn_nb.empty:
        rows.extend(_long_factor_rows(turn_nb, factor_id="HSGT_Turnover", evidence_class="proxy", unit="cny", window=1, layer="B"))

    out = pd.DataFrame(rows)
    assert_no_north_net_in_core(out)
    _ = etf_est
    _ = isolate_universe
    return out
