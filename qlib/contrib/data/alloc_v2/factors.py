# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""C/D 因子与 IndustryAllocationScore。

滚动分位只用当时已有窗口；权重先均等，样本外再估。
北向净买额不进核心因子。
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from .etf_metrics import build_etf_factor_panel
from .mappings import domestic_etf_codes, etf_to_industry_map, load_etf_master
from .registry import dataset_by_id
from .schema import SCHEMA_VERSION

FORMULA_VERSION = "alloc_factors@v3"
_log = logging.getLogger(__name__)


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


def aggregate_etf_panel_to_industry(
    panel: pd.DataFrame,
    code_to_industry: dict[str, str],
    *,
    how: str = "sum",
) -> pd.DataFrame:
    """将 ETF 代码面板按 primary_industry 聚合为申万行业主键。"""
    if panel.empty or not code_to_industry:
        return pd.DataFrame()
    pieces: list[pd.Series] = []
    for code, series in panel.items():
        industry = code_to_industry.get(str(code), "")
        if not industry or not str(industry).startswith("801"):
            continue
        pieces.append(series.rename(str(industry)))
    if not pieces:
        return pd.DataFrame()
    stacked = pd.concat(pieces, axis=1)
    grouped = stacked.T.groupby(level=0)
    if how == "max":
        return grouped.max().T
    return grouped.sum().T

def industry_turnover_delta(turn: pd.DataFrame, *, window: int = 20) -> pd.DataFrame:
    """成交占比变化（水平拥挤常均值回归，改用 Δ）。"""
    if turn.empty:
        return turn
    return turn - turn.shift(window)


def industry_residual_ret20(industry_ret20: pd.DataFrame, market_ret20: pd.Series) -> pd.DataFrame:
    """行业 20 日收益相对市场（沪深300）残差。"""
    if industry_ret20.empty or market_ret20.empty:
        return pd.DataFrame()
    aligned = industry_ret20.sub(market_ret20.reindex(industry_ret20.index), axis=0)
    return aligned


def industry_allocation_score(
    components: dict[str, pd.DataFrame],
    *,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """综合评分：权重默认均等；禁止用全样本事后调权宣称有效。"""
    if not components:
        return pd.DataFrame()
    # 跳过空组件后再等权，避免 flow 缺失时仍占权重
    usable = {name: frame for name, frame in components.items() if frame is not None and not frame.empty}
    if not usable:
        return pd.DataFrame()
    names = list(usable.keys())
    if weights is None:
        weights = {name: 1.0 / len(names) for name in names}
    score = None
    for name in names:
        frame = usable[name]
        z = frame.sub(frame.mean(axis=1), axis=0).div(frame.std(axis=1).replace(0, np.nan), axis=0)
        part = z * float(weights.get(name, 0.0))
        score = part if score is None else score.add(part, fill_value=0.0)
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
    available_at_lag_days: int = 0,
    available_at_panel: pd.DataFrame | None = None,
    quality_flag: str = "ok",
) -> list[dict[str, Any]]:
    """写出因子长表。份额衍生因子应传 available_at_lag_days=1 或 available_at_panel。"""
    rows: list[dict[str, Any]] = []
    for entity in panel.columns:
        for ts, value in panel[entity].items():
            if pd.isna(value):
                continue
            day = ts.date() if hasattr(ts, "date") else ts
            if available_at_panel is not None and entity in available_at_panel.columns:
                raw_at = available_at_panel[entity].get(ts)
                if pd.notna(raw_at):
                    avail = pd.Timestamp(raw_at).date()
                else:
                    avail = day + timedelta(days=available_at_lag_days)
            else:
                avail = day + timedelta(days=available_at_lag_days)
            rows.append(
                {
                    "dataset": "factor",
                    "entity_id": entity,
                    "event_date": day,
                    "announcement_date": day,
                    "available_at": avail,
                    "field": factor_id,
                    "value": float(value),
                    "unit": unit,
                    "freq": "day",
                    "evidence_class": evidence_class,
                    "quality_flag": quality_flag,
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
    code_to_industry = etf_to_industry_map(master)
    industry_flow20 = pd.DataFrame()

    # 行业 ETF 流：ETF 层估算配置 → 按 primary_industry 聚合 → 申万主键
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
                close_etf = etf_panel[etf_panel["factor_id"] == "Close"]
                nav_p = close_etf.pivot_table(
                    index=pd.to_datetime(close_etf["event_date"]), columns="entity_id", values="value", aggfunc="last"
                )
            keep = [c for c in est_p.columns if str(c) in domestics]
            if not keep and len(est_p.columns):
                # 有估算配置但无一映射到申万一级 → 显式日志，避免 silent factor_missing
                _log.warning(
                    "IndustryETFFlow20 skipped: %s ETF est columns but none in domestic thematic map (801*)",
                    len(est_p.columns),
                )
            est_p = est_p[keep] if keep else est_p.iloc[:, 0:0]
            sh_p = sh_p.reindex(columns=keep)
            nav_p = nav_p.reindex(columns=keep)
            aum = sh_p * nav_p
            # 先聚合成行业，再算 20 日流，避免 ETF 主键无法与申万收益对齐
            est_ind = aggregate_etf_panel_to_industry(est_p, code_to_industry)
            aum_ind = aggregate_etf_panel_to_industry(aum, code_to_industry)
            # 行业流可观测时点 = 成分 ETF available_at 的截面 max（通常为 T+1）
            if "available_at" in est.columns:
                est_at = est.copy()
                est_at["available_at"] = pd.to_datetime(est_at["available_at"])
                at_p = est_at.pivot_table(
                    index=pd.to_datetime(est_at["event_date"]),
                    columns="entity_id",
                    values="available_at",
                    aggfunc="max",
                )
                at_p = at_p.reindex(columns=keep) if keep else at_p.iloc[:, 0:0]
                at_ind = aggregate_etf_panel_to_industry(at_p, code_to_industry, how="max")
            else:
                at_ind = pd.DataFrame()
            flow_quality = "ok"
            if "quality_flag" in est.columns and (est["quality_flag"].astype(str) == "sparse_sample").any():
                flow_quality = "sparse_sample"
            if not est_ind.empty and not aum_ind.empty:
                industry_flow20 = industry_etf_flow20(est_ind, aum_ind, window=20)
                rows.extend(
                    _long_factor_rows(
                        industry_flow20,
                        factor_id="IndustryETFFlow20",
                        evidence_class="estimate",
                        unit="ratio",
                        window=20,
                        available_at_lag_days=1,
                        available_at_panel=at_ind if not at_ind.empty else None,
                        quality_flag=flow_quality,
                    )
                )
                for win in (120, 250):
                    for entity in industry_flow20.columns:
                        pct = rolling_percentile(industry_flow20[entity], win)
                        tmp = pct.to_frame(name=entity)
                        rows.extend(
                            _long_factor_rows(
                                tmp,
                                factor_id=f"IndustryETFFlow20_pct{win}",
                                evidence_class="estimate",
                                unit="pctile",
                                window=win,
                                available_at_lag_days=1,
                                available_at_panel=at_ind if not at_ind.empty else None,
                                quality_flag=flow_quality,
                            )
                        )
    amount = _pivot(work, "sw_industry", "amount")
    if not amount.empty:
        turn = industry_turnover_ratio(amount)
        rows.extend(_long_factor_rows(turn, factor_id="Industry_Turnover_Ratio", evidence_class="proxy", unit="ratio", window=1))
        turn_delta = industry_turnover_delta(turn, window=20)
        rows.extend(
            _long_factor_rows(turn_delta, factor_id="Industry_Turnover_Delta20", evidence_class="proxy", unit="ratio", window=20)
        )
        for win in (20, 60):
            rel = turn / turn.rolling(win, min_periods=max(5, win // 2)).mean()
            rows.extend(_long_factor_rows(rel, factor_id=f"Industry_Turnover_Rel{win}", evidence_class="proxy", unit="ratio", window=win))
        close = _pivot(work, "sw_industry", "close")
        ret20 = pd.DataFrame()
        resid20 = pd.DataFrame()
        if not close.empty:
            ret20 = close.pct_change(20, fill_method=None)
            rows.extend(_long_factor_rows(ret20, factor_id="Industry_RelRet20", evidence_class="proxy", unit="ret", window=20))
            # 市场基准：优先标准层沪深300，否则行业等权
            mkt = _pivot(work, "idx_global", "close")
            if not mkt.empty and "CSI300" in mkt.columns:
                market_ret20 = mkt["CSI300"].pct_change(20, fill_method=None)
            else:
                market_ret20 = close.mean(axis=1).pct_change(20, fill_method=None)
            resid20 = industry_residual_ret20(ret20, market_ret20)
            rows.extend(
                _long_factor_rows(resid20, factor_id="Industry_ResidualRet20", evidence_class="proxy", unit="ret", window=20)
            )

        # 综合分与 flow 解耦：Δ换手 + 残差收益 +（若有）行业流；等权，禁止全样本调权
        comps: dict[str, pd.DataFrame] = {}
        if not turn_delta.empty:
            comps["turn_delta"] = turn_delta
        if not resid20.empty:
            comps["resid_ret"] = resid20
        elif not ret20.empty:
            comps["relret"] = ret20
        if not industry_flow20.empty:
            comps["flow"] = industry_flow20.reindex(index=turn.index)
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
        rows.extend(
            _long_factor_rows(
                hold.pct_change(fill_method=None),
                factor_id="HSGT_HoldMcapChg",
                evidence_class="estimate",
                unit="ret",
                window=1,
                layer="B",
            )
        )
    turn_nb = _pivot(work, "hsgt_turnover", "amount")
    if not turn_nb.empty:
        rows.extend(_long_factor_rows(turn_nb, factor_id="HSGT_Turnover", evidence_class="proxy", unit="cny", window=1, layer="B"))

    out = pd.DataFrame(rows)
    assert_no_north_net_in_core(out)
    return out
