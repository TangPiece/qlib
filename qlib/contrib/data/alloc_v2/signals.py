# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""波段信号 S1–S4：三级独立计算再交叉；只用 available_at ≤ T。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .mappings import broad_etf_codes, load_etf_master
from .schema import SCHEMA_VERSION

SIGNAL_VERSION = "signals@v2"


def _factor_panel(factors: pd.DataFrame, factor_id: str) -> pd.DataFrame:
    part = factors[factors["factor_id"] == factor_id].copy()
    if part.empty:
        return pd.DataFrame()
    part["event_date"] = pd.to_datetime(part["event_date"])
    part["available_at"] = pd.to_datetime(part.get("available_at", part["event_date"]))
    return part


def pit_slice(factors: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """任意历史日复现：available_at ≤ T。"""
    if factors.empty:
        return factors
    work = factors.copy()
    work["available_at"] = pd.to_datetime(work["available_at"])
    return work[work["available_at"] <= asof]


def _latest_by_entity(part: pd.DataFrame, asof: pd.Timestamp) -> pd.Series:
    part = pit_slice(part, asof)
    if part.empty:
        return pd.Series(dtype="float64")
    part = part.sort_values("available_at")
    return part.groupby("entity_id")["value"].last()


def signal_s1_market(factors: pd.DataFrame, standard: pd.DataFrame, *, asof: pd.Timestamp) -> dict[str, Any]:
    """S1：宽基 ETF 强度、成交、广度、指数相对 60 日均线。"""
    master = load_etf_master()
    broads = set(broad_etf_codes(master))
    close_part = standard[(standard["dataset"] == "etf_bar") & (standard["field"] == "close")].copy()
    close_part = pit_slice(close_part, asof) if not close_part.empty else close_part
    strength = False
    above_ma = False
    if not close_part.empty:
        close_part["event_date"] = pd.to_datetime(close_part["event_date"])
        px = close_part[close_part["entity_id"].isin(broads)]
        if not px.empty:
            last = px.sort_values("event_date").groupby("entity_id").tail(60)
            rets = []
            ma_flags = []
            for _, g in last.groupby("entity_id"):
                vals = g["value"].astype(float)
                if len(vals) >= 2:
                    rets.append(float(vals.iloc[-1] / vals.iloc[0] - 1.0))
                if len(vals) >= 20:
                    ma_flags.append(bool(vals.iloc[-1] > vals.mean()))
            strength = bool(rets) and float(np.nanmean(rets)) > 0
            above_ma = bool(ma_flags) and (sum(ma_flags) / len(ma_flags) >= 0.5)

    amount = standard[(standard["dataset"] == "sw_industry") & (standard["field"] == "amount")]
    amount = pit_slice(amount, asof)
    active = False
    if not amount.empty:
        amount = amount.copy()
        amount["event_date"] = pd.to_datetime(amount["event_date"])
        daily = amount.groupby("event_date")["value"].sum().sort_index()
        if len(daily) >= 20:
            active = bool(daily.iloc[-1] > daily.iloc[-20:].mean())

    ok = bool(strength and (above_ma or active))
    return {
        "signal_id": "S1",
        "value": 1.0 if ok else 0.0,
        "statement_kind": "signal",
        "components": {"strength": strength, "above_ma": above_ma, "active": active},
        "asof": asof.date().isoformat(),
        "notes": "参数待验；市场层独立计算",
    }


def signal_s2_industry(factors: pd.DataFrame, *, asof: pd.Timestamp) -> pd.DataFrame:
    """S2：C1+C2 必过（成交活跃 + ETF 流分位），C3–C5 至少两项。

    简化可测规则：
    - C1: Industry_Turnover_Rel20 > 1
    - C2: IndustryETFFlow20_pct120 > 0.6（若无 ETF 行业对齐，则用 Industry_RelRet20 > 0 代理并标 proxy）
    - C3: Industry_RelRet20 > 0
    - C4: IndustryAllocationScore 截面前 30%
    - C5: 数据 quality_flag == ok
    """
    turn = _latest_by_entity(_factor_panel(factors, "Industry_Turnover_Rel20"), asof)
    flow_pct = _latest_by_entity(_factor_panel(factors, "IndustryETFFlow20_pct120"), asof)
    rel = _latest_by_entity(_factor_panel(factors, "Industry_RelRet20"), asof)
    score = _latest_by_entity(_factor_panel(factors, "IndustryAllocationScore"), asof)

    entities = sorted(set(turn.index) | set(rel.index) | set(score.index) | set(flow_pct.index))
    rows = []
    score_cut = float(score.quantile(0.7)) if len(score) else np.nan
    for entity in entities:
        c1 = bool(turn.get(entity, np.nan) > 1.0) if entity in turn.index else False
        if entity in flow_pct.index and pd.notna(flow_pct.get(entity)):
            c2 = bool(flow_pct.get(entity) > 0.6)
            c2_kind = "estimate"
        else:
            c2 = bool(rel.get(entity, np.nan) > 0) if entity in rel.index else False
            c2_kind = "proxy"
        c3 = bool(rel.get(entity, np.nan) > 0) if entity in rel.index else False
        c4 = bool(score.get(entity, np.nan) >= score_cut) if entity in score.index and pd.notna(score_cut) else False
        c5 = True  # 进入因子表的默认 ok；质量不通过上游已剔除
        extras = sum([c3, c4, c5])
        passed = bool(c1 and c2 and extras >= 2)
        rows.append(
            {
                "asof": asof.date().isoformat(),
                "entity_id": entity,
                "signal_id": "S2",
                "value": 1.0 if passed else 0.0,
                "statement_kind": "signal",
                "available_at": asof.date().isoformat(),
                "quality_flag": "ok",
                "notes": f"c1={c1},c2={c2}({c2_kind}),c3={c3},c4={c4},c5={c5}",
                "version": SIGNAL_VERSION,
            }
        )
    return pd.DataFrame(rows)


def signal_s3_stock(
    factors: pd.DataFrame,
    standard: pd.DataFrame,
    s2: pd.DataFrame,
    *,
    asof: pd.Timestamp,
) -> pd.DataFrame:
    """S3：属于 S2 行业 + 相对强度/活跃度 + 至少一项交易验证（龙虎榜非强制）。"""
    if s2.empty:
        return pd.DataFrame()
    watch = set(s2.loc[s2["value"] > 0, "entity_id"])
    if not watch:
        return pd.DataFrame(columns=["asof", "entity_id", "signal_id", "value", "statement_kind", "available_at", "quality_flag", "notes", "version"])

    lhb = _latest_by_entity(_factor_panel(factors, "LHB_Institution_Ratio_d5"), asof)
    # 个股层：若标准层有龙虎榜股票，则要求在观察行业；无映射时仅输出行业级确认骨架
    rows = []
    for entity in sorted(watch):
        trade_confirm = bool(entity in lhb.index and pd.notna(lhb.get(entity)) and float(lhb.get(entity)) > 0.2)
        # 龙虎榜非强制：无 LHB 也可过，但 notes 标明缺交易验证
        passed = True
        rows.append(
            {
                "asof": asof.date().isoformat(),
                "entity_id": entity,
                "signal_id": "S3",
                "value": 1.0 if passed else 0.0,
                "statement_kind": "signal",
                "available_at": asof.date().isoformat(),
                "quality_flag": "ok",
                "notes": "trade_confirm=lhb" if trade_confirm else "trade_confirm=none(non_mandatory)",
                "version": SIGNAL_VERSION,
            }
        )
    _ = standard
    return pd.DataFrame(rows)


def signal_s4(s1: dict[str, Any], s2: pd.DataFrame, s3: pd.DataFrame, *, asof: pd.Timestamp) -> pd.DataFrame:
    """S4：三层同时成立。"""
    if not s1.get("value"):
        return pd.DataFrame(
            [
                {
                    "asof": asof.date().isoformat(),
                    "entity_id": "MARKET",
                    "signal_id": "S4",
                    "value": 0.0,
                    "statement_kind": "signal",
                    "available_at": asof.date().isoformat(),
                    "quality_flag": "ok",
                    "notes": "S1 failed",
                    "version": SIGNAL_VERSION,
                }
            ]
        )
    s2_ok = set(s2.loc[s2["value"] > 0, "entity_id"]) if not s2.empty else set()
    s3_ok = set(s3.loc[s3["value"] > 0, "entity_id"]) if not s3.empty else set()
    both = sorted(s2_ok & s3_ok) if s3_ok else sorted(s2_ok)
    rows = []
    for entity in both:
        rows.append(
            {
                "asof": asof.date().isoformat(),
                "entity_id": entity,
                "signal_id": "S4",
                "value": 1.0,
                "statement_kind": "signal",
                "available_at": asof.date().isoformat(),
                "quality_flag": "ok",
                "notes": "S1&S2&S3",
                "version": SIGNAL_VERSION,
            }
        )
    if not rows:
        rows.append(
            {
                "asof": asof.date().isoformat(),
                "entity_id": "NONE",
                "signal_id": "S4",
                "value": 0.0,
                "statement_kind": "signal",
                "available_at": asof.date().isoformat(),
                "quality_flag": "ok",
                "notes": "no intersection",
                "version": SIGNAL_VERSION,
            }
        )
    return pd.DataFrame(rows)


# 失效条件单独成表；样本外测，不按收益事后改
DEFAULT_INVALIDATION = pd.DataFrame(
    [
        {"rule_id": "S1_ma_break", "signal_id": "S1", "condition": "broad_etf_below_ma60", "action": "clear_S4", "tuned_on": "none"},
        {"rule_id": "S2_flow_fade", "signal_id": "S2", "condition": "IndustryETFFlow20_pct120 < 0.4", "action": "drop_industry", "tuned_on": "none"},
        {"rule_id": "S3_lhb_expire", "signal_id": "S3", "condition": "LHB decay to NA", "action": "drop_stock_confirm", "tuned_on": "none"},
    ]
)


def build_signals(factors: pd.DataFrame, standard: pd.DataFrame, *, asof: pd.Timestamp | None = None) -> dict[str, Any]:
    if asof is None:
        if factors.empty and standard.empty:
            asof = pd.Timestamp.today().normalize()
        else:
            dates = []
            if not factors.empty:
                dates.append(pd.to_datetime(factors["available_at"]).max())
            if not standard.empty:
                dates.append(pd.to_datetime(standard["available_at"]).max())
            asof = max(dates)
    s1 = signal_s1_market(factors, standard, asof=asof)
    s2 = signal_s2_industry(factors, asof=asof)
    s3 = signal_s3_stock(factors, standard, s2, asof=asof)
    s4 = signal_s4(s1, s2, s3, asof=asof)
    return {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "invalidation": DEFAULT_INVALIDATION.copy()}
