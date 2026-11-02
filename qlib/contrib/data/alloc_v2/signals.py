# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""波段信号 S1–S4：三级独立计算再交叉；只用 available_at ≤ T。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .mappings import broad_etf_codes, load_etf_master
from .schema import SCHEMA_VERSION

SIGNAL_VERSION = "signals@v3"


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
    - C2: IndustryETFFlow20_pct120 > 0.6（有行业对齐流时用 estimate；否则 RelRet 代理）
    - C3: Industry_ResidualRet20 > 0（若无则 Industry_RelRet20 > 0）
    - C4: IndustryAllocationScore 截面前 30%
    - C5: 数据 quality_flag == ok
    """
    turn = _latest_by_entity(_factor_panel(factors, "Industry_Turnover_Rel20"), asof)
    flow_pct = _latest_by_entity(_factor_panel(factors, "IndustryETFFlow20_pct120"), asof)
    resid = _latest_by_entity(_factor_panel(factors, "Industry_ResidualRet20"), asof)
    rel = _latest_by_entity(_factor_panel(factors, "Industry_RelRet20"), asof)
    score = _latest_by_entity(_factor_panel(factors, "IndustryAllocationScore"), asof)

    entities = sorted(set(turn.index) | set(rel.index) | set(resid.index) | set(score.index) | set(flow_pct.index))
    rows = []
    score_cut = float(score.quantile(0.7)) if len(score) else np.nan
    for entity in entities:
        c1 = bool(turn.get(entity, np.nan) > 1.0) if entity in turn.index else False
        # 流因子可用则不再依赖收益代理，减少「涨了就算流入」循环论证
        if entity in flow_pct.index and pd.notna(flow_pct.get(entity)):
            c2 = bool(flow_pct.get(entity) > 0.6)
            c2_kind = "estimate"
        else:
            c2 = bool(rel.get(entity, np.nan) > 0) if entity in rel.index else False
            c2_kind = "proxy"
        if entity in resid.index and pd.notna(resid.get(entity)):
            c3 = bool(resid.get(entity) > 0)
        else:
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
    """S3：行业级确认骨架（当前无个股映射）。

    语义（显式）：
    - 对每个 S2=1 行业输出一行 S3=1（骨架放行），notes 标明是否有龙虎榜旁证；
    - 龙虎榜**不否决** S3（non_mandatory）；将来若改为强制，须同步改 S4 交集逻辑。
    - 因此当前「三层共振」在个股层等价于 S1∧S2，文档须与此一致。
    """
    if s2.empty:
        return pd.DataFrame()
    watch = set(s2.loc[s2["value"] > 0, "entity_id"])
    if not watch:
        return pd.DataFrame(
            columns=["asof", "entity_id", "signal_id", "value", "statement_kind", "available_at", "quality_flag", "notes", "version"]
        )

    lhb = _latest_by_entity(_factor_panel(factors, "LHB_Institution_Ratio_d5"), asof)
    lhb_available = not lhb.empty
    rows = []
    for entity in sorted(watch):
        trade_confirm = bool(entity in lhb.index and pd.notna(lhb.get(entity)) and float(lhb.get(entity)) > 0.2)
        # 骨架模式：S2 观察名单一律 S3=1；LHB 仅写入 notes
        rows.append(
            {
                "asof": asof.date().isoformat(),
                "entity_id": entity,
                "signal_id": "S3",
                "value": 1.0,
                "statement_kind": "signal",
                "available_at": asof.date().isoformat(),
                "quality_flag": "ok",
                "notes": (
                    "skeleton_pass;trade_confirm=lhb"
                    if trade_confirm
                    else f"skeleton_pass;trade_confirm=none(non_mandatory);lhb_data={'yes' if lhb_available else 'no'}"
                ),
                "version": SIGNAL_VERSION,
            }
        )
    _ = standard
    return pd.DataFrame(rows)


def signal_s4(s1: dict[str, Any], s2: pd.DataFrame, s3: pd.DataFrame, *, asof: pd.Timestamp) -> pd.DataFrame:
    """S4：三层同时成立。

    规则：
    - S1 必须过；
    - 若 S3 表为空（未跑/无观察名单）→ 仅用 S2（骨架降级）；
    - 若 S3 表非空 → 必须 S2∩S3；S3 全为 0 时不得回退到仅 S2。
    """
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
    if s3 is None or s3.empty:
        both = sorted(s2_ok)
        mode = "S1&S2(no_S3_table)"
    else:
        s3_ok = set(s3.loc[s3["value"] > 0, "entity_id"])
        both = sorted(s2_ok & s3_ok)
        mode = "S1&S2&S3"
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
                "notes": mode,
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
                "notes": "no intersection" if (s3 is not None and not s3.empty) else "S2 empty",
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
