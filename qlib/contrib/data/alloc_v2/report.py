# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""四本报告载荷：每格区分事实 / 估算 / 代理 / 推断 / 信号。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .paths import default_root
from .registry import catalog_rows
from .store import read_factor, read_research, read_standard


def _tag(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    if "statement_kind" not in out.columns:
        mapping = {"direct": "fact", "estimate": "estimate", "proxy": "proxy"}
        if "evidence_class" in out.columns:
            out["statement_kind"] = out["evidence_class"].map(mapping).fillna("inference")
        else:
            out["statement_kind"] = "fact"
    return out


def report_global(root: Path | None = None) -> pd.DataFrame:
    """报告1：全球与中国跨资产。"""
    standard = read_standard(root=root)
    if standard.empty:
        return standard
    keep = standard[standard["layer"].isin({"B", "A"}) | standard["dataset"].isin(
        {"idx_global", "yield_cn_us", "fx_usdcny", "gold_sge", "gold_gld", "cn_macro", "hsgt_turnover", "hsgt_hold", "hsgt_net"}
    )]
    # 净买额保留但标 reference_only
    out = _tag(keep)
    if not out.empty and "dataset" in out.columns:
        out.loc[out["dataset"] == "hsgt_net", "statement_kind"] = "inference"
        out.loc[out["dataset"] == "hsgt_net", "notes"] = "reference_only: 北向净买额不作核心因子"
    return out


def report_industry(root: Path | None = None) -> pd.DataFrame:
    """报告2：A 股行业配置（含套利标记、数据质量）。"""
    factors = read_factor("all", root=root)
    if factors.empty:
        return factors
    keep = factors[factors["factor_id"].isin(
        {
            "IndustryAllocationScore",
            "IndustryETFFlow20",
            "IndustryETFFlow20_pct120",
            "IndustryETFFlow20_pct250",
            "Industry_Turnover_Ratio",
            "Industry_Turnover_Rel20",
            "Industry_RelRet20",
            "Raw_Share_Change",
            "Estimated_Allocation",
            "ArbitrageRisk",
        }
    )]
    return _tag(keep)


def report_institution(root: Path | None = None) -> pd.DataFrame:
    """报告3：机构行为（带 asof 与主体口径）。"""
    standard = read_standard(root=root)
    factors = read_factor("all", root=root)
    parts = []
    if not standard.empty:
        parts.append(standard[standard["dataset"].isin({"fund_holding", "fund_scale_snapshot", "lhb", "block_trade", "margin", "holder_event"})])
    if not factors.empty:
        parts.append(factors[factors["factor_id"].str.startswith("LHB", na=False)])
    if not parts:
        return pd.DataFrame()
    return _tag(pd.concat(parts, ignore_index=True))


def report_swing(root: Path | None = None) -> dict[str, Any]:
    """报告4：波段研究 S0–S4 + 个股确认清单。"""
    return {
        "S1": read_research("signal_s1", root=root),
        "S2": read_research("signal_s2", root=root),
        "S3": read_research("signal_s3", root=root),
        "S4": read_research("signal_s4", root=root),
        "invalidation": read_research("signal_invalidation", root=root),
        "backtest": read_research("backtest_factors", root=root),
        "catalog": pd.DataFrame(catalog_rows()),
    }


def dashboard_payload(root: Path | None = None) -> dict[str, Any]:
    root = root or default_root()
    return {
        "global": report_global(root),
        "industry": report_industry(root),
        "institution": report_institution(root),
        "swing": report_swing(root),
        "standard": read_standard(root=root),
        "factors": read_factor("all", root=root),
    }
