# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""把计算结果整理成五个问题的监测表，供笔记本直接展示。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .catalog import SERIES_DEFINITIONS, catalog_rows
from .paths_local import default_staging_root, diag_dir, panel_dir
from .staging import read_long_rows


def load_outputs(root: Path | None = None) -> dict[str, pd.DataFrame]:
    root = root or default_staging_root()
    changes = diag_dir(root) / "changes.parquet"
    flow = diag_dir(root) / "flow_vs_price.parquet"
    alloc = diag_dir(root) / "allocation_evidence.parquet"
    return {
        "long": read_long_rows(root) if (panel_dir(root) / "long.parquet").is_file() or (root / "raw" / "long.parquet").is_file() else pd.DataFrame(),
        "changes": pd.read_parquet(changes) if changes.is_file() else pd.DataFrame(),
        "flow_vs_price": pd.read_parquet(flow) if flow.is_file() else pd.DataFrame(),
        "allocation": pd.read_parquet(alloc) if alloc.is_file() else pd.DataFrame(),
    }


def _select(changes: pd.DataFrame, *, layers: tuple[str, ...] | None = None, grades: tuple[str, ...] | None = None) -> pd.DataFrame:
    if changes.empty:
        return changes
    out = changes
    if layers:
        out = out[out["layer"].isin(layers)]
    if grades:
        out = out[out["evidence_grade"].isin(grades)]
    cols = [
        c
        for c in (
            "layer",
            "display_name",
            "series_id",
            "evidence_grade",
            "asset_class",
            "region",
            "investor",
            "freq",
            "asof",
            "last",
            "ret_5d",
            "ret_20d",
            "chg_5d",
            "chg_20d",
            "notes",
        )
        if c in out.columns
    ]
    return out[cols].copy()


def question_source_dest(changes: pd.DataFrame) -> pd.DataFrame:
    """Q1/Q2：全球与中国大类资产——流量用 L1/L2，价格对照用 L3。"""
    return _select(changes, layers=("A", "B"))


def question_cn_industry(changes: pd.DataFrame) -> pd.DataFrame:
    """Q 行业：风格/行业价格与 ETF 份额并列，避免只看涨幅。"""
    return _select(changes, layers=("C",))


def question_who_buys(changes: pd.DataFrame) -> pd.DataFrame:
    """Q3：按投资者类型看最新可得 asof，季/月数据不会伪装成当日。"""
    return _select(changes, layers=("D",))


def question_is_real(flow_vs_price: pd.DataFrame) -> pd.DataFrame:
    """Q5：价格上涨但份额/流量没跟上的，标 PRICE_DRIVEN。"""
    if flow_vs_price.empty:
        return flow_vs_price
    cols = [
        c
        for c in (
            "display_name",
            "series_id",
            "evidence_grade",
            "flag",
            "price_return",
            "level_change",
            "flow_est",
            "flow_proxy",
            "amount_change",
            "asof",
            "notes",
        )
        if c in flow_vs_price.columns
    ]
    return flow_vs_price[cols].sort_values(["flag", "series_id"]).reset_index(drop=True)


def question_why(changes: pd.DataFrame) -> pd.DataFrame:
    """Q4：利率、美元、黄金、股债相对表现，只提供并列事实。"""
    wanted = {
        "yld_us_10y",
        "yld_us_2y",
        "yld_cn_10y",
        "px_dxy",
        "fx_usdcny",
        "px_gold_sge",
        "gold_gld_change",
        "px_spx",
        "px_csi300",
        "px_hsi",
        "cn_lpr_1y",
        "cn_m2",
    }
    if changes.empty:
        return changes
    return changes[changes["series_id"].isin(wanted)].copy()


def unavailable_series(changes: pd.DataFrame) -> pd.DataFrame:
    """catalog 已启用但计算结果没有观测的序列。"""
    have = set(changes["series_id"]) if not changes.empty else set()
    rows = []
    for spec in SERIES_DEFINITIONS:
        if not spec.enabled:
            rows.append({**spec.as_dict(), "status": "disabled"})
            continue
        if spec.series_id not in have:
            rows.append({**spec.as_dict(), "status": "missing"})
    return pd.DataFrame(rows)


def dashboard_payload(root: Path | None = None) -> dict[str, Any]:
    """笔记本一次加载的全部表。"""
    data = load_outputs(root)
    return {
        **data,
        "q_source_dest": question_source_dest(data["changes"]),
        "q_industry": question_cn_industry(data["changes"]),
        "q_who": question_who_buys(data["changes"]),
        "q_why": question_why(data["changes"]),
        "q_real": question_is_real(data["flow_vs_price"]),
        "unavailable": unavailable_series(data["changes"]),
        "catalog": pd.DataFrame(catalog_rows()),
    }
