# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A 股财务因子 fund_cn_basic@v1：计算、质检、本地/R2 日宽表。不含 Factor Lab。"""

from __future__ import annotations

from typing import List, Tuple

from .catalog import (
    FACTOR_DEFINITIONS,
    FACTOR_SET,
    FACTOR_VERSION,
    SUPPORT_METRICS,
    backtest_eligible_factors,
    coverage_matrix,
    factor_ids,
    mvp_factors,
    schema_hash,
    wide_columns,
)
from .pipeline import run_compute_publish, run_coverage, run_pipeline, run_pit, run_stage

# 顺序与 catalog.FACTOR_DEFINITIONS 一致
FUND_CN_FACTOR_IDS: Tuple[str, ...] = tuple(factor_ids())
assert len(FUND_CN_FACTOR_IDS) == 30, len(FUND_CN_FACTOR_IDS)


def get_fund_cn_feature_config(*, mvp_only: bool = True) -> Tuple[List[str], List[str]]:
    """返回 QlibDataLoader 用的 ($roe 等, 列名) 配置。"""
    ids = factor_ids(mvp_only=mvp_only)
    fields = [f"${fid}" for fid in ids]
    return fields, list(ids)


__all__ = [
    "FACTOR_DEFINITIONS",
    "FACTOR_SET",
    "FACTOR_VERSION",
    "FUND_CN_FACTOR_IDS",
    "SUPPORT_METRICS",
    "backtest_eligible_factors",
    "coverage_matrix",
    "factor_ids",
    "get_fund_cn_feature_config",
    "mvp_factors",
    "run_compute_publish",
    "run_coverage",
    "run_pipeline",
    "run_pit",
    "run_stage",
    "schema_hash",
    "wide_columns",
]
