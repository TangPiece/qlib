# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""全球到 A 股资金流向监测：公开流量/持仓/代理指标，不写 qlib bin。"""

from .catalog import SERIES_DEFINITIONS, SERIES_SET, catalog_rows, series_ids
from .diagnostics import allocation_evidence_matrix, classify_price_vs_flow, price_adjusted_flow
from .pipeline import run_fetch, run_pipeline
from .report import dashboard_payload

__all__ = [
    "SERIES_DEFINITIONS",
    "SERIES_SET",
    "allocation_evidence_matrix",
    "catalog_rows",
    "classify_price_vs_flow",
    "dashboard_payload",
    "price_adjusted_flow",
    "run_fetch",
    "run_pipeline",
    "series_ids",
]
