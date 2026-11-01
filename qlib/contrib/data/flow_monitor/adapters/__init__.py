# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""资金流监测 adapters：每个源失败只跳过自己，不阻断其他序列。"""

from __future__ import annotations

from datetime import date

from .base import FlowRow
from .china_flow import ChinaFlowAdapter
from .etf_cn import EtfCnAdapter
from .global_macro import GlobalMacroAdapter
from .industry_cn import IndustryCnAdapter
from .network import prepare_network

ADAPTERS = (
    GlobalMacroAdapter,
    EtfCnAdapter,
    IndustryCnAdapter,
    ChinaFlowAdapter,
)


def fetch_all(*, start: date | None = None, end: date | None = None) -> tuple[list[FlowRow], dict[str, int]]:
    """按 adapter 拉数，返回行与各源成功条数。"""
    prepare_network()
    rows: list[FlowRow] = []
    counts: dict[str, int] = {}
    for cls in ADAPTERS:
        adapter = cls()
        part = adapter.fetch(start=start, end=end)
        counts[adapter.name] = len(part)
        rows.extend(part)
    return rows, counts
