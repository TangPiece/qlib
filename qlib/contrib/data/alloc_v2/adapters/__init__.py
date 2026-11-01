# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""alloc_v2 adapters：每个源失败只跳过自己。"""

from __future__ import annotations

from datetime import date

from ..network import prepare_network
from ..schema import StandardRow
from .china import ChinaAdapter
from .etf import EtfAdapter
from .global_macro import GlobalMacroAdapter
from .industry import IndustryAdapter
from .institution import InstitutionAdapter

ADAPTERS = (
    GlobalMacroAdapter,
    EtfAdapter,
    IndustryAdapter,
    ChinaAdapter,
    InstitutionAdapter,
)


def fetch_all(*, start: date | None = None, end: date | None = None) -> tuple[list[StandardRow], dict[str, int]]:
    prepare_network()
    rows: list[StandardRow] = []
    counts: dict[str, int] = {}
    for cls in ADAPTERS:
        adapter = cls()
        part = adapter.fetch(start=start, end=end)
        counts[adapter.name] = len(part)
        rows.extend(part)
    return rows, counts
