# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A 股多层次资金配置研究体系 V2：四层数据 + B/C/D 研究 + 波段信号。

份额变化 ≠ 净流入；北向成交额 ≠ 净买入；季报 ≠ 实时持仓。
"""

from .pipeline import run_fetch, run_factor, run_pipeline, run_signal
from .registry import DATASETS, catalog_rows
from .schema import SERIES_SET

__all__ = [
    "DATASETS",
    "SERIES_SET",
    "catalog_rows",
    "run_fetch",
    "run_factor",
    "run_pipeline",
    "run_signal",
]
