# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""L2 日频百因子：列名清单 + 计算流水线。

dump / handler 仍从本包导入 L2_DAILY_FACTOR_IDS，避免训练列和 bin 字段不一致。
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .catalog import ALL_GROUPS, FACTOR_DEFINITIONS, FACTOR_VERSION, factor_ids, wide_columns
from .pipeline import run, run_batch

# 顺序与 catalog.FACTOR_DEFINITIONS 一致，共 100 列
L2_DAILY_FACTOR_IDS: Tuple[str, ...] = tuple(factor_ids())

assert len(L2_DAILY_FACTOR_IDS) == 100, len(L2_DAILY_FACTOR_IDS)

_L2_GROUP_SLICES: Tuple[Tuple[str, str, int, int], ...] = (
    ("trade", "trade_basic", 0, 16),
    ("trade", "trade_direction", 16, 30),
    ("trade", "trade_large", 30, 42),
    ("book", "book_obi", 42, 64),
    ("order", "order_cancel", 64, 78),
    ("microstructure", "liquidity", 78, 91),
    ("microstructure", "order_flow", 91, 100),
)

L2_DAILY_FACTOR_GROUPS: Dict[str, Tuple[str, str]] = {}
for _group, _category, _start, _end in _L2_GROUP_SLICES:
    for _fid in L2_DAILY_FACTOR_IDS[_start:_end]:
        L2_DAILY_FACTOR_GROUPS[_fid] = (_group, _category)

assert len(L2_DAILY_FACTOR_GROUPS) == 100, len(L2_DAILY_FACTOR_GROUPS)


def get_l2_factor_group(factor_id: str) -> str:
    """返回因子所属计算组（trade/book/order/microstructure）。"""
    return L2_DAILY_FACTOR_GROUPS[factor_id][0]


def get_l2_factor_category(factor_id: str) -> str:
    """返回因子细分类（trade_basic、book_obi 等）。"""
    return L2_DAILY_FACTOR_GROUPS[factor_id][1]


def get_l2_feature_config() -> Tuple[List[str], List[str]]:
    """返回 QlibDataLoader 用的 ($l2_*, 列名) 配置。"""
    fields = [f"${fid}" for fid in L2_DAILY_FACTOR_IDS]
    names = list(L2_DAILY_FACTOR_IDS)
    return fields, names


__all__ = [
    "ALL_GROUPS",
    "FACTOR_DEFINITIONS",
    "FACTOR_VERSION",
    "L2_DAILY_FACTOR_GROUPS",
    "L2_DAILY_FACTOR_IDS",
    "factor_ids",
    "get_l2_factor_category",
    "get_l2_factor_group",
    "get_l2_feature_config",
    "run",
    "run_batch",
    "wide_columns",
]
