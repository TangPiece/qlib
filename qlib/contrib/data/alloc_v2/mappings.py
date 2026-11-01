# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""映射表：避免 ETF 联接双计、跨境混算、主体重复统计。

空表也写出列结构，后续探活/手工补全只填行，不改主键。
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .store import read_map, write_map

ETF_MASTER_COLUMNS = (
    "etf_code",
    "name",
    "benchmark",
    "primary_industry",
    "theme",
    "is_cross_border",
    "feeder_target",
    "exchange",
    "valid_from",
    "valid_to",
    "enabled",
    "notes",
)

ETF_INDUSTRY_WEIGHT_COLUMNS = (
    "etf_code",
    "industry_id",
    "weight",
    "asof",
    "source",
    "notes",
)

FEEDER_ETF_COLUMNS = (
    "feeder_code",
    "target_code",
    "valid_from",
    "valid_to",
    "notes",
)

ENTITY_MAP_COLUMNS = (
    "entity_id",
    "display_name",
    "entity_type",
    "parent_id",
    "group_id",
    "account_kind",
    "valid_from",
    "valid_to",
    "notes",
)

CROSS_BORDER_COLUMNS = (
    "etf_code",
    "event_date",
    "sub_status",
    "redeem_status",
    "quota_note",
    "overseas_open",
    "fx_rate",
    "iopv_ts",
    "nav_ts",
    "quality_flag",
    "notes",
)

# 宽基 / 行业 / 主题 / 黄金国债：代码以探活为准，联接与跨境先标清楚。
_SEED_ETFS: tuple[dict, ...] = (
    {"etf_code": "510050", "name": "上证50ETF", "benchmark": "000016", "primary_industry": "", "theme": "large", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "510300", "name": "沪深300ETF", "benchmark": "000300", "primary_industry": "", "theme": "broad", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "510500", "name": "中证500ETF", "benchmark": "000905", "primary_industry": "", "theme": "mid", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "512100", "name": "中证1000ETF", "benchmark": "000852", "primary_industry": "", "theme": "small", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "159922", "name": "中证2000ETF", "benchmark": "932000", "primary_industry": "", "theme": "micro", "is_cross_border": False, "feeder_target": "", "exchange": "SZ"},
    {"etf_code": "159915", "name": "创业板ETF", "benchmark": "399006", "primary_industry": "", "theme": "chinext", "is_cross_border": False, "feeder_target": "", "exchange": "SZ"},
    {"etf_code": "588000", "name": "科创50ETF", "benchmark": "000688", "primary_industry": "", "theme": "star50", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "512050", "name": "A500ETF", "benchmark": "000510", "primary_industry": "", "theme": "a500", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "518880", "name": "黄金ETF", "benchmark": "AU9999", "primary_industry": "gold", "theme": "gold", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "511260", "name": "十年国债ETF", "benchmark": "CGB10Y", "primary_industry": "bond", "theme": "bond", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "512480", "name": "半导体ETF", "benchmark": "H30184", "primary_industry": "801080", "theme": "semiconductor", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "516160", "name": "新能源ETF", "benchmark": "931151", "primary_industry": "801730", "theme": "new_energy", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "512800", "name": "银行ETF", "benchmark": "399986", "primary_industry": "801780", "theme": "bank", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "512880", "name": "证券ETF", "benchmark": "399975", "primary_industry": "801790", "theme": "broker", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "159928", "name": "消费ETF", "benchmark": "000932", "primary_industry": "801120", "theme": "consumer", "is_cross_border": False, "feeder_target": "", "exchange": "SZ"},
    {"etf_code": "515080", "name": "中证红利ETF", "benchmark": "000922", "primary_industry": "", "theme": "dividend", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "510880", "name": "红利ETF", "benchmark": "000015", "primary_industry": "", "theme": "dividend", "is_cross_border": False, "feeder_target": "", "exchange": "SH"},
    # 跨境 ETF：不与境内混算 IndustryETFFlow20
    {"etf_code": "513100", "name": "纳指ETF", "benchmark": "NDX", "primary_industry": "", "theme": "us_tech", "is_cross_border": True, "feeder_target": "", "exchange": "SH"},
    {"etf_code": "159920", "name": "恒生ETF", "benchmark": "HSI", "primary_industry": "", "theme": "hk", "is_cross_border": True, "feeder_target": "", "exchange": "SZ"},
    # 联接基金示例：规模不得与目标 ETF 份额相加
    {"etf_code": "000051", "name": "华夏沪深300ETF联接", "benchmark": "000300", "primary_industry": "", "theme": "feeder", "is_cross_border": False, "feeder_target": "510300", "exchange": "OF"},
)

_SEED_FEEDERS: tuple[dict, ...] = (
    {"feeder_code": "000051", "target_code": "510300", "valid_from": date(2012, 8, 14), "valid_to": None, "notes": "联接规模变化不与目标份额相加"},
)

_SEED_ENTITIES: tuple[dict, ...] = (
    {"entity_id": "mutual_fund", "display_name": "公募基金", "entity_type": "channel", "parent_id": "", "group_id": "", "account_kind": "product", "valid_from": date(2000, 1, 1), "valid_to": None, "notes": "按产品统计，不按管理人重复加总"},
    {"entity_id": "insurance_group", "display_name": "保险集团", "entity_type": "group", "parent_id": "", "group_id": "insurance", "account_kind": "group", "valid_from": date(2000, 1, 1), "valid_to": None, "notes": "集团/资管/产品不得双计"},
    {"entity_id": "northbound", "display_name": "沪深股通", "entity_type": "channel", "parent_id": "", "group_id": "foreign", "account_kind": "channel", "valid_from": date(2014, 11, 17), "valid_to": None, "notes": "通道口径，不是单一外资主体"},
    {"entity_id": "private_fund", "display_name": "私募", "entity_type": "channel", "parent_id": "", "group_id": "", "account_kind": "unavailable", "valid_from": date(2000, 1, 1), "valid_to": None, "notes": "无完整持仓，保持 unavailable"},
    {"entity_id": "wealth_mgmt", "display_name": "银行理财", "entity_type": "channel", "parent_id": "", "group_id": "", "account_kind": "unavailable", "valid_from": date(2000, 1, 1), "valid_to": None, "notes": "无完整持仓，保持 unavailable"},
    {"entity_id": "broker_prop", "display_name": "券商自营", "entity_type": "channel", "parent_id": "", "group_id": "", "account_kind": "unavailable", "valid_from": date(2000, 1, 1), "valid_to": None, "notes": "无完整持仓，保持 unavailable"},
)


def empty_frame(columns: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def seed_etf_master() -> pd.DataFrame:
    rows = []
    for item in _SEED_ETFS:
        rows.append(
            {
                **item,
                "valid_from": date(2018, 1, 1),
                "valid_to": None,
                "enabled": True,
                "notes": "跨境不混算境内行业流；联接不与目标相加" if item["is_cross_border"] or item["feeder_target"] else "",
            }
        )
    return pd.DataFrame(rows, columns=list(ETF_MASTER_COLUMNS))


def seed_feeder() -> pd.DataFrame:
    return pd.DataFrame(list(_SEED_FEEDERS), columns=list(FEEDER_ETF_COLUMNS))


def seed_entity_map() -> pd.DataFrame:
    return pd.DataFrame(list(_SEED_ENTITIES), columns=list(ENTITY_MAP_COLUMNS))


def seed_industry_weight() -> pd.DataFrame:
    """成分股权重映射主口径；主行业分类仅对照，两套不得加总。空表先占位。"""
    return empty_frame(ETF_INDUSTRY_WEIGHT_COLUMNS)


def seed_cross_border() -> pd.DataFrame:
    return empty_frame(CROSS_BORDER_COLUMNS)


def write_empty_maps(root: Path | None = None) -> dict[str, str]:
    """写出映射表（有种子, 无则空结构）。"""
    paths = {
        "etf_master": write_map(seed_etf_master(), name="etf_master", root=root),
        "etf_industry_weight": write_map(seed_industry_weight(), name="etf_industry_weight", root=root),
        "feeder_etf": write_map(seed_feeder(), name="feeder_etf", root=root),
        "entity_map": write_map(seed_entity_map(), name="entity_map", root=root),
        "cross_border_status": write_map(seed_cross_border(), name="cross_border_status", root=root),
    }
    return {key: str(path) for key, path in paths.items()}


def load_etf_master(root: Path | None = None) -> pd.DataFrame:
    frame = read_map("etf_master", root=root)
    return frame if not frame.empty else seed_etf_master()


def load_feeder(root: Path | None = None) -> pd.DataFrame:
    frame = read_map("feeder_etf", root=root)
    return frame if not frame.empty else seed_feeder()


def domestic_etf_codes(master: pd.DataFrame | None = None) -> list[str]:
    """境内、非联接 ETF，才允许进入 IndustryETFFlow20。"""
    frame = master if master is not None else seed_etf_master()
    mask = (~frame["is_cross_border"].fillna(False)) & (frame["feeder_target"].fillna("") == "")
    mask &= frame["theme"].isin({"semiconductor", "new_energy", "bank", "broker", "consumer", "dividend"})
    return [str(code) for code in frame.loc[mask, "etf_code"]]


def broad_etf_codes(master: pd.DataFrame | None = None) -> list[str]:
    frame = master if master is not None else seed_etf_master()
    mask = (~frame["is_cross_border"].fillna(False)) & (frame["feeder_target"].fillna("") == "")
    mask &= frame["theme"].isin({"broad", "large", "mid", "small", "micro", "chinext", "star50", "a500"})
    return [str(code) for code in frame.loc[mask, "etf_code"]]


def feeder_codes(master: pd.DataFrame | None = None) -> set[str]:
    frame = master if master is not None else seed_etf_master()
    return {str(code) for code in frame.loc[frame["feeder_target"].fillna("") != "", "etf_code"]}


def cross_border_codes(master: pd.DataFrame | None = None) -> set[str]:
    frame = master if master is not None else seed_etf_master()
    return {str(code) for code in frame.loc[frame["is_cross_border"].fillna(False), "etf_code"]}
