# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""V2.2 指标目录：从 registry 导出免费/禁用分栏，写入 maps。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .paths import maps_dir
from .registry import DATASETS, DatasetDef

# dataset → 授权标签（与 V22_INDICATOR_CATALOG.md 对齐）
_LICENSE: dict[str, str] = {
    "idx_global": "free",
    "yield_cn_us": "free",
    "fx_usdcny": "free",
    "gold_sge": "free",
    "gold_gld": "free",
    "cn_macro": "free",
    "vix_cboe": "free",
    "dxy": "free",
    "oil_brent": "free",
    "oil_wti": "free",
    "silver_sge": "free",
    "silver_etf": "free",
    "crypto_btc": "free",
    "us_real_rate": "free",
    "fed_balance": "free",
    "hsgt_turnover": "free",
    "hsgt_hold": "free",
    "hsgt_net": "free",
    "etf_bar": "free",
    "etf_nav": "free",
    "etf_shares": "free",
    "etf_status": "free",
    "sw_industry": "free",
    "industry_member": "free",
    "fund_holding": "free",
    "fund_scale_snapshot": "free",
    "lhb": "free",
    "block_trade": "free",
    "margin": "free",
    "holder_event": "free",
    "cftc_cot": "needs_license",
    "lme_metals": "needs_license",
    "comex_inventory": "needs_license",
    "ice_brent_depth": "needs_license",
    "baltic_freight": "needs_license",
    "smm_spot": "needs_license",
    "mysteel_steel": "needs_license",
    "wind_macro": "needs_license",
    "commodity_cn_specialty": "needs_license",
    "crypto_btc_flow": "unavailable",
    "qdii_scale": "unavailable",
    "ici_mmf": "unavailable",
    "epfr_fund": "unavailable",
    "fed_z1": "unavailable",
    "crb": "unavailable",
    "private_hold": "unavailable",
    "index_rebalance": "unavailable",
}


def _license_of(row: DatasetDef) -> str:
    if row.dataset in _LICENSE:
        return _LICENSE[row.dataset]
    if not row.enabled or row.role == "disabled":
        return "unavailable"
    return "free"


def indicator_catalog_frame() -> pd.DataFrame:
    """每个 registry dataset 一行：主源、频率、授权、启用状态。"""
    records: list[dict[str, Any]] = []
    for row in DATASETS:
        records.append(
            {
                "dataset": row.dataset,
                "display_name": row.display_name,
                "layer": row.layer,
                "source": row.source,
                "adapter": row.adapter,
                "freq": row.freq,
                "publish_lag": row.publish_lag,
                "evidence_class": row.evidence_class,
                "license": _license_of(row),
                "enabled": bool(row.enabled),
                "role": row.role,
                "notes": row.notes,
            }
        )
    return pd.DataFrame(records)


def write_indicator_catalog(root: Path | None = None) -> Path:
    """写出 maps/v22_indicator_catalog.csv，供 Phase A 验收。"""
    frame = indicator_catalog_frame()
    path = maps_dir(root) / "v22_indicator_catalog.csv"
    frame.to_csv(path, index=False)
    return path
