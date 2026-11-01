# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""把长表做成监测面板：窗口变化、价格调整流量、并发配置证据。

周/月/季序列保持原日期，不向前填充成假日频。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .catalog import series_by_id
from .diagnostics import (
    allocation_evidence_matrix,
    class_flow_scores,
    classify_price_vs_flow,
    price_adjusted_flow,
    share_to_flow_value,
    window_change,
    window_return,
)
from .paths_local import diag_dir, panel_dir
from .staging import read_long_rows, write_marker

WINDOWS = (5, 20, 60)


def _to_series(frame: pd.DataFrame, series_id: str) -> pd.Series:
    part = frame[frame["series_id"] == series_id]
    if part.empty:
        return pd.Series(dtype="float64")
    out = part.copy()
    out["date"] = pd.to_datetime(out["date"])
    return out.set_index("date")["value"].sort_index()


def build_change_table(long_df: pd.DataFrame) -> pd.DataFrame:
    """每条序列的最新值与 5/20/60 日变化；缺失窗口留空。"""
    lookup = series_by_id()
    rows: list[dict[str, Any]] = []
    for series_id, part in long_df.groupby("series_id"):
        spec = lookup.get(str(series_id))
        series = _to_series(part, str(series_id))
        if series.empty:
            continue
        row: dict[str, Any] = {
            "series_id": series_id,
            "display_name": spec.display_name if spec else series_id,
            "layer": spec.layer if spec else "",
            "asset_class": spec.asset_class if spec else "",
            "region": spec.region if spec else "",
            "investor": spec.investor if spec else "",
            "evidence_grade": spec.evidence_grade if spec else "",
            "freq": spec.freq if spec else "",
            "unit": spec.unit if spec else "",
            "asof": series.index.max().date().isoformat(),
            "last": float(series.iloc[-1]),
            "n_obs": int(len(series)),
            "notes": spec.notes if spec else "",
        }
        for days in WINDOWS:
            row[f"chg_{days}d"] = window_change(series, days)
            if spec is not None and spec.unit in {"price", "index", "cny/g"}:
                row[f"ret_{days}d"] = window_return(series, days)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["layer", "asset_class", "series_id"]).reset_index(drop=True)


def build_flow_vs_price(long_df: pd.DataFrame, *, days: int = 20) -> pd.DataFrame:
    """对有配对价格的份额/市值序列做价格调整，并标记 PRICE_DRIVEN 等。"""
    lookup = series_by_id()
    rows: list[dict[str, Any]] = []
    for series_id, part in long_df.groupby("series_id"):
        spec = lookup.get(str(series_id))
        if spec is None or not spec.pair_series_id:
            continue
        left = _to_series(part, str(series_id))
        right = _to_series(long_df, spec.pair_series_id)
        pair = lookup.get(spec.pair_series_id)
        price_series = right if pair is not None and pair.unit in {"price", "index", "cny/g"} else left
        level_series = left if spec.unit not in {"price", "index", "cny/g"} else right
        if spec.unit == "share":
            flow = share_to_flow_value(level_series, price_series)
            flow_window = window_change(level_series, days)
        elif spec.evidence_grade == "L2":
            flow = price_adjusted_flow(level_series, price_series)
            flow_window = window_change(flow, days) if not flow.empty else None
        else:
            flow = pd.Series(dtype="float64")
            flow_window = window_change(level_series, days)
        price_ret = window_return(price_series, days)
        amount_id = spec.series_id.replace("_shares", "_amount").replace("_hold_mcap", "_amount")
        amount_chg = window_change(_to_series(long_df, amount_id), days) if amount_id in set(long_df["series_id"]) else None
        flag = classify_price_vs_flow(
            price_change=price_ret,
            flow=flow_window,
            amount_change=amount_chg,
            has_l1=spec.evidence_grade == "L1",
        )
        rows.append(
            {
                "series_id": spec.series_id,
                "pair_series_id": spec.pair_series_id,
                "display_name": spec.display_name,
                "evidence_grade": spec.evidence_grade,
                "layer": spec.layer,
                "asset_class": spec.asset_class,
                "window_days": days,
                "level_change": window_change(level_series, days),
                "price_return": price_ret,
                "flow_est": None if flow.empty else window_change(flow, days),
                "flow_proxy": flow_window,
                "amount_change": amount_chg,
                "flag": flag,
                "asof": None if level_series.empty else level_series.index.max().date().isoformat(),
                "notes": spec.notes,
            }
        )
    return pd.DataFrame(rows)


def build_allocation_table(long_df: pd.DataFrame, *, days: int = 20) -> pd.DataFrame:
    scores = class_flow_scores(long_df, days=days)
    matrix = allocation_evidence_matrix(scores)
    out = matrix.reset_index().rename(columns={"index": "from_asset"})
    out.insert(0, "window_days", days)
    out.insert(1, "disclaimer", "并发配置变化证据，不是已证实的资金去向")
    for name, score in scores.items():
        out[f"score_{name}"] = score
    return out


def write_compute_outputs(long_df: pd.DataFrame, *, root: Path | None = None) -> dict[str, Any]:
    """写面板与诊断表；空数据也写空文件，方便笔记本判断。"""
    changes = build_change_table(long_df)
    flow_vs_price = build_flow_vs_price(long_df)
    alloc = build_allocation_table(long_df)
    panels = panel_dir(root)
    diags = diag_dir(root)
    long_path = panels / "long.parquet"
    long_df.to_parquet(long_path, index=False)
    changes_path = diags / "changes.parquet"
    flow_path = diags / "flow_vs_price.parquet"
    alloc_path = diags / "allocation_evidence.parquet"
    changes.to_parquet(changes_path, index=False)
    flow_vs_price.to_parquet(flow_path, index=False)
    alloc.to_parquet(alloc_path, index=False)
    summary = {
        "n_rows": int(len(long_df)),
        "n_series": int(long_df["series_id"].nunique()) if not long_df.empty else 0,
        "n_change_rows": int(len(changes)),
        "n_flow_vs_price": int(len(flow_vs_price)),
        "paths": {
            "long": str(long_path),
            "changes": str(changes_path),
            "flow_vs_price": str(flow_path),
            "allocation": str(alloc_path),
        },
    }
    write_marker("compute", summary, root=root)
    return summary


def run_compute(root: Path | None = None) -> dict[str, Any]:
    return write_compute_outputs(read_long_rows(root), root=root)
