# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""当日质检：空值率 + 抽样重算对账。失败则禁止写盘/上传。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .catalog import FACTOR_DEFINITIONS, mvp_factors
from .compute import build_daily_factor_panel
from .paths_local import default_qa_dir

REPORT_NULL_MAX = 0.30
VALUATION_NULL_MAX = 0.50
SAMPLE_KEYS = ("CNStock:600000", "CNStock:000001")


class DayQAFailed(RuntimeError):
    """单日质检未通过。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__("; ".join(payload.get("errors") or ["day qa failed"]))


def _factor_ids(*, mvp_only: bool) -> list[str]:
    if mvp_only:
        return [f.factor_id for f in mvp_factors()]
    return [f.factor_id for f in FACTOR_DEFINITIONS]


def _kind_ids(kind: str, *, mvp_only: bool) -> list[str]:
    wanted = set(_factor_ids(mvp_only=mvp_only))
    return [f.factor_id for f in FACTOR_DEFINITIONS if f.factor_id in wanted and f.kind == kind]


def factor_null_rates(wide: pd.DataFrame, columns: Sequence[str]) -> dict[str, float]:
    """各因子列空值率。"""
    out: dict[str, float] = {}
    for column in columns:
        if column not in wide.columns:
            out[column] = 1.0
            continue
        out[column] = float(pd.to_numeric(wide[column], errors="coerce").isna().mean())
    return out


def _close(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return bool(np.isclose(float(left), float(right), rtol=1e-5, atol=1e-8))


def _iso_day(value: Any) -> str:
    text = str(value).replace("-", "")
    if len(text) == 8 and text.isdigit():
        return f"{text[:4]}-{text[4:6]}-{text[6:8]}"
    return str(value)[:10]


def check_day(
    *,
    trading_date: str,
    wide: pd.DataFrame,
    pit_rows: Sequence[dict[str, Any]],
    market: pd.DataFrame | None,
    share_capital: pd.DataFrame | None = None,
    mvp_only: bool = True,
    qa_dir: Path | None = None,
) -> dict[str, Any]:
    """检查一日宽表。返回 {ok, errors, ...}。"""
    errors: list[str] = []
    columns = _factor_ids(mvp_only=mvp_only)
    present = [c for c in columns if c in wide.columns]
    if wide is None or wide.empty:
        errors.append("当日宽表为空")
    elif not present:
        errors.append("宽表缺少因子列")
    else:
        all_nan = bool(pd.DataFrame(wide[present]).isna().to_numpy().all())
        if all_nan:
            errors.append("全部因子全 NaN")

    report_cols = [c for c in _kind_ids("report", mvp_only=mvp_only) if c in (wide.columns if wide is not None else [])]
    valuation_cols = [c for c in _kind_ids("valuation", mvp_only=mvp_only) if c in (wide.columns if wide is not None else [])]
    report_rate = float(wide[report_cols].isna().to_numpy().mean()) if report_cols and wide is not None and not wide.empty else 1.0
    valuation_rate = (
        float(wide[valuation_cols].isna().to_numpy().mean()) if valuation_cols and wide is not None and not wide.empty else 1.0
    )
    if report_cols and report_rate >= REPORT_NULL_MAX:
        errors.append(f"报表因子空值率 {report_rate:.3f} ≥ {REPORT_NULL_MAX}")
    if valuation_cols and valuation_rate >= VALUATION_NULL_MAX:
        errors.append(f"估值因子空值率 {valuation_rate:.3f} ≥ {VALUATION_NULL_MAX}")

    samples: list[dict[str, Any]] = []
    if wide is not None and not wide.empty and "instrument_key" in wide.columns:
        keys = [k for k in SAMPLE_KEYS if k in set(wide["instrument_key"].astype(str))]
        others = sorted(set(wide["instrument_key"].astype(str)) - set(keys))
        day_key = str(trading_date).replace("-", "")
        rng = np.random.default_rng(int(day_key) if day_key.isdigit() else 0)
        extra = [str(x) for x in rng.choice(others, size=min(1, len(others)), replace=False)] if others else []
        iso = _iso_day(trading_date)
        for ik in keys + extra:
            live = build_daily_factor_panel(
                pit_rows,
                [iso],
                [ik],
                market,
                mvp_only=mvp_only,
                require_market=False,
                share_capital=share_capital,
            )
            table = wide[wide["instrument_key"].astype(str) == ik]
            row: dict[str, Any] = {"instrument_key": ik}
            if live.empty or table.empty:
                errors.append(f"{ik} 抽样宽表或重算为空")
                samples.append(row)
                continue
            live_row = live.iloc[0]
            table_row = table.iloc[0]
            for field in present[:8]:
                live_val = live_row[field] if field in live_row.index else float("nan")
                table_val = table_row[field] if field in table_row.index else float("nan")
                match = _close(live_val, table_val)
                row[field] = {
                    "live": None if pd.isna(live_val) else float(live_val),
                    "wide": None if pd.isna(table_val) else float(table_val),
                    "match": match,
                }
                if not match:
                    errors.append(f"{ik} {field} 现算与宽表不一致")
            samples.append(row)

    payload = {
        "ok": not errors,
        "date": _iso_day(trading_date),
        "n_symbols": int(len(wide)) if wide is not None else 0,
        "report_null_rate": report_rate,
        "valuation_null_rate": valuation_rate,
        "factor_null_rates": factor_null_rates(wide if wide is not None else pd.DataFrame(), present),
        "samples": samples,
        "errors": errors,
    }
    out_dir = Path(qa_dir) if qa_dir is not None else default_qa_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    day = str(trading_date).replace("-", "")
    (out_dir / f"fund_cn_day_qa_{day}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return payload


def require_day_ok(payload: dict[str, Any]) -> dict[str, Any]:
    """未通过则抛 DayQAFailed。"""
    if not payload.get("ok"):
        raise DayQAFailed(payload)
    return payload
