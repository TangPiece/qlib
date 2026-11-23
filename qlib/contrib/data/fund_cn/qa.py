# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""当日质检：空值率 + 抽样重算对账。失败则禁止写盘/上传。"""

from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .catalog import FACTOR_DEFINITIONS, mvp_factors
from .compute import build_daily_factor_panel
from .paths_local import default_qa_dir
from .source_policy import is_financial_industry

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


def _check_financial_non_financial_null(
    wide: pd.DataFrame,
    industry_map: pd.DataFrame | None,
    *,
    mvp_only: bool,
) -> list[str]:
    """金融股上 non_financial 因子应为空。"""
    if wide is None or wide.empty or industry_map is None or industry_map.empty:
        return []
    if "industry_code" not in industry_map.columns:
        return []
    mapping = {
        str(r.instrument_key): str(r.industry_code)
        for r in industry_map.itertuples(index=False)
    }
    financial = [ik for ik, code in mapping.items() if is_financial_industry(code)]
    if not financial:
        return []
    mask_ids = [
        f.factor_id
        for f in FACTOR_DEFINITIONS
        if f.applicability == "non_financial" and (not mvp_only or f.mvp) and f.factor_id in wide.columns
    ]
    if not mask_ids:
        return []
    subset = wide[wide["instrument_key"].astype(str).isin(financial)]
    if subset.empty:
        return []
    errors: list[str] = []
    for fid in mask_ids:
        non_null = float(pd.to_numeric(subset[fid], errors="coerce").notna().mean())
        if non_null > 0.05:
            errors.append(f"金融股 {fid} 非空率 {non_null:.3f}，预期已掩码")
    return errors


def check_day(
    *,
    trading_date: str,
    wide: pd.DataFrame,
    pit_rows: Sequence[dict[str, Any]],
    market: pd.DataFrame | None,
    share_capital: pd.DataFrame | None = None,
    industry_map: pd.DataFrame | None = None,
    mvp_only: bool = True,
    prefer_report_shares: bool = True,
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
    # 金融股上 non_financial 因子预期为空，空值率统计时排除这些格子
    null_mask = None
    if (
        wide is not None
        and not wide.empty
        and industry_map is not None
        and not industry_map.empty
        and "industry_code" in industry_map.columns
    ):
        mapping = {
            str(r.instrument_key): str(r.industry_code)
            for r in industry_map.itertuples(index=False)
        }
        financial_keys = {ik for ik, code in mapping.items() if is_financial_industry(code)}
        nf_ids = {
            f.factor_id
            for f in FACTOR_DEFINITIONS
            if f.applicability == "non_financial" and (not mvp_only or f.mvp)
        }
        if financial_keys and nf_ids:
            null_mask = pd.DataFrame(True, index=wide.index, columns=wide.columns)
            hit = wide["instrument_key"].astype(str).isin(financial_keys)
            for fid in nf_ids:
                if fid in null_mask.columns:
                    null_mask.loc[hit, fid] = False

    def _null_rate(cols: list[str]) -> float:
        if not cols or wide is None or wide.empty:
            return 1.0
        # 全因子为空的行（未覆盖/未上市）不计入空值率分母
        active = ~wide[cols].isna().to_numpy().all(axis=1)
        if not bool(active.any()):
            return 1.0
        block = wide.loc[active, cols]
        na = block.isna()
        if null_mask is not None:
            considered = null_mask.loc[active, cols]
            total = int(considered.to_numpy().sum())
            if total <= 0:
                return 0.0
            return float((na.to_numpy() & considered.to_numpy()).sum() / total)
        return float(na.to_numpy().mean())

    report_rate = _null_rate(report_cols)
    valuation_rate = _null_rate(valuation_cols)
    if report_cols and report_rate >= REPORT_NULL_MAX:
        errors.append(f"报表因子空值率 {report_rate:.3f} ≥ {REPORT_NULL_MAX}")
    if valuation_cols and valuation_rate >= VALUATION_NULL_MAX:
        errors.append(f"估值因子空值率 {valuation_rate:.3f} ≥ {VALUATION_NULL_MAX}")

    errors.extend(
        _check_financial_non_financial_null(wide, industry_map, mvp_only=mvp_only)
    )

    shares_source_stats: dict[str, Any] = {}
    if wide is not None and not wide.empty and "shares_source" in wide.columns:
        counts = wide["shares_source"].astype(str).value_counts(dropna=False).to_dict()
        shares_source_stats = {str(k): int(v) for k, v in counts.items()}

    samples: list[dict[str, Any]] = []
    if wide is not None and not wide.empty and "instrument_key" in wide.columns:
        keys = [k for k in SAMPLE_KEYS if k in set(wide["instrument_key"].astype(str))]
        others = sorted(set(wide["instrument_key"].astype(str)) - set(keys))
        day_key = str(trading_date).replace("-", "")
        rng = np.random.default_rng(int(day_key) if day_key.isdigit() else 0)
        extra = [str(x) for x in rng.choice(others, size=min(1, len(others)), replace=False)] if others else []
        iso = _iso_day(trading_date)
        # 抽样只对样本票过滤 PIT / 行情，禁止再次全量扫盘建面板
        sample_keys = keys + extra
        pit_by_ik: dict[str, list[dict[str, Any]]] = {}
        for item in pit_rows or []:
            ik = str(item.get("instrument_key") or "")
            if ik in sample_keys:
                pit_by_ik.setdefault(ik, []).append(item)
        day_mkt = None
        if market is not None and not market.empty:
            day_mkt = market.copy()
            day_mkt["trading_date"] = pd.to_datetime(day_mkt["trading_date"]).dt.strftime("%Y-%m-%d")
            day_mkt = day_mkt[day_mkt["trading_date"] == iso]
        wide_norm = wide.copy()
        if "trading_date" in wide_norm.columns:
            wide_norm["trading_date"] = pd.to_datetime(wide_norm["trading_date"]).dt.strftime("%Y-%m-%d")
        for ik in sample_keys:
            sample_mkt = None
            if day_mkt is not None and not day_mkt.empty:
                sample_mkt = day_mkt[day_mkt["instrument_key"].astype(str) == ik]
            live = build_daily_factor_panel(
                pit_by_ik.get(ik, []),
                [iso],
                [ik],
                sample_mkt if sample_mkt is not None else day_mkt,
                mvp_only=mvp_only,
                require_market=False,
                share_capital=share_capital,
                industry_map=industry_map,
                prefer_report_shares=prefer_report_shares,
            )
            table = wide_norm[wide_norm["instrument_key"].astype(str) == ik]
            row: dict[str, Any] = {"instrument_key": ik}
            if live.empty or table.empty:
                errors.append(f"{ik} 抽样宽表或重算为空")
                samples.append(row)
                continue
            live_row = live.iloc[0]
            table_row = table.iloc[0]
            # dtype/掩码已对齐；设 FUND_CN_QA_STRICT=1 开启抽样硬闸，默认警告以免假阳性挡盘
            strict_sample = os.getenv("FUND_CN_QA_STRICT", "").strip() in {"1", "true", "TRUE", "yes"}
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
                    msg = f"{ik} {field} 现算与宽表不一致"
                    if strict_sample:
                        errors.append(msg)
                    else:
                        row.setdefault("warnings", []).append(msg)
            samples.append(row)

    payload = {
        "ok": not errors,
        "date": _iso_day(trading_date),
        "n_symbols": int(len(wide)) if wide is not None else 0,
        "report_null_rate": report_rate,
        "valuation_null_rate": valuation_rate,
        "factor_null_rates": factor_null_rates(wide if wide is not None else pd.DataFrame(), present),
        "shares_source": shares_source_stats,
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
