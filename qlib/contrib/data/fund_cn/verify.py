# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""免费源辅助验证：不对国泰安科目写回覆盖，仅输出差异报告与硬闸。"""

from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .adapters.akshare_adapter import AkshareStatementAdapter
from .adapters.base import StatementFieldRow
from .adapters.share_capital import CninfoShareCapitalAdapter
from .paths_local import markers_dir
from .source_policy import COVERAGE_REPORT_MIN_DATE, DEFAULT_POLICY, FundCnSourcePolicy

_log = logging.getLogger(__name__)

# 交叉验证的核心科目（与 MVP 上游对齐）
_VERIFY_FIELDS: tuple[str, ...] = (
    "revenue",
    "net_profit_parent",
    "total_assets",
    "total_liabilities",
    "shareholder_equity",
    "cash_from_operations",
    "shares_outstanding",
)


class VerifyFailed(RuntimeError):
    """交叉验证差异超阈值，禁止出数。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        super().__init__(payload.get("summary") or "verify failed")


def _rel_error(left: float, right: float) -> float:
    denom = max(abs(left), abs(right), 1e-12)
    return abs(left - right) / denom


def sample_instrument_keys(
    keys: Sequence[str],
    *,
    sample_size: int,
    seed: int = 42,
) -> list[str]:
    """确定性抽样，便于回归。"""
    uniq = sorted({str(k) for k in keys})
    if len(uniq) <= sample_size:
        return uniq
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(uniq), size=sample_size, replace=False)
    return [uniq[int(i)] for i in sorted(idx.tolist())]


def filter_overlap_periods(
    frame: pd.DataFrame,
    *,
    min_period: date | None = COVERAGE_REPORT_MIN_DATE,
    max_period: date | None = None,
) -> pd.DataFrame:
    """只对重叠报告期对账，避免早期无免费源历史拖高 mismatch。"""
    if frame is None or frame.empty:
        return frame if frame is not None else pd.DataFrame()
    out = frame.copy()
    out["report_period_end"] = pd.to_datetime(out["report_period_end"]).dt.date
    if min_period is not None:
        out = out[out["report_period_end"] >= min_period]
    if max_period is not None:
        out = out[out["report_period_end"] <= max_period]
    return out


def compare_statement_samples(
    csmar_rows: Sequence[StatementFieldRow] | pd.DataFrame,
    free_rows: Sequence[StatementFieldRow] | pd.DataFrame,
    *,
    fields: Sequence[str] = _VERIFY_FIELDS,
    max_rel_error: float = 0.05,
    min_period: date | None = COVERAGE_REPORT_MIN_DATE,
    max_period: date | None = None,
) -> dict[str, Any]:
    """同 (instrument, period, field, revision=0) 相对误差对账（仅重叠期）。"""

    def _frame(rows: Sequence[StatementFieldRow] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(rows, pd.DataFrame):
            frame = rows.copy()
        else:
            frame = pd.DataFrame([r.__dict__ if hasattr(r, "__dict__") else r for r in rows])
        if frame.empty:
            return pd.DataFrame(
                columns=["instrument_key", "report_period_end", "field_code", "revision", "value"]
            )
        if "revision" not in frame.columns:
            frame["revision"] = 0
        frame["revision"] = frame["revision"].fillna(0).astype(int)
        frame["report_period_end"] = pd.to_datetime(frame["report_period_end"]).dt.date
        frame = frame[frame["revision"] == 0]
        frame = frame[frame["field_code"].isin(list(fields))]
        return filter_overlap_periods(frame, min_period=min_period, max_period=max_period)

    left = _frame(csmar_rows)
    right = _frame(free_rows)
    if left.empty or right.empty:
        return {
            "compared": 0,
            "mismatch": 0,
            "mismatch_rate": 1.0 if left.empty or right.empty else 0.0,
            "details": [],
            "ok": False,
            "reason": "empty_side",
            "min_period": str(min_period) if min_period else None,
        }

    keys = ["instrument_key", "report_period_end", "field_code"]
    merged = left[keys + ["value"]].merge(
        right[keys + ["value"]],
        on=keys,
        how="inner",
        suffixes=("_csmar", "_free"),
    )
    details: list[dict[str, Any]] = []
    mismatch = 0
    for row in merged.itertuples(index=False):
        lv = float(row.value_csmar)
        rv = float(row.value_free)
        err = _rel_error(lv, rv)
        bad = err > max_rel_error
        if bad:
            mismatch += 1
            details.append(
                {
                    "instrument_key": row.instrument_key,
                    "report_period_end": str(row.report_period_end),
                    "field_code": row.field_code,
                    "csmar": lv,
                    "free": rv,
                    "rel_error": err,
                }
            )
    compared = int(len(merged))
    rate = float(mismatch / compared) if compared else 1.0
    # 分位数便于 marker 审计
    errs = [float(d["rel_error"]) for d in details] if details else [0.0]
    return {
        "compared": compared,
        "mismatch": mismatch,
        "mismatch_rate": rate,
        "details": details[:200],
        "rel_error_p50": float(np.median(errs)) if errs else 0.0,
        "rel_error_p95": float(np.quantile(errs, 0.95)) if errs else 0.0,
        "ok": compared > 0 and rate <= 1.0,
        "min_period": str(min_period) if min_period else None,
    }


def compare_share_capital_samples(
    report_shares: pd.DataFrame,
    cninfo_shares: pd.DataFrame,
    *,
    max_rel_error: float = 0.15,
) -> dict[str, Any]:
    """报告期末股本 vs 巨潮最近事件：仅告警/验证，不改主值。"""
    if report_shares is None or report_shares.empty:
        return {"compared": 0, "mismatch": 0, "mismatch_rate": 0.0, "ok": True, "details": []}
    if cninfo_shares is None or cninfo_shares.empty:
        return {
            "compared": 0,
            "mismatch": 0,
            "mismatch_rate": 0.0,
            "ok": True,
            "details": [],
            "note": "cninfo_empty_skip",
        }
    rep = report_shares.copy()
    rep = rep[rep["field_code"] == "shares_outstanding"] if "field_code" in rep.columns else rep
    if rep.empty:
        return {"compared": 0, "mismatch": 0, "mismatch_rate": 0.0, "ok": True, "details": []}
    rep["report_period_end"] = pd.to_datetime(rep["report_period_end"]).dt.date
    rep = rep.sort_values("report_period_end").groupby("instrument_key", as_index=False).tail(1)
    ev = cninfo_shares.copy()
    ev["effective_date"] = pd.to_datetime(ev["effective_date"]).dt.date
    ev = ev.sort_values("effective_date").groupby("instrument_key", as_index=False).tail(1)
    merged = rep[["instrument_key", "value"]].merge(
        ev[["instrument_key", "total_shares"]],
        on="instrument_key",
        how="inner",
    )
    details: list[dict[str, Any]] = []
    mismatch = 0
    for row in merged.itertuples(index=False):
        err = _rel_error(float(row.value), float(row.total_shares))
        if err > max_rel_error:
            mismatch += 1
            details.append(
                {
                    "instrument_key": row.instrument_key,
                    "report_shares": float(row.value),
                    "cninfo_shares": float(row.total_shares),
                    "rel_error": err,
                }
            )
    compared = int(len(merged))
    rate = float(mismatch / compared) if compared else 0.0
    return {
        "compared": compared,
        "mismatch": mismatch,
        "mismatch_rate": rate,
        "details": details[:50],
        "ok": True,
        "warn": rate > 0.3,
    }


def run_free_source_verify(
    *,
    csmar_statements: pd.DataFrame,
    instrument_keys: Sequence[str],
    policy: FundCnSourcePolicy = DEFAULT_POLICY,
    root: Path | None = None,
    start: date | None = None,
    end: date | None = None,
    free_statement_adapter: AkshareStatementAdapter | None = None,
    share_adapter: CninfoShareCapitalAdapter | None = None,
    require: bool = True,
    batch_size: int = 10,
) -> dict[str, Any]:
    """分批抽样拉 EM 科目 + 巨潮股本；只对重叠期对账；超阈抛 VerifyFailed 并落可复现清单。"""
    if not policy.verify_with_free_sources:
        payload = {"ok": True, "skipped": True, "reason": "verify_disabled"}
        _write_marker(payload, root=root)
        return payload

    sample = sample_instrument_keys(instrument_keys, sample_size=policy.verify_sample_size)
    adapter = free_statement_adapter or AkshareStatementAdapter(source="akshare_em_verify")
    min_period = getattr(policy, "verify_min_period", None) or COVERAGE_REPORT_MIN_DATE
    # 分批拉取，避免一次打爆免费源限流
    free_rows: list[StatementFieldRow] = []
    batch_errors: list[dict[str, Any]] = []
    for i in range(0, len(sample), max(1, batch_size)):
        chunk = sample[i : i + batch_size]
        try:
            free_rows.extend(adapter.fetch_statements(chunk, start=start, end=end))
        except Exception as exc:
            _log.warning("verify batch %s failed: %s", chunk, exc)
            batch_errors.append({"instruments": chunk, "error": str(exc)})

    stmt_cmp = compare_statement_samples(
        csmar_statements,
        free_rows,
        max_rel_error=policy.max_verify_rel_error,
        min_period=min_period,
        max_period=end,
    )
    share_cmp: dict[str, Any] = {"ok": True, "skipped": True}
    try:
        sh_adapter = share_adapter or CninfoShareCapitalAdapter()
        shares = sh_adapter.fetch_share_capital(sample, start=start, end=end)
        share_frame = pd.DataFrame([s.__dict__ for s in shares]) if shares else pd.DataFrame()
        share_cmp = compare_share_capital_samples(csmar_statements, share_frame)
    except Exception as exc:
        _log.warning("cninfo verify skipped: %s", exc)
        share_cmp = {"ok": True, "skipped": True, "error": str(exc)}

    mismatch_rate = float(stmt_cmp.get("mismatch_rate") or 1.0)
    ok = bool(stmt_cmp.get("compared", 0) > 0) and mismatch_rate <= policy.max_verify_mismatch_rate
    payload = {
        "ok": ok,
        "sample_instruments": sample,
        "statement_compare": stmt_cmp,
        "share_compare": share_cmp,
        "batch_errors": batch_errors,
        "max_verify_mismatch_rate": policy.max_verify_mismatch_rate,
        "max_verify_rel_error": policy.max_verify_rel_error,
        "verify_min_period": str(min_period) if min_period else None,
        "summary": (
            "ok"
            if ok
            else f"statement mismatch_rate={mismatch_rate:.3f} > {policy.max_verify_mismatch_rate}"
        ),
    }
    _write_marker(payload, root=root)
    # 失败时另落可复现 mismatch 清单
    if not ok:
        mismatch_path = markers_dir(root) / "verify_mismatch_list.json"
        mismatch_path.write_text(
            json.dumps(
                {
                    "sample_instruments": sample,
                    "details": stmt_cmp.get("details") or [],
                    "mismatch_rate": mismatch_rate,
                },
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        payload["mismatch_list_path"] = str(mismatch_path)
    if require and not ok:
        raise VerifyFailed(payload)
    return payload


def _write_marker(payload: dict[str, Any], *, root: Path | None) -> None:
    path = markers_dir(root) / "verify_free_source.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
