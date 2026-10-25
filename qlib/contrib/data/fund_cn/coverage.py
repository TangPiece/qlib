"""Coverage / quality gates — 出数前挡住公告/数值缺失超阈值的批次。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .catalog import FACTOR_DEFINITIONS, FundFactorDef, mvp_factors

# Map every catalog metric_deps entry → upstream raw statement field_code(s).
# Missing keys must raise (never silently fail coverage with empty frames).
# "|" means alternative sources; "+" means every component is required.
_GROSS = "gross_profit|operating_cost"
_OPEX = "operating_expenses|selling_expense+admin_expense+rd_expense+financial_expense"
METRIC_TO_STATEMENT_FIELDS: dict[str, tuple[str, ...]] = {
    # Ratios / derived metrics
    "roe": ("net_profit_parent", "shareholder_equity"),
    "roa": ("net_profit_parent", "total_assets"),
    "gross_margin": (_GROSS, "revenue"),
    "net_margin": ("net_profit_parent", "revenue"),
    "operating_margin": ("operating_profit", "revenue"),
    "expense_ratio": (_OPEX, "revenue"),
    "debt_to_asset": ("total_liabilities", "total_assets"),
    "current_ratio": ("current_assets", "current_liabilities"),
    "interest_bearing_debt_ratio": ("interest_bearing_debt", "total_assets"),
    "ar_to_revenue": ("accounts_receivable", "revenue"),
    "inventory_to_revenue": ("inventory", "revenue"),
    "asset_turnover": ("revenue", "total_assets"),
    "ocf_to_net_profit": ("cash_from_operations", "net_profit_parent"),
    "ocf_to_revenue": ("cash_from_operations", "revenue"),
    "capex_ratio": ("capex", "revenue"),
    "accruals_ratio": ("net_profit_parent", "cash_from_operations", "total_assets"),
    "roic": ("operating_profit", "shareholder_equity", "interest_bearing_debt"),
    "earnings_volatility": ("net_profit_parent",),
    # Growth
    "revenue_yoy": ("revenue",),
    "net_profit_yoy": ("net_profit_parent",),
    "deducted_net_profit_yoy": ("net_profit_deducted",),
    "total_assets_yoy": ("total_assets",),
    "equity_yoy": ("shareholder_equity",),
    "ocf_yoy": ("cash_from_operations",),
    # TTM / levels used as deps
    "net_income_ttm": ("net_profit_parent",),
    "revenue_ttm": ("revenue",),
    "cash_from_operations_ttm": ("cash_from_operations",),
    "cash_dividend_ttm": ("cash_dividend",),
    "operating_profit_ttm": ("operating_profit",),
    "free_cash_flow_ttm": ("cash_from_operations", "capex"),
    "gross_profit_ttm": (_GROSS, "revenue"),
    "operating_expenses_ttm": (_OPEX,),
    "capex_ttm": ("capex",),
    "free_cash_flow": ("cash_from_operations", "capex"),
    "shares_outstanding": ("shares_outstanding",),
    "shareholder_equity": ("shareholder_equity",),
    "cash_dividend": ("cash_dividend",),
}


@dataclass
class CoverageReport:
    metric_code: str
    periods: int
    instruments: int
    missing_announce_rate: float
    value_missing_rate: float
    raw_ok: bool
    pit_ok: bool
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric_code": self.metric_code,
            "periods": self.periods,
            "instruments": self.instruments,
            "missing_announce_rate": self.missing_announce_rate,
            "value_missing_rate": self.value_missing_rate,
            "RAW_OK": "ok" if self.raw_ok else "fail",
            "PIT_OK": "ok" if self.pit_ok else "fail",
            "details": self.details,
        }


DEFAULT_MAX_MISSING_ANNOUNCE = 0.05
DEFAULT_MAX_VALUE_MISSING = 0.30


def resolve_statement_fields(metric_dep: str) -> tuple[str, ...]:
    """Resolve a catalog metric_deps entry to raw statement fields; raise if unmapped."""
    if metric_dep not in METRIC_TO_STATEMENT_FIELDS:
        raise KeyError(
            f"metric_deps {metric_dep!r} missing from METRIC_TO_STATEMENT_FIELDS"
        )
    return METRIC_TO_STATEMENT_FIELDS[metric_dep]


def evaluate_metric_coverage(
    statements: pd.DataFrame,
    announce: pd.DataFrame,
    metric_field: str,
    *,
    max_missing_announce: float = DEFAULT_MAX_MISSING_ANNOUNCE,
    max_value_missing: float = DEFAULT_MAX_VALUE_MISSING,
) -> CoverageReport:
    """Evaluate RAW/PIT gates for one statement field feeding metrics."""
    if statements is None or statements.empty:
        return CoverageReport(
            metric_code=metric_field,
            periods=0,
            instruments=0,
            missing_announce_rate=1.0,
            value_missing_rate=1.0,
            raw_ok=False,
            pit_ok=False,
            details={"reason": "empty_statements"},
        )

    frame = statements.copy()
    frame = frame[frame["field_code"] == metric_field]
    if "revision" not in frame.columns:
        frame["revision"] = 0
    frame["revision"] = frame["revision"].fillna(0).astype(int)
    instruments = int(frame["instrument_key"].nunique()) if not frame.empty else 0
    periods = int(frame["report_period_end"].nunique()) if not frame.empty else 0
    if frame.empty:
        value_missing_rate = 1.0
    else:
        values = pd.to_numeric(frame["value"], errors="coerce")
        # Inf 与 NaN 一样不能进入可交易因子，coverage 必须在出数前拦截。
        value_missing_rate = float((values.isna() | ~np.isfinite(values)).mean())

    if announce is None or announce.empty:
        missing_announce_rate = 1.0
    else:
        # Coverage 必须复用指标生成的 revision 级公告契约，禁止 rev1
        # 借用 rev0 的公告时间而被误判为 PIT-ready。
        join_keys = ["instrument_key", "report_period_end", "revision"]
        keys = frame[join_keys].drop_duplicates()
        ann = announce.copy()
        if "revision" not in ann.columns:
            ann["revision"] = 0
        ann["revision"] = ann["revision"].fillna(0).astype(int)
        ann["report_period_end"] = pd.to_datetime(ann["report_period_end"]).dt.date
        keys = keys.copy()
        keys["report_period_end"] = pd.to_datetime(keys["report_period_end"]).dt.date
        merged = keys.merge(
            ann[join_keys].drop_duplicates(),
            on=join_keys,
            how="left",
            indicator=True,
        )
        missing_announce_rate = float((merged["_merge"] == "left_only").mean()) if len(merged) else 1.0

    raw_ok = instruments > 0 and periods > 0 and value_missing_rate <= max_value_missing
    pit_ok = raw_ok and missing_announce_rate <= max_missing_announce
    return CoverageReport(
        metric_code=metric_field,
        periods=periods,
        instruments=instruments,
        missing_announce_rate=missing_announce_rate,
        value_missing_rate=value_missing_rate,
        raw_ok=raw_ok,
        pit_ok=pit_ok,
        details={
            "max_missing_announce": max_missing_announce,
            "max_value_missing": max_value_missing,
        },
    )


def evaluate_factor_gates(
    factor: FundFactorDef,
    statements: pd.DataFrame,
    announce: pd.DataFrame,
) -> dict[str, Any]:
    """Aggregate coverage across statement fields for a factor's upstream metrics.

    Multi-field deps (e.g. pe_ttm) require **all** upstream raw fields to pass.
    """
    requirements: list[str] = []
    for dep in factor.metric_deps:
        requirements.extend(resolve_statement_fields(dep))
    requirements = sorted(set(requirements))
    reports: list[CoverageReport] = []
    raw_ok = bool(requirements)
    pit_ok = bool(requirements)
    for requirement in requirements:
        alternatives: list[CoverageReport] = []
        for alternative in requirement.split("|"):
            component_reports = [
                evaluate_metric_coverage(statements, announce, field)
                for field in alternative.split("+")
            ]
            if len(component_reports) == 1:
                alternatives.append(component_reports[0])
                continue
            # 派生字段只有在所有组成项均通过时才可用；取最差缺失率，
            # 防止单一费用分项把不完整合计误判为 RAW/PIT 合格。
            alternatives.append(
                CoverageReport(
                    metric_code=alternative,
                    periods=min(report.periods for report in component_reports),
                    instruments=min(report.instruments for report in component_reports),
                    missing_announce_rate=max(
                        report.missing_announce_rate for report in component_reports
                    ),
                    value_missing_rate=max(
                        report.value_missing_rate for report in component_reports
                    ),
                    raw_ok=all(report.raw_ok for report in component_reports),
                    pit_ok=all(report.pit_ok for report in component_reports),
                    details={
                        "all_components_required": [
                            report.metric_code for report in component_reports
                        ]
                    },
                )
            )
        # Best alternative wins (PIT first, then RAW).
        best = max(alternatives, key=lambda r: (r.pit_ok, r.raw_ok))
        reports.append(best)
        raw_ok = raw_ok and best.raw_ok
        pit_ok = pit_ok and best.pit_ok
    return {
        "factor_id": factor.factor_id,
        "RAW_OK": "ok" if raw_ok else "fail",
        "COMPUTE_OK": factor.compute_ok,
        "PIT_OK": "ok" if pit_ok else "fail",
        "eligible_pit_safe": raw_ok and pit_ok and factor.compute_ok == "ok",
        "reports": [r.as_dict() for r in reports],
    }


def coverage_summary(
    statements: pd.DataFrame,
    announce: pd.DataFrame,
    *,
    factors: Sequence[FundFactorDef] | None = None,
) -> dict[str, Any]:
    targets = list(factors) if factors is not None else list(mvp_factors())
    rows = [evaluate_factor_gates(f, statements, announce) for f in targets]
    eligible = [r["factor_id"] for r in rows if r["eligible_pit_safe"]]
    return {
        "factor_count": len(rows),
        "eligible_count": len(eligible),
        "eligible_factor_ids": eligible,
        "factors": rows,
        "catalog_size": len(FACTOR_DEFINITIONS),
    }


class CoverageFailed(RuntimeError):
    """MVP 覆盖率未过硬闸，禁止写 PIT / 日宽表。"""

    def __init__(self, summary: dict[str, Any], failed: list[str]) -> None:
        self.summary = summary
        self.failed = failed
        super().__init__(f"MVP coverage failed: {failed}")


def assert_mvp_coverage(summary: dict[str, Any]) -> None:
    """MVP 任一因子未过 RAW/PIT 则整批停止。"""
    failed = [
        str(row["factor_id"])
        for row in summary.get("factors") or []
        if not row.get("eligible_pit_safe")
    ]
    if failed:
        raise CoverageFailed(summary, failed)


def assert_pit_safe_allowed(factor_id: str, summary: dict[str, Any]) -> None:
    """Raise if factor is not coverage-eligible for PIT_SAFE registration."""
    for row in summary.get("factors") or []:
        if row["factor_id"] == factor_id:
            if not row.get("eligible_pit_safe"):
                raise PermissionError(
                    f"factor {factor_id} failed coverage gates: "
                    f"RAW={row['RAW_OK']} COMPUTE={row['COMPUTE_OK']} PIT={row['PIT_OK']}"
                )
            return
    raise KeyError(factor_id)
