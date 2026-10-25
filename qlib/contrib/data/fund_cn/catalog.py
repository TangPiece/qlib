"""fund_cn_basic@v1 catalog — MVP + full-30 gate status (not L2 hot-table)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

FACTOR_SET = "fund_cn_basic@v1"
FACTOR_SET_CODE = "fund_cn_basic"
FACTOR_SET_VERSION = "v1"
FACTOR_VERSION = 1
EXCHANGE_DEFAULT = "CN"

FactorKind = Literal["report", "valuation"]
GateFlag = Literal["pending", "ok", "fail"]
Applicability = Literal["all_a_shares", "non_financial"]


@dataclass(frozen=True)
class FundFactorDef:
    """Single A-share fundamental factor definition."""

    factor_id: str
    display_name: str
    category: str
    kind: FactorKind
    formula: str
    metric_deps: tuple[str, ...]
    market_deps: tuple[str, ...] = ()
    unit: str = "ratio"
    mvp: bool = False
    enabled: bool = False
    information_policy: Literal["PIT_SAFE", "NON_PIT", "UNKNOWN"] = "UNKNOWN"
    raw_ok: GateFlag = "pending"
    compute_ok: GateFlag = "pending"
    pit_ok: GateFlag = "pending"
    notes: str = ""
    applicability: Applicability = "all_a_shares"

    @property
    def factor_ref(self) -> str:
        return f"{self.factor_id}@1.0.0"

    @property
    def all_gates_ok(self) -> bool:
        return self.raw_ok == "ok" and self.compute_ok == "ok" and self.pit_ok == "ok"


def _report(
    factor_id: str,
    display_name: str,
    category: str,
    formula: str,
    metrics: tuple[str, ...],
    *,
    mvp: bool = False,
    compute_ok: GateFlag = "pending",
    notes: str = "",
    applicability: Applicability = "all_a_shares",
) -> FundFactorDef:
    enabled = mvp
    policy: Literal["PIT_SAFE", "NON_PIT", "UNKNOWN"] = "PIT_SAFE" if mvp else "UNKNOWN"
    return FundFactorDef(
        factor_id=factor_id,
        display_name=display_name,
        category=category,
        kind="report",
        formula=formula,
        metric_deps=metrics,
        mvp=mvp,
        enabled=enabled,
        information_policy=policy,
        raw_ok="ok" if mvp else "pending",
        compute_ok=compute_ok if not mvp else "ok",
        pit_ok="ok" if mvp else "pending",
        notes=notes,
        applicability=applicability,
    )


def _valuation(
    factor_id: str,
    display_name: str,
    category: str,
    formula: str,
    metrics: tuple[str, ...],
    *,
    mvp: bool = False,
    compute_ok: GateFlag = "pending",
    notes: str = "",
    applicability: Applicability = "all_a_shares",
) -> FundFactorDef:
    enabled = mvp
    policy: Literal["PIT_SAFE", "NON_PIT", "UNKNOWN"] = "PIT_SAFE" if mvp else "UNKNOWN"
    return FundFactorDef(
        factor_id=factor_id,
        display_name=display_name,
        category=category,
        kind="valuation",
        formula=formula,
        metric_deps=metrics,
        market_deps=("close",),
        mvp=mvp,
        enabled=enabled,
        information_policy=policy,
        raw_ok="ok" if mvp else "pending",
        compute_ok=compute_ok if not mvp else "ok",
        pit_ok="ok" if mvp else "pending",
        notes=notes,
        applicability=applicability,
    )


# Supporting PIT metrics (not user-facing factors) needed for formulas / valuation.
SUPPORT_METRICS: tuple[str, ...] = (
    "revenue",
    "net_profit_parent",
    "net_profit_deducted",
    "operating_profit",
    "gross_profit",
    "operating_expenses",
    "total_assets",
    "total_liabilities",
    "shareholder_equity",
    "current_assets",
    "current_liabilities",
    "cash_from_operations",
    "capex",
    "free_cash_flow",
    "accounts_receivable",
    "inventory",
    "interest_bearing_debt",
    "shares_outstanding",
    "cash_dividend",
    "net_income_ttm",
    "revenue_ttm",
    "cash_from_operations_ttm",
    "cash_dividend_ttm",
    "operating_profit_ttm",
    "free_cash_flow_ttm",
    "gross_profit_ttm",
    "operating_expenses_ttm",
    "capex_ttm",
)

_TTM_NOTE = "time_basis=ttm numerator / point_in_time balance; None until 4 consecutive quarters"

FACTOR_DEFINITIONS: tuple[FundFactorDef, ...] = (
    # --- MVP quality ---
    _report(
        "roe",
        "ROE",
        "quality",
        "net_income_ttm / shareholder_equity",
        ("roe",),
        mvp=True,
        notes=f"{_TTM_NOTE}; denominator=end-of-period equity (not average equity)",
    ),
    _report(
        "roa",
        "ROA",
        "quality",
        "net_income_ttm / total_assets",
        ("roa",),
        mvp=True,
        notes=f"{_TTM_NOTE}; denominator=end-of-period assets (not average assets)",
    ),
    _report(
        "gross_margin",
        "毛利率",
        "quality",
        "gross_profit_ttm / revenue_ttm",
        ("gross_margin",),
        mvp=True,
        notes="time_basis=ttm; gross_profit derived as revenue - operating_cost when absent",
        applicability="non_financial",
    ),
    _report(
        "net_margin",
        "净利率",
        "quality",
        "net_income_ttm / revenue_ttm",
        ("net_margin",),
        mvp=True,
        notes="time_basis=ttm",
    ),
    _report(
        "debt_to_asset",
        "资产负债率",
        "leverage",
        "total_liabilities / total_assets",
        ("debt_to_asset",),
        mvp=True,
        notes="time_basis=point_in_time (balance sheet)",
    ),
    # --- MVP growth ---
    _report(
        "revenue_yoy",
        "营收同比增长率",
        "growth",
        "(revenue_sq - revenue_sq_lag4) / |revenue_sq_lag4|",
        ("revenue_yoy",),
        mvp=True,
        notes="time_basis=sq; single-quarter YoY after YTD→SQ conversion",
    ),
    _report(
        "net_profit_yoy",
        "归母净利润同比增长率",
        "growth",
        "(net_profit_sq - net_profit_sq_lag4) / |net_profit_sq_lag4|",
        ("net_profit_yoy",),
        mvp=True,
        notes=(
            "time_basis=sq; denominator=|lagged|; None if lagged≈0; negative bases "
            "represent loss narrowing/widening rather than ordinary profit growth"
        ),
    ),
    _report(
        "total_assets_yoy",
        "总资产增长率",
        "growth",
        "(total_assets - total_assets_lag4) / |total_assets_lag4|",
        ("total_assets_yoy",),
        mvp=True,
        notes="time_basis=point_in_time YoY",
    ),
    # --- MVP cashflow ---
    _report(
        "ocf_to_net_profit",
        "经营现金流/净利润",
        "cashflow",
        "cash_from_operations_ttm / net_income_ttm",
        ("ocf_to_net_profit",),
        mvp=True,
        notes="time_basis=ttm",
    ),
    _report(
        "ocf_to_revenue",
        "经营现金流/营业收入",
        "cashflow",
        "cash_from_operations_ttm / revenue_ttm",
        ("ocf_to_revenue",),
        mvp=True,
        notes="time_basis=ttm",
    ),
    # --- MVP valuation ---
    _valuation(
        "pe_ttm",
        "PE_TTM",
        "valuation",
        "(close * shares_outstanding) / net_income_ttm",
        ("net_income_ttm", "shares_outstanding"),
        mvp=True,
        notes="negative earnings produce negative raw PE; consumers must not read it as normal PE",
    ),
    _valuation(
        "pb",
        "PB",
        "valuation",
        "(close * shares_outstanding) / shareholder_equity",
        ("shareholder_equity", "shares_outstanding"),
        mvp=True,
    ),
    _valuation(
        "ps_ttm",
        "PS_TTM",
        "valuation",
        "(close * shares_outstanding) / revenue_ttm",
        ("revenue_ttm", "shares_outstanding"),
        mvp=True,
    ),
    # --- Full-30 remainder (formulas ready; gates pending until field audit) ---
    _valuation(
        "pcf",
        "PCF",
        "valuation",
        "(close * shares_outstanding) / cash_from_operations_ttm",
        ("cash_from_operations_ttm", "shares_outstanding"),
        compute_ok="ok",
        notes=(
            "OCF TTM from single-quarter sum / FY cum shortcut; negative OCF produces "
            "negative raw PCF"
        ),
    ),
    _valuation(
        "dividend_yield",
        "股息率",
        "valuation",
        "cash_dividend_ttm / (close * shares_outstanding)",
        ("cash_dividend_ttm", "shares_outstanding"),
        compute_ok="pending",
        notes="Requires cash_dividend staging field / dividend adapter",
    ),
    _report(
        "operating_margin",
        "营业利润率",
        "quality",
        "operating_profit_ttm / revenue_ttm",
        ("operating_margin",),
        compute_ok="ok",
        notes="time_basis=ttm",
        applicability="non_financial",
    ),
    _report(
        "expense_ratio",
        "费用率",
        "quality",
        "operating_expenses_ttm / revenue_ttm",
        ("expense_ratio",),
        compute_ok="ok",
        notes=(
            "time_basis=ttm; broad period-expense ratio = selling + admin + R&D + "
            "financial; all four components required when no direct total is provided"
        ),
        applicability="non_financial",
    ),
    _report(
        "roic",
        "投入资本营业利润率（近似 ROIC）",
        "quality",
        "operating_profit_ttm / (equity + interest_bearing_debt)",
        ("roic",),
        compute_ok="ok",
        notes=(
            "not standard ROIC: pre-tax operating profit / end-of-period "
            "(equity + interest-bearing debt); no NOPAT, cash or average-capital adjustment; "
            "None when debt components are incomplete"
        ),
    ),
    _report(
        "deducted_net_profit_yoy",
        "扣非净利润同比增长率",
        "growth",
        "(deducted_sq - deducted_sq_lag4) / |deducted_sq_lag4|",
        ("deducted_net_profit_yoy",),
        compute_ok="ok",
        notes=(
            "denominator=|lagged|; negative bases describe loss state changes, "
            "not ordinary positive-profit growth"
        ),
    ),
    _report(
        "equity_yoy",
        "净资产增长率",
        "growth",
        "(equity - equity_lag4) / |equity_lag4|",
        ("equity_yoy",),
        compute_ok="ok",
    ),
    _report(
        "ocf_yoy",
        "经营现金流增长率",
        "growth",
        "(ocf_sq - ocf_sq_lag4) / |ocf_sq_lag4|",
        ("ocf_yoy",),
        compute_ok="ok",
    ),
    _valuation(
        "fcf_yield",
        "自由现金流收益率",
        "cashflow",
        "free_cash_flow_ttm / (close * shares_outstanding)",
        ("free_cash_flow_ttm", "shares_outstanding"),
        compute_ok="ok",
        notes="time_basis=ttm",
    ),
    _report(
        "capex_ratio",
        "资本开支率",
        "cashflow",
        "capex_ttm / revenue_ttm",
        ("capex_ratio",),
        compute_ok="ok",
        notes="time_basis=ttm",
        applicability="non_financial",
    ),
    _report(
        "current_ratio",
        "流动比率",
        "leverage",
        "current_assets / current_liabilities",
        ("current_ratio",),
        compute_ok="ok",
        applicability="non_financial",
    ),
    _report(
        "interest_bearing_debt_ratio",
        "有息负债率",
        "leverage",
        "interest_bearing_debt / total_assets",
        ("interest_bearing_debt_ratio",),
        compute_ok="ok",
        notes="Requires short/long debt + bonds mapped to interest_bearing_debt",
    ),
    _report(
        "ar_to_revenue",
        "应收账款/营收",
        "efficiency",
        "accounts_receivable / revenue_ttm",
        ("ar_to_revenue",),
        compute_ok="ok",
        notes="time_basis=point_in_time / ttm; scale ratio, not receivables turnover",
        applicability="non_financial",
    ),
    _report(
        "inventory_to_revenue",
        "存货/营收",
        "efficiency",
        "inventory / revenue_ttm",
        ("inventory_to_revenue",),
        compute_ok="ok",
        notes="time_basis=point_in_time / ttm; scale ratio, not inventory turnover",
        applicability="non_financial",
    ),
    _report(
        "asset_turnover",
        "总资产周转率",
        "efficiency",
        "revenue_ttm / total_assets",
        ("asset_turnover",),
        compute_ok="ok",
        notes="time_basis=ttm",
        applicability="non_financial",
    ),
    _report(
        "earnings_volatility",
        "盈利波动率",
        "quality",
        "std(net_profit_sq[-8:]) / |mean|  (requires 8 consecutive quarters)",
        ("earnings_volatility",),
        compute_ok="ok",
        notes="Fixed 8 consecutive quarters; any gap → None; skipped when |mean|<1e-6",
    ),
    _report(
        "accruals_ratio",
        "应计利润率",
        "quality",
        "(net_income_ttm - cash_from_operations_ttm) / total_assets",
        ("accruals_ratio",),
        compute_ok="ok",
        notes=_TTM_NOTE,
    ),
)


def factor_ids(*, mvp_only: bool = False, enabled_only: bool = False) -> list[str]:
    rows = FACTOR_DEFINITIONS
    if mvp_only:
        rows = tuple(f for f in rows if f.mvp)
    if enabled_only:
        rows = tuple(f for f in rows if f.enabled)
    return [f.factor_id for f in rows]


def get_factor(factor_id: str) -> FundFactorDef | None:
    for item in FACTOR_DEFINITIONS:
        if item.factor_id == factor_id:
            return item
    return None


def mvp_factors() -> tuple[FundFactorDef, ...]:
    return tuple(f for f in FACTOR_DEFINITIONS if f.mvp)


def backtest_eligible_factors() -> tuple[FundFactorDef, ...]:
    """Only factors with all three gates green may enter default PIT_SAFE set.

    Honors runtime gate overlays from ``promote_factor_to_pit_safe``.
    """
    out: list[FundFactorDef] = []
    for base in FACTOR_DEFINITIONS:
        factor = effective_factor(base.factor_id)
        if factor.all_gates_ok and factor.enabled:
            out.append(factor)
    return tuple(out)


def wide_columns(*, mvp_only: bool = False) -> list[str]:
    """Panel columns: MVP set or enabled/backtest-eligible (same policy as compute)."""
    if mvp_only:
        ids = factor_ids(mvp_only=True)
    else:
        ids = [f.factor_id for f in backtest_eligible_factors()]
        if not ids:
            ids = [f.factor_id for f in FACTOR_DEFINITIONS if f.mvp or f.compute_ok == "ok"]
    return ["trading_date", "instrument_key", *ids]


def schema_hash() -> str:
    import hashlib

    payload = "|".join(
        f"{f.factor_id}:{f.kind}:{f.formula}:{f.raw_ok}:{f.compute_ok}:{f.pit_ok}:{f.applicability}"
        for f in FACTOR_DEFINITIONS
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def coverage_matrix() -> list[dict]:
    """Product checklist with three-state gates (replaces theoretical OK table)."""
    return [
        {
            "factor_id": f.factor_id,
            "display_name": f.display_name,
            "kind": f.kind,
            "mvp": f.mvp,
            "enabled": f.enabled,
            "RAW_OK": f.raw_ok,
            "COMPUTE_OK": f.compute_ok,
            "PIT_OK": f.pit_ok,
            "information_policy": f.information_policy,
            "applicability": f.applicability,
            "notes": f.notes,
        }
        for f in FACTOR_DEFINITIONS
    ]


@dataclass
class GateUpdate:
    factor_id: str
    raw_ok: GateFlag | None = None
    compute_ok: GateFlag | None = None
    pit_ok: GateFlag | None = None
    enabled: bool | None = None
    information_policy: Literal["PIT_SAFE", "NON_PIT", "UNKNOWN"] | None = None


# Mutable overlay for runtime gate promotion; catalog tuple stays frozen.
_GATE_OVERLAY: dict[str, dict[str, Any]] = {}


def apply_gate_update(update: GateUpdate) -> FundFactorDef:
    """Record runtime gate promotion; returns effective view (does not mutate frozen defs)."""
    base = get_factor(update.factor_id)
    if base is None:
        raise KeyError(update.factor_id)
    overlay = _GATE_OVERLAY.setdefault(update.factor_id, {})
    if update.raw_ok is not None:
        overlay["raw_ok"] = update.raw_ok
    if update.compute_ok is not None:
        overlay["compute_ok"] = update.compute_ok
    if update.pit_ok is not None:
        overlay["pit_ok"] = update.pit_ok
    if update.enabled is not None:
        overlay["enabled"] = update.enabled
    if update.information_policy is not None:
        overlay["information_policy"] = update.information_policy
    return effective_factor(update.factor_id)


def effective_factor(factor_id: str) -> FundFactorDef:
    base = get_factor(factor_id)
    if base is None:
        raise KeyError(factor_id)
    overlay = _GATE_OVERLAY.get(factor_id) or {}
    if not overlay:
        return base
    return FundFactorDef(
        factor_id=base.factor_id,
        display_name=base.display_name,
        category=base.category,
        kind=base.kind,
        formula=base.formula,
        metric_deps=base.metric_deps,
        market_deps=base.market_deps,
        unit=base.unit,
        mvp=base.mvp,
        enabled=bool(overlay.get("enabled", base.enabled)),
        information_policy=overlay.get("information_policy", base.information_policy),
        raw_ok=overlay.get("raw_ok", base.raw_ok),
        compute_ok=overlay.get("compute_ok", base.compute_ok),
        pit_ok=overlay.get("pit_ok", base.pit_ok),
        notes=base.notes,
        applicability=base.applicability,
    )


def promote_factor_to_pit_safe(factor_id: str) -> FundFactorDef:
    """Mark gates ok, enable factor, and set PIT_SAFE after field audit."""
    return apply_gate_update(
        GateUpdate(
            factor_id=factor_id,
            raw_ok="ok",
            compute_ok="ok",
            pit_ok="ok",
            enabled=True,
            information_policy="PIT_SAFE",
        )
    )

