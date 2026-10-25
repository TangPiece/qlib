"""Phase 3 helpers — promote remaining factors after field audits."""

from __future__ import annotations

from typing import Iterable

from .catalog import (
    FACTOR_DEFINITIONS,
    FundFactorDef,
    GateUpdate,
    apply_gate_update,
    coverage_matrix,
    effective_factor,
    promote_factor_to_pit_safe,
)


def pending_audit_factors() -> list[FundFactorDef]:
    """Factors not yet all-green (need RAW/COMPUTE/PIT field audit)."""
    return [f for f in FACTOR_DEFINITIONS if not f.all_gates_ok]


def compute_ready_pending_pit() -> list[FundFactorDef]:
    """Formulas ready but disclosure/raw gates still pending."""
    return [
        f
        for f in FACTOR_DEFINITIONS
        if f.compute_ok == "ok" and (f.raw_ok != "ok" or f.pit_ok != "ok")
    ]


def promote_many(factor_ids: Iterable[str]) -> list[FundFactorDef]:
    return [promote_factor_to_pit_safe(fid) for fid in factor_ids]


def mark_raw_ok(factor_id: str) -> FundFactorDef:
    return apply_gate_update(GateUpdate(factor_id=factor_id, raw_ok="ok"))


def mark_pit_ok(factor_id: str) -> FundFactorDef:
    return apply_gate_update(GateUpdate(factor_id=factor_id, pit_ok="ok"))


def audit_status_table() -> list[dict]:
    """Three-state checklist for the full 30-factor product list."""
    rows = coverage_matrix()
    for row in rows:
        eff = effective_factor(row["factor_id"])
        row["RAW_OK"] = eff.raw_ok
        row["COMPUTE_OK"] = eff.compute_ok
        row["PIT_OK"] = eff.pit_ok
        row["all_gates_ok"] = eff.all_gates_ok
    return rows
