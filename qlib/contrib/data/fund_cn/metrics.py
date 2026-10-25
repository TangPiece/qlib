"""Compute PIT metric rows from staged statement fields + announce calendar.

PIT rules:
- Statement snapshots are keyed by (instrument, period, revision).
- Announce joins on the same revision (no last-write-wins backfill onto older revisions).
- Every observation is evaluated against the state visible at its own
  ``available_time``: per (field, fiscal quarter) the latest revision visible.
- Single-quarter values are re-differenced from that visible cumulative state
  (so a later restatement of Q3 changes the Q4 single quarter used afterwards).
- TTM / YoY / volatility windows require consecutive fiscal quarters; any gap → None.

Flow / income / cashflow YTD cumulatives are converted to single-quarter before
YoY and TTM. Balance-sheet fields stay point-in-time. Ratio factors use TTM
numerators over point-in-time balance denominators (cross-sectionally comparable
across Q1/H1/Q3/FY disclosure windows).
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
import pandas as pd

from .adapters.announce import as_utc
from .numeric import finite_float, safe_divide

QKey = tuple[int, int]

# YTD cumulative flow fields (income / cashflow). Balance sheet excluded.
# cash_dividend is period occurrence (not YTD) — excluded so TTM sums raw periods.
_FLOW_FIELDS: tuple[str, ...] = (
    "revenue",
    "net_profit_parent",
    "net_profit_deducted",
    "operating_profit",
    "gross_profit",
    "operating_expenses",
    "cash_from_operations",
    "capex",
    "free_cash_flow",
)

_BALANCE_FIELDS: tuple[str, ...] = (
    "total_assets",
    "total_liabilities",
    "shareholder_equity",
    "current_assets",
    "current_liabilities",
    "accounts_receivable",
    "inventory",
    "interest_bearing_debt",
    "shares_outstanding",
)

_OCCURRENCE_FIELDS: tuple[str, ...] = ("cash_dividend",)

# flow field → TTM metric code
_TTM_METRICS: tuple[tuple[str, str], ...] = (
    ("net_profit_parent", "net_income_ttm"),
    ("revenue", "revenue_ttm"),
    ("cash_from_operations", "cash_from_operations_ttm"),
    ("operating_profit", "operating_profit_ttm"),
    ("free_cash_flow", "free_cash_flow_ttm"),
    ("gross_profit", "gross_profit_ttm"),
    ("operating_expenses", "operating_expenses_ttm"),
    ("capex", "capex_ttm"),
)

_OPEX_COMPONENTS: tuple[str, ...] = (
    "selling_expense",
    "admin_expense",
    "rd_expense",
    "financial_expense",
)


def _safe_div(num: float | None, den: float | None) -> float | None:
    """Fundamental ratio guard shared with valuation calculations."""
    return safe_divide(num, den)


def _yoy_growth(current: float | None, lagged: float | None) -> float | None:
    """Single-quarter YoY: (current - lagged) / |lagged|. None if lagged==0."""
    current_num = finite_float(current)
    lagged_num = finite_float(lagged)
    if current_num is None or lagged_num is None:
        return None
    return safe_divide(current_num - lagged_num, abs(lagged_num))


def _prev_q(key: QKey) -> QKey:
    fy, q = key
    return (fy, q - 1) if q > 1 else (fy - 1, 4)


def pivot_statements(statements: pd.DataFrame) -> pd.DataFrame:
    """Long statements → wide fields keyed by instrument + period + revision."""
    if statements is None or statements.empty:
        return pd.DataFrame()
    frame = statements.copy()
    frame["report_period_end"] = pd.to_datetime(frame["report_period_end"]).dt.date
    if "revision" not in frame.columns:
        frame["revision"] = 0
    frame["revision"] = frame["revision"].fillna(0).astype(int)
    # Keep each revision snapshot; last row within same revision wins.
    wide = (
        frame.sort_values(
            ["instrument_key", "report_period_end", "revision", "field_code"]
        )
        .drop_duplicates(
            ["instrument_key", "report_period_end", "revision", "field_code"],
            keep="last",
        )
        .pivot_table(
            index=[
                "instrument_key",
                "report_period_end",
                "fiscal_year",
                "fiscal_quarter",
                "revision",
            ],
            columns="field_code",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )
    wide.columns.name = None
    return wide


def derive_fields(wide: pd.DataFrame) -> pd.DataFrame:
    """Fill derivable cumulative fields when the source lacks them.

    - gross_profit = revenue − operating_cost
    - operating_expenses = selling + admin + R&D + financial expenses
    - free_cash_flow = cash_from_operations − |capex| (on YTD cumulatives, before differencing)
    """
    if wide is None or wide.empty:
        return wide
    out = wide.copy()

    def _existing(name: str) -> pd.Series:
        if name in out.columns:
            return out[name]
        return pd.Series(np.nan, index=out.index, dtype=float)

    if "revenue" in out.columns and "operating_cost" in out.columns:
        out["gross_profit"] = _existing("gross_profit").fillna(out["revenue"] - out["operating_cost"])
    components = [c for c in _OPEX_COMPONENTS if c in out.columns]
    if len(components) == len(_OPEX_COMPONENTS):
        # 分项口径要求四项齐全；部分求和会把“缺字段”伪装成低费用率。
        derived = out[components].sum(axis=1, min_count=len(_OPEX_COMPONENTS))
        out["operating_expenses"] = _existing("operating_expenses").fillna(derived)
    if "cash_from_operations" in out.columns and "capex" in out.columns:
        derived = out["cash_from_operations"] - out["capex"].abs()
        out["free_cash_flow"] = _existing("free_cash_flow").fillna(derived)
    return out


def attach_announce(wide: pd.DataFrame, announce: pd.DataFrame) -> pd.DataFrame:
    """Join disclosure times on matching revision only."""
    if wide is None or wide.empty:
        return pd.DataFrame()
    out = wide.copy()
    out["report_period_end"] = pd.to_datetime(out["report_period_end"]).dt.date
    if "revision" not in out.columns:
        out["revision"] = 0
    out["revision"] = out["revision"].fillna(0).astype(int)

    if announce is None or announce.empty:
        out["publish_time"] = pd.NaT
        out["available_time"] = pd.NaT
        out["is_restatement"] = out["revision"] > 0
        out["announce_source"] = ""
        out["announce_record_id"] = ""
        return out

    ann = announce.copy()
    ann["report_period_end"] = pd.to_datetime(ann["report_period_end"]).dt.date
    if "revision" not in ann.columns:
        ann["revision"] = 0
    ann["revision"] = ann["revision"].fillna(0).astype(int)
    ann["publish_time"] = pd.to_datetime(ann["publish_time"], utc=True)
    ann["available_time"] = pd.to_datetime(ann["available_time"], utc=True)
    # Same (period, revision) → earliest available_time (first disclosure wins).
    ann = ann.sort_values(
        ["instrument_key", "report_period_end", "revision", "available_time"]
    )
    ann = ann.drop_duplicates(
        ["instrument_key", "report_period_end", "revision"], keep="first"
    )

    merged = out.merge(
        ann[
            [
                "instrument_key",
                "report_period_end",
                "revision",
                "publish_time",
                "available_time",
                "is_restatement",
                "source",
                "source_record_id",
            ]
        ].rename(
            columns={"source": "announce_source", "source_record_id": "announce_record_id"}
        ),
        on=["instrument_key", "report_period_end", "revision"],
        how="left",
    )
    if "is_restatement" not in merged.columns:
        merged["is_restatement"] = merged["revision"] > 0
    else:
        merged["is_restatement"] = merged["is_restatement"].fillna(merged["revision"] > 0)
    return merged


class _PitState:
    """Visible cumulative values per (field, fiscal quarter) at the current KT."""

    def __init__(self) -> None:
        self._best: dict[str, dict[QKey, tuple[int, int, float]]] = {}

    def update(self, field: str, key: QKey, revision: int, at_ns: int, value: float) -> None:
        slot = self._best.setdefault(field, {})
        current = slot.get(key)
        if current is None or (revision, at_ns) >= (current[0], current[1]):
            slot[key] = (revision, at_ns, value)

    def cum(self, field: str, key: QKey) -> float | None:
        hit = self._best.get(field, {}).get(key)
        return None if hit is None else hit[2]

    def sq(self, field: str, key: QKey) -> float | None:
        """Single quarter re-differenced from visible cumulatives (None on gap)."""
        cur = self.cum(field, key)
        if cur is None:
            return None
        if key[1] == 1:
            return cur
        prev = self.cum(field, (key[0], key[1] - 1))
        return None if prev is None else cur - prev

    def sq_window(self, field: str, key: QKey, n: int) -> list[float] | None:
        """n consecutive single quarters ending at ``key`` (oldest first), or None."""
        values: list[float] = []
        k = key
        for _ in range(n):
            v = self.sq(field, k)
            if v is None:
                return None
            values.append(v)
            k = _prev_q(k)
        return values[::-1]

    def ttm(self, field: str, key: QKey) -> float | None:
        if key[1] == 4:
            fy = self.cum(field, key)
            if fy is not None:
                return fy
        window = self.sq_window(field, key, 4)
        return None if window is None else float(sum(window))

    def occurrence_ttm(self, field: str, key: QKey) -> float | None:
        """Sum period-occurrence values over the 4-quarter window; None if none reported."""
        total = 0.0
        seen = False
        k = key
        for _ in range(4):
            v = self.cum(field, k)
            if v is not None:
                total += v
                seen = True
            k = _prev_q(k)
        return total if seen else None


def _num(value: Any) -> float | None:
    return finite_float(value)


def _at_ns(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    return int(pd.Timestamp(value).value)


def _iter_pit_rows(
    frame: pd.DataFrame,
    fields: tuple[str, ...],
) -> Iterator[tuple[dict[str, Any], _PitState | None]]:
    """Yield (row, state) per instrument row.

    Rows sharing the same available_time are applied to the state before any of
    them is evaluated. If the same report period has multiple revisions at that
    timestamp, only the highest revision is effective because lower revisions had
    no independent visibility interval. Rows without available_time yield state=None.
    """
    for _, sub in frame.groupby("instrument_key", sort=False):
        records = sub.to_dict("records")
        timed = [r for r in records if _at_ns(r.get("available_time")) is not None]
        for r in records:
            if _at_ns(r.get("available_time")) is None:
                yield r, None
        timed.sort(key=lambda r: (_at_ns(r["available_time"]), int(r.get("revision") or 0)))
        state = _PitState()
        i = 0
        while i < len(timed):
            at = _at_ns(timed[i]["available_time"])
            j = i
            while j < len(timed) and _at_ns(timed[j]["available_time"]) == at:
                j += 1
            raw_batch = timed[i:j]
            effective_by_period: dict[Any, dict[str, Any]] = {}
            for row in raw_batch:
                period_key = row["report_period_end"]
                current = effective_by_period.get(period_key)
                if current is None or int(row.get("revision") or 0) > int(
                    current.get("revision") or 0
                ):
                    effective_by_period[period_key] = row
            batch = list(effective_by_period.values())
            for r in batch:
                key = (int(r["fiscal_year"]), int(r["fiscal_quarter"]))
                rev = int(r.get("revision") or 0)
                for field in fields:
                    v = _num(r.get(field))
                    if v is not None:
                        state.update(field, key, rev, at, v)
            for r in batch:
                yield r, state
            i = j


def to_single_quarter(wide: pd.DataFrame, flow_fields: tuple[str, ...] = _FLOW_FIELDS) -> pd.DataFrame:
    """Add ``{field}__sq`` / ``{field}__cum`` display columns from YTD cumulatives.

    Q1 → cum. Qn → this row's cum − latest prior-quarter cum visible at the row's
    ``available_time`` (cross-revision PIT). Missing / not-yet-visible prior quarter
    → None. Negative single quarters are allowed. Metrics re-difference at query
    time via ``_PitState`` and do not rely on these columns.
    """
    if wide is None or wide.empty:
        return wide
    out = wide.copy()
    out["report_period_end"] = pd.to_datetime(out["report_period_end"]).dt.date
    if "revision" not in out.columns:
        out["revision"] = 0
    out["revision"] = out["revision"].fillna(0).astype(int)
    if "available_time" in out.columns:
        out["available_time"] = pd.to_datetime(out["available_time"], utc=True)
    else:
        out["available_time"] = pd.NaT
    present = tuple(f for f in flow_fields if f in out.columns)
    if not present:
        return out
    out = out.reset_index(drop=True)
    out["_row_id"] = np.arange(len(out))

    sq_values: dict[str, np.ndarray] = {f: np.full(len(out), np.nan) for f in present}
    for row, state in _iter_pit_rows(out, present):
        rid = int(row["_row_id"])
        q = int(row["fiscal_quarter"])
        for field in present:
            cum = _num(row.get(field))
            if cum is None:
                continue
            if q == 1:
                sq_values[field][rid] = cum
                continue
            if state is None:
                continue
            prev = state.cum(field, (int(row["fiscal_year"]), q - 1))
            if prev is not None:
                sq_values[field][rid] = cum - prev
    for field in present:
        out[f"{field}__sq"] = [None if np.isnan(v) else float(v) for v in sq_values[field]]
        out[f"{field}__cum"] = out[field]
    return out.drop(columns=["_row_id"])


def compute_period_metrics(wide_with_announce: pd.DataFrame) -> list[dict[str, Any]]:
    """Emit pit_fundamental-shaped metric dicts (one row per metric observation)."""
    if wide_with_announce is None or wide_with_announce.empty:
        return []

    frame = derive_fields(wide_with_announce)
    frame["report_period_end"] = pd.to_datetime(frame["report_period_end"]).dt.date
    if "revision" not in frame.columns:
        frame["revision"] = 0
    frame["revision"] = frame["revision"].fillna(0).astype(int)
    frame["available_time"] = pd.to_datetime(frame["available_time"], utc=True)

    tracked = tuple(
        f
        for f in (*_FLOW_FIELDS, *_BALANCE_FIELDS, *_OCCURRENCE_FIELDS)
        if f in frame.columns
    )
    metrics: list[dict[str, Any]] = []

    for row, state in _iter_pit_rows(frame, tracked):
        if state is None:
            continue
        ik = str(row["instrument_key"])
        pe = row["report_period_end"]
        fy = int(row.get("fiscal_year") or pe.year)
        fq = int(row.get("fiscal_quarter") or ((pe.month - 1) // 3 + 1))
        key: QKey = (fy, fq)
        revision = int(row.get("revision") or 0)
        at = row["available_time"]
        pt = row.get("publish_time")
        if pt is None or pd.isna(pt):
            pt = at
        base = {
            "instrument_key": ik,
            "report_period_start": None,
            "report_period_end": pe,
            "fiscal_year": fy,
            "fiscal_quarter": fq,
            "publish_time": as_utc(pd.Timestamp(pt).to_pydatetime()),
            "available_time": as_utc(pd.Timestamp(at).to_pydatetime()),
            "currency": "CNY",
            "revision": revision,
            "is_restatement": bool(row.get("is_restatement") or False),
            "source": str(row.get("announce_source") or "fund_cn"),
        }
        record_id = row.get("announce_record_id")
        if record_id is not None and pd.isna(record_id):
            record_id = None

        def emit(metric_code: str, value: float | None, unit: str = "ratio") -> None:
            normalized = finite_float(value)
            if normalized is None:
                return
            item = dict(base)
            item["metric_code"] = metric_code
            item["value"] = normalized
            item["unit"] = unit
            item["source_record_id"] = str(
                record_id or f"{ik}:{metric_code}:{pe.isoformat()}:r{revision}"
            )
            metrics.append(item)

        get = lambda f: _num(row.get(f))  # noqa: E731

        # Reported levels (flows = YTD cumulative as disclosed; balance = point-in-time).
        for code in (*_FLOW_FIELDS, *_BALANCE_FIELDS, *_OCCURRENCE_FIELDS):
            emit(code, get(code), "shares" if code == "shares_outstanding" else "currency")

        ttm = {metric: state.ttm(field, key) for field, metric in _TTM_METRICS}
        for metric, value in ttm.items():
            emit(metric, value, "currency")
        emit("cash_dividend_ttm", state.occurrence_ttm("cash_dividend", key), "currency")

        ni_ttm = ttm["net_income_ttm"]
        rev_ttm = ttm["revenue_ttm"]
        ocf_ttm = ttm["cash_from_operations_ttm"]
        op_ttm = ttm["operating_profit_ttm"]

        total_assets = get("total_assets")
        equity = get("shareholder_equity")
        ib_debt = get("interest_bearing_debt")

        # Ratios: decimal (0.15 = 15%); TTM numerators over point-in-time balances.
        emit("roe", _safe_div(ni_ttm, equity))
        emit("roa", _safe_div(ni_ttm, total_assets))
        emit("gross_margin", _safe_div(ttm["gross_profit_ttm"], rev_ttm))
        emit("net_margin", _safe_div(ni_ttm, rev_ttm))
        emit("operating_margin", _safe_div(op_ttm, rev_ttm))
        emit("expense_ratio", _safe_div(ttm["operating_expenses_ttm"], rev_ttm))
        emit("debt_to_asset", _safe_div(get("total_liabilities"), total_assets))
        emit("current_ratio", _safe_div(get("current_assets"), get("current_liabilities")))
        emit("interest_bearing_debt_ratio", _safe_div(ib_debt, total_assets))
        emit("ar_to_revenue", _safe_div(get("accounts_receivable"), rev_ttm))
        emit("inventory_to_revenue", _safe_div(get("inventory"), rev_ttm))
        emit("asset_turnover", _safe_div(rev_ttm, total_assets))
        emit("ocf_to_net_profit", _safe_div(ocf_ttm, ni_ttm))
        emit("ocf_to_revenue", _safe_div(ocf_ttm, rev_ttm))
        emit("capex_ratio", _safe_div(ttm["capex_ttm"], rev_ttm))
        if ni_ttm is not None and ocf_ttm is not None:
            emit("accruals_ratio", _safe_div(ni_ttm - ocf_ttm, total_assets))
        if equity is not None and ib_debt is not None:
            emit("roic", _safe_div(op_ttm, equity + ib_debt))

        # YoY: single quarter (flows) or point-in-time (balance), same quarter last year.
        lag_key: QKey = (fy - 1, fq)
        for field, metric in (
            ("revenue", "revenue_yoy"),
            ("net_profit_parent", "net_profit_yoy"),
            ("net_profit_deducted", "deducted_net_profit_yoy"),
            ("cash_from_operations", "ocf_yoy"),
        ):
            emit(metric, _yoy_growth(state.sq(field, key), state.sq(field, lag_key)))
        for field, metric in (
            ("total_assets", "total_assets_yoy"),
            ("shareholder_equity", "equity_yoy"),
        ):
            emit(metric, _yoy_growth(state.cum(field, key), state.cum(field, lag_key)))

        window = state.sq_window("net_profit_parent", key, 8)
        if window is not None:
            mean = float(np.mean(window))
            std = float(np.std(window, ddof=0))
            if abs(mean) >= 1e-6:
                emit("earnings_volatility", _safe_div(std, abs(mean)))

    return metrics


def statements_to_pit_metrics(
    statements: pd.DataFrame,
    announce: pd.DataFrame,
) -> list[dict[str, Any]]:
    wide = pivot_statements(statements)
    joined = attach_announce(wide, announce)
    return compute_period_metrics(joined)
