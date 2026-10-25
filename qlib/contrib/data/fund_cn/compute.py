"""Daily factor panel: report forward-fill + valuation join on Canonical market.

Asof lookups use per-(instrument, metric) sorted arrays + searchsorted (Phase D).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .catalog import FACTOR_DEFINITIONS, backtest_eligible_factors, mvp_factors
from .numeric import finite_float, safe_divide

_log = logging.getLogger(__name__)


class MarketRequiredError(ValueError):
    """Raised when valuation factors are requested but market panel is missing."""


def _as_date(value: Any) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.to_datetime(value).date()


def pit_metrics_to_frame(rows: Sequence[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(
            columns=[
                "instrument_key",
                "metric_code",
                "available_time",
                "report_period_end",
                "revision",
                "value",
            ]
        )
    frame = pd.DataFrame(list(rows))
    frame["available_time"] = pd.to_datetime(frame["available_time"], utc=True)
    frame["report_period_end"] = pd.to_datetime(frame["report_period_end"]).dt.date
    if "revision" not in frame.columns:
        frame["revision"] = 0
    return frame


AsofEntry = tuple[np.ndarray, np.ndarray, np.ndarray]

# Report values older than this (trade day − report_period_end) are not forward-filled.
DEFAULT_MAX_STALENESS_DAYS = 548


def _to_ns(series: pd.Series) -> np.ndarray:
    """Epoch nanoseconds regardless of pandas datetime resolution (s/ms/us/ns)."""
    values = pd.to_datetime(series, utc=True)
    return values.dt.as_unit("ns").astype("int64").to_numpy()


def _stepwise_latest(
    times_ns: np.ndarray,
    rank_keys: Sequence[tuple],
    values: np.ndarray,
    periods_ns: np.ndarray,
) -> AsofEntry:
    """Replay events in time order; state = value with the greatest rank key seen so far.

    Late-arriving events with a lower rank (e.g. a restatement of an older period)
    do not replace the visible state.
    """
    out_values = np.empty(len(values), dtype=float)
    out_periods = np.empty(len(values), dtype="int64")
    best_key: tuple | None = None
    best_value = np.nan
    best_period = np.iinfo("int64").min
    for i, key in enumerate(rank_keys):
        if best_key is None or key >= best_key:
            best_key, best_value, best_period = key, values[i], periods_ns[i]
        out_values[i] = best_value
        out_periods[i] = best_period
    return times_ns, out_values, out_periods


def _build_asof_index(pit: pd.DataFrame) -> dict[tuple[str, str], AsofEntry]:
    """Map (instrument_key, metric_code) → stepwise (time_ns, value, period_ns) arrays.

    At any KT the visible value is the latest report period, then latest revision,
    then latest available_time — never "whatever was published last".
    """
    if pit is None or pit.empty:
        return {}
    frame = pit.copy()
    frame["available_time"] = pd.to_datetime(frame["available_time"], utc=True)
    frame["_period"] = pd.to_datetime(frame["report_period_end"])
    frame["revision"] = frame["revision"].fillna(0).astype(int)
    frame = frame.sort_values(
        ["instrument_key", "metric_code", "available_time", "_period", "revision"]
    )
    index: dict[tuple[str, str], AsofEntry] = {}
    for (ik, metric), group in frame.groupby(["instrument_key", "metric_code"], sort=False):
        times = _to_ns(group["available_time"])
        periods = _to_ns(group["_period"])
        revisions = group["revision"].to_numpy()
        keys = list(zip(periods.tolist(), revisions.tolist(), times.tolist()))
        values = group["value"].astype(float).to_numpy()
        index[(str(ik), str(metric))] = _stepwise_latest(times, keys, values, periods)
    return index


def latest_metric_asof_fast(
    asof_index: dict[tuple[str, str], AsofEntry],
    *,
    instrument_key: str,
    metric_code: str,
    knowledge_time: datetime,
    max_staleness_days: int | None = None,
) -> float | None:
    entry = asof_index.get((instrument_key, metric_code))
    if entry is None:
        return None
    times, values, periods = entry
    kt = knowledge_time if knowledge_time.tzinfo else knowledge_time.replace(tzinfo=timezone.utc)
    kt_ns = int(pd.Timestamp(kt).value)
    idx = int(np.searchsorted(times, kt_ns, side="right")) - 1
    if idx < 0:
        return None
    if max_staleness_days is not None:
        age_days = (kt_ns - int(periods[idx])) / 86_400e9
        if age_days > max_staleness_days:
            return None
    return finite_float(values[idx])


def latest_metric_asof(
    pit: pd.DataFrame,
    *,
    instrument_key: str,
    metric_code: str,
    knowledge_time: datetime,
) -> float | None:
    """Compatibility wrapper — rebuilds the asof index each call.

    Do **not** use on hot paths (daily panels). Prefer ``_build_asof_index`` once
    then ``latest_metric_asof_fast``, or pass a prebuilt index from the caller.
    """
    return latest_metric_asof_fast(
        _build_asof_index(pit),
        instrument_key=instrument_key,
        metric_code=metric_code,
        knowledge_time=knowledge_time,
    )


def resolve_factor_subset(
    *,
    mvp_only: bool = True,
    factor_ids_subset: Sequence[str] | None = None,
) -> list[str]:
    """Single source of truth for which factor columns to materialize."""
    if factor_ids_subset is not None:
        return list(factor_ids_subset)
    if mvp_only:
        return [f.factor_id for f in mvp_factors()]
    eligible = [f.factor_id for f in backtest_eligible_factors()]
    if eligible:
        return eligible
    return [f.factor_id for f in FACTOR_DEFINITIONS if f.mvp or f.compute_ok == "ok"]


def _build_share_index(share_capital: pd.DataFrame | None) -> dict[str, AsofEntry]:
    """instrument_key → stepwise total shares, usable from max(effective_date, announce_time).

    Among usable events the latest effective_date wins (then latest announcement),
    so a late-announced historical correction never rolls the series back.
    """
    if share_capital is None or share_capital.empty:
        return {}
    frame = share_capital.copy()
    frame["_eff"] = pd.to_datetime(frame["effective_date"]).dt.tz_localize("UTC")
    frame["_ann"] = pd.to_datetime(frame["announce_time"], utc=True)
    frame["_usable"] = frame[["_eff", "_ann"]].max(axis=1)
    frame = frame.sort_values(["instrument_key", "_usable", "_eff", "_ann"])
    index: dict[str, AsofEntry] = {}
    for ik, group in frame.groupby("instrument_key", sort=False):
        times = _to_ns(group["_usable"])
        eff = _to_ns(group["_eff"])
        ann = _to_ns(group["_ann"])
        keys = list(zip(eff.tolist(), ann.tolist()))
        values = group["total_shares"].astype(float).to_numpy()
        index[str(ik)] = _stepwise_latest(times, keys, values, eff)
    return index


def forward_fill_report_factors(
    pit_rows: Sequence[dict[str, Any]],
    trading_dates: Sequence[date | str],
    instrument_keys: Sequence[str],
    *,
    factor_ids_subset: Sequence[str] | None = None,
    mvp_only: bool = True,
    max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
) -> pd.DataFrame:
    """Broadcast latest PIT metric (available_time <= trade day close) onto daily panel.

    Values whose report period ended more than ``max_staleness_days`` before the
    trade day are dropped (suspended / delisted issuers stop reporting).
    """
    wanted = resolve_factor_subset(mvp_only=mvp_only, factor_ids_subset=factor_ids_subset)
    report_ids = [
        f.factor_id
        for f in FACTOR_DEFINITIONS
        if f.factor_id in wanted and f.kind == "report"
    ]
    asof_index = _build_asof_index(pit_metrics_to_frame(pit_rows))
    records: list[dict[str, Any]] = []
    for day in (_as_date(d) for d in trading_dates):
        kt = datetime(day.year, day.month, day.day, 7, 0, tzinfo=timezone.utc)
        for ik in instrument_keys:
            row: dict[str, Any] = {"trading_date": day.isoformat(), "instrument_key": ik}
            for fid in report_ids:
                row[fid] = latest_metric_asof_fast(
                    asof_index,
                    instrument_key=ik,
                    metric_code=fid,
                    knowledge_time=kt,
                    max_staleness_days=max_staleness_days,
                )
            records.append(row)
    if not records:
        return pd.DataFrame(columns=["trading_date", "instrument_key", *report_ids])
    return pd.DataFrame.from_records(records)


def compute_valuation_factors(
    pit_rows: Sequence[dict[str, Any]],
    market: pd.DataFrame,
    *,
    factor_ids_subset: Sequence[str] | None = None,
    mvp_only: bool = True,
    require_market: bool = True,
    share_capital: pd.DataFrame | None = None,
    max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
) -> pd.DataFrame:
    """
    Valuation factors: ``mcap = raw (unadjusted) close × total shares``.

    ``market`` columns: instrument_key, trading_date, close — close must be
    **unadjusted** (Canonical bars store raw prices).
    Total shares come from ``share_capital`` events (effective & announced by the
    trade day); when an issuer has no visible event the report-period
    ``shares_outstanding`` PIT metric is used and a warning is logged once.
    When ``factor_ids_subset`` is omitted, ``mvp_only`` selects the default set.
    """
    wanted = resolve_factor_subset(mvp_only=mvp_only, factor_ids_subset=factor_ids_subset)
    valuation_ids = [
        f.factor_id
        for f in FACTOR_DEFINITIONS
        if f.factor_id in wanted and f.kind == "valuation"
    ]
    if not valuation_ids:
        return pd.DataFrame(columns=["trading_date", "instrument_key"])

    if market is None or getattr(market, "empty", True):
        if require_market:
            raise MarketRequiredError(
                "valuation factors require a non-empty market panel "
                f"(requested: {valuation_ids})"
            )
        return pd.DataFrame(columns=["trading_date", "instrument_key", *valuation_ids])

    asof_index = _build_asof_index(pit_metrics_to_frame(pit_rows))
    share_index = _build_share_index(share_capital)
    mkt = market.copy()
    mkt["trading_date"] = pd.to_datetime(mkt["trading_date"]).dt.date
    records: list[dict[str, Any]] = []
    fallback_keys: set[str] = set()

    def _div(num: float | None, den: float | None) -> float | None:
        """Keep valuation output finite and reject economically unstable denominators."""
        return safe_divide(num, den)

    def _shares(ik: str, kt: datetime) -> float | None:
        entry = share_index.get(ik)
        if entry is not None:
            times, values, _ = entry
            idx = int(np.searchsorted(times, int(pd.Timestamp(kt).value), side="right")) - 1
            if idx >= 0:
                return finite_float(values[idx])
        fallback_keys.add(ik)
        return latest_metric_asof_fast(
            asof_index,
            instrument_key=ik,
            metric_code="shares_outstanding",
            knowledge_time=kt,
            max_staleness_days=max_staleness_days,
        )

    for day, day_frame in mkt.groupby("trading_date", sort=True):
        kt = datetime(day.year, day.month, day.day, 7, 0, tzinfo=timezone.utc)
        for bar in day_frame.itertuples(index=False):
            ik = str(bar.instrument_key)
            close = finite_float(bar.close)

            def pit(metric: str) -> float | None:
                return latest_metric_asof_fast(
                    asof_index,
                    instrument_key=ik,
                    metric_code=metric,
                    knowledge_time=kt,
                    max_staleness_days=max_staleness_days,
                )

            shares = _shares(ik, kt)
            equity = pit("shareholder_equity")
            net_ttm = pit("net_income_ttm")
            rev_ttm = pit("revenue_ttm")
            ocf_ttm = pit("cash_from_operations_ttm")
            fcf_ttm = pit("free_cash_flow_ttm")
            dividend_ttm = pit("cash_dividend_ttm")
            mcap = (
                finite_float(close * shares)
                if close is not None and shares is not None
                else None
            )
            values = {
                "pe_ttm": _div(mcap, net_ttm),
                "pb": _div(mcap, equity),
                "ps_ttm": _div(mcap, rev_ttm),
                "pcf": _div(mcap, ocf_ttm),
                "fcf_yield": _div(fcf_ttm, mcap),
                "dividend_yield": _div(dividend_ttm, mcap),
            }
            row: dict[str, Any] = {
                "trading_date": day.isoformat() if hasattr(day, "isoformat") else str(day),
                "instrument_key": ik,
            }
            for fid in valuation_ids:
                row[fid] = values.get(fid)
            records.append(row)
    if fallback_keys:
        _log.warning(
            "share_capital events missing for %d instruments; fell back to report shares_outstanding "
            "(stale after 送转/增发): %s",
            len(fallback_keys),
            sorted(fallback_keys)[:10],
        )
    return pd.DataFrame.from_records(records)


def merge_factor_panels(
    report_panel: pd.DataFrame,
    valuation_panel: pd.DataFrame,
) -> pd.DataFrame:
    if report_panel is None or report_panel.empty:
        return valuation_panel.copy() if valuation_panel is not None else pd.DataFrame()
    if valuation_panel is None or valuation_panel.empty:
        return report_panel.copy()
    return report_panel.merge(
        valuation_panel,
        on=["trading_date", "instrument_key"],
        how="outer",
    )


def build_daily_factor_panel(
    pit_rows: Sequence[dict[str, Any]],
    trading_dates: Sequence[date | str],
    instrument_keys: Sequence[str],
    market: pd.DataFrame | None = None,
    *,
    mvp_only: bool = True,
    require_market: bool | None = None,
    share_capital: pd.DataFrame | None = None,
    max_staleness_days: int | None = DEFAULT_MAX_STALENESS_DAYS,
) -> pd.DataFrame:
    subset = resolve_factor_subset(mvp_only=mvp_only, factor_ids_subset=None)
    has_valuation = any(
        f.factor_id in subset and f.kind == "valuation" for f in FACTOR_DEFINITIONS
    )
    must_have_market = require_market if require_market is not None else has_valuation

    report = forward_fill_report_factors(
        pit_rows,
        trading_dates,
        instrument_keys,
        factor_ids_subset=subset,
        max_staleness_days=max_staleness_days,
    )
    valuation = compute_valuation_factors(
        pit_rows,
        market if market is not None else pd.DataFrame(),
        factor_ids_subset=subset,
        require_market=must_have_market,
        share_capital=share_capital,
        max_staleness_days=max_staleness_days,
    )
    panel = merge_factor_panels(report, valuation)
    cols = ["trading_date", "instrument_key"] + list(subset)
    for column in subset:
        if column not in panel.columns:
            panel[column] = np.nan
    return panel[cols]
