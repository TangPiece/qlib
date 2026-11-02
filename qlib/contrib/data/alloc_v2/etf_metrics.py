# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ETF 三类指标同时保留：原始份额变化、估算配置、套利风险标记。

份额变化 ≠ 净流入；ΔShares × NAV 是估算，必须打 estimate。
联接与跨境不进入境内行业流。
份额/NAV 的 available_at（通常 T+1）必须传到因子行，禁止压成 event_date。
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from .mappings import cross_border_codes, feeder_codes, load_etf_master
from .schema import SCHEMA_VERSION

FORMULA_VERSION = "etf_metrics@v3"

# 停更后最多前向填充的交易日数；超过则视为缺失，避免「假零流」
FFILL_LIMIT_DAYS = 7


def _pivot_field(standard: pd.DataFrame, field: str) -> pd.DataFrame:
    part = standard[standard["field"] == field].copy()
    if part.empty:
        return pd.DataFrame()
    part["event_date"] = pd.to_datetime(part["event_date"])
    return part.pivot_table(index="event_date", columns="entity_id", values="value", aggfunc="last")


def _pivot_available_at(standard: pd.DataFrame, field: str) -> pd.DataFrame:
    """按 event_date×entity 取 available_at（同日多条取 max）。"""
    part = standard[standard["field"] == field].copy()
    if part.empty or "available_at" not in part.columns:
        return pd.DataFrame()
    part["event_date"] = pd.to_datetime(part["event_date"])
    part["available_at"] = pd.to_datetime(part["available_at"])
    return part.pivot_table(index="event_date", columns="entity_id", values="available_at", aggfunc="max")


def _pivot_quality(standard: pd.DataFrame, field: str) -> pd.DataFrame:
    """保留标准层 quality_flag（如 sparse_sample）。"""
    part = standard[standard["field"] == field].copy()
    if part.empty or "quality_flag" not in part.columns:
        return pd.DataFrame()
    part["event_date"] = pd.to_datetime(part["event_date"])
    # 同日取「更差」标记：sparse_sample > ok
    def _agg(flags: pd.Series) -> str:
        vals = [str(v) for v in flags.dropna().tolist()]
        if any(v == "sparse_sample" for v in vals):
            return "sparse_sample"
        if any(v == "stale_ffill" for v in vals):
            return "stale_ffill"
        return vals[-1] if vals else "ok"

    return part.pivot_table(index="event_date", columns="entity_id", values="quality_flag", aggfunc=_agg)


def raw_share_change(shares: pd.DataFrame) -> pd.DataFrame:
    """原始份额差分，不乘价格。"""
    return shares.sort_index().diff()


def estimated_allocation(shares: pd.DataFrame, nav: pd.DataFrame) -> pd.DataFrame:
    """ΔShares × NAV：估算配置，不是现金净流入。"""
    aligned_shares, aligned_nav = shares.align(nav, join="inner")
    return aligned_shares.diff() * aligned_nav


def price_adjusted_flow(aum: pd.Series, price: pd.Series) -> pd.Series:
    """flow_est = ΔAUM - AUM_lag * r。价格上涨造成的账面增大会被扣掉。"""
    aligned = pd.concat({"aum": aum, "price": price}, axis=1).sort_index().dropna()
    if aligned.empty:
        return pd.Series(dtype="float64")
    ret = aligned["price"].pct_change(fill_method=None)
    return (aligned["aum"] - aligned["aum"].shift(1) * (1.0 + ret)).rename("flow_est")


def premium(close: pd.DataFrame, nav_or_iopv: pd.DataFrame) -> pd.DataFrame:
    aligned_px, aligned_nav = close.align(nav_or_iopv, join="inner")
    return aligned_px / aligned_nav - 1.0


def arbitrage_flags(
    *,
    close: pd.DataFrame,
    nav: pd.DataFrame,
    shares: pd.DataFrame,
    amount: pd.DataFrame,
    index_px: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """折溢价、成交/规模、份额与指数涨跌背离。"""
    prem = premium(close, nav)
    aum = shares * nav
    turnover = amount.align(aum, join="inner")[0] / aum.replace(0, np.nan)
    share_ret = shares.pct_change(fill_method=None)
    px_ret = close.pct_change(fill_method=None)
    diverge = (share_ret * px_ret) < 0
    rows: list[dict[str, Any]] = []
    for entity in prem.columns:
        for ts, value in prem[entity].dropna().items():
            aum_v = aum[entity].get(ts, np.nan) if entity in aum.columns else np.nan
            to_v = turnover[entity].get(ts, np.nan) if entity in turnover.columns else np.nan
            flag = "ok"
            if pd.notna(value) and abs(float(value)) > 0.02:
                flag = "premium_wide"
            if pd.notna(to_v) and float(to_v) > 0.5:
                flag = "high_turnover"
            if entity in diverge.columns and bool(diverge[entity].get(ts, False)):
                flag = "share_price_diverge"
            rows.append(
                {
                    "entity_id": entity,
                    "event_date": ts.date() if hasattr(ts, "date") else ts,
                    "premium": float(value) if pd.notna(value) else np.nan,
                    "aum": float(aum_v) if pd.notna(aum_v) else np.nan,
                    "turnover_to_aum": float(to_v) if pd.notna(to_v) else np.nan,
                    "quality_flag": flag,
                    "evidence_class": "proxy",
                    "version": FORMULA_VERSION,
                }
            )
    _ = index_px
    return pd.DataFrame(rows)


def isolate_universe(frame: pd.DataFrame, *, master: pd.DataFrame | None = None) -> pd.DataFrame:
    """去掉联接与跨境，避免双计/混算。"""
    if frame.empty:
        return frame
    blocked = feeder_codes(master) | cross_border_codes(master)
    col = "entity_id" if "entity_id" in frame.columns else None
    if col:
        return frame[~frame[col].isin(blocked)].copy()
    keep = [c for c in frame.columns if str(c) not in blocked]
    return frame[keep]


def _as_date(ts) -> Any:
    return ts.date() if hasattr(ts, "date") else ts


def _avail_to_date(val, *, fallback) -> Any:
    if pd.isna(val):
        return fallback
    ts = pd.Timestamp(val)
    return ts.date()


def build_etf_factor_panel(standard: pd.DataFrame, *, master: pd.DataFrame | None = None) -> pd.DataFrame:
    """从标准层生成 ETF 因子长表；available_at 继承份额/NAV 的 T+1 可观测时点。"""
    if standard.empty:
        return pd.DataFrame()
    etf = standard[standard["dataset"].isin({"etf_bar", "etf_nav", "etf_shares"})].copy()
    if etf.empty:
        return pd.DataFrame()
    master = master if master is not None else load_etf_master()
    shares_raw = isolate_universe(_pivot_field(etf, "shares"), master=master)
    nav_raw = isolate_universe(_pivot_field(etf, "nav"), master=master)
    close = isolate_universe(_pivot_field(etf, "close"), master=master)
    amount = isolate_universe(_pivot_field(etf, "amount"), master=master)
    share_at_raw = isolate_universe(_pivot_available_at(etf, "shares"), master=master)
    nav_at_raw = isolate_universe(_pivot_available_at(etf, "nav"), master=master)
    share_q_raw = isolate_universe(_pivot_quality(etf, "shares"), master=master)
    if shares_raw.empty:
        return pd.DataFrame()

    calendar = shares_raw.index
    if not close.empty:
        calendar = calendar.union(close.index)
    if not nav_raw.empty:
        calendar = calendar.union(nav_raw.index)
    calendar = calendar.sort_values()

    # 有限 ffill：周频抽样可补齐数日；停更后变为 NaN，避免假零流
    shares = shares_raw.reindex(calendar).ffill(limit=FFILL_LIMIT_DAYS)
    nav = nav_raw.reindex(calendar).ffill(limit=FFILL_LIMIT_DAYS) if not nav_raw.empty else nav_raw
    close = close.reindex(calendar).ffill(limit=FFILL_LIMIT_DAYS) if not close.empty else close
    amount = amount.reindex(calendar) if not amount.empty else amount
    share_at = share_at_raw.reindex(calendar).ffill(limit=FFILL_LIMIT_DAYS) if not share_at_raw.empty else share_at_raw
    nav_at = nav_at_raw.reindex(calendar).ffill(limit=FFILL_LIMIT_DAYS) if not nav_at_raw.empty else nav_at_raw
    share_q = share_q_raw.reindex(calendar).ffill(limit=FFILL_LIMIT_DAYS) if not share_q_raw.empty else share_q_raw

    # 标记：值来自 ffill（原始份额日无观测）→ stale_ffill，便于下游降权
    observed = shares_raw.reindex(calendar).notna()
    raw = raw_share_change(shares)
    est = estimated_allocation(shares, nav if not nav.empty else close)

    rows: list[dict[str, Any]] = []
    for entity in shares.columns:
        for ts in shares.index:
            share_v = shares[entity].get(ts, np.nan)
            if pd.isna(share_v):
                continue
            day = _as_date(ts)
            raw_v = raw[entity].get(ts, np.nan) if entity in raw.columns else np.nan
            est_v = est[entity].get(ts, np.nan) if entity in est.columns else np.nan
            nav_v = nav[entity].get(ts, np.nan) if (not nav.empty and entity in nav.columns) else np.nan
            close_v = close[entity].get(ts, np.nan) if (not close.empty and entity in close.columns) else np.nan
            amount_v = amount[entity].get(ts, np.nan) if (not amount.empty and entity in amount.columns) else np.nan

            share_avail = (
                _avail_to_date(share_at[entity].get(ts), fallback=day + timedelta(days=1))
                if (not share_at.empty and entity in share_at.columns)
                else day + timedelta(days=1)
            )
            nav_avail = (
                _avail_to_date(nav_at[entity].get(ts), fallback=share_avail)
                if (not nav_at.empty and entity in nav_at.columns and pd.notna(nav_at[entity].get(ts)))
                else share_avail
            )
            # 估算配置需份额与净值都可观测
            est_avail = max(share_avail, nav_avail) if isinstance(share_avail, type(nav_avail)) else share_avail

            qflag = "ok"
            if not share_q.empty and entity in share_q.columns and pd.notna(share_q[entity].get(ts)):
                qflag = str(share_q[entity].get(ts))
            if entity in observed.columns and not bool(observed[entity].get(ts, False)):
                # 当日无原始份额点，依赖 ffill
                qflag = "stale_ffill" if qflag == "ok" else qflag

            def _row(field: str, value: float, *, factor_id: str, evidence: str, unit: str, available_at, quality: str) -> dict[str, Any]:
                return {
                    "dataset": "etf_factor",
                    "entity_id": entity,
                    "event_date": day,
                    "announcement_date": day,
                    "available_at": available_at,
                    "field": field,
                    "value": float(value),
                    "unit": unit,
                    "freq": "day",
                    "evidence_class": evidence,
                    "quality_flag": quality,
                    "version": SCHEMA_VERSION,
                    "formula_version": FORMULA_VERSION,
                    "source": "computed",
                    "layer": "C",
                    "factor_id": factor_id,
                    "window": 1,
                }

            rows.append(
                _row("shares", float(share_v), factor_id="RawShares", evidence="direct", unit="share", available_at=share_avail, quality=qflag)
            )
            if pd.notna(raw_v):
                rows.append(
                    _row(
                        "raw_share_change",
                        float(raw_v),
                        factor_id="Raw_Share_Change",
                        evidence="direct",
                        unit="share",
                        available_at=share_avail,
                        quality=qflag,
                    )
                )
            if pd.notna(est_v):
                rows.append(
                    _row(
                        "estimated_allocation",
                        float(est_v),
                        factor_id="Estimated_Allocation",
                        evidence="estimate",
                        unit="cny",
                        available_at=est_avail,
                        quality=qflag,
                    )
                )
            if pd.notna(nav_v):
                rows.append(
                    _row("nav", float(nav_v), factor_id="NAV", evidence="direct", unit="nav", available_at=nav_avail, quality="ok")
                )
            if pd.notna(close_v):
                rows.append(
                    _row("close", float(close_v), factor_id="Close", evidence="proxy", unit="price", available_at=day, quality="ok")
                )
            if pd.notna(amount_v):
                rows.append(
                    _row("amount", float(amount_v), factor_id="Amount", evidence="proxy", unit="cny", available_at=day, quality="ok")
                )

    flags = arbitrage_flags(
        close=close,
        nav=nav if not nav.empty else close,
        shares=shares,
        amount=amount if not amount.empty else shares * 0,
    )
    if not flags.empty:
        for rec in flags.to_dict("records"):
            rows.append(
                {
                    "dataset": "etf_factor",
                    "entity_id": rec["entity_id"],
                    "event_date": rec["event_date"],
                    "announcement_date": rec["event_date"],
                    "available_at": rec["event_date"],
                    "field": "arbitrage_flag",
                    "value": rec["premium"],
                    "unit": "pct",
                    "freq": "day",
                    "evidence_class": "proxy",
                    "quality_flag": rec["quality_flag"],
                    "version": SCHEMA_VERSION,
                    "source": "computed",
                    "layer": "C",
                    "factor_id": "ArbitrageRisk",
                    "window": 1,
                    "formula_version": FORMULA_VERSION,
                }
            )
    return pd.DataFrame(rows)
