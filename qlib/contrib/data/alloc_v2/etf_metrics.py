# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""ETF 三类指标同时保留：原始份额变化、估算配置、套利风险标记。

份额变化 ≠ 净流入；ΔShares × NAV 是估算，必须打 estimate。
联接与跨境不进入境内行业流。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .mappings import cross_border_codes, feeder_codes, load_etf_master
from .schema import SCHEMA_VERSION

FORMULA_VERSION = "etf_metrics@v2"


def _pivot_field(standard: pd.DataFrame, field: str) -> pd.DataFrame:
    part = standard[standard["field"] == field].copy()
    if part.empty:
        return pd.DataFrame()
    part["event_date"] = pd.to_datetime(part["event_date"])
    return part.pivot_table(index="event_date", columns="entity_id", values="value", aggfunc="last")


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
    ret = aligned["price"].pct_change()
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
    share_ret = shares.pct_change()
    px_ret = close.pct_change()
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


def build_etf_factor_panel(standard: pd.DataFrame, *, master: pd.DataFrame | None = None) -> pd.DataFrame:
    """从标准层生成 ETF 因子长表，三类指标并列。"""
    if standard.empty:
        return pd.DataFrame()
    etf = standard[standard["dataset"].isin({"etf_bar", "etf_nav", "etf_shares"})].copy()
    if etf.empty:
        return pd.DataFrame()
    master = master if master is not None else load_etf_master()
    shares = isolate_universe(_pivot_field(etf, "shares"), master=master)
    nav = isolate_universe(_pivot_field(etf, "nav"), master=master)
    close = isolate_universe(_pivot_field(etf, "close"), master=master)
    amount = isolate_universe(_pivot_field(etf, "amount"), master=master)
    if shares.empty:
        return pd.DataFrame()
    raw = raw_share_change(shares)
    est = estimated_allocation(shares, nav if not nav.empty else close)
    rows: list[dict[str, Any]] = []
    for entity in shares.columns:
        for ts in shares.index:
            raw_v = raw[entity].get(ts, np.nan) if entity in raw.columns else np.nan
            est_v = est[entity].get(ts, np.nan) if entity in est.columns else np.nan
            nav_v = nav[entity].get(ts, np.nan) if entity in nav.columns else np.nan
            close_v = close[entity].get(ts, np.nan) if entity in close.columns else np.nan
            amount_v = amount[entity].get(ts, np.nan) if entity in amount.columns else np.nan
            share_v = shares[entity].get(ts, np.nan)
            day = ts.date() if hasattr(ts, "date") else ts
            common = {
                "dataset": "etf_factor",
                "entity_id": entity,
                "event_date": day,
                "announcement_date": day,
                "available_at": day,
                "freq": "day",
                "quality_flag": "ok",
                "version": SCHEMA_VERSION,
                "formula_version": FORMULA_VERSION,
                "source": "computed",
                "layer": "C",
            }
            rows.append({**common, "field": "shares", "value": share_v, "unit": "share", "evidence_class": "direct", "factor_id": "RawShares", "window": 1})
            rows.append({**common, "field": "raw_share_change", "value": raw_v, "unit": "share", "evidence_class": "direct", "factor_id": "Raw_Share_Change", "window": 1})
            rows.append({**common, "field": "estimated_allocation", "value": est_v, "unit": "cny", "evidence_class": "estimate", "factor_id": "Estimated_Allocation", "window": 1})
            if pd.notna(nav_v):
                rows.append({**common, "field": "nav", "value": nav_v, "unit": "nav", "evidence_class": "direct", "factor_id": "NAV", "window": 1})
            if pd.notna(close_v):
                rows.append({**common, "field": "close", "value": close_v, "unit": "price", "evidence_class": "proxy", "factor_id": "Close", "window": 1})
            if pd.notna(amount_v):
                rows.append({**common, "field": "amount", "value": amount_v, "unit": "cny", "evidence_class": "proxy", "factor_id": "Amount", "window": 1})
    flags = arbitrage_flags(close=close, nav=nav if not nav.empty else close, shares=shares, amount=amount if not amount.empty else shares * 0)
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
