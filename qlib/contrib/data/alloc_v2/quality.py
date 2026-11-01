# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""质量检查：完整性 / 合理性 / 一致性 / 连续性。异常标记保留，不静默删除。"""

from __future__ import annotations

from typing import Any

import pandas as pd


def mark_quality(frame: pd.DataFrame) -> pd.DataFrame:
    """给标准层打 quality_flag；不丢行。"""
    if frame.empty:
        return frame
    out = frame.copy()
    if "quality_flag" not in out.columns:
        out["quality_flag"] = "ok"
    values = pd.to_numeric(out["value"], errors="coerce")
    missing = values.isna()
    out.loc[missing, "quality_flag"] = "missing"
    # 份额/净值/价格不应为负；成交额可为 0
    numeric_fields = {"close", "nav", "iopv", "shares", "aum"}
    if "field" in out.columns:
        bad = out["field"].isin(numeric_fields) & (values < 0)
        out.loc[bad, "quality_flag"] = "inconsistent"
    return out


def continuity_gaps(frame: pd.DataFrame, *, freq: str = "day") -> pd.DataFrame:
    """按 entity+field 找日频缺口；周/月/季不按交易日要求连续。"""
    if frame.empty or freq != "day":
        return pd.DataFrame(columns=["entity_id", "field", "gap_start", "gap_end", "n_missing"])
    rows: list[dict[str, Any]] = []
    work = frame.copy()
    work["event_date"] = pd.to_datetime(work["event_date"])
    for (entity, field), part in work.groupby(["entity_id", "field"]):
        days = part["event_date"].sort_values().drop_duplicates()
        if len(days) < 2:
            continue
        expected = pd.bdate_range(days.min(), days.max())
        missing = expected.difference(days)
        if len(missing) == 0:
            continue
        rows.append(
            {
                "entity_id": entity,
                "field": field,
                "gap_start": missing.min().date().isoformat(),
                "gap_end": missing.max().date().isoformat(),
                "n_missing": int(len(missing)),
            }
        )
    return pd.DataFrame(rows)


def quality_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"n_rows": 0, "n_ok": 0, "n_flagged": 0}
    flags = frame["quality_flag"] if "quality_flag" in frame.columns else pd.Series(["ok"] * len(frame))
    return {
        "n_rows": int(len(frame)),
        "n_ok": int((flags == "ok").sum()),
        "n_flagged": int((flags != "ok").sum()),
        "flags": {str(k): int(v) for k, v in flags.value_counts().to_dict().items()},
    }
