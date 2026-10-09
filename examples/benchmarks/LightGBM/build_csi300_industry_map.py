#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Build CSI300 x 申万一级 industry PIT interval mapping via AKShare.

Primary source: akshare.stock_industry_clf_hist_sw (申万宏源 StockClassifyUse_stock.xls),
which provides per-stock industry entry dates. Internal 6-digit industry codes are
mapped to SW L1 index codes (801xxx). Intervals are intersected with local
csi300.txt membership and calendars/day.txt.

Usage:
    .venv/bin/python examples/benchmarks/LightGBM/build_csi300_industry_map.py
    .venv/bin/python examples/benchmarks/LightGBM/build_csi300_industry_map.py --open-end-as-cal-end
"""
from __future__ import annotations

import argparse
import io
import json
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests

DIRNAME = Path(__file__).absolute().resolve().parent
DEFAULT_QLIB_DIR = Path("~/.qlib/qlib_data/cn_data").expanduser()
DEFAULT_OUT_DIR = DIRNAME / "data"
OPEN_END = pd.Timestamp("9999-12-31")
STANDARD = "SW"
STANDARD_VERSION = "2021"
SOURCE = "akshare"
SW_HIST_URL = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
AK_APIS = ["stock_industry_clf_hist_sw"]

# Internal L1 prefix (first 2 digits of 6-digit SW industry code) -> (index_code, name).
# Includes SW2021 L1 and SW2014-only prefixes still present in historical rows.
SW_L1_BY_PREFIX: Dict[str, Tuple[str, str]] = {
    # SW2021
    "11": ("801010", "农林牧渔"),
    "22": ("801030", "基础化工"),
    "23": ("801040", "钢铁"),
    "24": ("801050", "有色金属"),
    "27": ("801080", "电子"),
    "28": ("801880", "汽车"),
    "33": ("801110", "家用电器"),
    "34": ("801120", "食品饮料"),
    "35": ("801130", "纺织服饰"),
    "36": ("801140", "轻工制造"),
    "37": ("801150", "医药生物"),
    "41": ("801160", "公用事业"),
    "42": ("801170", "交通运输"),
    "43": ("801180", "房地产"),
    "45": ("801200", "商贸零售"),
    "46": ("801210", "社会服务"),
    "48": ("801780", "银行"),
    "49": ("801790", "非银金融"),
    "51": ("801230", "综合"),
    "61": ("801710", "建筑材料"),
    "62": ("801720", "建筑装饰"),
    "63": ("801730", "电力设备"),
    "64": ("801890", "机械设备"),
    "65": ("801740", "国防军工"),
    "71": ("801750", "计算机"),
    "72": ("801760", "传媒"),
    "73": ("801770", "通信"),
    "74": ("801950", "煤炭"),
    "75": ("801960", "石油石化"),
    "76": ("801970", "环保"),
    "77": ("801980", "美容护理"),
    # SW2014-era prefixes still appearing before 2021 reclass
    "21": ("801020", "采掘"),
    "25": ("801710", "建筑材料"),
    "26": ("801890", "机械设备"),
    "31": ("801880", "汽车"),
    "32": ("801750", "计算机"),
    "44": ("801780", "银行"),
    "47": ("801790", "非银金融"),
}

PIT_RULE = (
    "Source Excel gives industry entry dates (计入日期) per stock without explicit "
    "exit dates. For each stock, rows are sorted by entry date; start_date = entry; "
    "end_date = day before the next entry; the last segment uses open end 9999-12-31 "
    "before intersecting with CSI300 membership and the local calendar. "
    "6-digit SW industry codes are mapped to L1 index codes via code prefix."
)
LIMITATIONS = (
    "Official SW workbook is a point-in-time classification history, but exit dates "
    "are inferred. Some SW2014 prefixes are mapped to the closest L1 index code "
    "(e.g. 采掘 801020; 信息设备-era 32xx -> 计算机 801750). "
    "Coverage depends on workbook completeness for delisted names."
)


def log(msg: str) -> None:
    print(msg, flush=True)


def to_qlib_symbol(raw: str) -> Optional[str]:
    s = str(raw).strip().upper().replace(".SH", "").replace(".SZ", "").replace(".BJ", "")
    if not s.isdigit() or len(s) != 6:
        return None
    if s.startswith(("5", "6", "9")):
        return f"SH{s}"
    if s.startswith(("0", "1", "2", "3")):
        return f"SZ{s}"
    if s.startswith(("4", "8")):
        return f"BJ{s}"
    return None


def map_to_l1(industry_code: str) -> Optional[Tuple[str, str]]:
    code = str(industry_code).strip()
    if len(code) < 2:
        return None
    return SW_L1_BY_PREFIX.get(code[:2])


def load_calendar(qlib_dir: Path) -> List[pd.Timestamp]:
    path = qlib_dir / "calendars" / "day.txt"
    if not path.exists():
        raise FileNotFoundError(f"Calendar not found: {path}")
    days = [
        pd.Timestamp(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not days:
        raise RuntimeError(f"Empty calendar: {path}")
    return days


def load_csi300_membership(qlib_dir: Path) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    path = qlib_dir / "instruments" / "csi300.txt"
    if not path.exists():
        raise FileNotFoundError(f"CSI300 instruments not found: {path}")
    membership: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        sym, start, end = parts[0].strip().upper(), parts[1].strip(), parts[2].strip()
        membership[sym].append((pd.Timestamp(start), pd.Timestamp(end)))
    for sym in membership:
        membership[sym].sort(key=lambda x: x[0])
    return dict(membership)


def fetch_sw_hist_via_akshare() -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_industry_clf_hist_sw()
    if df is None or df.empty:
        raise RuntimeError("stock_industry_clf_hist_sw returned empty")
    # akshare renames to symbol / start_date / industry_code / update_time
    rename = {}
    if "股票代码" in df.columns:
        rename["股票代码"] = "symbol"
    if "计入日期" in df.columns:
        rename["计入日期"] = "start_date"
    if "行业代码" in df.columns:
        rename["行业代码"] = "industry_code"
    if rename:
        df = df.rename(columns=rename)
    return df


def fetch_sw_hist_direct() -> pd.DataFrame:
    """Same URL as akshare.stock_industry_clf_hist_sw, with SSL fallback."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    last_err: Optional[Exception] = None
    for verify in (True, False):
        try:
            with warnings.catch_warnings():
                if not verify:
                    warnings.simplefilter("ignore")
                r = requests.get(SW_HIST_URL, headers=headers, timeout=120, verify=verify)
            r.raise_for_status()
            df = pd.read_excel(
                io.BytesIO(r.content),
                dtype={"股票代码": "str", "行业代码": "str"},
            )
            df = df.rename(
                columns={
                    "股票代码": "symbol",
                    "计入日期": "start_date",
                    "行业代码": "industry_code",
                    "更新日期": "update_time",
                }
            )
            return df
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            log(f"direct download verify={verify} failed: {exc}")
    raise RuntimeError(f"Failed to download SW industry workbook: {last_err}")


def load_sw_classification_history() -> Tuple[pd.DataFrame, str]:
    try:
        df = fetch_sw_hist_via_akshare()
        return df, "akshare.stock_industry_clf_hist_sw"
    except Exception as exc:  # noqa: BLE001
        log(f"akshare.stock_industry_clf_hist_sw failed: {exc}")
        log("falling back to direct download of the same SW workbook URL")
        df = fetch_sw_hist_direct()
        return df, "direct:" + SW_HIST_URL


def build_join_events(raw: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    warnings_list: List[str] = []
    rows: List[dict] = []
    unmapped = 0
    for _, r in raw.iterrows():
        instrument = to_qlib_symbol(r["symbol"])
        if instrument is None:
            continue
        start = pd.to_datetime(r["start_date"], errors="coerce")
        if pd.isna(start):
            continue
        mapped = map_to_l1(r["industry_code"])
        if mapped is None:
            unmapped += 1
            continue
        ind_code, ind_name = mapped
        rows.append(
            {
                "instrument": instrument,
                "industry_code": ind_code,
                "industry_name": ind_name,
                "join_date": pd.Timestamp(start).normalize(),
                "raw_industry_code": str(r["industry_code"]).strip(),
            }
        )
    if unmapped:
        warnings_list.append(f"unmapped raw industry codes rows: {unmapped}")
    if not rows:
        raise RuntimeError("No join events after L1 mapping")
    events = pd.DataFrame(rows)
    # Same join_date: keep last raw row (latest workbook update order)
    events = events.sort_values(["instrument", "join_date", "raw_industry_code"])
    events = events.drop_duplicates(subset=["instrument", "join_date"], keep="last")
    return events, warnings_list


def merge_adjacent_same_industry(iv: pd.DataFrame) -> pd.DataFrame:
    if iv.empty:
        return iv
    out: List[dict] = []
    for _, g in iv.groupby("instrument", sort=True):
        g = g.sort_values("start_date").reset_index(drop=True)
        cur = g.iloc[0].to_dict()
        for i in range(1, len(g)):
            row = g.iloc[i].to_dict()
            contiguous = row["start_date"] <= cur["end_date"] + timedelta(days=1)
            same = row["industry_code"] == cur["industry_code"]
            if same and contiguous:
                cur["end_date"] = max(cur["end_date"], row["end_date"])
            else:
                out.append(cur)
                cur = row
        out.append(cur)
    return pd.DataFrame(out)


def infer_industry_intervals(events: pd.DataFrame) -> pd.DataFrame:
    records: List[dict] = []
    for instrument, g in events.groupby("instrument", sort=True):
        g = g.sort_values(["join_date", "industry_code"]).reset_index(drop=True)
        for i, row in g.iterrows():
            start = row["join_date"]
            if i + 1 < len(g):
                end = g.loc[i + 1, "join_date"] - timedelta(days=1)
            else:
                end = OPEN_END
            if end < start:
                end = start
            records.append(
                {
                    "instrument": instrument,
                    "industry_code": row["industry_code"],
                    "industry_name": row["industry_name"],
                    "start_date": start,
                    "end_date": end,
                }
            )
    return merge_adjacent_same_industry(pd.DataFrame(records))


def intersect_intervals(
    a_start: pd.Timestamp,
    a_end: pd.Timestamp,
    b_start: pd.Timestamp,
    b_end: pd.Timestamp,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    if start <= end:
        return start, end
    return None


def intersect_with_csi300(
    industry_iv: pd.DataFrame,
    membership: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]],
    cal_start: pd.Timestamp,
    cal_end: pd.Timestamp,
) -> pd.DataFrame:
    rows: List[dict] = []
    for _, row in industry_iv.iterrows():
        inst = row["instrument"]
        if inst not in membership:
            continue
        for m_start, m_end in membership[inst]:
            m_start = max(m_start, cal_start)
            m_end = min(m_end, cal_end)
            hit = intersect_intervals(row["start_date"], row["end_date"], m_start, m_end)
            if hit is None:
                continue
            start, end = hit
            rows.append(
                {
                    "instrument": inst,
                    "industry_code": row["industry_code"],
                    "industry_name": row["industry_name"],
                    "start_date": start,
                    "end_date": end,
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=["instrument", "industry_code", "industry_name", "start_date", "end_date"]
        )
    return merge_adjacent_same_industry(pd.DataFrame(rows))


def validate_intervals(df: pd.DataFrame) -> List[str]:
    warnings_list: List[str] = []
    if df.empty:
        return ["empty interval table"]
    overlap_n = 0
    for _, g in df.groupby("instrument"):
        g = g.sort_values("start_date")
        prev_end = None
        for _, row in g.iterrows():
            if prev_end is not None and row["start_date"] <= prev_end:
                overlap_n += 1
                break
            prev_end = row["end_date"]
    if overlap_n:
        warnings_list.append(f"instruments with overlapping intervals: {overlap_n}")
    return warnings_list


def build_output(intervals: pd.DataFrame, asof: str) -> pd.DataFrame:
    out = intervals.copy() if not intervals.empty else intervals
    if out.empty:
        out = pd.DataFrame(
            columns=[
                "instrument",
                "industry_code",
                "industry_name",
                "start_date",
                "end_date",
                "standard",
                "standard_version",
                "source",
                "asof",
            ]
        )
        return out
    out["standard"] = STANDARD
    out["standard_version"] = STANDARD_VERSION
    out["source"] = SOURCE
    out["asof"] = asof
    out["start_date"] = pd.to_datetime(out["start_date"]).dt.strftime("%Y-%m-%d")
    out["end_date"] = pd.to_datetime(out["end_date"]).dt.strftime("%Y-%m-%d")
    cols = [
        "instrument",
        "industry_code",
        "industry_name",
        "start_date",
        "end_date",
        "standard",
        "standard_version",
        "source",
        "asof",
    ]
    return out[cols].sort_values(["instrument", "start_date"]).reset_index(drop=True)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build CSI300 SW L1 industry PIT intervals (AKShare SW workbook)"
    )
    p.add_argument("--qlib-dir", type=Path, default=DEFAULT_QLIB_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--open-end-as-cal-end",
        action="store_true",
        help="Replace 9999-12-31 with calendar last day in output intervals",
    )
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    qlib_dir = args.qlib_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log(f"qlib_dir: {qlib_dir}")
    calendar = load_calendar(qlib_dir)
    cal_start, cal_end = calendar[0], calendar[-1]
    log(f"calendar: {cal_start.date()} -> {cal_end.date()} ({len(calendar)} days)")

    membership = load_csi300_membership(qlib_dir)
    csi_symbols = sorted(membership)
    log(f"csi300 unique symbols: {len(csi_symbols)}")

    asof = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    log("loading SW industry classification history ...")
    raw, fetch_method = load_sw_classification_history()
    log(f"raw rows: {len(raw)} via {fetch_method}")

    events, map_warnings = build_join_events(raw)
    log(
        f"join events: {len(events)} "
        f"(unique instruments: {events['instrument'].nunique()})"
    )

    industry_iv = infer_industry_intervals(events)
    log(f"raw industry intervals: {len(industry_iv)}")

    intersected = intersect_with_csi300(industry_iv, membership, cal_start, cal_end)
    if args.open_end_as_cal_end and not intersected.empty:
        mask = intersected["end_date"] >= OPEN_END
        intersected.loc[mask, "end_date"] = cal_end

    covered = set(intersected["instrument"]) if not intersected.empty else set()
    missing = sorted(set(csi_symbols) - covered)
    coverage = len(covered) / len(csi_symbols) if csi_symbols else 0.0
    log(f"CSI300 coverage: {len(covered)}/{len(csi_symbols)} ({coverage:.1%})")
    if missing:
        log(f"missing sample ({min(10, len(missing))}): {missing[:10]}")

    warnings_list = map_warnings + validate_intervals(intersected)
    for w in warnings_list:
        log(f"WARN: {w}")

    out_df = build_output(intersected, asof=asof)
    parquet_path = out_dir / "csi300_sw_l1_industry_intervals.parquet"
    meta_path = out_dir / "csi300_sw_l1_industry_meta.json"
    out_df.to_parquet(parquet_path, index=False)

    meta = {
        "asof": asof,
        "qlib_dir": str(qlib_dir),
        "calendar_start": str(cal_start.date()),
        "calendar_end": str(cal_end.date()),
        "calendar_days": len(calendar),
        "csi300_symbols": len(csi_symbols),
        "interval_rows": int(len(out_df)),
        "covered_symbols": len(covered),
        "coverage_ratio": round(coverage, 6),
        "missing_symbols_count": len(missing),
        "missing_symbols_sample": missing[:50],
        "standard": STANDARD,
        "standard_version": STANDARD_VERSION,
        "source": SOURCE,
        "akshare_apis": AK_APIS,
        "fetch_method": fetch_method,
        "sw_hist_url": SW_HIST_URL,
        "open_end_sentinel": str(OPEN_END.date()),
        "end_date_convention": (
            "calendar last day"
            if args.open_end_as_cal_end
            else "9999-12-31 for still-active before clip; "
            "after CSI300 intersect clipped to membership/calendar"
        ),
        "pit_inference_rule": PIT_RULE,
        "limitations": LIMITATIONS,
        "validation_warnings": warnings_list,
        "output_parquet": str(parquet_path),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    log(f"wrote: {parquet_path} ({len(out_df)} rows)")
    log(f"wrote: {meta_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        log("interrupted")
        raise SystemExit(130)
    except Exception as exc:  # noqa: BLE001
        log(f"FATAL: {exc}")
        raise SystemExit(1)
