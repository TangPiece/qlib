# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""评估 L2Daily 百因子质量：覆盖、IC/ICIR、相关性，并逐条列出问题数据。

口径与 workflow_config_lightgbm_L2Daily.yaml 对齐：CSI300、
2017-01-03 ~ 2022-01-28，标签 Ref($close,-2)/Ref($close,-1)-1。
不做 Fillna(0)；当日有效截面 < 30 的 IC 记为无效。

Example:
    .venv/bin/python examples/benchmarks/LightGBM/eval_l2_daily_factors.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# 允许直接 python 本脚本时找到 qlib 包
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import qlib
from qlib.config import REG_CN
from qlib.contrib.data.l2_daily import (
    L2_DAILY_FACTOR_GROUPS,
    L2_DAILY_FACTOR_IDS,
    get_l2_factor_category,
    get_l2_factor_group,
)
from qlib.data import D
from qlib.utils import code_to_fname

# ---------- 与训练 YAML 对齐的默认口径 ----------
DEFAULT_PROVIDER = "~/.qlib/qlib_data/cn_data"
DEFAULT_MARKET = "csi300"
DEFAULT_START = "2017-01-03"
DEFAULT_END = "2022-01-28"
DEFAULT_LABEL = "Ref($close, -2)/Ref($close, -1) - 1"
DEFAULT_L2_ROOT = (
    Path("/Users/900game/Desktop/ai/QuantDinger/backend_api_python")
    / "data"
    / "level2_staging"
    / "factors"
    / "l2_daily"
)
MIN_CS = 30
LOW_COVERAGE = 0.50
CHRONIC_COVERAGE = 0.80
HIGH_CORR = 0.80
SEGMENTS: Dict[str, Tuple[str, str]] = {
    "all": (DEFAULT_START, DEFAULT_END),
    "train": ("2017-01-03", "2019-12-31"),
    "valid": ("2020-01-01", "2020-12-31"),
    "test": ("2021-01-01", "2022-01-28"),
}
# 已有模型级 IC，只作对照，不重新训练
MODEL_IC_REF = {
    "LightGBM_L2Daily_20261024": {
        "IC": 0.0098,
        "ICIR": 0.0576,
        "Rank IC": -0.0016,
        "Rank ICIR": -0.0098,
    }
}

ISSUE_ALL_NAN = "all_nan_day"
ISSUE_LOW_COV = "low_coverage_day"
ISSUE_SPARSE = "sparse_cs"
ISSUE_CHRONIC = "chronic_sparse"
ISSUE_STATUS = "status_mismatch"
ISSUE_MISSING_BIN = "missing_bin"


def _ts(text: str) -> pd.Timestamp:
    return pd.Timestamp(text)


def _datetime_level(index: pd.MultiIndex) -> int:
    """qlib 面板有时是 (datetime, instrument)，有时反过来。"""
    for i, name in enumerate(index.names):
        if name in {"datetime", "date"}:
            return i
    # 名称缺失时看第一层样本能不能当日期
    sample = index.get_level_values(0)[:8]
    parsed = pd.to_datetime(sample, errors="coerce")
    return 0 if parsed.notna().all() else 1


def _instrument_level(index: pd.MultiIndex) -> int:
    return 1 - _datetime_level(index)


def _json_safe(value: Any) -> Any:
    """把 numpy / Timestamp 转成 JSON 可写类型。"""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if (np.isnan(number) or np.isinf(number)) else number
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 L2Daily 百因子质量并列出问题数据")
    parser.add_argument("--provider_uri", default=DEFAULT_PROVIDER)
    parser.add_argument("--market", default=DEFAULT_MARKET)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument(
        "--l2_root",
        default=str(DEFAULT_L2_ROOT),
        help="l2_daily 根目录（含 _markers/ 与 v1/）",
    )
    parser.add_argument(
        "--out_dir",
        default=str(Path(__file__).resolve().parent / "eval_output"),
    )
    return parser.parse_args()


def init_qlib(provider_uri: str) -> Path:
    """初始化 qlib，返回展开后的数据根目录。"""
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    return Path(provider_uri).expanduser().resolve()


def load_panel(market: str, start: str, end: str) -> Tuple[pd.DataFrame, pd.Series]:
    """一次拉齐 100 个 L2 因子和标签；列名改成 factor_id。"""
    fields = [f"${fid}" for fid in L2_DAILY_FACTOR_IDS] + [DEFAULT_LABEL]
    print(f"loading {len(fields)} fields {market} {start}~{end}", flush=True)
    raw = D.features(D.instruments(market), fields, start_time=start, end_time=end, freq="day")
    if raw is None or raw.empty:
        raise RuntimeError("D.features returned empty panel")
    raw.columns = list(L2_DAILY_FACTOR_IDS) + ["label"]
    features = raw[list(L2_DAILY_FACTOR_IDS)]
    label = raw["label"]
    date_lv = _datetime_level(raw.index)
    inst_lv = _instrument_level(raw.index)
    print(
        f"panel rows={len(raw)} dates={raw.index.get_level_values(date_lv).nunique()} "
        f"inst={raw.index.get_level_values(inst_lv).nunique()} "
        f"index_names={list(raw.index.names)}",
        flush=True,
    )
    return features, label


def _align_wide(factor: pd.Series, label: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """把 MultiIndex Series 展开成 date × instrument，列集合对齐。"""
    date_lv = _datetime_level(factor.index)
    inst_lv = _instrument_level(factor.index)
    factor_wide = factor.unstack(level=inst_lv)
    label_wide = label.unstack(level=inst_lv)
    factor_wide.index = _as_datetime_index(factor_wide.index)
    label_wide.index = _as_datetime_index(label_wide.index)
    columns = factor_wide.columns.intersection(label_wide.columns)
    return factor_wide.loc[:, columns], label_wide.loc[:, columns]


def _row_pearson(left: np.ndarray, right: np.ndarray, min_cs: int) -> np.ndarray:
    """按行算 Pearson，忽略 NaN；有效截面 < min_cs 记 NaN。"""
    mask = np.isfinite(left) & np.isfinite(right)
    count = mask.sum(axis=1)
    left = np.where(mask, left, np.nan)
    right = np.where(mask, right, np.nan)
    # 常数截面 denom=0，后面再挡掉
    with np.errstate(invalid="ignore", divide="ignore"):
        left_mean = np.nanmean(left, axis=1, keepdims=True)
        right_mean = np.nanmean(right, axis=1, keepdims=True)
        left_d = np.where(mask, left - left_mean, 0.0)
        right_d = np.where(mask, right - right_mean, 0.0)
        cov = (left_d * right_d).sum(axis=1)
        var_l = (left_d * left_d).sum(axis=1)
        var_r = (right_d * right_d).sum(axis=1)
        denom = np.sqrt(var_l * var_r)
    out = np.full(left.shape[0], np.nan, dtype=np.float64)
    ok = (count >= min_cs) & (denom > 0)
    out[ok] = cov[ok] / denom[ok]
    return out


def _as_datetime_index(index: pd.Index) -> pd.DatetimeIndex:
    """qlib 面板日期有时是 str，统一成 Timestamp 才能按窗口切片。"""
    return pd.DatetimeIndex(pd.to_datetime(index))


def daily_ic_series(factor: pd.Series, label: pd.Series, min_cs: int = MIN_CS) -> Tuple[pd.Series, pd.Series]:
    """返回按日 Pearson IC 与 Spearman Rank IC。"""
    factor_wide, label_wide = _align_wide(factor, label)
    dates = _as_datetime_index(factor_wide.index)
    ic = pd.Series(_row_pearson(factor_wide.to_numpy(dtype=np.float64), label_wide.to_numpy(dtype=np.float64), min_cs), index=dates)
    # Spearman = 先截面 rank 再 Pearson
    factor_rank = factor_wide.rank(axis=1, method="average")
    label_rank = label_wide.rank(axis=1, method="average")
    ric = pd.Series(
        _row_pearson(factor_rank.to_numpy(dtype=np.float64), label_rank.to_numpy(dtype=np.float64), min_cs),
        index=dates,
    )
    return ic, ric


def ic_stats(series: pd.Series) -> Dict[str, Optional[float]]:
    """mean / ICIR / t-stat；有效日不足则全空。"""
    clean = series.dropna()
    n_days = int(len(clean))
    if n_days < 2:
        return {"mean": None, "std": None, "icir": None, "tstat": None, "n_days": n_days}
    mean = float(clean.mean())
    std = float(clean.std(ddof=1))
    icir = mean / std if std > 0 else None
    tstat = mean / (std / np.sqrt(n_days)) if std > 0 else None
    return {"mean": mean, "std": std, "icir": icir, "tstat": tstat, "n_days": n_days}


def daily_coverage(factor: pd.Series) -> pd.DataFrame:
    """每个交易日的有效样本数 / 宇宙数 / 缺测率。"""
    date_lv = _datetime_level(factor.index)
    n_valid = factor.notna().groupby(level=date_lv).sum().astype(int)
    n_universe = factor.groupby(level=date_lv).size().astype(int)
    frame = pd.DataFrame({"n_valid": n_valid, "n_universe": n_universe})
    frame.index = _as_datetime_index(frame.index)
    frame["missing_rate"] = 1.0 - frame["n_valid"] / frame["n_universe"].clip(lower=1)
    return frame


def slice_by_range(series: pd.Series, start: str, end: str) -> pd.Series:
    index = series.index
    if isinstance(index, pd.MultiIndex):
        dates = _as_datetime_index(index.get_level_values(_datetime_level(index)))
        mask = (dates >= _ts(start)) & (dates <= _ts(end))
        return series.loc[mask]
    sliced = series.copy()
    sliced.index = _as_datetime_index(sliced.index)
    return sliced.loc[_ts(start) : _ts(end)]


def year_coverage(cover: pd.DataFrame) -> Dict[str, float]:
    """分年有效率（按日有效率再平均，避免股票数膨胀主导）。"""
    out: Dict[str, float] = {}
    years = cover.index.year
    for year in sorted(set(years)):
        part = cover.loc[years == year]
        valid = float(part["n_valid"].sum())
        total = float(part["n_universe"].sum())
        out[str(int(year))] = valid / total if total else 0.0
    return out


def classify_day_issue(n_valid: int, n_universe: int) -> Optional[str]:
    """互斥：全空 > 截面过稀 > 低覆盖。"""
    if n_universe <= 0:
        return ISSUE_ALL_NAN
    if n_valid == 0:
        return ISSUE_ALL_NAN
    if 0 < n_valid < MIN_CS:
        return ISSUE_SPARSE
    valid_rate = n_valid / n_universe
    if valid_rate < LOW_COVERAGE:
        return ISSUE_LOW_COV
    return None


def load_markers(marker_dir: Path, start: str, end: str) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """读取窗口内 _markers/{YYYYMMDD}_{group}.json。"""
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if not marker_dir.is_dir():
        print(f"marker dir missing: {marker_dir}", flush=True)
        return out
    start_key = _ts(start).strftime("%Y%m%d")
    end_key = _ts(end).strftime("%Y%m%d")
    for path in marker_dir.glob("*.json"):
        stem = path.stem
        if "_" not in stem:
            continue
        day, group = stem.split("_", 1)
        if len(day) != 8 or not day.isdigit() or day < start_key or day > end_key:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        out[(day, group)] = payload
    print(f"markers loaded={len(out)} from {marker_dir}", flush=True)
    return out


def check_missing_bins(qlib_dir: Path, instruments: Sequence[str]) -> List[Dict[str, Any]]:
    """CSI300 成分缺少 l2_*.day.bin 的逐条记录。"""
    features_root = qlib_dir / "features"
    issues: List[Dict[str, Any]] = []
    for inst in instruments:
        folder = features_root / code_to_fname(inst)
        missing = [fid for fid in L2_DAILY_FACTOR_IDS if not (folder / f"{fid}.day.bin").exists()]
        if not missing:
            continue
        # 整组缺则按因子列出，note 标明股票
        for fid in missing:
            group, category = L2_DAILY_FACTOR_GROUPS[fid]
            issues.append(
                _issue_row(
                    date="",
                    factor_id=fid,
                    group=group,
                    category=category,
                    issue_type=ISSUE_MISSING_BIN,
                    n_valid=0,
                    n_universe=1,
                    missing_rate=1.0,
                    marker_status="",
                    symbol_fail=None,
                    note=f"missing bin {inst}",
                )
            )
    return issues


def _issue_row(
    *,
    date: str,
    factor_id: str,
    group: str,
    category: str,
    issue_type: str,
    n_valid: int,
    n_universe: int,
    missing_rate: float,
    marker_status: str,
    symbol_fail: Optional[int],
    note: str,
) -> Dict[str, Any]:
    return {
        "date": date,
        "factor_id": factor_id,
        "group": group,
        "category": category,
        "issue_type": issue_type,
        "n_valid": int(n_valid),
        "n_universe": int(n_universe),
        "missing_rate": float(missing_rate),
        "marker_status": marker_status,
        "symbol_fail": symbol_fail,
        "note": note,
    }


def evaluate_factor(
    factor_id: str,
    factor: pd.Series,
    label: pd.Series,
    markers: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
    """单因子覆盖 + 分段 IC + 问题日枚举。"""
    group = get_l2_factor_group(factor_id)
    category = get_l2_factor_category(factor_id)
    cover = daily_coverage(factor)
    overall_valid = float(cover["n_valid"].sum())
    overall_total = float(cover["n_universe"].sum())
    overall_rate = overall_valid / overall_total if overall_total else 0.0

    ic_all, ric_all = daily_ic_series(factor, label)
    segment_metrics: Dict[str, Any] = {}
    for name, (start, end) in SEGMENTS.items():
        ic_part = ic_stats(slice_by_range(ic_all, start, end))
        ric_part = ic_stats(slice_by_range(ric_all, start, end))
        segment_metrics[name] = {
            "IC": ic_part["mean"],
            "ICIR": ic_part["icir"],
            "IC_tstat": ic_part["tstat"],
            "IC_n_days": ic_part["n_days"],
            "RankIC": ric_part["mean"],
            "RankICIR": ric_part["icir"],
            "RankIC_tstat": ric_part["tstat"],
            "RankIC_n_days": ric_part["n_days"],
        }

    rank_ic = segment_metrics["all"]["RankIC"]
    direction = "+" if rank_ic is not None and rank_ic > 0 else ("-" if rank_ic is not None else "")

    issues: List[Dict[str, Any]] = []
    if overall_rate < CHRONIC_COVERAGE:
        issues.append(
            _issue_row(
                date="",
                factor_id=factor_id,
                group=group,
                category=category,
                issue_type=ISSUE_CHRONIC,
                n_valid=int(overall_valid),
                n_universe=int(overall_total),
                missing_rate=1.0 - overall_rate,
                marker_status="",
                symbol_fail=None,
                note=f"window valid_rate={overall_rate:.4f} < {CHRONIC_COVERAGE}",
            )
        )

    for date, row in cover.iterrows():
        n_valid = int(row["n_valid"])
        n_universe = int(row["n_universe"])
        missing_rate = float(row["missing_rate"])
        issue_type = classify_day_issue(n_valid, n_universe)
        if issue_type is None:
            continue
        day_key = pd.Timestamp(date).strftime("%Y%m%d")
        marker = markers.get((day_key, group), {})
        issues.append(
            _issue_row(
                date=pd.Timestamp(date).strftime("%Y-%m-%d"),
                factor_id=factor_id,
                group=group,
                category=category,
                issue_type=issue_type,
                n_valid=n_valid,
                n_universe=n_universe,
                missing_rate=missing_rate,
                marker_status=str(marker.get("status") or ""),
                symbol_fail=marker.get("symbol_fail"),
                note="",
            )
        )

    n_all_nan = int((cover["n_valid"] == 0).sum())
    n_sparse = int(((cover["n_valid"] > 0) & (cover["n_valid"] < MIN_CS)).sum())
    record = {
        "factor_id": factor_id,
        "group": group,
        "category": category,
        "coverage": overall_rate,
        "coverage_by_year": year_coverage(cover),
        "n_all_nan_days": n_all_nan,
        "n_sparse_days": n_sparse,
        "n_valid_cs_days": int((cover["n_valid"] >= MIN_CS).sum()),
        "direction": direction,
        **{f"{seg}_{k}": v for seg, metrics in segment_metrics.items() for k, v in metrics.items()},
        "segments": segment_metrics,
    }
    return record, issues, cover


def group_status_mismatches(
    covers: Dict[str, pd.DataFrame],
    markers: Dict[Tuple[str, str], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """marker 标 done，但该组在 CSI300 上整组全空或低覆盖。"""
    group_factors: Dict[str, List[str]] = defaultdict(list)
    for fid in L2_DAILY_FACTOR_IDS:
        group_factors[get_l2_factor_group(fid)].append(fid)

    # 所有因子覆盖表的日期并集
    all_dates = pd.Index([])
    for cover in covers.values():
        all_dates = all_dates.union(cover.index)
    all_dates = all_dates.sort_values()

    issues: List[Dict[str, Any]] = []
    for group, factor_ids in group_factors.items():
        for date in all_dates:
            day_key = pd.Timestamp(date).strftime("%Y%m%d")
            marker = markers.get((day_key, group), {})
            if str(marker.get("status") or "") != "done":
                continue
            n_valid_list: List[int] = []
            n_universe_list: List[int] = []
            for fid in factor_ids:
                cover = covers[fid]
                if date not in cover.index:
                    n_valid_list.append(0)
                    n_universe_list.append(0)
                    continue
                n_valid_list.append(int(cover.loc[date, "n_valid"]))
                n_universe_list.append(int(cover.loc[date, "n_universe"]))
            n_valid = int(np.sum(n_valid_list))
            n_universe = int(np.sum(n_universe_list))
            missing_rate = 1.0 - (n_valid / n_universe) if n_universe else 1.0
            all_empty = n_valid == 0
            low = (not all_empty) and missing_rate > (1.0 - LOW_COVERAGE)
            if not (all_empty or low):
                continue
            issues.append(
                _issue_row(
                    date=pd.Timestamp(date).strftime("%Y-%m-%d"),
                    factor_id="*",
                    group=group,
                    category="",
                    issue_type=ISSUE_STATUS,
                    n_valid=n_valid,
                    n_universe=n_universe,
                    missing_rate=missing_rate,
                    marker_status="done",
                    symbol_fail=marker.get("symbol_fail"),
                    note="marker done but CSI300 group empty/low coverage",
                )
            )
    return issues


def monthly_mean_spearman(features: pd.DataFrame) -> pd.DataFrame:
    """每月取最后一个交易日做截面 Spearman，再对月份平均。"""
    # 复制一份并把日期层转成 Timestamp，避免 xs 时 str/Timestamp 对不上
    frame = features.copy()
    date_lv = _datetime_level(frame.index)
    inst_lv = _instrument_level(frame.index)
    frame.index = pd.MultiIndex.from_arrays(
        [_as_datetime_index(frame.index.get_level_values(date_lv)), frame.index.get_level_values(inst_lv)],
        names=["datetime", "instrument"],
    )
    unique_dates = pd.DatetimeIndex(sorted(frame.index.get_level_values(0).unique()))
    month_ends = unique_dates.to_series().groupby(unique_dates.to_period("M")).max()
    acc: Optional[pd.DataFrame] = None
    n_used = 0
    for day in month_ends:
        block = frame.xs(day, level=0)
        if block.dropna(how="all").shape[0] < MIN_CS:
            continue
        corr = block.corr(method="spearman")
        acc = corr if acc is None else acc.add(corr)
        n_used += 1
    if acc is None or n_used == 0:
        return pd.DataFrame(index=list(L2_DAILY_FACTOR_IDS), columns=list(L2_DAILY_FACTOR_IDS), dtype=float)
    print(f"corr months used={n_used}", flush=True)
    return acc / n_used


def high_corr_pairs(corr: pd.DataFrame) -> List[Dict[str, Any]]:
    """|ρ|>0.8 的上三角对。"""
    pairs: List[Dict[str, Any]] = []
    ids = list(corr.columns)
    values = corr.to_numpy(dtype=np.float64)
    for i, left in enumerate(ids):
        for j in range(i + 1, len(ids)):
            rho = values[i, j]
            if not np.isfinite(rho) or abs(rho) <= HIGH_CORR:
                continue
            right = ids[j]
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "rho": float(rho),
                    "left_group": get_l2_factor_group(left),
                    "right_group": get_l2_factor_group(right),
                    "same_group": get_l2_factor_group(left) == get_l2_factor_group(right),
                }
            )
    pairs.sort(key=lambda item: abs(item["rho"]), reverse=True)
    return pairs


def group_summary(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 group / category 汇总中位 |Rank IC| 与覆盖。"""
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        buckets[(rec["group"], rec["category"])].append(rec)
    rows: List[Dict[str, Any]] = []
    for (group, category), items in sorted(buckets.items()):
        rank_abs = [abs(x["all_RankIC"]) for x in items if x["all_RankIC"] is not None]
        icir = [x["all_ICIR"] for x in items if x["all_ICIR"] is not None]
        rows.append(
            {
                "group": group,
                "category": category,
                "n_factors": len(items),
                "median_abs_rank_ic": float(np.median(rank_abs)) if rank_abs else None,
                "median_icir": float(np.median(icir)) if icir else None,
                "median_coverage": float(np.median([x["coverage"] for x in items])),
                "n_rank_ic_ge_0_02": int(sum(1 for x in items if x["all_RankIC"] is not None and abs(x["all_RankIC"]) >= 0.02)),
                "n_icir_ge_0_3": int(sum(1 for x in items if x["all_ICIR"] is not None and x["all_ICIR"] >= 0.3)),
            }
        )
    return rows


def issues_summary(issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """各类型条数、最严重日期 / 因子。"""
    counts: Dict[str, int] = defaultdict(int)
    by_day: Dict[str, int] = defaultdict(int)
    by_factor: Dict[str, int] = defaultdict(int)
    for item in issues:
        counts[item["issue_type"]] += 1
        if item["date"]:
            by_day[item["date"]] += 1
        if item["factor_id"] and item["factor_id"] != "*":
            by_factor[item["factor_id"]] += 1
    worst_days = [
        {"date": date, "n_issues": n}
        for date, n in sorted(by_day.items(), key=lambda kv: kv[1], reverse=True)[:20]
    ]
    worst_factors = [
        {"factor_id": fid, "n_issues": n, "group": get_l2_factor_group(fid)}
        for fid, n in sorted(by_factor.items(), key=lambda kv: kv[1], reverse=True)[:20]
    ]
    return {"counts": dict(counts), "worst_days": worst_days, "worst_factors": worst_factors}


def issues_by_day(issues: List[Dict[str, Any]]) -> pd.DataFrame:
    """按日 × 组透视：坏了几列、有哪些类型。"""
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for item in issues:
        if not item["date"]:
            continue
        grouped[(item["date"], item["group"])].append(item)
    for (date, group), items in sorted(grouped.items()):
        types = sorted({x["issue_type"] for x in items})
        factor_ids = sorted({x["factor_id"] for x in items if x["factor_id"] != "*"})
        missing = [x["missing_rate"] for x in items]
        rows.append(
            {
                "date": date,
                "group": group,
                "n_issues": len(items),
                "n_factors": len(factor_ids),
                "issue_types": "|".join(types),
                "max_missing_rate": max(missing) if missing else None,
                "marker_status": next((x["marker_status"] for x in items if x["marker_status"]), ""),
                "has_status_mismatch": any(x["issue_type"] == ISSUE_STATUS for x in items),
            }
        )
    return pd.DataFrame(rows)


def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """质量 csv 用的扁平行，去掉嵌套。"""
    row = {k: v for k, v in record.items() if k not in {"segments", "coverage_by_year"}}
    for year, rate in record.get("coverage_by_year", {}).items():
        row[f"coverage_{year}"] = rate
    return row


def write_outputs(
    out_dir: Path,
    payload: Dict[str, Any],
    records: List[Dict[str, Any]],
    issues: List[Dict[str, Any]],
) -> None:
    """写出 JSON + 三份 CSV。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    quality_path = out_dir / "l2_daily_factor_quality.json"
    quality_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame([flatten_record(x) for x in records]).to_csv(out_dir / "l2_daily_factor_quality.csv", index=False)
    pd.DataFrame(issues).to_csv(out_dir / "l2_daily_data_issues.csv", index=False)
    by_day = issues_by_day(issues)
    by_day.to_csv(out_dir / "l2_daily_data_issues_by_day.csv", index=False)
    print(f"wrote {quality_path}", flush=True)
    print(f"wrote {out_dir / 'l2_daily_data_issues.csv'} rows={len(issues)}", flush=True)
    print(f"wrote {out_dir / 'l2_daily_data_issues_by_day.csv'} rows={len(by_day)}", flush=True)


def main() -> None:
    args = parse_args()
    qlib_dir = init_qlib(args.provider_uri)
    l2_root = Path(args.l2_root).expanduser()
    marker_dir = l2_root / "_markers" if (l2_root / "_markers").is_dir() else l2_root.parent / "_markers"

    features, label = load_panel(args.market, args.start, args.end)
    date_lv = _datetime_level(features.index)
    inst_lv = _instrument_level(features.index)
    instruments = sorted(features.index.get_level_values(inst_lv).unique())
    n_dates = int(features.index.get_level_values(date_lv).nunique())
    markers = load_markers(marker_dir, args.start, args.end)

    records: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    covers: Dict[str, pd.DataFrame] = {}
    for i, factor_id in enumerate(L2_DAILY_FACTOR_IDS, start=1):
        print(f"[{i:03d}/100] {factor_id}", flush=True)
        record, factor_issues, cover = evaluate_factor(factor_id, features[factor_id], label, markers)
        records.append(record)
        issues.extend(factor_issues)
        covers[factor_id] = cover

    print("checking status mismatches / missing bins / correlation", flush=True)
    issues.extend(group_status_mismatches(covers, markers))
    issues.extend(check_missing_bins(qlib_dir, instruments))

    corr = monthly_mean_spearman(features)
    pairs = high_corr_pairs(corr)
    groups = group_summary(records)
    summary = issues_summary(issues)

    n_rank = sum(1 for x in records if x["all_RankIC"] is not None and abs(x["all_RankIC"]) >= 0.02)
    n_icir = sum(1 for x in records if x["all_ICIR"] is not None and x["all_ICIR"] >= 0.3)

    payload = {
        "meta": {
            "universe": args.market,
            "start": args.start,
            "end": args.end,
            "label": DEFAULT_LABEL,
            "min_cross_section": MIN_CS,
            "n_instruments": len(instruments),
            "n_days": n_dates,
            "n_factors": len(records),
            "n_rank_ic_ge_0_02": n_rank,
            "n_icir_ge_0_3": n_icir,
            "model_ic_ref": MODEL_IC_REF,
            "note_202206": "2022-06 source is incomplete (order/microstructure missing on 20220617); excluded from this window",
        },
        "factors": [{k: _json_safe(v) if k != "segments" and k != "coverage_by_year" else v for k, v in rec.items()} for rec in records],
        "groups": groups,
        "corr_pairs": pairs[:200],
        "issues_summary": summary,
    }
    # 嵌套里的 numpy 再扫一遍
    payload = json.loads(json.dumps(payload, default=_json_safe))
    write_outputs(Path(args.out_dir), payload, records, issues)
    print(
        f"done factors={len(records)} issues={len(issues)} "
        f"|RankIC|>=0.02:{n_rank} ICIR>=0.3:{n_icir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
