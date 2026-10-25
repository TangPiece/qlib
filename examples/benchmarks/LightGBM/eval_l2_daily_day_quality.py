"""评估单日 L2 宽表质量：覆盖、抽样对账、CSI300 单日 Rank IC。

只读 parquet / OHLCV，不写 bin，不重跑四组全市场计算。

Example:
    .venv/bin/python examples/benchmarks/LightGBM/eval_l2_daily_day_quality.py --dates 20220601,20220602
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qlib.contrib.data.l2_daily import (  # noqa: E402
    L2_DAILY_FACTOR_IDS,
    get_l2_factor_category,
    get_l2_factor_group,
)
from qlib.contrib.data.l2_daily.catalog import ALL_GROUPS, factor_ids_for_group  # noqa: E402
from qlib.contrib.data.l2_daily.csv_io import read_csv_bytes  # noqa: E402
from qlib.contrib.data.l2_daily.engine import source_file_path  # noqa: E402
from qlib.contrib.data.l2_daily.groups.trade import calc_trade_factors  # noqa: E402
from qlib.contrib.data.l2_daily._prep import prepare_trades  # noqa: E402


def _to_qlib(symbol: str) -> str | None:
    text = str(symbol)
    if "." not in text:
        return None
    code, exch = text.rsplit(".", 1)
    if not code.isdigit():
        return None
    return f"{exch.upper()}{code}"


def _close(left, right) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    return bool(np.isclose(float(left), float(right), rtol=1e-5, atol=1e-8))


def _wide_path(day: str) -> Path:
    """v1/YYYY/YYYYMM/YYYYMMDD.parquet。"""
    return _REPO_ROOT / "data/level2_staging/factors/l2_daily/v1" / day[:4] / day[:6] / f"{day}.parquet"


def _iso(day: str) -> str:
    """YYYYMMDD → YYYY-MM-DD，给 qlib 日历用。"""
    return f"{day[:4]}-{day[4:6]}-{day[6:8]}"


def _shift_day(day: str, delta: int) -> str:
    """按自然日平移，用来找邻日宽表和次日收益窗口。"""
    return (datetime.strptime(day, "%Y%m%d") + timedelta(days=delta)).strftime("%Y%m%d")


def evaluate_day(day: str) -> dict:
    """评估一日宽表，写 JSON 并返回摘要。"""
    wide_path = _wide_path(day)
    if not wide_path.is_file():
        raise FileNotFoundError(f"缺少日宽表: {wide_path}")
    raw_root = _REPO_ROOT / "data/level2_raw"
    mark_root = _REPO_ROOT / "data/level2_staging/factors/l2_daily/_markers"
    out_path = _REPO_ROOT / "examples/benchmarks/LightGBM/eval_output" / f"l2_daily_quality_{day}.json"

    df = pd.read_parquet(wide_path)
    markers = {}
    for group in ALL_GROUPS:
        path = mark_root / f"{day}_{group}.json"
        markers[group] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None

    factor_rows = []
    for fid in L2_DAILY_FACTOR_IDS:
        series = pd.to_numeric(df[fid], errors="coerce")
        finite = series.replace([np.inf, -np.inf], np.nan)
        n_valid = int(finite.notna().sum())
        factor_rows.append(
            {
                "factor_id": fid,
                "group": get_l2_factor_group(fid),
                "category": get_l2_factor_category(fid),
                "n_valid": n_valid,
                "coverage": n_valid / len(df),
                "n_inf": int(np.isinf(series.fillna(0)).sum()),
                "mean": None if n_valid == 0 else float(finite.mean()),
                "std": None if n_valid < 2 else float(finite.std()),
                "p50": None if n_valid == 0 else float(finite.median()),
                "p01": None if n_valid == 0 else float(finite.quantile(0.01)),
                "p99": None if n_valid == 0 else float(finite.quantile(0.99)),
                "n_unique": int(finite.nunique(dropna=True)),
            }
        )
    stats = pd.DataFrame(factor_rows)

    group_null = {}
    group_any = {}
    for group in ALL_GROUPS:
        cols = factor_ids_for_group(group)
        sub = df[cols]
        group_null[group] = float(sub.isna().to_numpy().mean())
        group_any[group] = float(sub.notna().any(axis=1).mean())

    neighbors = {}
    for other in (_shift_day(day, -1), _shift_day(day, 1)):
        path = (
            _REPO_ROOT
            / "data/level2_staging/factors/l2_daily/v1"
            / other[:4]
            / other[:6]
            / f"{other}.parquet"
        )
        if not path.is_file():
            neighbors[other] = None
            continue
        other_df = pd.read_parquet(path)
        rec = {"n": int(len(other_df))}
        for group in ALL_GROUPS:
            cols = [c for c in factor_ids_for_group(group) if c in other_df.columns]
            rec[group] = float(other_df[cols].isna().to_numpy().mean()) if cols else None
        rec["trade_count_cov"] = (
            float(other_df["l2_trade_count"].notna().mean()) if "l2_trade_count" in other_df.columns else None
        )
        neighbors[other] = rec

    trade_fail = df.loc[df["l2_trade_count"].isna(), "symbol"].astype(str).tolist()

    # 抽样对账：锚点 + 3 只随机，只算 trade 组，避免进程池
    rng = np.random.default_rng(int(day))
    extras = [c for c in df["symbol"].astype(str) if c not in ("600000.SH", "000001.SZ")]
    picked = ["600000.SH", "000001.SZ", *list(rng.choice(extras, size=3, replace=False))]
    live_rows = []
    for code in picked:
        path = source_file_path(raw_root, day, code, "逐笔成交")
        prep = 0
        live: dict = {}
        if path is not None and path.suffix.lower() == ".csv":
            trades = read_csv_bytes(path.read_bytes(), "逐笔成交")
            prep = int(len(prepare_trades(trades, code)))
            live = calc_trade_factors(trades, code)
        table = df[df["symbol"] == code]
        live_count = live.get("l2_trade_count", float("nan"))
        wide_count = float(table["l2_trade_count"].iloc[0]) if len(table) else float("nan")
        live_vwap = live.get("l2_vwap", float("nan"))
        wide_vwap = float(table["l2_vwap"].iloc[0]) if len(table) else float("nan")
        live_rows.append(
            {
                "symbol": code,
                "prepare_trades": prep,
                "live_count": None if pd.isna(live_count) else float(live_count),
                "wide_count": None if pd.isna(wide_count) else float(wide_count),
                "count_match": _close(live_count, wide_count),
                "vwap_match": _close(live_vwap, wide_vwap),
            }
        )

    import qlib
    from qlib.config import REG_CN
    from qlib.data import D

    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    iso = _iso(day)
    next_iso = _iso(_shift_day(day, 3))
    instruments = D.list_instruments(
        instruments=D.instruments("csi300"),
        start_time=iso,
        end_time=next_iso,
        as_list=True,
    )
    label = D.features(
        instruments,
        ["Ref($close, -1)/$close - 1"],
        start_time=iso,
        end_time=iso,
        freq="day",
    )
    label.columns = ["ret"]
    lab = label.reset_index()
    inst_col = "instrument" if "instrument" in lab.columns else lab.columns[0]
    lab_map = dict(zip(lab[inst_col].astype(str), lab["ret"]))

    work = df.copy()
    work["qlib"] = work["symbol"].map(_to_qlib)
    work = work[work["qlib"].isin(set(instruments))].copy()
    work["ret"] = work["qlib"].map(lab_map)
    y = pd.to_numeric(work["ret"], errors="coerce")

    ic_rows = []
    for fid in L2_DAILY_FACTOR_IDS:
        x = pd.to_numeric(work[fid], errors="coerce")
        mask = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        n = int(mask.sum())
        if n < 30:
            ic_rows.append({"factor_id": fid, "group": get_l2_factor_group(fid), "n": n, "ic": None, "rank_ic": None})
            continue
        xx = x[mask].to_numpy(dtype=float)
        yy = y[mask].to_numpy(dtype=float)
        ic = float(np.corrcoef(xx, yy)[0, 1]) if xx.std() > 0 and yy.std() > 0 else None
        rx = pd.Series(xx).rank().to_numpy()
        ry = pd.Series(yy).rank().to_numpy()
        ric = float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else None
        ic_rows.append({"factor_id": fid, "group": get_l2_factor_group(fid), "n": n, "ic": ic, "rank_ic": ric})
    ic_df = pd.DataFrame(ic_rows)

    payload = {
        "date": day,
        "n_symbols": int(len(df)),
        "n_factors": 100,
        "markers": markers,
        "group_null_rates": group_null,
        "group_symbol_any_valid": group_any,
        "trade_count_coverage": float(df["l2_trade_count"].notna().mean()),
        "trade_fail_n": len(trade_fail),
        "trade_fail_sample": trade_fail[:20],
        "neighbors": neighbors,
        "live_samples": live_rows,
        "csi300_overlap": int(len(work)),
        "csi300_with_label": int(y.notna().sum()),
        "low_coverage_factors": stats[stats["coverage"] < 0.80].sort_values("coverage").to_dict(orient="records"),
        "near_constant_factors": stats[(stats["n_unique"] <= 2) & (stats["n_valid"] > 0)][
            ["factor_id", "group", "n_valid", "n_unique", "coverage"]
        ].to_dict(orient="records"),
        "factor_stats": stats.to_dict(orient="records"),
        "factor_ic": ic_rows,
        "mean_abs_rank_ic": float(ic_df["rank_ic"].abs().mean()) if ic_df["rank_ic"].notna().any() else None,
        "mean_abs_ic": float(ic_df["ic"].abs().mean()) if ic_df["ic"].notna().any() else None,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)
    print("group_null", {k: round(v, 4) for k, v in group_null.items()}, flush=True)
    print("markers", {k: (v or {}).get("status") for k, v in markers.items()}, flush=True)
    print("low_cov", [r["factor_id"] for r in payload["low_coverage_factors"]], flush=True)
    ranked = ic_df.dropna(subset=["rank_ic"]).assign(abs_ric=lambda x: x["rank_ic"].abs())
    print(ranked.sort_values("abs_ric", ascending=False).head(10).to_string(index=False), flush=True)
    print("live", live_rows, flush=True)
    print("neighbors", neighbors, flush=True)
    print("trade_fail", len(trade_fail), trade_fail[:10], flush=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="评估单日 L2 宽表质量")
    parser.add_argument("--dates", default="20180731", help="逗号分隔 YYYYMMDD")
    args = parser.parse_args(argv)
    dates = [item.strip() for item in args.dates.split(",") if item.strip()]
    failed = 0
    for day in dates:
        try:
            evaluate_day(day)
        except Exception as exc:
            failed += 1
            print(f"[{day}] 评估失败: {exc}", flush=True)
            raise
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
