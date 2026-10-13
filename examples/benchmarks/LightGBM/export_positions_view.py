#!/usr/bin/env python3
"""Export backtest positions to JSON for the shared day-browser HTML.

Reads positions and analysis artifacts from an MLflow experiment recorder,
labels hold/buy/sell, attaches stock names and buy dates, and writes
``data/runs/{exp_name}_positions.json``. View via the console shared page::

    /positions_view.html?data=/data/{exp_name}_positions.json

Example (from qlib repo root)::

    .venv/bin/python examples/benchmarks/LightGBM/export_positions_view.py \\
      --exp_name=rolling_lgb_7y1y_20261009
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import qlib
from qlib.config import C
from qlib.constant import REG_CN
from qlib.contrib.evaluate import risk_analysis
from qlib.data import D
from qlib.workflow import R

DIRNAME = Path(__file__).absolute().resolve().parent
DEFAULT_REF_DIR = DIRNAME / "data" / "ref"
DEFAULT_OUT_DIR = DIRNAME / "data" / "runs"
DEFAULT_EXP_NAME = "rolling_lgb_7y1y_20261009"
NAME_MAP_PATH = DEFAULT_REF_DIR / "stock_name_map.json"
DEFAULT_INDUSTRY_MAP = DEFAULT_REF_DIR / "csi300_sw_l1_industry_intervals.parquet"
POSITIONS_ARTIFACT = "portfolio_analysis/positions_normal_1day.pkl"
REPORT_ARTIFACT = "portfolio_analysis/report_normal_1day.pkl"

# instrument -> [(start, end, industry_code, industry_name), ...]
IndustryIntervals = Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, str, str]]]


def _latest_recorder(exp_name: str, recorder_id: Optional[str] = None):
    if recorder_id:
        return R.get_recorder(recorder_id=recorder_id, experiment_name=exp_name)
    exp = R.get_exp(experiment_name=exp_name, create=False)
    recorders = exp.list_recorders(rtype="list")
    if not recorders:
        raise RuntimeError(f"No recorders found for experiment {exp_name!r}")
    finished = [r for r in recorders if getattr(r, "status", None) == "FINISHED"]
    candidates = finished or list(recorders)
    candidates.sort(key=lambda r: getattr(r, "start_time", None) or "", reverse=True)
    return candidates[0]


def _to_qlib_code(code: str) -> str:
    code = str(code).strip()
    if code.startswith(("SH", "SZ", "BJ")):
        return code
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) != 6:
        return code
    if digits.startswith(("5", "6", "9")):
        return "SH" + digits
    return "SZ" + digits


def load_or_build_name_map(cache_path: Path = NAME_MAP_PATH) -> Dict[str, str]:
    """Load SH/SZ -> name map; try akshare if cache missing."""
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return {str(k): str(v) for k, v in data.items()}
        except (json.JSONDecodeError, OSError) as exc:
            warnings.warn(f"failed to read name map {cache_path}: {exc}")

    name_map: Dict[str, str] = {}
    try:
        import akshare as ak

        df = ak.stock_info_a_code_name()
        # columns typically: code, name
        code_col = "code" if "code" in df.columns else df.columns[0]
        name_col = "name" if "name" in df.columns else df.columns[1]
        for _, row in df.iterrows():
            qcode = _to_qlib_code(row[code_col])
            name = str(row[name_col]).strip()
            if qcode and name:
                name_map[qcode] = name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(name_map, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote name map {cache_path} ({len(name_map)} names)")
    except Exception as exc:  # noqa: BLE001 — best-effort optional dependency
        warnings.warn(f"stock name map unavailable ({exc}); names will be blank")
    return name_map


def _stock_info(pos) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for code in pos.get_stock_list():
        info = pos.position[code]
        out[code] = {
            "amount": float(info.get("amount")) if info.get("amount") is not None else None,
            "price": float(info.get("price")) if info.get("price") is not None else None,
            "weight": float(info.get("weight")) if info.get("weight") is not None else None,
            "count_day": int(info.get("count_day")) if info.get("count_day") is not None else None,
        }
    return out


def load_industry_intervals(path: Path) -> Optional[IndustryIntervals]:
    """Load PIT SW L1 intervals; return None if missing/unreadable."""
    path = Path(path).expanduser()
    if not path.exists():
        warnings.warn(
            f"industry map not found: {path}; day.industries will be empty. "
            "Run build_csi300_industry_map.py first."
        )
        return None
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"failed to read industry map {path}: {exc}")
        return None
    required = {"instrument", "industry_code", "industry_name", "start_date", "end_date"}
    missing = required - set(df.columns)
    if missing:
        warnings.warn(f"industry map missing columns {sorted(missing)}; skipping")
        return None
    out: IndustryIntervals = defaultdict(list)
    for row in df.itertuples(index=False):
        inst = str(getattr(row, "instrument")).strip().upper()
        code = str(getattr(row, "industry_code")).strip()
        name = str(getattr(row, "industry_name")).strip() or code
        start = pd.Timestamp(getattr(row, "start_date")).normalize()
        end = pd.Timestamp(getattr(row, "end_date")).normalize()
        out[inst].append((start, end, code, name))
    for inst in out:
        out[inst].sort(key=lambda x: (x[1], x[0]))
    return dict(out)


def lookup_industry(
    intervals: Optional[IndustryIntervals],
    instrument: str,
    trade_date: pd.Timestamp,
) -> Tuple[str, str]:
    """Return (industry_code, industry_name); unknown -> ('', '未知')."""
    if not intervals:
        return "", "未知"
    trade_date = pd.Timestamp(trade_date).normalize()
    key = str(instrument).strip().upper()
    ivs = intervals.get(key)
    if not ivs:
        return "", "未知"

    covering = [iv for iv in ivs if iv[0] <= trade_date <= iv[1]]
    if covering:
        covering.sort(key=lambda x: x[0])
        _, _, code, name = covering[-1]
        return code, name

    earlier = [iv for iv in ivs if iv[1] < trade_date]
    if earlier:
        earlier.sort(key=lambda x: x[1])
        _, _, code, name = earlier[-1]
        return code, name

    return "", "未知"


def aggregate_industries(hold_items: List[dict], date_str: str, intervals: Optional[IndustryIntervals]) -> List[dict]:
    """Sum hold weights by SW L1 industry (desc by weight)."""
    buckets: Dict[str, Dict[str, Any]] = {}
    trade_date = pd.Timestamp(date_str)
    for item in hold_items:
        code = item.get("instrument") or ""
        ind_code, ind_name = lookup_industry(intervals, code, trade_date)
        item["industry_code"] = ind_code
        item["industry_name"] = ind_name
        w = item.get("weight")
        w = float(w) if w is not None and not (isinstance(w, float) and np.isnan(w)) else 0.0
        key = ind_code or "__UNKNOWN__"
        if key not in buckets:
            buckets[key] = {
                "industry_code": ind_code,
                "industry_name": ind_name,
                "weight": 0.0,
                "n": 0,
            }
        buckets[key]["weight"] += w
        buckets[key]["n"] += 1
    rows = list(buckets.values())
    rows.sort(key=lambda r: (-float(r["weight"]), str(r["industry_name"])))
    return rows


def _fetch_adj_raw_prices(
    keys: List[Tuple[str, str]],
) -> Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]]:
    """Map (instrument, YYYY-MM-DD) -> (adj_close, raw_price).

    ``adj_close`` is qlib ``$close`` (factor-adjusted). ``raw_price`` is
    ``$close / $factor`` (approx. exchange nominal close).

    Uses a single kernel: macOS spawn + python -c/<stdin> breaks qlib's multiprocessing.
    """
    out: Dict[Tuple[str, str], Tuple[Optional[float], Optional[float]]] = {
        k: (None, None) for k in keys
    }
    if not keys:
        return out
    instruments = sorted({code for code, _ in keys})
    start = min(d for _, d in keys)
    end = max(d for _, d in keys)
    prev_kernels = C.kernels
    try:
        C.kernels = 1
        price_df = D.features(
            instruments, ["$close", "$factor"], start_time=start, end_time=end, freq="day"
        )
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"failed to load adj/raw prices: {exc}")
        return out
    finally:
        C.kernels = prev_kernels
    if price_df is None or price_df.empty:
        return out
    for (code, date_str) in keys:
        try:
            row = price_df.loc[(code, pd.Timestamp(date_str))]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            adj = row["$close"]
            fac = row["$factor"]
            adj_f = float(adj) if pd.notna(adj) else None
            raw_f = None
            if adj_f is not None and pd.notna(fac) and float(fac) != 0.0:
                raw_f = adj_f / float(fac)
            out[(code, date_str)] = (adj_f, raw_f)
        except (KeyError, TypeError, ValueError, IndexError, ZeroDivisionError):
            out[(code, date_str)] = (None, None)
    return out


def build_day_frames(
    positions: Dict,
    name_map: Optional[Dict[str, str]] = None,
    industry_intervals: Optional[IndustryIntervals] = None,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Return long parquet rows + per-day HTML payload (with name / buy_date / sell return)."""
    name_map = name_map or {}
    dates = sorted(positions.keys())
    rows: List[dict] = []
    days_payload: List[dict] = []
    prev_codes: set = set()
    prev_info: Dict[str, Dict[str, Any]] = {}
    buy_dates: Dict[str, str] = {}
    buy_prices: Dict[str, Optional[float]] = {}
    sell_keys: List[Tuple[str, str]] = []

    for dt in dates:
        pos = positions[dt]
        date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
        cash = float(pos.get_cash())
        account_value = float(pos.position.get("now_account_value", cash))
        cur_info = _stock_info(pos)
        cur_codes = set(cur_info)

        buys = sorted(cur_codes - prev_codes)
        sells = sorted(prev_codes - cur_codes)
        holds = sorted(cur_codes & prev_codes)

        sell_buy_dates = {c: buy_dates.get(c) for c in sells}
        sell_buy_prices = {c: buy_prices.get(c) for c in sells}
        for code in buys:
            buy_dates[code] = date_str
            buy_prices[code] = cur_info[code].get("price")
        for code in sells:
            buy_dates.pop(code, None)
            buy_prices.pop(code, None)
            sell_keys.append((code, date_str))

        def _item(
            code: str,
            info: Dict[str, Any],
            action: str,
            buy_date: Optional[str],
            *,
            buy_price: Optional[float] = None,
            sell_price: Optional[float] = None,
            return_rate: Optional[float] = None,
        ) -> dict:
            return {
                "instrument": code,
                "name": name_map.get(code) or "",
                "buy_date": buy_date,
                "amount": info.get("amount"),
                "price": info.get("price"),  # qlib adj close ($close)
                "raw_price": None,  # filled later: $close / $factor
                "weight": info.get("weight"),
                "count_day": info.get("count_day"),
                "action": action,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "sell_raw_price": None,
                "return_rate": return_rate,
            }

        day_hold = [
            _item(c, cur_info[c], "hold" if c in holds else "buy", buy_dates.get(c))
            for c in sorted(cur_codes)
        ]
        day_buy = [_item(c, cur_info[c], "buy", buy_dates.get(c)) for c in buys]
        day_sell = [
            _item(
                c,
                prev_info[c],
                "sell",
                sell_buy_dates.get(c),
                buy_price=sell_buy_prices.get(c),
            )
            for c in sells
        ]

        def _row(item: dict, action: Optional[str] = None) -> dict:
            return {
                "datetime": date_str,
                "instrument": item.get("instrument"),
                "name": item.get("name"),
                "buy_date": item.get("buy_date"),
                "amount": item.get("amount"),
                "price": item.get("price"),
                "weight": item.get("weight"),
                "count_day": item.get("count_day"),
                "cash": cash,
                "account_value": account_value,
                "action": action if action is not None else item.get("action"),
                "buy_price": item.get("buy_price"),
                "sell_price": item.get("sell_price"),
                "return_rate": item.get("return_rate"),
            }

        for item in day_hold:
            rows.append(_row(item))
        for item in day_sell:
            rows.append(_row(item, action="sell"))

        if not cur_codes and not sells:
            rows.append(
                {
                    "datetime": date_str,
                    "instrument": None,
                    "name": None,
                    "buy_date": None,
                    "amount": None,
                    "price": None,
                    "weight": None,
                    "count_day": None,
                    "cash": cash,
                    "account_value": account_value,
                    "action": None,
                    "buy_price": None,
                    "sell_price": None,
                    "return_rate": None,
                }
            )

        industries = aggregate_industries(day_hold, date_str, industry_intervals)

        days_payload.append(
            {
                "date": date_str,
                "cash": cash,
                "account_value": account_value,
                "n_hold": len(cur_codes),
                "n_buy": len(buys),
                "n_sell": len(sells),
                "hold": day_hold,
                "buy": day_buy,
                "sell": day_sell,
                "industries": industries,
            }
        )
        prev_codes = cur_codes
        prev_info = cur_info

    price_keys: List[Tuple[str, str]] = []
    for day in days_payload:
        d = day["date"]
        for bucket in ("hold", "buy", "sell"):
            for item in day[bucket]:
                code = item.get("instrument")
                if code:
                    price_keys.append((code, d))
    # unique while preserving order
    seen = set()
    uniq_keys: List[Tuple[str, str]] = []
    for k in price_keys:
        if k not in seen:
            seen.add(k)
            uniq_keys.append(k)

    price_map = _fetch_adj_raw_prices(uniq_keys)
    for day in days_payload:
        d = day["date"]
        for bucket in ("hold", "buy"):
            for item in day[bucket]:
                code = item.get("instrument")
                if not code:
                    continue
                adj, raw = price_map.get((code, d), (None, None))
                if item.get("price") is None and adj is not None:
                    item["price"] = adj
                item["raw_price"] = raw
        for item in day["sell"]:
            code = item.get("instrument")
            if not code:
                continue
            adj, raw = price_map.get((code, d), (None, None))
            buy_price = item.get("buy_price")
            item["sell_price"] = adj
            item["sell_raw_price"] = raw
            item["raw_price"] = raw
            if buy_price and adj and buy_price != 0:
                item["return_rate"] = float(adj) / float(buy_price) - 1.0
            else:
                item["return_rate"] = None

    for row in rows:
        code = row.get("instrument")
        date_str = row.get("datetime")
        if not code or not date_str:
            continue
        adj, raw = price_map.get((code, date_str), (None, None))
        if row.get("price") is None and adj is not None:
            row["price"] = adj
        row["raw_price"] = raw
        if row.get("action") == "sell":
            buy_price = row.get("buy_price")
            row["sell_price"] = adj
            row["sell_raw_price"] = raw
            if buy_price and adj and buy_price != 0:
                row["return_rate"] = float(adj) / float(buy_price) - 1.0
            else:
                row["return_rate"] = None

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["datetime", "action", "instrument"], kind="mergesort").reset_index(drop=True)
    return df, days_payload


def _series_ic_stats(ic: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    ic = ic.dropna()
    if ic.empty:
        return None, None
    mean = float(ic.mean())
    std = float(ic.std())
    icir = float(mean / std) if std and not np.isnan(std) else None
    return mean, icir


def _port_block(analysis_df: pd.DataFrame, key: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    try:
        block = analysis_df.loc[key]
        for metric in ("mean", "std", "annualized_return", "information_ratio", "max_drawdown"):
            try:
                val = block.loc[metric]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                out[metric] = float(val) if pd.notna(val) else None
            except (KeyError, IndexError, TypeError, ValueError):
                out[metric] = None
    except KeyError:
        pass
    return out


def collect_metrics(recorder) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "IC": None,
        "ICIR": None,
        "Rank IC": None,
        "Rank ICIR": None,
        "port": {
            "benchmark": {},
            "excess_without_cost": {},
            "excess_with_cost": {},
        },
        "indicators": {},
    }
    try:
        ic = recorder.load_object("sig_analysis/ic.pkl")
        ric = recorder.load_object("sig_analysis/ric.pkl")
        metrics["IC"], metrics["ICIR"] = _series_ic_stats(ic)
        metrics["Rank IC"], metrics["Rank ICIR"] = _series_ic_stats(ric)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"sig_analysis metrics unavailable: {exc}")

    try:
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
        metrics["port"]["benchmark"] = _port_block(analysis_df, "benchmark")
        # qlib may use these index labels:
        for src, dst in [
            ("excess_return_without_cost", "excess_without_cost"),
            ("excess_return_with_cost", "excess_with_cost"),
        ]:
            metrics["port"][dst] = _port_block(analysis_df, src)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"port_analysis metrics unavailable: {exc}")

    try:
        ind = recorder.load_object("portfolio_analysis/indicator_analysis_1day.pkl")
        if isinstance(ind, pd.DataFrame):
            metrics["indicators"] = {
                str(k): (float(v) if pd.notna(v) else None)
                for k, v in ind["value"].items()
            }
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"indicator_analysis unavailable: {exc}")

    return metrics


def annualize(r: pd.Series) -> float:
    """Annualized return using qlib day-freq risk_analysis (mean * 238)."""
    r = r.dropna()
    if r.empty:
        return float("nan")
    return float(risk_analysis(r, freq="day").loc["annualized_return", "risk"])


def collect_yearly_excess(recorder) -> Dict[str, Any]:
    """Yearly excess table + concentration warning from report_normal."""
    out: Dict[str, Any] = {"rows": [], "concentration_warning": None}
    try:
        report = recorder.load_object(REPORT_ARTIFACT)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"report_normal unavailable for yearly excess: {exc}")
        return out

    excess = report["return"] - report["bench"]
    excess_cost = report["return"] - report["bench"] - report["cost"]
    # Ensure DatetimeIndex for .year grouping
    if not isinstance(excess.index, pd.DatetimeIndex):
        excess.index = pd.to_datetime(excess.index)
        excess_cost.index = pd.to_datetime(excess_cost.index)

    by_year = excess.groupby(excess.index.year).apply(annualize)
    by_year_cost = excess_cost.groupby(excess_cost.index.year).apply(annualize)
    days_by_year = excess.groupby(excess.index.year).size()
    cum_cost_by_year = excess_cost.groupby(excess_cost.index.year).sum()

    years = sorted(set(by_year.index) | set(by_year_cost.index))
    rows: List[Dict[str, Any]] = []
    for y in years:
        ann = float(by_year.get(y, float("nan")))
        ann_cost = float(by_year_cost.get(y, float("nan")))
        cum_cost = float(cum_cost_by_year.get(y, float("nan")))
        n_days = int(days_by_year.get(y, 0))
        rows.append(
            {
                "year": int(y),
                "annualized_without_cost": None if np.isnan(ann) else ann,
                "annualized_with_cost": None if np.isnan(ann_cost) else ann_cost,
                "cum_excess_with_cost": None if np.isnan(cum_cost) else cum_cost,
                "n_days": n_days,
            }
        )
    out["rows"] = rows

    # Concentration: one year ≥ 50% of sum of positive with-cost annualized excess
    pos = [(r["year"], r["annualized_with_cost"]) for r in rows if r["annualized_with_cost"] and r["annualized_with_cost"] > 0]
    pos_sum = sum(v for _, v in pos)
    if pos and pos_sum > 0:
        top_year, top_val = max(pos, key=lambda t: t[1])
        share = top_val / pos_sum
        if share >= 0.5:
            out["concentration_warning"] = (
                f"{top_year} 年含成本年化超额占各年正贡献之和的 {share:.0%}；"
                "若大部分改善集中在一两年，警惕是特定行情（如风格切换）喂出来的。"
            )
    return out



def export(
    exp_name: str,
    recorder_id: Optional[str] = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    provider_uri: str = "~/.qlib/qlib_data/cn_data",
    name_map_path: Path = NAME_MAP_PATH,
    industry_map_path: Path = DEFAULT_INDUSTRY_MAP,
) -> Path:
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    recorder = _latest_recorder(exp_name, recorder_id)
    rid = recorder.info.get("id") if hasattr(recorder, "info") else getattr(recorder, "id", None)
    if rid is None:
        rid = str(recorder)
    print(f"using experiment={exp_name!r} recorder={rid}")

    name_map = load_or_build_name_map(Path(name_map_path))
    industry_intervals = load_industry_intervals(Path(industry_map_path))
    if industry_intervals:
        print(f"loaded industry map ({len(industry_intervals)} instruments)")
    positions = recorder.load_object(POSITIONS_ARTIFACT)
    _, days = build_day_frames(
        positions, name_map=name_map, industry_intervals=industry_intervals
    )
    metrics = collect_metrics(recorder)
    yearly_excess = collect_yearly_excess(recorder)

    dates = [d["date"] for d in days]
    payload = {
        "exp_name": exp_name,
        "recorder_id": str(rid),
        "start": dates[0] if dates else "",
        "end": dates[-1] if dates else "",
        "first_account_value": days[0]["account_value"] if days else None,
        "last_account_value": days[-1]["account_value"] if days else None,
        "metrics": metrics,
        "yearly_excess": yearly_excess,
        "days": days,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = exp_name.replace("/", "_")
    json_path = out_dir / f"{safe_name}_positions.json"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"wrote {json_path} ({len(days)} days)")
    print(
        "view: start console server, then open "
        f"/positions_view.html?data=/data/{json_path.name}"
    )
    if yearly_excess.get("concentration_warning"):
        print(f"WARNING: {yearly_excess['concentration_warning']}")
    return json_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Export positions JSON for shared day browser")
    parser.add_argument("--exp_name", default=DEFAULT_EXP_NAME, help="MLflow experiment name")
    parser.add_argument("--recorder_id", default=None, help="Optional recorder id; default=latest FINISHED")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--name_map",
        type=Path,
        default=NAME_MAP_PATH,
        help=f"Stock name map JSON (default: {NAME_MAP_PATH})",
    )
    parser.add_argument(
        "--provider_uri",
        default="~/.qlib/qlib_data/cn_data",
        help="qlib data provider uri",
    )
    parser.add_argument(
        "--industry_map",
        type=Path,
        default=DEFAULT_INDUSTRY_MAP,
        help=f"SW L1 industry intervals parquet (default: {DEFAULT_INDUSTRY_MAP})",
    )
    args = parser.parse_args(argv)
    export(
        exp_name=args.exp_name,
        recorder_id=args.recorder_id,
        out_dir=args.out_dir,
        provider_uri=args.provider_uri,
        name_map_path=args.name_map,
        industry_map_path=args.industry_map,
    )


if __name__ == "__main__":
    main()
