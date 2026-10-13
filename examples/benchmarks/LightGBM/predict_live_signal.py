#!/usr/bin/env python3
"""Produce next-session TopkDropout live signals from an online-tagged recorder.

Uses OnlineToolR (not full OnlineManager): tag the latest trainable window as
online, optionally extend ``pred.pkl`` to the local calendar end, then apply the
same TopkDropout rule as the rolling yaml (topk=10, n_drop=1).

Output: ``data/{exp_name}_live_signal.json``

Example (repo root)::

    .venv/bin/python examples/benchmarks/LightGBM/predict_live_signal.py \\
      --exp_name=rolling_lgb_7y1y_20261013_1013
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.workflow import R
from qlib.workflow.online.utils import OnlineToolR

DIRNAME = Path(__file__).absolute().resolve().parent
if str(DIRNAME) not in sys.path:
    sys.path.insert(0, str(DIRNAME))

from export_positions_view import (  # noqa: E402
    NAME_MAP_PATH,
    POSITIONS_ARTIFACT,
    _latest_recorder,
    load_or_build_name_map,
)

DEFAULT_OUT_DIR = DIRNAME / "data"
DEFAULT_TOPK = 10
DEFAULT_N_DROP = 1
DEFAULT_HOLD_THRESH = 1


def _safe_name(exp_name: str) -> str:
    return exp_name.replace("/", "_")


def _recorder_has(rec, name: str) -> bool:
    try:
        rec.load_object(name)
        return True
    except Exception:  # noqa: BLE001 — probe artifact presence
        return False


def _find_model_recorder(exp_name: str, recorder_id: Optional[str] = None):
    """Prefer a FINISHED recorder that has params + pred (trainable window)."""
    if recorder_id:
        return R.get_recorder(recorder_id=recorder_id, experiment_name=exp_name)

    exp = R.get_exp(experiment_name=exp_name, create=False)
    recorders = exp.list_recorders(rtype="list")
    if not recorders:
        raise RuntimeError(f"No recorders found for experiment {exp_name!r}")

    finished = [r for r in recorders if getattr(r, "status", None) == "FINISHED"]
    candidates = finished or list(recorders)
    with_model: List[Any] = []
    for rec in candidates:
        if _recorder_has(rec, "params.pkl") and _recorder_has(rec, "pred.pkl"):
            with_model.append(rec)
    pool = with_model or candidates
    pool.sort(key=lambda r: getattr(r, "start_time", None) or "", reverse=True)
    return pool[0]


def _pred_to_series(pred: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(pred, pd.DataFrame):
        return pred.iloc[:, 0]
    return pred


def _scores_on_date(pred: pd.DataFrame | pd.Series, signal_date: pd.Timestamp) -> pd.Series:
    s = _pred_to_series(pred)
    if not isinstance(s.index, pd.MultiIndex):
        raise RuntimeError("pred.pkl index must be MultiIndex(datetime, instrument)")
    day = s.xs(signal_date, level="datetime")
    day = day.dropna()
    if day.empty:
        raise RuntimeError(f"No prediction scores on signal_date={signal_date.date()}")
    return day


def _latest_signal_date(pred: pd.DataFrame | pd.Series) -> pd.Timestamp:
    s = _pred_to_series(pred).dropna()
    if s.empty:
        raise RuntimeError("pred.pkl is empty after dropna")
    return pd.Timestamp(s.index.get_level_values("datetime").max()).normalize()


def _next_trading_day(signal_date: pd.Timestamp) -> Optional[str]:
    # future=True so we can suggest an execute date even when signal_date is calendar tip
    cal = D.calendar(start_time=signal_date, end_time=None, freq="day", future=True)
    if cal is None or len(cal) == 0:
        return None
    sig = pd.Timestamp(signal_date).normalize()
    for d in cal:
        ts = pd.Timestamp(d).normalize()
        if ts > sig:
            return ts.strftime("%Y-%m-%d")
    return None


def _pick_day_payload(days: List[dict], asof: Optional[pd.Timestamp]) -> Optional[dict]:
    """Prefer holdings on signal day (backtest snapshot that day); else last day <= asof."""
    if not days:
        return None
    if asof is None:
        return days[-1]
    asof = pd.Timestamp(asof).normalize()
    best = None
    for d in days:
        dt = pd.Timestamp(d.get("date")).normalize()
        if dt == asof:
            return d
        if dt <= asof:
            best = d
    return best or days[-1]


def _holdings_from_day(day: dict) -> Tuple[str, List[str], Dict[str, int]]:
    codes = [str(x["instrument"]) for x in (day.get("hold") or []) if x.get("instrument")]
    count_day: Dict[str, int] = {}
    for x in day.get("hold") or []:
        code = x.get("instrument")
        if code is None:
            continue
        cd = x.get("count_day")
        if cd is not None:
            count_day[str(code)] = int(cd)
    return str(day.get("date") or ""), codes, count_day


def _load_holdings_from_json(
    path: Path, asof: Optional[pd.Timestamp] = None
) -> Tuple[Optional[str], List[str], Dict[str, int]]:
    if not path.exists():
        return None, [], {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    days = payload.get("days") or []
    day = _pick_day_payload(days, asof)
    if not day:
        return None, [], {}
    return _holdings_from_day(day)


def _load_holdings_from_recorder(
    exp_name: str, asof: Optional[pd.Timestamp] = None
) -> Tuple[Optional[str], List[str], Dict[str, int]]:
    try:
        rec = _latest_recorder(exp_name)
        positions = rec.load_object(POSITIONS_ARTIFACT)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"cannot load positions from recorder: {exc}")
        return None, [], {}
    if not positions:
        return None, [], {}
    dates = sorted(positions.keys())
    if asof is not None:
        asof = pd.Timestamp(asof).normalize()
        chosen = None
        for d in dates:
            ts = pd.Timestamp(d).normalize()
            if ts == asof:
                chosen = d
                break
            if ts <= asof:
                chosen = d
        last_dt = chosen if chosen is not None else dates[-1]
    else:
        last_dt = dates[-1]
    pos = positions[last_dt]
    codes = list(pos.get_stock_list())
    count_day: Dict[str, int] = {}
    for code in codes:
        info = pos.position.get(code) or {}
        if info.get("count_day") is not None:
            count_day[code] = int(info["count_day"])
    return pd.Timestamp(last_dt).strftime("%Y-%m-%d"), codes, count_day


def topk_dropout_actions(
    pred_score: pd.Series,
    current_holdings: Sequence[str],
    *,
    topk: int = DEFAULT_TOPK,
    n_drop: int = DEFAULT_N_DROP,
    hold_thresh: int = DEFAULT_HOLD_THRESH,
    count_day: Optional[Dict[str, int]] = None,
) -> Tuple[List[str], List[str], List[str]]:
    """Mirror TopkDropoutStrategy (method_buy=top, method_sell=bottom) without exchange checks."""
    count_day = count_day or {}
    score = pred_score.astype(float).dropna()
    held = [c for c in current_holdings if c in score.index]
    last = score.reindex(held).sort_values(ascending=False).index
    n_buy_candi = n_drop + topk - len(last)
    today = list(score[~score.index.isin(last)].sort_values(ascending=False).index)
    if n_buy_candi > 0:
        today = today[:n_buy_candi]
    else:
        today = []
    comb = score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index
    sell_candi = last[last.isin(list(comb)[-n_drop:])] if n_drop > 0 else pd.Index([])
    sells: List[str] = []
    for code in sell_candi:
        if count_day.get(code, hold_thresh) < hold_thresh:
            continue
        sells.append(str(code))
    buys = [str(c) for c in today[: len(sells) + topk - len(last)]]
    holds = [str(c) for c in held if c not in sells]
    return buys, sells, holds


def _item(code: str, score: Optional[float], name_map: Dict[str, str]) -> Dict[str, Any]:
    return {
        "instrument": code,
        "name": name_map.get(code) or "",
        "score": None if score is None or pd.isna(score) else float(score),
    }


def run(
    exp_name: str,
    recorder_id: Optional[str] = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    provider_uri: str = "~/.qlib/qlib_data/cn_data",
    name_map_path: Path = NAME_MAP_PATH,
    topk: int = DEFAULT_TOPK,
    n_drop: int = DEFAULT_N_DROP,
    hold_thresh: int = DEFAULT_HOLD_THRESH,
    skip_update: bool = False,
) -> Path:
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(exp_name)

    model_rec = _find_model_recorder(exp_name, recorder_id)
    rid = model_rec.info.get("id") if hasattr(model_rec, "info") else getattr(model_rec, "id", None)
    print(f"using experiment={exp_name!r} recorder={rid}")

    tool = OnlineToolR(exp_name)
    tool.reset_online_tag(model_rec, exp_name=exp_name)
    updated = False
    if not skip_update:
        try:
            tool.update_online_pred(exp_name=exp_name)
            updated = True
            print("update_online_pred finished")
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"update_online_pred failed ({exc}); using existing pred.pkl")

    pred = model_rec.load_object("pred.pkl")
    signal_ts = _latest_signal_date(pred)
    scores = _scores_on_date(pred, signal_ts)
    signal_date = signal_ts.strftime("%Y-%m-%d")
    execute_date = _next_trading_day(signal_ts)

    # Holdings for TopkDropout should be the book *before* execute_date trades:
    # use signal_date snapshot (same day the score is from), not a later backtest day.
    pos_json = out_dir / f"{safe}_positions.json"
    hold_date, holdings, count_day = _load_holdings_from_json(pos_json, asof=signal_ts)
    if not holdings:
        hold_date, holdings, count_day = _load_holdings_from_recorder(exp_name, asof=signal_ts)

    buys, sells, holds = topk_dropout_actions(
        scores,
        holdings,
        topk=topk,
        n_drop=n_drop,
        hold_thresh=hold_thresh,
        count_day=count_day,
    )

    name_map = load_or_build_name_map(Path(name_map_path))
    score_map = {str(i): float(v) for i, v in scores.items()}

    note = (
        "信号日分数用于下一交易日执行（对齐 TopkDropout shift=1）。"
        "本结果不是回测成交复盘；勿把 positions 页上的成交日买卖当成刚生成的实盘单。"
    )
    if execute_date is None:
        note += " 日历上暂无信号日之后的交易日，execute_date 为空。"

    payload = {
        "exp_name": exp_name,
        "recorder_id": str(rid),
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "signal_date": signal_date,
        "execute_date": execute_date,
        "holdings_asof": hold_date,
        "pred_updated": updated,
        "topk": topk,
        "n_drop": n_drop,
        "hold_thresh": hold_thresh,
        "n_buy": len(buys),
        "n_sell": len(sells),
        "n_hold": len(holds),
        "buys": [_item(c, score_map.get(c), name_map) for c in buys],
        "sells": [_item(c, score_map.get(c), name_map) for c in sells],
        "holds": [_item(c, score_map.get(c), name_map) for c in holds],
        "note": note,
    }

    out_path = out_dir / f"{safe}_live_signal.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        f"signal_date={signal_date} execute_date={execute_date} "
        f"buy={len(buys)} sell={len(sells)} hold={len(holds)}"
    )
    return out_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="OnlineToolR live TopkDropout signal export")
    parser.add_argument("--exp_name", required=True, help="MLflow experiment name")
    parser.add_argument("--recorder_id", default=None, help="Optional recorder id")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name_map", type=Path, default=NAME_MAP_PATH)
    parser.add_argument("--provider_uri", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--n_drop", type=int, default=DEFAULT_N_DROP)
    parser.add_argument("--hold_thresh", type=int, default=DEFAULT_HOLD_THRESH)
    parser.add_argument(
        "--skip_update",
        action="store_true",
        help="Do not call update_online_pred; only use existing pred.pkl",
    )
    args = parser.parse_args(argv)
    run(
        exp_name=args.exp_name,
        recorder_id=args.recorder_id,
        out_dir=args.out_dir,
        provider_uri=args.provider_uri,
        name_map_path=args.name_map,
        topk=args.topk,
        n_drop=args.n_drop,
        hold_thresh=args.hold_thresh,
        skip_update=args.skip_update,
    )


if __name__ == "__main__":
    main()
