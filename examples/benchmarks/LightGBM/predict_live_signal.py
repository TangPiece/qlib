#!/usr/bin/env python3
"""Produce next-session TopkDropout live signals from an online-tagged recorder.

Rolling ensemble experiments (``rolling_lgb_*``) only store concatenated pred;
trainable windows live in the companion ``rolling_models_*`` experiment (logged
as param ``exp_name`` on the ensemble recorder). This script resolves that
window, uses OnlineToolR to extend ``pred.pkl`` to the local calendar end, then
applies the same TopkDropout rule as the training yaml (``topk`` / ``n_drop``
from ``port_analysis_config.strategy.kwargs``, falling back to 10 / 1).

Output:
  - ``examples/console/data/runs/{exp_name}_live_signal.json``
  - ``examples/console/data/runs/{exp_name}_live_book.json`` (滚动持仓账本)

Example (repo root)::

    .venv/bin/python examples/benchmarks/LightGBM/predict_live_signal.py \\
      --exp_name=rolling_lgb_7y1y_20261013
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D
from qlib.workflow import R
from qlib.workflow.online.utils import OnlineToolR
from ruamel.yaml import YAML

_TZ_SH = ZoneInfo("Asia/Shanghai")

DIRNAME = Path(__file__).absolute().resolve().parent
REPO_ROOT = DIRNAME.parents[2]
if str(DIRNAME) not in sys.path:
    sys.path.insert(0, str(DIRNAME))

from export_positions_view import (  # noqa: E402
    NAME_MAP_PATH,
    POSITIONS_ARTIFACT,
    _latest_recorder,
    load_or_build_name_map,
)

# 与 export_positions_view 一致，写入 examples/console/data/runs
DEFAULT_OUT_DIR = DIRNAME.parents[1] / "console" / "data" / "runs"
DEFAULT_TOPK = 10
DEFAULT_N_DROP = 1
DEFAULT_HOLD_THRESH = 1

# 持仓来源标记，写入 live_signal 供页面展示
SOURCE_LIVE_BOOK = "live_book"
SOURCE_POSITIONS_JSON = "positions_json"
SOURCE_RECORDER = "recorder"

_CONF_PATH_RE = re.compile(r"--conf_path(?:=|\s+)(\S+)")


def _safe_name(exp_name: str) -> str:
    return exp_name.replace("/", "_")


def _recorder_has(rec, name: str) -> bool:
    try:
        rec.load_object(name)
        return True
    except Exception:  # noqa: BLE001 — probe artifact presence
        return False


def _recorder_id(rec) -> str:
    if hasattr(rec, "info") and isinstance(rec.info, dict):
        return str(rec.info.get("id") or "")
    return str(getattr(rec, "id", "") or "")


def _get_ensemble_recorder(exp_name: str, recorder_id: Optional[str] = None):
    if recorder_id:
        return R.get_recorder(recorder_id=recorder_id, experiment_name=exp_name)
    return _latest_recorder(exp_name)


def _rolling_models_exp_name(ensemble_rec) -> Optional[str]:
    """Companion rolling_models_* experiment name logged by Rolling._ens_rolling."""
    try:
        params = ensemble_rec.list_params() or {}
    except Exception:  # noqa: BLE001
        return None
    name = params.get("exp_name")
    if not name:
        return None
    name = str(name).strip()
    if name.startswith("rolling_models_"):
        return name
    return None


def _conf_raw_from_argv(argv_text: str) -> Optional[Path]:
    """从 cmd-sys.argv 取出 --conf_path 原文（文件可以已不存在）。"""
    m = _CONF_PATH_RE.search(argv_text or "")
    if not m:
        return None
    raw = m.group(1).strip().strip("'\"")
    if not raw:
        return None
    return Path(raw)


def _existing_conf_file(path: Path) -> Optional[Path]:
    """在 cwd / 仓库根 / LightGBM 目录下解析仍存在的 yaml。"""
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.extend(
            [
                Path.cwd() / path,
                REPO_ROOT / path,
                DIRNAME / path.name,
            ]
        )
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return None


def _fallback_rolling_yaml(recorded: Path) -> Optional[Path]:
    """记录的 conf 已删除时，在同目录找滚动 yaml。

    优先 ``workflow_config_*rolling_7y1y.yaml``，排除文件名含 ``10w`` 的资金规模变体
    （那份通常是另一套 topk）。多个命中时取文件名最短的一份。
    """
    parents = []
    if recorded.is_absolute():
        parents.append(recorded.parent)
    else:
        parents.append((REPO_ROOT / recorded).parent)
        parents.append((Path.cwd() / recorded).parent)
    hits: List[Path] = []
    seen = set()
    for parent in parents:
        try:
            key = parent.resolve() if parent.exists() else parent
        except OSError:
            continue
        if key in seen or not parent.is_dir():
            continue
        seen.add(key)
        for p in parent.glob("workflow_config_*rolling_7y1y.yaml"):
            if "10w" in p.name:
                continue
            hits.append(p.resolve())
    if not hits:
        return None
    hits.sort(key=lambda p: (len(p.name), p.name))
    return hits[0]


def _conf_path_from_argv(argv_text: str) -> Optional[Path]:
    """Parse --conf_path from a recorded cmd-sys.argv string.

    原文件不存在时回落到同目录滚动 yaml（见 ``_fallback_rolling_yaml``）。
    """
    raw = _conf_raw_from_argv(argv_text)
    if raw is None:
        return None
    found = _existing_conf_file(raw)
    if found is not None:
        return found
    fallback = _fallback_rolling_yaml(raw)
    if fallback is not None:
        print(f"training yaml missing ({raw}); fallback {fallback}")
        return fallback
    return None


def _strategy_kwargs_from_yaml(conf_path: Path) -> Dict[str, Any]:
    yaml = YAML(typ="safe", pure=True)
    with conf_path.open("r", encoding="utf-8") as f:
        conf = yaml.load(f) or {}
    port = conf.get("port_analysis_config") or {}
    strategy = port.get("strategy") or {}
    kwargs = strategy.get("kwargs") or {}
    return dict(kwargs) if isinstance(kwargs, dict) else {}


def _strategy_params_from_recorder(ensemble_rec) -> Tuple[Optional[int], Optional[int]]:
    """Return (topk, n_drop) from training yaml linked via cmd-sys.argv; else (None, None)."""
    try:
        params = ensemble_rec.list_params() or {}
    except Exception:  # noqa: BLE001
        return None, None
    argv_text = params.get("cmd-sys.argv")
    if not argv_text:
        return None, None
    conf_path = _conf_path_from_argv(str(argv_text))
    if conf_path is None:
        return None, None
    try:
        kwargs = _strategy_kwargs_from_yaml(conf_path)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"cannot read strategy kwargs from {conf_path}: {exc}")
        return None, None
    topk = kwargs.get("topk")
    n_drop = kwargs.get("n_drop")
    try:
        topk_i = int(topk) if topk is not None else None
    except (TypeError, ValueError):
        topk_i = None
    try:
        n_drop_i = int(n_drop) if n_drop is not None else None
    except (TypeError, ValueError):
        n_drop_i = None
    return topk_i, n_drop_i


def _resolve_topk_n_drop(
    ensemble_rec,
    topk: Optional[int],
    n_drop: Optional[int],
) -> Tuple[int, int]:
    """CLI 优先；否则用训练 yaml。yaml 缺失时抛错，避免静默退回 10/1。"""
    yaml_topk, yaml_n_drop = (None, None)
    if topk is None or n_drop is None:
        yaml_topk, yaml_n_drop = _strategy_params_from_recorder(ensemble_rec)
        missing = []
        if topk is None and yaml_topk is None:
            missing.append("topk")
        if n_drop is None and yaml_n_drop is None:
            missing.append("n_drop")
        if missing:
            raise RuntimeError(
                "无法解析 "
                + "/".join(missing)
                + "：训练 yaml 不存在或未写策略参数。请恢复 conf，或显式传入 --topk/--n_drop。"
            )
    resolved_topk = int(topk) if topk is not None else int(yaml_topk)
    resolved_n_drop = int(n_drop) if n_drop is not None else int(yaml_n_drop)
    if topk is None or n_drop is None:
        print(f"strategy params: topk={resolved_topk} n_drop={resolved_n_drop} (from training yaml)")
    return resolved_topk, resolved_n_drop


def _test_segment_end(rec) -> Any:
    """Return test-segment end (None = open-ended); 'missing' if task absent."""
    try:
        task = rec.load_object("task")
        test = task["dataset"]["kwargs"]["segments"]["test"]
        if isinstance(test, (list, tuple)) and len(test) >= 2:
            return test[1]
        return "missing"
    except Exception:  # noqa: BLE001
        return "missing"


def _find_trainable_window(models_exp: str):
    """Pick latest window with params + dataset + pred; prefer open-ended test."""
    exp = R.get_exp(experiment_name=models_exp, create=False)
    recorders = exp.list_recorders(rtype="list")
    if not recorders:
        raise RuntimeError(f"No recorders found for model experiment {models_exp!r}")

    finished = [r for r in recorders if getattr(r, "status", None) == "FINISHED"]
    pool = finished or list(recorders)
    trainable: List[Tuple[Any, Any]] = []
    for rec in pool:
        if not (
            _recorder_has(rec, "params.pkl")
            and _recorder_has(rec, "dataset")
            and _recorder_has(rec, "pred.pkl")
        ):
            continue
        trainable.append((rec, _test_segment_end(rec)))

    if not trainable:
        raise RuntimeError(
            f"No trainable window (params.pkl + dataset + pred.pkl) in {models_exp!r}"
        )

    open_ended = [r for r, end in trainable if end is None]
    if open_ended:
        open_ended.sort(key=lambda r: str(getattr(r, "start_time", None) or ""), reverse=True)
        return open_ended[0]

    dated = [(r, end) for r, end in trainable if end not in (None, "missing")]
    if dated:
        dated.sort(key=lambda x: pd.Timestamp(x[1]), reverse=True)
        return dated[0][0]

    trainable.sort(key=lambda x: str(getattr(x[0], "start_time", None) or ""), reverse=True)
    return trainable[0][0]


def _find_model_recorder_in_exp(exp_name: str):
    """Fallback when ensemble itself stores a trainable recorder (non-rolling)."""
    exp = R.get_exp(experiment_name=exp_name, create=False)
    recorders = exp.list_recorders(rtype="list")
    if not recorders:
        raise RuntimeError(f"No recorders found for experiment {exp_name!r}")

    finished = [r for r in recorders if getattr(r, "status", None) == "FINISHED"]
    candidates = finished or list(recorders)
    with_model: List[Any] = []
    for rec in candidates:
        if (
            _recorder_has(rec, "params.pkl")
            and _recorder_has(rec, "dataset")
            and _recorder_has(rec, "pred.pkl")
        ):
            with_model.append(rec)
    if not with_model:
        raise RuntimeError(
            f"Experiment {exp_name!r} has no trainable recorder and no rolling_models_* link"
        )
    with_model.sort(key=lambda r: getattr(r, "start_time", None) or "", reverse=True)
    return with_model[0]


def _resolve_model_source(
    exp_name: str, recorder_id: Optional[str] = None
) -> Tuple[str, Any, Any]:
    """Return (models_exp_name, model_recorder, ensemble_recorder)."""
    ensemble_rec = _get_ensemble_recorder(exp_name, recorder_id)
    models_exp = _rolling_models_exp_name(ensemble_rec)
    if models_exp:
        model_rec = _find_trainable_window(models_exp)
        return models_exp, model_rec, ensemble_rec
    # Same-exp trainable model (non-rolling / already has params)
    if _recorder_has(ensemble_rec, "params.pkl") and _recorder_has(ensemble_rec, "dataset"):
        return exp_name, ensemble_rec, ensemble_rec
    model_rec = _find_model_recorder_in_exp(exp_name)
    return exp_name, model_rec, ensemble_rec


def _pred_to_series(pred: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(pred, pd.DataFrame):
        return pred.iloc[:, 0]
    return pred


def _scores_on_date(pred: pd.DataFrame | pd.Series, signal_date: pd.Timestamp) -> pd.Series:
    s = _pred_to_series(pred)
    if not isinstance(s.index, pd.MultiIndex):
        raise RuntimeError("pred.pkl index must be MultiIndex(datetime, instrument)")
    try:
        day = s.xs(signal_date, level="datetime")
    except KeyError as exc:
        # 缺该日分数时 xs 抛 KeyError，统一成 RuntimeError 以便写入友好错误
        raise RuntimeError(f"No prediction scores on signal_date={signal_date.date()}") from exc
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
    """Prefer holdings on signal day; else last day <= asof。

    若没有任何 date <= asof，返回 None（禁止回退到未来日，避免把执行后仓当信号日仓）。
    """
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
    return best


def _holdings_from_day(
    day: dict,
) -> Tuple[str, List[str], Dict[str, int], Dict[str, str]]:
    """从单日 payload 解析代码、count_day、buy_date。"""
    codes = [str(x["instrument"]) for x in (day.get("hold") or []) if x.get("instrument")]
    count_day: Dict[str, int] = {}
    buy_dates: Dict[str, str] = {}
    hold_date = str(day.get("date") or "")
    for x in day.get("hold") or []:
        code = x.get("instrument")
        if code is None:
            continue
        c = str(code)
        cd = x.get("count_day")
        if cd is not None:
            count_day[c] = int(cd)
        bd = x.get("buy_date")
        if bd:
            buy_dates[c] = str(bd)
        elif hold_date and c not in buy_dates:
            # 旧 live_book / 缺字段时用快照日近似，避免丢买入日
            buy_dates[c] = hold_date
    return hold_date, codes, count_day, buy_dates


def _load_holdings_from_json(
    path: Path, asof: Optional[pd.Timestamp] = None
) -> Tuple[Optional[str], List[str], Dict[str, int], Dict[str, str]]:
    if not path.exists():
        return None, [], {}, {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    days = payload.get("days") or []
    day = _pick_day_payload(days, asof)
    if not day:
        return None, [], {}, {}
    return _holdings_from_day(day)


def _live_book_path(out_dir: Path, safe: str) -> Path:
    """滚动持仓账本路径：{exp}_live_book.json。"""
    return Path(out_dir) / f"{safe}_live_book.json"


def _live_signal_path(out_dir: Path, safe: str) -> Path:
    return Path(out_dir) / f"{safe}_live_signal.json"


def _shanghai_today() -> date:
    """Asia/Shanghai 自然日，用于预测与写入持仓解锁。"""
    return datetime.now(_TZ_SH).date()


def _local_calendar_has(day: str) -> bool:
    """本地行情日历是否已包含该日。future=False，未下载的交易日不算已有数据。"""
    ts = pd.Timestamp(day).normalize()
    cal = D.calendar(start_time=ts, end_time=ts, freq="day", future=False)
    if cal is None or len(cal) == 0:
        return False
    return any(pd.Timestamp(d).normalize() == ts for d in cal)


def _local_calendar_latest() -> Optional[str]:
    """本地行情日历末日（不含 future）。"""
    cal = D.calendar(start_time="2008-01-01", end_time=None, freq="day", future=False)
    if cal is None or len(cal) == 0:
        return None
    return pd.Timestamp(cal[-1]).normalize().strftime("%Y-%m-%d")


def _json_max_date(path: Path) -> Optional[str]:
    """读取 days[] 中最大 date；文件不存在或为空则 None。"""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None
    days = payload.get("days") or []
    dates = [str(d.get("date") or "") for d in days if d.get("date")]
    return max(dates) if dates else None


def _book_tip(out_dir: Path, safe: str) -> str:
    """当前持仓末日：live_book 优先，否则 positions.json。"""
    tip = _json_max_date(_live_book_path(out_dir, safe))
    if tip:
        return tip
    tip = _json_max_date(Path(out_dir) / f"{safe}_positions.json")
    if tip:
        return tip
    raise RuntimeError(
        "无法确定持仓 tip：缺少 live_book 与 positions.json，请先导出回测持仓。"
    )


def _codes_from_signal_items(items: Optional[List[dict]]) -> List[str]:
    out: List[str] = []
    for x in items or []:
        code = x.get("instrument") if isinstance(x, dict) else None
        if code:
            out.append(str(code))
    return out


def _reset_live_book(out_dir: Path, safe: str) -> bool:
    """删除 live_book，使下次预测从回测仓重开。返回是否删除了文件。"""
    path = _live_book_path(out_dir, safe)
    if path.exists():
        path.unlink()
        print(f"reset live_book: removed {path}")
        return True
    print("reset live_book: no existing file")
    return False


def _load_holdings_from_recorder(
    exp_name: str, asof: Optional[pd.Timestamp] = None
) -> Tuple[Optional[str], List[str], Dict[str, int], Dict[str, str]]:
    try:
        rec = _latest_recorder(exp_name)
        positions = rec.load_object(POSITIONS_ARTIFACT)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"cannot load positions from recorder: {exc}")
        return None, [], {}, {}
    if not positions:
        return None, [], {}, {}
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
        # 与 _pick_day_payload 一致：没有 <= asof 的日则放弃，勿用未来仓
        if chosen is None:
            return None, [], {}, {}
        last_dt = chosen
    else:
        last_dt = dates[-1]
    pos = positions[last_dt]
    codes = list(pos.get_stock_list())
    hold_date = pd.Timestamp(last_dt).strftime("%Y-%m-%d")
    count_day: Dict[str, int] = {}
    buy_dates: Dict[str, str] = {}
    for code in codes:
        info = pos.position.get(code) or {}
        if info.get("count_day") is not None:
            count_day[code] = int(info["count_day"])
        # recorder 无独立 buy_date 字段时用快照日近似
        buy_dates[code] = hold_date
    return hold_date, codes, count_day, buy_dates


def _resolve_holdings(
    *,
    out_dir: Path,
    safe: str,
    exp_name: str,
    asof: pd.Timestamp,
) -> Tuple[Optional[str], List[str], Dict[str, int], Dict[str, str], str]:
    """按优先级取持仓：live_book → positions.json → recorder。

    返回 (hold_date, codes, count_day, buy_dates, holdings_source)。
    仅使用 date <= asof 的快照，避免把执行后仓当作信号日输入。
    """
    book_path = _live_book_path(out_dir, safe)
    hold_date, holdings, count_day, buy_dates = _load_holdings_from_json(book_path, asof=asof)
    if holdings:
        return hold_date, holdings, count_day, buy_dates, SOURCE_LIVE_BOOK

    pos_json = Path(out_dir) / f"{safe}_positions.json"
    hold_date, holdings, count_day, buy_dates = _load_holdings_from_json(pos_json, asof=asof)
    if holdings:
        return hold_date, holdings, count_day, buy_dates, SOURCE_POSITIONS_JSON

    hold_date, holdings, count_day, buy_dates = _load_holdings_from_recorder(exp_name, asof=asof)
    if holdings:
        return hold_date, holdings, count_day, buy_dates, SOURCE_RECORDER
    return None, [], {}, {}, SOURCE_POSITIONS_JSON


def _roll_holdings_after_actions(
    holds: Sequence[str],
    buys: Sequence[str],
    *,
    execute_date: str,
    count_day: Optional[Dict[str, int]] = None,
    buy_dates: Optional[Dict[str, str]] = None,
) -> Tuple[List[str], Dict[str, int], Dict[str, str]]:
    """执行日成交后滚动仓：保留股 count_day+1 并继承 buy_date；新买入 count_day=1、buy_date=执行日。"""
    prev_cd = count_day or {}
    prev_bd = buy_dates or {}
    new_count: Dict[str, int] = {}
    new_buy_dates: Dict[str, str] = {}
    # 先写持有，再追加买入（买入覆盖同码时以买入为准）
    for code in holds:
        c = str(code)
        new_count[c] = int(prev_cd.get(c, 0)) + 1
        new_buy_dates[c] = prev_bd.get(c) or execute_date
    for code in buys:
        c = str(code)
        new_count[c] = 1
        new_buy_dates[c] = execute_date
    return list(new_count.keys()), new_count, new_buy_dates


def _upsert_live_book_day(
    path: Path,
    *,
    exp_name: str,
    execute_date: str,
    signal_date: str,
    holdings: Sequence[str],
    count_day: Dict[str, int],
    buy_dates: Dict[str, str],
) -> None:
    """写入/覆盖 live_book 中 execute_date 对应条目（含显式 buy_date）。"""
    payload: Dict[str, Any] = {"exp_name": exp_name, "days": []}
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"corrupt live_book, recreating: {exc}")
            payload = {"exp_name": exp_name, "days": []}
    days: List[dict] = list(payload.get("days") or [])
    # 去掉同日旧条目，再追加新快照（同信号重跑可幂等覆盖）
    days = [d for d in days if str(d.get("date") or "") != execute_date]
    hold_rows = [
        {
            "instrument": str(c),
            "count_day": int(count_day.get(str(c), 1)),
            "buy_date": str(buy_dates.get(str(c)) or execute_date),
        }
        for c in holdings
    ]
    days.append(
        {
            "date": execute_date,
            "source": "predict_roll",
            "from_signal_date": signal_date,
            "hold": hold_rows,
        }
    )
    days.sort(key=lambda d: str(d.get("date") or ""))
    payload["exp_name"] = exp_name
    payload["updated_at"] = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    payload["days"] = days
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote live_book {path} date={execute_date} n_hold={len(holdings)}")

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


def apply_book(
    exp_name: str,
    out_dir: Path = DEFAULT_OUT_DIR,
    provider_uri: str = "~/.qlib/qlib_data/cn_data",
) -> Path:
    """按已有 live_signal 写入 live_book。

    条件：本地行情末日 >= 今天，且信号的执行日正好是今天（当前日已有预测）。
    """
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    out_dir = Path(out_dir)
    safe = _safe_name(exp_name)
    out_path = _live_signal_path(out_dir, safe)
    if not out_path.exists():
        raise RuntimeError(f"缺少信号文件: {out_path}，请先点「预测」")

    sig = json.loads(out_path.read_text(encoding="utf-8"))
    if sig.get("error"):
        raise RuntimeError(f"信号含错误，无法写入持仓: {sig.get('error')}")
    signal_date = sig.get("signal_date")
    execute_date = sig.get("execute_date")
    if not signal_date or not execute_date:
        raise RuntimeError("信号缺少 signal_date / execute_date")

    today = _shanghai_today()
    today_s = today.isoformat()
    cal_latest = _local_calendar_latest()
    # 本地最近交易日须已覆盖今天；且必须已有「执行日=今天」的预测信号
    if not cal_latest or cal_latest < today_s:
        raise RuntimeError(
            f"写入持仓须本地行情末日 >= 今天（本地末日 {cal_latest or '无'}，今天 {today_s}）"
        )
    if str(execute_date) != today_s:
        raise RuntimeError(
            f"写入持仓须当前日已有预测信号（信号执行日 {execute_date}，今天 {today_s}）"
        )

    # 若账本 tip 已 >= 执行日，视为已写入，禁止用旧信号再滚一次
    tip = _json_max_date(_live_book_path(out_dir, safe))
    if tip and tip >= str(execute_date):
        raise RuntimeError(
            f"live_book tip={tip} 已覆盖执行日 {execute_date}，无需重复写入"
        )

    buys = _codes_from_signal_items(sig.get("buys"))
    sells = _codes_from_signal_items(sig.get("sells"))
    holds = _codes_from_signal_items(sig.get("holds"))
    # holds 应已排除 sells；再保险一次
    sell_set = set(sells)
    holds = [c for c in holds if c not in sell_set]

    hold_date, _holdings, count_day, buy_dates, holdings_source = _resolve_holdings(
        out_dir=out_dir,
        safe=safe,
        exp_name=exp_name,
        asof=pd.Timestamp(signal_date),
    )
    rolled_codes, rolled_cd, rolled_bd = _roll_holdings_after_actions(
        holds,
        buys,
        execute_date=str(execute_date),
        count_day=count_day,
        buy_dates=buy_dates,
    )
    _upsert_live_book_day(
        _live_book_path(out_dir, safe),
        exp_name=exp_name,
        execute_date=str(execute_date),
        signal_date=str(signal_date),
        holdings=rolled_codes,
        count_day=rolled_cd,
        buy_dates=rolled_bd,
    )

    sig["book_rolled"] = True
    sig["book_tip"] = str(execute_date)
    sig["holdings_asof"] = hold_date
    sig["holdings_source"] = holdings_source
    note = str(sig.get("note") or "")
    applied = (
        f" 已于 {today.isoformat()} 按信号写入 live_book（执行日 {execute_date}）。"
    )
    if "已于" not in note:
        sig["note"] = note + applied
    out_path.write_text(json.dumps(sig, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"apply_book ok execute_date={execute_date} n_hold={len(rolled_codes)} "
        f"today={today.isoformat()}"
    )
    return out_path


def run(
    exp_name: str,
    recorder_id: Optional[str] = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    provider_uri: str = "~/.qlib/qlib_data/cn_data",
    name_map_path: Path = NAME_MAP_PATH,
    topk: Optional[int] = None,
    n_drop: Optional[int] = None,
    hold_thresh: int = DEFAULT_HOLD_THRESH,
    skip_update: bool = False,
    no_roll: bool = False,
    reset_book: bool = False,
) -> Path:
    """预测：仅针对持仓 tip 的下一交易日；默认可不写账本（no_roll）。"""
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe = _safe_name(exp_name)

    # 先于读仓重置，避免本轮仍读到旧 live_book
    if reset_book:
        _reset_live_book(out_dir, safe)

    book_tip = _book_tip(out_dir, safe)
    tip_has_data = _local_calendar_has(book_tip)
    signal_ts = pd.Timestamp(book_tip).normalize()
    signal_date = signal_ts.strftime("%Y-%m-%d")
    # 只预测 tip 的下一交易日，禁止跨日
    execute_date = _next_trading_day(signal_ts)
    # 本地须有 tip（信号日/上一交易日）数据；名单只针对 tip 的下一交易日
    if not tip_has_data:
        raise RuntimeError(
            f"本地行情尚无持仓 tip {book_tip}，请先更新行情后再预测下一交易日"
        )
    if not execute_date:
        raise RuntimeError(f"持仓 tip={book_tip} 之后无下一交易日，无法预测")

    models_exp, model_rec, ensemble_rec = _resolve_model_source(exp_name, recorder_id)
    model_rid = _recorder_id(model_rec)
    ens_rid = _recorder_id(ensemble_rec)
    topk, n_drop = _resolve_topk_n_drop(ensemble_rec, topk, n_drop)
    print(
        f"ensemble={exp_name!r} recorder={ens_rid}; "
        f"model_exp={models_exp!r} model_recorder={model_rid}; "
        f"book_tip={book_tip} → execute={execute_date}"
    )

    tool = OnlineToolR(models_exp)
    tool.reset_online_tag(model_rec, exp_name=models_exp)
    updated = False
    update_error: Optional[str] = None
    if not skip_update:
        try:
            tool.update_online_pred(exp_name=models_exp)
            updated = True
            print("update_online_pred finished")
        except Exception as exc:  # noqa: BLE001
            update_error = str(exc)
            print(f"update_online_pred failed: {update_error}", file=sys.stderr)

    out_path = _live_signal_path(out_dir, safe)

    if update_error:
        note = (
            "预测更新失败：未能把分数续算到本地日历末日。"
            "请检查 rolling_models 窗口是否含 params.pkl/dataset，以及本地行情是否完整。"
        )
        payload = {
            "exp_name": exp_name,
            "recorder_id": ens_rid,
            "model_exp_name": models_exp,
            "model_recorder_id": model_rid,
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "book_tip": book_tip,
            "signal_date": signal_date,
            "execute_date": execute_date,
            "holdings_asof": None,
            "holdings_source": None,
            "pred_updated": False,
            "book_rolled": False,
            "error": update_error,
            "topk": topk,
            "n_drop": n_drop,
            "hold_thresh": hold_thresh,
            "n_buy": 0,
            "n_sell": 0,
            "n_hold": 0,
            "buys": [],
            "sells": [],
            "holds": [],
            "note": note,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out_path} (error)")
        raise RuntimeError(f"update_online_pred failed: {update_error}")

    pred = model_rec.load_object("pred.pkl")
    # 必须用 tip 日分数，禁止跳到 pred 日历更远的日子
    try:
        scores = _scores_on_date(pred, signal_ts)
    except RuntimeError as exc:
        pred_tip = None
        try:
            pred_tip = _latest_signal_date(pred).strftime("%Y-%m-%d")
        except Exception:  # noqa: BLE001
            pass
        msg = (
            f"pred 尚无持仓 tip 日 {signal_date} 的分数"
            + (f"（pred 末日={pred_tip}）" if pred_tip else "")
            + "。请先更新行情后再预测下一交易日。"
        )
        payload = {
            "exp_name": exp_name,
            "recorder_id": ens_rid,
            "model_exp_name": models_exp,
            "model_recorder_id": model_rid,
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "book_tip": book_tip,
            "signal_date": signal_date,
            "execute_date": execute_date,
            "holdings_asof": None,
            "holdings_source": None,
            "pred_updated": updated,
            "book_rolled": False,
            "error": msg,
            "topk": topk,
            "n_drop": n_drop,
            "hold_thresh": hold_thresh,
            "n_buy": 0,
            "n_sell": 0,
            "n_hold": 0,
            "buys": [],
            "sells": [],
            "holds": [],
            "note": msg,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out_path} (error)")
        raise RuntimeError(msg) from exc

    hold_date, holdings, count_day, buy_dates, holdings_source = _resolve_holdings(
        out_dir=out_dir, safe=safe, exp_name=exp_name, asof=signal_ts
    )

    buys, sells, holds = topk_dropout_actions(
        scores,
        holdings,
        topk=topk,
        n_drop=n_drop,
        hold_thresh=hold_thresh,
        count_day=count_day,
    )

    book_rolled = False
    if not no_roll and execute_date:
        rolled_codes, rolled_cd, rolled_bd = _roll_holdings_after_actions(
            holds,
            buys,
            execute_date=execute_date,
            count_day=count_day,
            buy_dates=buy_dates,
        )
        _upsert_live_book_day(
            _live_book_path(out_dir, safe),
            exp_name=exp_name,
            execute_date=execute_date,
            signal_date=signal_date,
            holdings=rolled_codes,
            count_day=rolled_cd,
            buy_dates=rolled_bd,
        )
        book_rolled = True

    name_map = load_or_build_name_map(Path(name_map_path))
    score_map = {str(i): float(v) for i, v in scores.items()}

    note = (
        "信号日=持仓 tip，执行日=tip 下一交易日（对齐 TopkDropout shift=1）。"
        "本结果不是回测成交复盘。预测仅生成 tip 的下一交易日名单；写入持仓须本地行情末日>=今天且信号执行日=今天。"
        f" 分数来自模型实验 {models_exp}。"
        f" book_tip={book_tip}；持仓来源={holdings_source}（asof={hold_date or '—'}）。"
    )
    if book_rolled:
        note += f" 已按本轮买卖滚动写入 live_book（执行日 {execute_date}）。"
    elif no_roll:
        note += " （未写入 live_book。本地行情末日>=今天且信号执行日=今天后再点「写入持仓」。）"
    if skip_update:
        note += " （跳过了 update_online_pred，使用已有 pred.pkl。）"
    if reset_book:
        note += " （本轮已 --reset_book，从回测仓重开。）"

    payload = {
        "exp_name": exp_name,
        "recorder_id": ens_rid,
        "model_exp_name": models_exp,
        "model_recorder_id": model_rid,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "book_tip": book_tip,
        "signal_date": signal_date,
        "execute_date": execute_date,
        "holdings_asof": hold_date,
        "holdings_source": holdings_source,
        "pred_updated": updated,
        "book_rolled": book_rolled,
        "error": None,
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

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out_path}")
    print(
        f"book_tip={book_tip} signal_date={signal_date} execute_date={execute_date} "
        f"buy={len(buys)} sell={len(sells)} hold={len(holds)} "
        f"pred_updated={updated} holdings_source={holdings_source} book_rolled={book_rolled}"
    )
    return out_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="OnlineToolR live TopkDropout signal export")
    parser.add_argument("--exp_name", required=True, help="MLflow experiment name")
    parser.add_argument("--recorder_id", default=None, help="Optional recorder id")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--name_map", type=Path, default=NAME_MAP_PATH)
    parser.add_argument("--provider_uri", default="~/.qlib/qlib_data/cn_data")
    parser.add_argument(
        "--topk",
        type=int,
        default=None,
        help="TopkDropout topk; default: training yaml, else 10",
    )
    parser.add_argument(
        "--n_drop",
        type=int,
        default=None,
        help="TopkDropout n_drop; default: training yaml, else 1",
    )
    parser.add_argument("--hold_thresh", type=int, default=DEFAULT_HOLD_THRESH)
    parser.add_argument(
        "--skip_update",
        action="store_true",
        help="Do not call update_online_pred; only use existing pred.pkl",
    )
    parser.add_argument(
        "--no_roll",
        action="store_true",
        help="只出信号，不写入 live_book 滚动持仓",
    )
    parser.add_argument(
        "--reset_book",
        action="store_true",
        help="删除该实验 live_book 后从回测仓重开",
    )
    parser.add_argument(
        "--apply_book",
        action="store_true",
        help="按已有 live_signal 写入 live_book（须日历日>=执行日）",
    )
    args = parser.parse_args(argv)
    if args.apply_book:
        apply_book(
            exp_name=args.exp_name,
            out_dir=args.out_dir,
            provider_uri=args.provider_uri,
        )
        return
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
        no_roll=args.no_roll,
        reset_book=args.reset_book,
    )


if __name__ == "__main__":
    main()
