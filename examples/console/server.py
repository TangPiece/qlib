#!/usr/bin/env python3
"""公用模型训练控制台（stdlib only）。

服务于 examples/benchmarks 下各模型：滚动训练 / 官方 qrun / 数据更新 / 导出。

Usage (from qlib repo root)::

    .venv/bin/python examples/console/server.py
    # open http://127.0.0.1:8765
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from urllib.parse import unquote, urlparse

CONSOLE_DIR = Path(__file__).absolute().resolve().parent  # examples/console
EXAMPLES_DIR = CONSOLE_DIR.parent
REPO_ROOT = EXAMPLES_DIR.parent
BENCHMARKS_DIR = EXAMPLES_DIR / "benchmarks"
LGBM_DIR = BENCHMARKS_DIR / "LightGBM"
DATA_DIR = CONSOLE_DIR / "data" / "runs"
# 历史结果收藏（按导出文件名），放在 data/ 下与 runs 并列
FAVORITES_PATH = CONSOLE_DIR / "data" / "favorites.json"
# Train jobs run with cwd=repo root, so qlib's default file store lands here.
MLRUNS_DIR = REPO_ROOT / "mlruns"
JOBS_DIR = CONSOLE_DIR / ".jobs"
DEFAULT_QLIB_DIR = Path("~/.qlib/qlib_data/cn_data").expanduser()
HOST = "127.0.0.1"
PORT = 8765

# macOS 上 torch/sklearn/lightgbm 各带一份 libomp，多线程易 SIGSEGV；训练统一单线程 + 允许重复加载
_SAFE_OMP_ENV: Dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "KMP_DUPLICATE_LIB_OK": "TRUE",
}


def _rolling_entry(
    model_dir_name: str,
    *,
    dataset: str,
    yaml_names: Dict[str, str],
    use_omp: bool = True,
    note: str = "",
    default_horizon: int = 1,
    label: Optional[str] = None,
    exp_prefixes: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """按统一命名规则构造一条滚动注册项。

    model_id = {modelkey}_rolling
    exp_prefix = {ModelDir}_{dataset}_rolling_7y1y
    config key = 7y1y / 7y1y_{suffix}

    exp_prefixes 只覆盖个别 config key；未列出的 key 用模型默认前缀。
    """
    model_dir = BENCHMARKS_DIR / model_dir_name
    default_prefix = f"{model_dir_name}_{dataset}_rolling_7y1y"
    overrides = exp_prefixes or {}
    return {
        "label": label or f"{model_dir_name} 滚动",
        "script": model_dir / "rolling_7y1y.py",
        "exp_prefix": default_prefix,
        "exp_prefixes": {key: overrides.get(key, default_prefix) for key in yaml_names},
        "default_horizon": default_horizon,
        "configs": {key: model_dir / fname for key, fname in yaml_names.items()},
        "env": dict(_SAFE_OMP_ENV) if use_omp else {},
        "note": note,
    }


# 滚动训练注册表：命名见 _rolling_entry；model_id = {modelkey}_rolling
ROLLING_MODELS: Dict[str, Dict[str, Any]] = {
    "lightgbm_rolling": _rolling_entry(
        "LightGBM",
        dataset="Alpha158",
        yaml_names={
            "7y1y": "workflow_config_lightgbm_Alpha158_rolling_7y1y.yaml",
            "7y1y_10000w": "workflow_config_lightgbm_Alpha158_rolling_7y1y_10000w.yaml",
        },
        use_omp=True,
    ),
    "adarnn_rolling": _rolling_entry(
        "ADARNN",
        dataset="Alpha360",
        yaml_names={
            "7y1y": "workflow_config_adarnn_Alpha360_rolling_7y1y.yaml",
            "7y1y_10w": "workflow_config_adarnn_Alpha360_rolling_7y1y_10w.yaml",
        },
    ),
    "catboost_rolling": _rolling_entry(
        "CatBoost",
        dataset="Alpha158",
        yaml_names={"7y1y": "workflow_config_catboost_Alpha158_rolling_7y1y.yaml"},
    ),
    "xgboost_rolling": _rolling_entry(
        "XGBoost",
        dataset="Alpha158",
        yaml_names={"7y1y": "workflow_config_xgboost_Alpha158_rolling_7y1y.yaml"},
    ),
    "doubleensemble_rolling": _rolling_entry(
        "DoubleEnsemble",
        dataset="Alpha158",
        yaml_names={
            "7y1y": "workflow_config_doubleensemble_Alpha158_rolling_7y1y.yaml",
            # Alpha360：10 万 / 1 亿两套资金，实验名前缀与 Alpha158 分开
            "7y1y_alpha360_10w": "workflow_config_doubleensemble_Alpha360_rolling_7y1y_10w.yaml",
            "7y1y_alpha360_10000w": "workflow_config_doubleensemble_Alpha360_rolling_7y1y_10000w.yaml",
        },
        exp_prefixes={
            "7y1y_alpha360_10w": "DoubleEnsemble_Alpha360_rolling_7y1y_10w",
            "7y1y_alpha360_10000w": "DoubleEnsemble_Alpha360_rolling_7y1y_10000w",
        },
    ),
    "linear_rolling": _rolling_entry(
        "Linear",
        dataset="Alpha158",
        yaml_names={"7y1y": "workflow_config_linear_Alpha158_rolling_7y1y.yaml"},
    ),
    "mlp_rolling": _rolling_entry(
        "MLP",
        dataset="Alpha158",
        yaml_names={"7y1y": "workflow_config_mlp_Alpha158_rolling_7y1y.yaml"},
    ),
    "tabnet_rolling": _rolling_entry(
        "TabNet",
        dataset="Alpha158",
        yaml_names={"7y1y": "workflow_config_TabNet_Alpha158_rolling_7y1y.yaml"},
    ),
    "tft_rolling": _rolling_entry(
        "TFT",
        dataset="Alpha158",
        yaml_names={"7y1y": "workflow_config_tft_Alpha158_rolling_7y1y.yaml"},
        note="官方说明需要 GPU，且历史仅支持 Python 3.6–3.7。",
    ),
    "add_rolling": _rolling_entry(
        "ADD",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_add_Alpha360_rolling_7y1y.yaml"},
    ),
    "alstm_rolling": _rolling_entry(
        "ALSTM",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_alstm_Alpha360_rolling_7y1y.yaml"},
    ),
    "gats_rolling": _rolling_entry(
        "GATs",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_gats_Alpha360_rolling_7y1y.yaml"},
    ),
    "gru_rolling": _rolling_entry(
        "GRU",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_gru_Alpha360_rolling_7y1y.yaml"},
    ),
    "lstm_rolling": _rolling_entry(
        "LSTM",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_lstm_Alpha360_rolling_7y1y.yaml"},
    ),
    "localformer_rolling": _rolling_entry(
        "Localformer",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_localformer_Alpha360_rolling_7y1y.yaml"},
    ),
    "tcn_rolling": _rolling_entry(
        "TCN",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_tcn_Alpha360_rolling_7y1y.yaml"},
    ),
    "transformer_rolling": _rolling_entry(
        "Transformer",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_transformer_Alpha360_rolling_7y1y.yaml"},
    ),
    "krnn_rolling": _rolling_entry(
        "KRNN",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_krnn_Alpha360_rolling_7y1y.yaml"},
    ),
    "sfm_rolling": _rolling_entry(
        "SFM",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_sfm_Alpha360_rolling_7y1y.yaml"},
    ),
    "sandwich_rolling": _rolling_entry(
        "Sandwich",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_sandwich_Alpha360_rolling_7y1y.yaml"},
    ),
    "hist_rolling": _rolling_entry(
        "HIST",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_hist_Alpha360_rolling_7y1y.yaml"},
        note="需 stock2concept / stock_index 等外部 npy 与预训练权重。",
    ),
    "igmtf_rolling": _rolling_entry(
        "IGMTF",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_igmtf_Alpha360_rolling_7y1y.yaml"},
        note="需预训练 LSTM pickle（model_path）。",
    ),
    "tcts_rolling": _rolling_entry(
        "TCTS",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_tcts_Alpha360_rolling_7y1y.yaml"},
        note="多 label；请保持 horizon 与 yaml 一致，勿随意改 horizon。",
    ),
    "tra_rolling": _rolling_entry(
        "TRA",
        dataset="Alpha360",
        yaml_names={"7y1y": "workflow_config_tra_Alpha360_rolling_7y1y.yaml"},
        note="使用 MTSDatasetH / router；需按 TRA README 准备数据。",
    ),
    "generalptnn_rolling": _rolling_entry(
        "GeneralPtNN",
        dataset="gru",
        yaml_names={"7y1y": "workflow_config_generalptnn_gru_rolling_7y1y.yaml"},
        note="基于 GeneralPtNN GRU 配置。",
    ),
}
DEFAULT_ROLLING_MODEL_ID = "lightgbm_rolling"
# LightGBM 目录里的本地扩展，不作为「官方单次 workflow」展示。
_LGBM_LOCAL_YAML_MARKERS = ("rolling_", "multi_freq", "configurable_dataset")
# 扫描模型目录时跳过控制台自身与共用滚动包
_SKIP_MODEL_DIRS = frozenset({"console", "__pycache__", "_rolling"})
_MODEL_NOTES = {
    "TFT": "官方说明需要 GPU，且仅支持 Python 3.6–3.7，仍允许执行。",
}
_DATASET_TOKENS = ("early_stop", "Alpha360", "Alpha158", "L2Daily", "csi500", "multi_pass", "full")

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_META_NAME_RE = re.compile(r"(?m)^name:\s*(.*?)\s*$")
_RESULT_JSON_SUFFIX = "_positions.json"
_RESULT_HTML_SUFFIX = "_positions_view.html"
# Companion experiment created by Rolling._train_rolling_tasks.
_ROLLING_PREFIX = "rolling_models_"
_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_active_job_id: Optional[str] = None
# History delete touches the export file and mlruns together.
_results_lock = threading.Lock()


def _load_update_module():
    path = LGBM_DIR / "update_cn_data_daily.py"
    spec = importlib.util.spec_from_file_location("lgbm_update_cn_data_daily", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_update_mod = _load_update_module()


def _json_response(handler: SimpleHTTPRequestHandler, code: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: SimpleHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def _day_market(body: Dict[str, Any]) -> Dict[str, Any]:
    """把 {keys:[{instrument,date}]} 交给报价脚本，返回复权价、现价和行业。"""
    raw_keys = body.get("keys") or []
    if not isinstance(raw_keys, list):
        raise ValueError("keys must be a list")
    if len(raw_keys) > 800:
        raise ValueError("too many keys")
    keys = []
    seen = set()
    for item in raw_keys:
        if not isinstance(item, dict):
            continue
        code = str(item.get("instrument") or "").strip().upper()
        day = str(item.get("date") or "").strip()
        if not _DATE_RE.match(day) or not re.fullmatch(r"[A-Z0-9]{4,12}", code):
            continue
        pair = (code, day)
        if pair in seen:
            continue
        seen.add(pair)
        keys.append({"instrument": code, "date": day})
    if not keys:
        return {"quotes": []}
    proc = subprocess.run(
        [sys.executable, str(LGBM_DIR / "quote_live_day.py")],
        input=json.dumps({"keys": keys}),
        text=True,
        capture_output=True,
        cwd=str(REPO_ROOT),
        timeout=180,
    )
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-8:]
        raise RuntimeError("day_market failed: " + " | ".join(tail))
    line = ""
    for row in (proc.stdout or "").splitlines():
        row = row.strip()
        if row.startswith("{"):
            line = row
    if not line:
        raise RuntimeError("day_market returned no JSON")
    payload = json.loads(line)
    if not isinstance(payload, dict):
        raise RuntimeError("day_market JSON must be an object")
    return payload


def _model_class_from_yaml(text: str) -> Optional[str]:
    """取 task.model 下第一行 class，兼容 2 空格与 4 空格缩进。"""
    in_task = False
    model_indent: Optional[int] = None
    for line in text.splitlines():
        if re.match(r"^task:\s*$", line):
            in_task = True
            model_indent = None
            continue
        if not in_task:
            continue
        if model_indent is None:
            matched = re.match(r"^( +)model:\s*$", line)
            if matched:
                model_indent = len(matched.group(1))
                continue
            if line and not line.startswith((" ", "#")):
                in_task = False
            continue
        matched = re.match(r"^( +)class:\s*(\S+)\s*$", line)
        if matched and len(matched.group(1)) > model_indent:
            return matched.group(2)
        if line.strip() and not line.lstrip().startswith("#"):
            indent = len(line) - len(line.lstrip(" "))
            if indent <= model_indent:
                return None
    return None


def _is_qrun_yaml(text: str) -> bool:
    """官方 workflow 需同时具备 qlib_init 与 task.model / task.dataset，才能交给 qrun。"""
    return (
        re.search(r"(?m)^qlib_init:\s*$", text) is not None
        and re.search(r"(?m)^task:\s*$", text) is not None
        and re.search(r"(?m)^[ \t]+model:\s*$", text) is not None
        and re.search(r"(?m)^[ \t]+dataset:\s*$", text) is not None
    )


def _dataset_label(filename: str) -> str:
    """从文件名抽出数据集标记，供默认实验名使用。"""
    stem = Path(filename).stem
    found = [token for token in _DATASET_TOKENS if token in stem]
    if found:
        return "_".join(found)
    if stem.startswith("workflow_config_"):
        return stem[len("workflow_config_") :]
    return stem


def _skip_official_yaml(model_name: str, filename: str) -> bool:
    """滚动模板 / 本机 local / LightGBM 扩展不进官方 qrun 列表。"""
    stem = Path(filename).stem
    # 统一滚动命名：*_rolling_* 只出现在 ROLLING_MODELS
    if "_rolling_" in stem or stem.endswith("_rolling"):
        return True
    # 任意模型的 *_local*.yaml（含 local_10w）留给本机专用入口
    if "_local" in stem:
        return True
    if model_name != "LightGBM":
        return False
    return any(marker in filename for marker in _LGBM_LOCAL_YAML_MARKERS)


def _rolling_model_entry(model_id: str) -> Dict[str, Any]:
    """把注册表项转成 /api/models 用的前端结构。"""
    spec = ROLLING_MODELS[model_id]
    return {
        "id": model_id,
        "label": spec["label"],
        "kind": "rolling",
        "editable": True,
        "note": spec.get("note") or "",
        "exp_prefix": spec["exp_prefix"],
        "default_horizon": spec["default_horizon"],
        "yamls": [],
    }


def _list_official_models() -> List[Dict[str, Any]]:
    """扫描 benchmarks 各模型目录顶层 workflow_config_*.yaml。

    先返回滚动注册表项，再返回每个可 qrun 的官方模型及其 YAML。
    不读取 TRA/configs 等子目录。
    """
    models: List[Dict[str, Any]] = [_rolling_model_entry(mid) for mid in ROLLING_MODELS]
    if not BENCHMARKS_DIR.is_dir():
        return models

    official: List[Dict[str, Any]] = []
    for model_dir in sorted(BENCHMARKS_DIR.iterdir(), key=lambda p: p.name.lower()):
        if not model_dir.is_dir() or model_dir.name.startswith(".") or model_dir.name in _SKIP_MODEL_DIRS:
            continue
        yamls: List[Dict[str, Any]] = []
        for path in sorted(model_dir.glob("workflow_config_*.yaml")):
            if not path.is_file() or _skip_official_yaml(model_dir.name, path.name):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if not _is_qrun_yaml(text):
                continue
            model_class = _model_class_from_yaml(text) or ""
            yamls.append(
                {
                    "name": path.name,
                    "label": f"{path.name} · {model_class}" if model_class else path.name,
                    "dataset": _dataset_label(path.name),
                    "model_class": model_class,
                    "path": str(path.relative_to(REPO_ROOT)),
                }
            )
        if not yamls:
            continue
        official.append(
            {
                "id": model_dir.name,
                "label": model_dir.name,
                "kind": "qrun",
                "editable": False,
                "note": _MODEL_NOTES.get(model_dir.name, ""),
                "yamls": yamls,
            }
        )
    models.extend(official)
    return models


def _official_model(model_id: str) -> Dict[str, Any]:
    """按目录名取官方模型；调用方传入的 id 必须来自扫描结果。"""
    for model in _list_official_models():
        if model["id"] == model_id and model["kind"] == "qrun":
            return model
    raise ValueError(f"unknown model: {model_id}")


def _resolve_official_yaml(model_id: str, yaml_name: str) -> Path:
    """只接受该模型扫描列表中的文件名，避免任意路径。"""
    name = (yaml_name or "").strip()
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError("yaml must be a workflow filename from the model list")
    model = _official_model(model_id)
    match = next((item for item in model["yamls"] if item["name"] == name), None)
    if match is None:
        raise ValueError(f"yaml {name} is not listed for {model_id}")
    path = (REPO_ROOT / match["path"]).resolve()
    path.relative_to(BENCHMARKS_DIR.resolve())
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return path


def _default_qrun_exp_name(model_id: str, yaml_name: str) -> str:
    """官方单次训练的默认实验名：模型_数据集_日期。"""
    dataset = _dataset_label(yaml_name)
    stamp = datetime.now().strftime("%Y%m%d")
    return f"{model_id}_{dataset}_{stamp}"


def _rolling_configs(model_id: Optional[str] = None) -> Dict[str, Path]:
    """当前滚动模型的 config key -> path；未知 id 回退到默认 LightGBM 滚动。"""
    mid = (model_id or "").strip() or DEFAULT_ROLLING_MODEL_ID
    if mid not in ROLLING_MODELS:
        mid = DEFAULT_ROLLING_MODEL_ID
    return dict(ROLLING_MODELS[mid]["configs"])


def _resolve_config(key_or_path: Optional[str], model_id: Optional[str] = None) -> Path:
    """解析滚动 YAML：优先按当前滚动模型的 configs key，否则允许 benchmarks 下相对路径。"""
    configs = _rolling_configs(model_id)
    if not key_or_path:
        return next(iter(configs.values()))
    if key_or_path in configs:
        return configs[key_or_path]
    # 也允许用任意已注册滚动模型的 key（写入结束日时前端可能只传 key）
    for spec in ROLLING_MODELS.values():
        if key_or_path in spec["configs"]:
            return spec["configs"][key_or_path]
    path = Path(key_or_path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    try:
        path.relative_to(BENCHMARKS_DIR.resolve())
    except ValueError as exc:
        raise ValueError(f"config must be under {BENCHMARKS_DIR}") from exc
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return path


def _parse_yaml_ends(config_path: Path) -> Dict[str, Optional[str]]:
    """解析 data_handler / backtest / segments.test 结束日。

    兼容 2/4 空格缩进、可选引号日期，以及 ``test: [a, b]`` / 多行 list。
    """
    text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    out: Dict[str, Optional[str]] = {
        "data_handler_end": None,
        "backtest_end": None,
        "test_end": None,
    }
    date = r"[\"']?(\d{4}-\d{2}-\d{2})[\"']?"
    m = re.search(
        rf"(?m)^data_handler_config:[^\n]*\n(?:[^\n]*\n)*?[ \t]+end_time:\s*{date}",
        text,
    )
    if m:
        out["data_handler_end"] = m.group(1)
    m = re.search(
        rf"(?m)^[ \t]+backtest:\s*\n(?:[^\n]*\n)*?[ \t]+end_time:\s*{date}",
        text,
    )
    if m:
        out["backtest_end"] = m.group(1)
    m = re.search(rf"(?m)^[ \t]*test:\s*\[[^\],]+,\s*{date}\s*\]", text)
    if m:
        out["test_end"] = m.group(1)
    else:
        # ruamel 多行: test:\n        - '2016-01-01'\n        - '2026-10-16'
        m = re.search(
            rf"(?m)^[ \t]*test:\s*\n[ \t]*-[ \t]*{date}\s*\n[ \t]*-[ \t]*{date}",
            text,
        )
        if m:
            out["test_end"] = m.group(2)
    return out


def _calendar_bounds() -> Dict[str, Optional[str]]:
    try:
        first, latest = _update_mod.read_calendar_bounds(DEFAULT_QLIB_DIR)
        return {"first": first, "latest": latest}
    except Exception as exc:  # noqa: BLE001
        return {"first": None, "latest": None, "error": str(exc)}


def _job_snapshot(job_id: str, log_tail: int = 200) -> Dict[str, Any]:
    with _jobs_lock:
        job = dict(_jobs.get(job_id) or {})
    if not job:
        return {"id": job_id, "error": "not found"}
    log_path = Path(job.get("log_path") or "")
    log_text = ""
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        log_text = "\n".join(lines[-log_tail:])
    return {
        "id": job["id"],
        "kind": job.get("kind"),
        "status": job.get("status"),
        "command": job.get("command"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "exit_code": job.get("exit_code"),
        "error": job.get("error"),
        "log": log_text,
    }


def _set_active(job_id: Optional[str]) -> None:
    global _active_job_id
    with _jobs_lock:
        _active_job_id = job_id


def _get_active() -> Optional[str]:
    with _jobs_lock:
        return _active_job_id


def _start_job(kind: str, command: List[str], env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """启动后台任务；env 会合并进当前进程环境。

    训练、预测、写入持仓会加载 torch/lightgbm，自动注入 _SAFE_OMP_ENV，避免 macOS 多份 libomp 导致 SIGSEGV。
    """
    if _get_active():
        raise RuntimeError("another job is already running")

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    log_path = JOBS_DIR / f"{job_id}.log"
    job = {
        "id": job_id,
        "kind": kind,
        "status": "running",
        "command": command,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "exit_code": None,
        "error": None,
        "log_path": str(log_path),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _set_active(job_id)
    # 先铺 OpenMP 防护，再叠调用方 env（可覆盖单项）
    run_env = {
        **os.environ,
        **(_SAFE_OMP_ENV if kind in ("train", "predict", "apply_book") else {}),
        **(env or {}),
    }

    def _runner() -> None:
        try:
            with log_path.open("w", encoding="utf-8") as logf:
                logf.write("$ " + " ".join(command) + "\n\n")
                logf.flush()
                proc = subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=run_env,
                )
                code = proc.wait()
            with _jobs_lock:
                _jobs[job_id]["exit_code"] = code
                _jobs[job_id]["status"] = "finished" if code == 0 else "failed"
                _jobs[job_id]["finished_at"] = datetime.now().isoformat(timespec="seconds")
                if code != 0:
                    _jobs[job_id]["error"] = f"exit code {code}"
        except Exception as exc:  # noqa: BLE001
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(exc)
                _jobs[job_id]["finished_at"] = datetime.now().isoformat(timespec="seconds")
            try:
                with log_path.open("a", encoding="utf-8") as logf:
                    logf.write(f"\n[console] job failed: {exc}\n")
            except OSError:
                pass
        finally:
            _set_active(None)

    threading.Thread(target=_runner, daemon=True).start()
    return _job_snapshot(job_id)


def _is_safe_result_filename(name: str) -> bool:
    """拒绝路径穿越，且必须是导出结果文件名。"""
    if not name or "/" in name or "\\" in name or ".." in name or name != Path(name).name:
        return False
    return name.endswith(_RESULT_JSON_SUFFIX) or name.endswith(_RESULT_HTML_SUFFIX)


def _load_favorites() -> Set[str]:
    """读取 favorites.json；丢弃非法或不存在的文件名。"""
    path = FAVORITES_PATH
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return set()
    names = payload.get("filenames") if isinstance(payload, dict) else None
    if not isinstance(names, list):
        return set()
    data_root = DATA_DIR.resolve()
    out: Set[str] = set()
    for raw in names:
        name = str(raw or "").strip()
        if not _is_safe_result_filename(name):
            continue
        try:
            target = (DATA_DIR / name).resolve()
            target.relative_to(data_root)
        except (OSError, ValueError):
            continue
        if target.is_file():
            out.add(name)
    return out


def _save_favorites(filenames: Set[str]) -> None:
    """写入收藏列表（仅保留合法文件名）。"""
    cleaned = sorted(n for n in filenames if _is_safe_result_filename(n))
    FAVORITES_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"filenames": cleaned}
    FAVORITES_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _set_favorite(filename: str, favorited: bool) -> Dict[str, Any]:
    """切换一条历史结果的收藏状态；文件必须仍存在。"""
    name = str(filename or "").strip()
    if not _is_safe_result_filename(name):
        raise ValueError("invalid filename")
    target = _resolve_result_file(name)
    with _results_lock:
        favs = _load_favorites()
        if favorited:
            favs.add(target.name)
        else:
            favs.discard(target.name)
        _save_favorites(favs)
    return {"ok": True, "filename": target.name, "favorited": favorited}


def _list_results() -> List[Dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    favorites = _load_favorites()
    rows: List[Dict[str, Any]] = []

    for path in DATA_DIR.glob("*_positions.json"):
        st = path.stat()
        name = path.name
        exp_name = name[: -len("_positions.json")] if name.endswith("_positions.json") else name
        rows.append(
            {
                "filename": name,
                "exp_name": exp_name,
                "kind": "json",
                "url": f"/positions_view.html?data=/data/{name}",
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "mtime_ts": st.st_mtime,
                "size_bytes": st.st_size,
                "size_mb": round(st.st_size / (1024 * 1024), 2),
                "favorited": name in favorites,
            }
        )

    # Legacy self-contained HTML (read-only history; new exports write JSON only)
    for path in DATA_DIR.glob("*_positions_view.html"):
        st = path.stat()
        name = path.name
        exp_name = (
            name[: -len("_positions_view.html")] if name.endswith("_positions_view.html") else name
        )
        rows.append(
            {
                "filename": name,
                "exp_name": exp_name,
                "kind": "html_legacy",
                "url": f"/data/{name}",
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "mtime_ts": st.st_mtime,
                "size_bytes": st.st_size,
                "size_mb": round(st.st_size / (1024 * 1024), 2),
                "favorited": name in favorites,
            }
        )

    # 收藏优先，同组内按修改时间降序
    rows.sort(key=lambda r: (0 if r.get("favorited") else 1, -(r.get("mtime_ts") or 0)))
    for r in rows:
        r.pop("mtime_ts", None)
    return rows


def _result_exp_name(filename: str) -> str:
    """Experiment name encoded in an exported history filename."""
    if filename.endswith(_RESULT_JSON_SUFFIX):
        return filename[: -len(_RESULT_JSON_SUFFIX)]
    if filename.endswith(_RESULT_HTML_SUFFIX):
        return filename[: -len(_RESULT_HTML_SUFFIX)]
    raise ValueError("filename must be *_positions.json or *_positions_view.html")


def _resolve_result_file(filename: str) -> Path:
    """Resolve a history export under DATA_DIR, rejecting path traversal."""
    name = (filename or "").strip()
    if not name or "/" in name or "\\" in name or ".." in name:
        raise ValueError("invalid filename")
    exp_name = _result_exp_name(name)
    if not exp_name or "/" in exp_name or "\\" in exp_name or ".." in exp_name:
        raise ValueError("invalid filename")
    root = DATA_DIR.resolve()
    target = (DATA_DIR / name).resolve()
    target.relative_to(root)
    if not target.is_file():
        raise FileNotFoundError(f"result not found: {name}")
    return target


def _read_meta_name(exp_dir: Path) -> Optional[str]:
    """Read the experiment name from an MLflow file-store meta.yaml."""
    meta = exp_dir / "meta.yaml"
    if not meta.is_file():
        return None
    match = _META_NAME_RE.search(meta.read_text(encoding="utf-8", errors="replace"))
    if not match:
        return None
    raw = match.group(1).strip()
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
        raw = raw[1:-1]
    if raw in {"", "null", "None", "~"}:
        return None
    return raw


def _iter_experiment_dirs(mlruns: Path):
    """Yield (experiment_id, directory) for active experiments and .trash copies."""
    if not mlruns.is_dir():
        return
    for child in mlruns.iterdir():
        if not child.is_dir():
            continue
        if child.name == ".trash":
            for trashed in child.iterdir():
                if trashed.is_dir():
                    yield trashed.name, trashed
            continue
        yield child.name, child


def _rolling_models_linked(exp_dir: Path) -> Set[str]:
    """rolling_models_* names logged as param exp_name on this experiment's runs."""
    linked: Set[str] = set()
    if not exp_dir.is_dir():
        return linked
    for param in exp_dir.glob("*/params/exp_name"):
        if not param.is_file():
            continue
        value = param.read_text(encoding="utf-8", errors="replace").strip()
        if value.startswith(_ROLLING_PREFIX):
            linked.add(value)
    return linked


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _rm_under(path: Path, root: Path) -> None:
    """Delete a directory that must stay inside root. Never remove id 0 or root itself."""
    resolved = path.resolve()
    root_resolved = root.resolve()
    resolved.relative_to(root_resolved)
    trash = (root_resolved / ".trash").resolve()
    if resolved in {root_resolved, trash}:
        raise ValueError(f"refuse to delete {resolved}")
    # MLflow's default experiment is id 0; removing it breaks the file store.
    if resolved.name == "0" and resolved.parent in {root_resolved, trash}:
        raise ValueError("refuse to delete the default MLflow experiment")
    shutil.rmtree(resolved)


def _purge_named_experiments(mlruns: Path, names: Set[str]) -> List[str]:
    """Permanently remove experiment directories whose meta name is in names.

    Includes ``.trash`` copies. MLflow's delete_experiment only soft-deletes and
    then refuses to recreate the same name.
    """
    deleted: List[str] = []
    if not names or not mlruns.is_dir():
        return deleted
    root = mlruns.resolve()
    for exp_id, exp_dir in list(_iter_experiment_dirs(mlruns)):
        if exp_id == "0" or not exp_dir.exists():
            continue
        if _read_meta_name(exp_dir) not in names:
            continue
        rel = _display_path(exp_dir)
        _rm_under(exp_dir, root)
        deleted.append(rel)
    return deleted


def _delete_result(filename: str) -> Dict[str, Any]:
    """Delete one history row, its ensemble experiment, and unreferenced rolling models.

    The keep set is exp names still present in the history list after this file is
    gone. The ensemble experiment is removed only when no remaining row uses it.
    ``rolling_models_*`` directories stay only when a kept ensemble records that
    name in ``params/exp_name``. Other ensemble experiments that are not this row
    are left in place, including ones that were never exported.
    """
    with _results_lock:
        # Re-check under the lock so a second delete cannot purge after the file is gone.
        target = _resolve_result_file(filename)
        name = target.name
        exp_name = _result_exp_name(name)
        is_json = name.endswith(_RESULT_JSON_SUFFIX)
        # Read links before unlinking so a failed purge can be retried from the UI.
        keep = {row["exp_name"] for row in _list_results() if row["filename"] != name}
        by_name: Dict[str, List[Path]] = {}
        for exp_id, exp_dir in _iter_experiment_dirs(MLRUNS_DIR):
            if exp_id == "0":
                continue
            meta_name = _read_meta_name(exp_dir)
            if meta_name:
                by_name.setdefault(meta_name, []).append(exp_dir)

        referenced: Set[str] = set()
        for kept in keep:
            for exp_dir in by_name.get(kept, []):
                referenced |= _rolling_models_linked(exp_dir)

        deleted: List[str] = []
        if exp_name not in keep:
            deleted.extend(_purge_named_experiments(MLRUNS_DIR, {exp_name}))

        orphans = {
            meta_name
            for meta_name in by_name
            if meta_name.startswith(_ROLLING_PREFIX) and meta_name not in referenced
        }
        deleted.extend(_purge_named_experiments(MLRUNS_DIR, orphans))

        target.unlink()
        deleted.append(_display_path(target))

        # 同步从收藏列表移除已删文件
        favs = _load_favorites()
        if name in favs:
            favs.discard(name)
            _save_favorites(favs)

        if is_json:
            signal = (DATA_DIR / f"{exp_name}_live_signal.json").resolve()
            signal.relative_to(DATA_DIR.resolve())
            if signal.is_file():
                signal.unlink()
                deleted.append(_display_path(signal))

    return {"ok": True, "deleted": deleted}


def _cleanup_job_logs() -> List[str]:
    """删除 .jobs 下已结束任务的日志；跳过当前仍在运行的 job log。"""
    deleted_logs: List[str] = []
    if not JOBS_DIR.is_dir():
        return deleted_logs

    keep_path: Optional[Path] = None
    active_id = _get_active()
    if active_id:
        with _jobs_lock:
            job = _jobs.get(active_id) or {}
        log_path = job.get("log_path")
        if log_path:
            keep_path = Path(log_path).resolve()

    for path in sorted(JOBS_DIR.glob("*.log")):
        try:
            resolved = path.resolve()
            if keep_path is not None and resolved == keep_path:
                continue
            resolved.relative_to(JOBS_DIR.resolve())
            path.unlink()
            deleted_logs.append(_display_path(path))
        except (OSError, ValueError):
            continue
    return deleted_logs


def _cleanup_unused_mlruns() -> Dict[str, Any]:
    """Permanently delete MLflow experiments not referenced by the history list.

    Keeps every exp_name still shown in the history table, the ``rolling_models_*``
    experiments those ensembles link via ``params/exp_name``, and experiment id 0.
    Does not touch ``data/runs`` export files.
    Also removes finished job logs under ``.jobs/``.
    """
    with _results_lock:
        keep = {row["exp_name"] for row in _list_results()}
        by_name: Dict[str, List[Path]] = {}
        for exp_id, exp_dir in _iter_experiment_dirs(MLRUNS_DIR):
            if exp_id == "0":
                continue
            meta_name = _read_meta_name(exp_dir)
            if meta_name:
                by_name.setdefault(meta_name, []).append(exp_dir)

        referenced: Set[str] = set()
        for kept in keep:
            for exp_dir in by_name.get(kept, []):
                referenced |= _rolling_models_linked(exp_dir)

        retain = keep | referenced
        orphans = {meta_name for meta_name in by_name if meta_name not in retain}
        deleted = _purge_named_experiments(MLRUNS_DIR, orphans)
        kept_list = sorted(retain)

    deleted_logs = _cleanup_job_logs()
    return {"ok": True, "kept": kept_list, "deleted": deleted, "deleted_logs": deleted_logs}


def _status_payload(
    config_key: str = "7y1y",
    model: Optional[str] = None,
    yaml_name: Optional[str] = None,
) -> Dict[str, Any]:
    """状态条展示的结束日跟当前训练目标 YAML：滚动用 config，官方模型用 model+yaml。"""
    model_id = (model or "").strip() or DEFAULT_ROLLING_MODEL_ID
    if model_id in ROLLING_MODELS:
        configs = _rolling_configs(model_id)
        # 当前模型没有该 key 时回退到该模型第一项
        if config_key not in configs:
            config_key = next(iter(configs.keys()))
        config_path = _resolve_config(config_key, model_id)
        resolved_key = config_key
        yaml_out = None
        configs_out = {k: str(v.relative_to(REPO_ROOT)) for k, v in configs.items()}
        exp_prefixes_out = dict(ROLLING_MODELS[model_id].get("exp_prefixes") or {})
    else:
        config_path = _resolve_official_yaml(model_id, yaml_name or "")
        resolved_key = None
        yaml_out = Path(yaml_name).name if yaml_name else None
        configs_out = {
            k: str(v.relative_to(REPO_ROOT))
            for k, v in _rolling_configs(DEFAULT_ROLLING_MODEL_ID).items()
        }
        exp_prefixes_out = {}
    cal = _calendar_bounds()
    active = _get_active()
    return {
        "today": datetime.now().date().isoformat(),
        "calendar": cal,
        "config_key": resolved_key,
        "model": model_id,
        "yaml": yaml_out,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "yaml_ends": _parse_yaml_ends(config_path),
        "configs": configs_out,
        "exp_prefixes": exp_prefixes_out,
        "active_job_id": active,
        "active_job": _job_snapshot(active) if active else None,
        "python": sys.executable,
        "repo_root": str(REPO_ROOT),
    }


class ConsoleHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(CONSOLE_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/status":
            from urllib.parse import parse_qs

            qs = parse_qs(parsed.query)
            model = (qs.get("model") or [DEFAULT_ROLLING_MODEL_ID])[0]
            configs = _rolling_configs(model if model in ROLLING_MODELS else DEFAULT_ROLLING_MODEL_ID)
            default_key = next(iter(configs.keys()))
            config_key = (qs.get("config") or [default_key])[0]
            yaml_name = (qs.get("yaml") or [None])[0]
            try:
                _json_response(
                    self,
                    200,
                    _status_payload(config_key, model=model, yaml_name=yaml_name),
                )
            except Exception as exc:  # noqa: BLE001
                _json_response(self, 400, {"error": str(exc)})
            return

        if path == "/api/models":
            _json_response(self, 200, {"models": _list_official_models()})
            return

        if path == "/api/results":
            _json_response(self, 200, {"results": _list_results()})
            return

        if path.startswith("/api/jobs/"):
            job_id = path[len("/api/jobs/") :].strip("/")
            snap = _job_snapshot(job_id)
            code = 404 if snap.get("error") == "not found" else 200
            _json_response(self, code, snap)
            return

        if path == "/" or path == "/index.html":
            return self._serve_file(CONSOLE_DIR / "index.html", "text/html; charset=utf-8")

        if path == "/positions_view.html":
            return self._serve_file(CONSOLE_DIR / "positions_view.html", "text/html; charset=utf-8")

        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            target = (CONSOLE_DIR / rel).resolve()
            try:
                target.relative_to(CONSOLE_DIR)
            except ValueError:
                self.send_error(403)
                return
            return self._serve_file(target)

        if path.startswith("/data/"):
            rel = path[len("/data/") :]
            target = (DATA_DIR / rel).resolve()
            try:
                target.relative_to(DATA_DIR)
            except ValueError:
                self.send_error(403)
                return
            return self._serve_file(target)

        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            body = _read_json(self)
        except Exception as exc:  # noqa: BLE001
            _json_response(self, 400, {"error": f"invalid JSON: {exc}"})
            return

        try:
            if path == "/api/set-end-date":
                date = str(body.get("date") or "").strip()
                if not _DATE_RE.match(date):
                    raise ValueError("date must be YYYY-MM-DD")
                model_id = str(body.get("model") or DEFAULT_ROLLING_MODEL_ID).strip()
                config_path = _resolve_config(body.get("config"), model_id)
                changes = _update_mod.update_config_end_times(config_path, date, dry_run=False)
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "date": date,
                        "config": str(config_path.relative_to(REPO_ROOT)),
                        "changes": changes,
                    },
                )
                return

            if path == "/api/update-data":
                cmd = [sys.executable, str(LGBM_DIR / "update_cn_data_daily.py")]
                if body.get("skip_download"):
                    cmd.append("--skip-download")
                if body.get("force"):
                    cmd.append("--force")
                configs = body.get("configs")
                model_id = str(body.get("model") or DEFAULT_ROLLING_MODEL_ID).strip()
                if isinstance(configs, list) and configs:
                    cmd.append("--config")
                    for c in configs:
                        cmd.append(str(_resolve_config(str(c), model_id)))
                job = _start_job("update-data", cmd)
                _json_response(self, 200, {"ok": True, "job": job})
                return

            if path == "/api/train":
                model = str(body.get("model") or DEFAULT_ROLLING_MODEL_ID).strip()
                exp_name = str(body.get("exp_name") or "").strip()
                if "/" in exp_name or "\\" in exp_name or ".." in exp_name:
                    raise ValueError("invalid exp_name")
                if model in ROLLING_MODELS:
                    # 滚动：各模型脚本 + 对应 YAML
                    spec = ROLLING_MODELS[model]
                    config_key = body.get("config") or body.get("conf_path")
                    config_path = _resolve_config(config_key, model)
                    if not exp_name:
                        stamp = datetime.now().strftime("%Y%m%d")
                        # 未传实验名时跟当前 config key 的前缀，避免 Alpha360 写成 Alpha158
                        prefixes = spec.get("exp_prefixes") or {}
                        prefix = prefixes.get(config_key) or spec["exp_prefix"]
                        exp_name = f"{prefix}_{stamp}"
                    horizon = int(body.get("horizon") or spec["default_horizon"])
                    conf_rel = config_path.relative_to(REPO_ROOT)
                    cmd = [
                        sys.executable,
                        str(spec["script"]),
                        f"--conf_path={conf_rel}",
                        f"--exp_name={exp_name}",
                        f"--horizon={horizon}",
                        "run",
                    ]
                    job = _start_job("train", cmd, env=dict(spec.get("env") or {}))
                else:
                    # 官方单次：不改 YAML，交给 qrun
                    yaml_name = str(body.get("yaml") or "").strip()
                    config_path = _resolve_official_yaml(model, yaml_name)
                    if not exp_name:
                        exp_name = _default_qrun_exp_name(model, yaml_name)
                    conf_rel = config_path.relative_to(REPO_ROOT)
                    cmd = [
                        sys.executable,
                        "-m",
                        "qlib.cli.run",
                        str(conf_rel),
                        f"--experiment_name={exp_name}",
                    ]
                    # OpenMP 防护由 _start_job(kind=train) 统一注入
                    job = _start_job("train", cmd)
                _json_response(self, 200, {"ok": True, "job": job, "exp_name": exp_name, "model": model})
                return

            if path == "/api/export":
                exp_name = str(body.get("exp_name") or "").strip()
                if not exp_name:
                    raise ValueError("exp_name is required")
                cmd = [
                    sys.executable,
                    str(LGBM_DIR / "export_positions_view.py"),
                    f"--exp_name={exp_name}",
                ]
                job = _start_job("export", cmd)
                _json_response(self, 200, {"ok": True, "job": job, "exp_name": exp_name})
                return

            if path == "/api/results/delete":
                try:
                    payload = _delete_result(str(body.get("filename") or ""))
                except FileNotFoundError as exc:
                    _json_response(self, 404, {"error": str(exc)})
                    return
                _json_response(self, 200, payload)
                return

            if path == "/api/results/favorite":
                # body: {filename, favorited: true|false}
                payload = _set_favorite(
                    str(body.get("filename") or ""),
                    bool(body.get("favorited")),
                )
                _json_response(self, 200, payload)
                return

            if path == "/api/mlruns/cleanup":
                payload = _cleanup_unused_mlruns()
                _json_response(self, 200, payload)
                return

            if path == "/api/predict":
                # 脚本内校验：本地已有持仓 tip 行情；只预测 tip 的下一交易日
                exp_name = str(body.get("exp_name") or "").strip()
                if not exp_name:
                    raise ValueError("exp_name is required")
                cmd = [
                    sys.executable,
                    str(LGBM_DIR / "predict_live_signal.py"),
                    f"--exp_name={exp_name}",
                ]
                recorder_id = str(body.get("recorder_id") or "").strip()
                if recorder_id:
                    cmd.append(f"--recorder_id={recorder_id}")
                if body.get("skip_update"):
                    cmd.append("--skip_update")
                # 控制台默认只出信号；显式 no_roll=false 才立即滚动
                if body.get("no_roll", True):
                    cmd.append("--no_roll")
                if body.get("reset_book"):
                    cmd.append("--reset_book")
                if body.get("topk") is not None:
                    cmd.append(f"--topk={int(body['topk'])}")
                if body.get("n_drop") is not None:
                    cmd.append(f"--n_drop={int(body['n_drop'])}")
                job = _start_job("predict", cmd)
                safe = exp_name.replace("/", "_")
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "job": job,
                        "exp_name": exp_name,
                        "signal_url": f"/data/{safe}_live_signal.json",
                    },
                )
                return

            if path == "/api/day_market":
                # 账本日补价：子进程查复权收盘 / 现价 / 行业，本进程不 import qlib
                _json_response(self, 200, _day_market(body))
                return

            if path == "/api/apply_book":
                # 按现有 live_signal 写入 live_book（脚本内校验：本地末日≥建议执行日）
                exp_name = str(body.get("exp_name") or "").strip()
                if not exp_name:
                    raise ValueError("exp_name is required")
                cmd = [
                    sys.executable,
                    str(LGBM_DIR / "predict_live_signal.py"),
                    f"--exp_name={exp_name}",
                    "--apply_book",
                ]
                job = _start_job("apply_book", cmd)
                safe = exp_name.replace("/", "_")
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "job": job,
                        "exp_name": exp_name,
                        "signal_url": f"/data/{safe}_live_signal.json",
                    },
                )
                return

            _json_response(self, 404, {"error": f"unknown endpoint {path}"})
        except RuntimeError as exc:
            _json_response(self, 409, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            _json_response(self, 400, {"error": str(exc)})

    def _serve_file(self, path: Path, content_type: Optional[str] = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404, f"File not found: {path.name}")
            return
        data = path.read_bytes()
        if content_type is None:
            if path.suffix == ".html":
                content_type = "text/html; charset=utf-8"
            elif path.suffix == ".js":
                content_type = "application/javascript; charset=utf-8"
            elif path.suffix == ".css":
                content_type = "text/css; charset=utf-8"
            elif path.suffix == ".json":
                content_type = "application/json; charset=utf-8"
            else:
                content_type = "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), ConsoleHandler)
    print(f"Model training console: http://{HOST}:{PORT}", flush=True)
    print(f"repo root: {REPO_ROOT}", flush=True)
    print(f"python:    {sys.executable}", flush=True)
    print("Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
