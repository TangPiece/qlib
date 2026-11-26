#!/usr/bin/env python3
"""扫描 benchmarks 官方 yaml，生成五套因子 × 20万/10/1 的单次训练配置。

用法（仓库根）::

    .venv/bin/python examples/benchmarks/_factor_sweep/gen_configs.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ruamel.yaml import YAML

DIR = Path(__file__).resolve().parent
BENCHMARKS = DIR.parent
GENERATED = DIR / "generated"
MANIFEST = DIR / "manifest.json"

# 因子键 → handler 类名；含 L2 的用短窗
FACTOR_HANDLERS = {
    "alpha158": "Alpha158",
    "alpha360": "Alpha360",
    "fund_cn": "FundCn",
    "l2": "L2Daily",
    "mix": "MixFour",
}
FACTOR_ORDER = ["alpha158", "alpha360", "fund_cn", "l2", "mix"]
L2_FACTORS = {"l2", "mix"}

# 无 L2：早起始；含 L2：2017+
EARLY = {
    "handler_start": "2008-01-01",
    "handler_end": "2026-11-11",
    "fit_start": "2008-01-01",
    "fit_end": "2014-12-31",
    "train": ["2008-01-01", "2014-12-31"],
    "valid": ["2015-01-01", "2016-12-31"],
    "test": ["2017-01-01", "2026-11-11"],
    "bt_start": "2017-01-01",
    "bt_end": "2026-11-11",
}
L2_WIN = {
    "handler_start": "2017-01-03",
    "handler_end": "2026-11-11",
    "fit_start": "2017-01-03",
    "fit_end": "2022-12-30",
    "train": ["2017-01-03", "2022-12-30"],
    "valid": ["2023-01-03", "2023-12-29"],
    "test": ["2024-01-02", "2026-11-11"],
    "bt_start": "2024-01-02",
    "bt_end": "2026-11-11",
}

# 先快后慢
PHASE1 = ["Linear", "LightGBM", "XGBoost", "CatBoost", "DoubleEnsemble"]
PHASE2 = ["MLP", "GeneralPtNN", "TabNet"]
PHASE3 = [
    "LSTM",
    "GRU",
    "ALSTM",
    "TCN",
    "Transformer",
    "Localformer",
    "GATs",
    "SFM",
    "KRNN",
    "Sandwich",
    "ADARNN",
    "ADD",
    "TCTS",
    "TRA",
]
# 缺环境/预训练时默认跳过（run_sweep 可再探测）
SKIP_MODELS = ["TFT", "HIST", "IGMTF"]

MODEL_PHASE = {}
for m in PHASE1:
    MODEL_PHASE[m] = 1
for m in PHASE2:
    MODEL_PHASE[m] = 2
for m in PHASE3:
    MODEL_PHASE[m] = 3
for m in SKIP_MODELS:
    MODEL_PHASE[m] = 0


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.default_flow_style = False
    return y


def _pick_template(model_dir: Path) -> Optional[Path]:
    """优先 Alpha158 官方单次 yaml，否则 Alpha360；跳过 rolling/本地扩展。"""
    cands_158 = sorted(model_dir.glob("workflow_config_*Alpha158.yaml"))
    cands_360 = sorted(model_dir.glob("workflow_config_*Alpha360.yaml"))
    # 排除 early_stop / csi500 / full 等变体，优先最短文件名
    def score(p: Path) -> Tuple[int, int, str]:
        name = p.name.lower()
        penalty = 0
        if "csi500" in name:
            penalty += 10
        if "early_stop" in name:
            penalty += 5
        if "full" in name:
            penalty += 3
        if "rolling" in name:
            penalty += 100
        return (penalty, len(name), name)

    for pool in (cands_158, cands_360):
        pool = [p for p in pool if "rolling" not in p.name.lower()]
        if pool:
            return sorted(pool, key=score)[0]
    # GeneralPtNN 等无 Alpha 后缀
    others = [
        p
        for p in model_dir.glob("workflow_config_*.yaml")
        if "rolling" not in p.name.lower()
        and "csi500" not in p.name.lower()
    ]
    if others:
        return sorted(others, key=score)[0]
    return None


def _strip_filter_col(processors: Any) -> Any:
    """非 Alpha158 时去掉绑死列名的 FilterCol。"""
    if not isinstance(processors, list):
        return processors
    out = []
    for p in processors:
        if isinstance(p, dict) and p.get("class") == "FilterCol":
            continue
        out.append(p)
    return out


def _patch_config(doc: Dict[str, Any], factor_key: str, handler_cls: str) -> Dict[str, Any]:
    """写入统一资金仓位、时间窗与 handler。"""
    cfg = copy.deepcopy(doc)
    win = L2_WIN if factor_key in L2_FACTORS else EARLY

    dh = cfg.get("data_handler_config")
    if isinstance(dh, dict):
        dh["start_time"] = win["handler_start"]
        dh["end_time"] = win["handler_end"]
        dh["fit_start_time"] = win["fit_start"]
        dh["fit_end_time"] = win["fit_end"]
        if factor_key != "alpha158":
            if "infer_processors" in dh:
                dh["infer_processors"] = _strip_filter_col(dh["infer_processors"])

    port = cfg.get("port_analysis_config")
    if isinstance(port, dict):
        strat = port.get("strategy") or {}
        kwargs = strat.get("kwargs") or {}
        kwargs["topk"] = 10
        kwargs["n_drop"] = 1
        strat["kwargs"] = kwargs
        port["strategy"] = strat
        bt = port.get("backtest") or {}
        bt["start_time"] = win["bt_start"]
        bt["end_time"] = win["bt_end"]
        bt["account"] = 200000
        port["backtest"] = bt

    task = cfg.get("task") or {}
    # CPU 上 CatBoost Poisson bootstrap 不可用 → Bernoulli
    model = task.get("model") or {}
    mkw = model.get("kwargs")
    if isinstance(mkw, dict):
        if str(mkw.get("bootstrap_type", "")).lower() == "poisson":
            mkw["bootstrap_type"] = "Bernoulli"
        # macOS 上过高并行易把 DoubleEnsemble/LGB 打崩，限制线程
        for key in ("num_threads", "thread_count", "n_jobs"):
            if key in mkw and isinstance(mkw[key], int) and mkw[key] > 8:
                mkw[key] = 8
        # DoubleEnsemble：joblib/SR/FS 在本机易 SIGSEGV，收敛到稳妥子集
        if model.get("class") == "DEnsembleModel":
            mkw["num_threads"] = 1
            mkw["num_models"] = 2
            mkw["sub_weights"] = [1, 1]
            mkw["epochs"] = 12
            mkw["enable_fs"] = False
            mkw["enable_sr"] = False
        model["kwargs"] = mkw
        task["model"] = model

    ds = task.get("dataset") or {}
    dskw = ds.get("kwargs") or {}
    handler = dskw.get("handler") or {}
    handler["class"] = handler_cls
    handler["module_path"] = "qlib.contrib.data.handler"
    # kwargs 走锚点 *data_handler_config；若已展开则同步时间
    hkw = handler.get("kwargs")
    if isinstance(hkw, dict):
        hkw["start_time"] = win["handler_start"]
        hkw["end_time"] = win["handler_end"]
        hkw["fit_start_time"] = win["fit_start"]
        hkw["fit_end_time"] = win["fit_end"]
        if factor_key != "alpha158" and "infer_processors" in hkw:
            hkw["infer_processors"] = _strip_filter_col(hkw["infer_processors"])
        handler["kwargs"] = hkw
    dskw["handler"] = handler
    dskw["segments"] = {
        "train": list(win["train"]),
        "valid": list(win["valid"]),
        "test": list(win["test"]),
    }
    ds["kwargs"] = dskw
    task["dataset"] = ds
    cfg["task"] = task
    return cfg


def discover_models() -> List[Tuple[str, Path]]:
    """返回 (模型名, 模板 yaml 路径)。"""
    found: List[Tuple[str, Path]] = []
    for d in sorted(BENCHMARKS.iterdir()):
        if not d.is_dir() or d.name.startswith("_") or d.name in {".git"}:
            continue
        # 跳过非模型目录
        if d.name in {"README.md"}:
            continue
        tpl = _pick_template(d)
        if tpl is None:
            continue
        found.append((d.name, tpl))
    return found


def main() -> None:
    yaml = _yaml()
    GENERATED.mkdir(parents=True, exist_ok=True)
    jobs: List[Dict[str, Any]] = []

    for model_name, tpl in discover_models():
        with tpl.open("r", encoding="utf-8") as f:
            doc = yaml.load(f)
        out_dir = GENERATED / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        phase = MODEL_PHASE.get(model_name, 3)
        for factor_key in FACTOR_ORDER:
            handler_cls = FACTOR_HANDLERS[factor_key]
            patched = _patch_config(doc, factor_key, handler_cls)
            out_name = f"{model_name.lower()}_{factor_key}_20w.yaml"
            out_path = out_dir / out_name
            with out_path.open("w", encoding="utf-8") as f:
                yaml.dump(patched, f)
            exp_name = f"{model_name}_{factor_key}_20w_10n1"
            status = "skipped" if phase == 0 else "pending"
            skip_reason = "missing_pretrained_or_env" if phase == 0 else ""
            jobs.append(
                {
                    "model": model_name,
                    "factor_key": factor_key,
                    "phase": phase,
                    "yaml": str(out_path.relative_to(BENCHMARKS.parent.parent)),
                    "exp_name": exp_name,
                    "status": status,
                    "skip_reason": skip_reason,
                    "template": str(tpl.relative_to(BENCHMARKS.parent.parent)),
                    "error": "",
                }
            )

    # 按 phase、模型顺序、因子顺序排序
    model_rank = {m: i for i, m in enumerate(PHASE1 + PHASE2 + PHASE3 + SKIP_MODELS)}
    factor_rank = {f: i for i, f in enumerate(FACTOR_ORDER)}

    def sort_key(j: Dict[str, Any]):
        return (
            j["phase"] if j["phase"] > 0 else 99,
            model_rank.get(j["model"], 50),
            factor_rank.get(j["factor_key"], 50),
        )

    jobs.sort(key=sort_key)
    manifest = {
        "account": 200000,
        "topk": 10,
        "n_drop": 1,
        "factor_order": FACTOR_ORDER,
        "n_jobs": len(jobs),
        "jobs": jobs,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(jobs)} jobs -> {MANIFEST}")
    print(f"generated under {GENERATED}")


if __name__ == "__main__":
    main()
