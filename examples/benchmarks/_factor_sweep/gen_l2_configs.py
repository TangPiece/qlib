#!/usr/bin/env python3
"""为 KRNN / Sandwich 生成 L2 专用 yaml（写入 generated_l2/，不改官方与旧 generated）。

用法（仓库根）::

    .venv/bin/python examples/benchmarks/_factor_sweep/gen_l2_configs.py
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict

from ruamel.yaml import YAML

from qlib.contrib.data.l2_daily import get_l2_feature_config

DIR = Path(__file__).resolve().parent
BENCHMARKS = DIR.parent
OUT_ROOT = DIR / "generated_l2"

# 与 gen_configs.L2_WIN 一致
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

# 模型目录名 → 官方 Alpha360 模板（KRNN/Sandwich 无 Alpha158 官方单次）
MODELS = {
    "KRNN": "workflow_config_krnn_Alpha360.yaml",
    "Sandwich": "workflow_config_sandwich_Alpha360.yaml",
}


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.default_flow_style = False
    return y


def _patch_l2(doc: Dict[str, Any], fea_dim: int) -> Dict[str, Any]:
    """写入 L2 handler、短窗、20万/10/1，以及正确 fea_dim。"""
    cfg = copy.deepcopy(doc)
    win = L2_WIN

    dh = cfg.get("data_handler_config")
    if isinstance(dh, dict):
        dh["start_time"] = win["handler_start"]
        dh["end_time"] = win["handler_end"]
        dh["fit_start_time"] = win["fit_start"]
        dh["fit_end_time"] = win["fit_end"]

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
    model = task.get("model") or {}
    mkw = model.get("kwargs")
    if isinstance(mkw, dict):
        # KRNN / Sandwich 用 fea_dim 作为每步特征维；L2 为扁平 100 列 → fea_dim=100
        mkw["fea_dim"] = int(fea_dim)
        model["kwargs"] = mkw
        task["model"] = model

    ds = task.get("dataset") or {}
    dskw = ds.get("kwargs") or {}
    handler = dskw.get("handler") or {}
    handler["class"] = "L2Daily"
    handler["module_path"] = "qlib.contrib.data.handler"
    hkw = handler.get("kwargs")
    if isinstance(hkw, dict):
        hkw["start_time"] = win["handler_start"]
        hkw["end_time"] = win["handler_end"]
        hkw["fit_start_time"] = win["fit_start"]
        hkw["fit_end_time"] = win["fit_end"]
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


def main() -> None:
    yaml = _yaml()
    _, names = get_l2_feature_config()
    fea_dim = len(names)
    print(f"L2 fea_dim={fea_dim}")

    for model_name, tpl_name in MODELS.items():
        tpl = BENCHMARKS / model_name / tpl_name
        if not tpl.exists():
            raise FileNotFoundError(tpl)
        with tpl.open("r", encoding="utf-8") as f:
            doc = yaml.load(f)
        patched = _patch_l2(doc, fea_dim)
        out_dir = OUT_ROOT / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{model_name.lower()}_l2_20w_feadim.yaml"
        with out_path.open("w", encoding="utf-8") as f:
            yaml.dump(patched, f)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
