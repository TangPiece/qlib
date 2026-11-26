#!/usr/bin/env python3
"""串行跑 generated_l2 下 KRNN/Sandwich L2 专用 yaml，并 export 到控制台历史。

用法（仓库根）::

    PYTHONUNBUFFERED=1 .venv/bin/python examples/benchmarks/_factor_sweep/run_l2_feadim.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

DIR = Path(__file__).resolve().parent
REPO = DIR.parents[2]
EXPORT = REPO / "examples" / "benchmarks" / "LightGBM" / "export_positions_view.py"
LOG_DIR = DIR / "logs"
PYTHON = sys.executable

JOBS = [
    ("KRNN", DIR / "generated_l2" / "KRNN" / "krnn_l2_20w_feadim.yaml"),
    ("Sandwich", DIR / "generated_l2" / "Sandwich" / "sandwich_l2_20w_feadim.yaml"),
]


def main() -> None:
    date_tag = datetime.now().strftime("%Y%m%d")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MPLCONFIGDIR", str(DIR / ".mplconfig"))

    for model, yaml_path in JOBS:
        if not yaml_path.exists():
            raise FileNotFoundError(yaml_path)
        exp = f"{model}_l2_20w_10n1_feadim_{date_tag}"
        train_log = LOG_DIR / f"{exp}_train.log"
        export_log = LOG_DIR / f"{exp}_export.log"
        print(f"\n=== {model} L2 feadim -> {exp} ===", flush=True)
        with train_log.open("w", encoding="utf-8") as logf:
            logf.write(f"CMD train {yaml_path} exp={exp}\n\n")
            logf.flush()
            rc = subprocess.run(
                [PYTHON, "-m", "qlib.cli.run", str(yaml_path), f"--experiment_name={exp}"],
                cwd=str(REPO),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
            ).returncode
        if rc != 0:
            print(f"FAIL train exit={rc} log={train_log}", flush=True)
            sys.exit(rc)
        print(f"train ok, exporting...", flush=True)
        with export_log.open("w", encoding="utf-8") as logf:
            rc2 = subprocess.run(
                [PYTHON, str(EXPORT), f"--exp_name={exp}"],
                cwd=str(REPO),
                stdout=logf,
                stderr=subprocess.STDOUT,
                env=env,
            ).returncode
        if rc2 != 0:
            print(f"FAIL export exit={rc2} log={export_log}", flush=True)
            sys.exit(rc2)
        print(f"SUCCESS {exp}", flush=True)
    print("\nall L2 feadim jobs done", flush=True)


if __name__ == "__main__":
    main()
