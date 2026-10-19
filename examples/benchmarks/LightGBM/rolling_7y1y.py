# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Sliding rolling retraining: 7y train + 1y valid -> predict next ~2y.

Naming: NyMy = N years train + M years valid (rolling step stays ~2y trading days).

Example:
    python examples/benchmarks/LightGBM/rolling_7y1y.py \\
      --conf_path=examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_rolling_7y1y.yaml \\
      --exp_name=LightGBM_Alpha158_rolling_7y1y_20261019 \\
      --horizon=1 \\
      run

Prerequisite: ~/.qlib/qlib_data/cn_data must cover the yaml end_time
(use update_cn_data_daily.py to refresh data and sync end dates).
"""
from __future__ import annotations

import sys
from pathlib import Path

# 允许从本目录直接运行时导入共用滚动模块
_BENCHMARKS = Path(__file__).absolute().resolve().parents[1]
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

from _rolling.rolling_7y1y_common import run_rolling_cli  # noqa: E402

DIRNAME = Path(__file__).absolute().resolve().parent
DEFAULT_CONF = DIRNAME / "workflow_config_lightgbm_Alpha158_rolling_7y1y.yaml"

if __name__ == "__main__":
    run_rolling_cli(DEFAULT_CONF, default_horizon=1, set_omp_threads=True)
