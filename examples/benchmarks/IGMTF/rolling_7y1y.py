# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
IGMTF Alpha360 滑动滚动：7y train + 1y valid -> 预测下一段 ~2y。

NOTE: 需预训练 LSTM pickle（model_path）。

Example:
    python examples/benchmarks/IGMTF/rolling_7y1y.py \\
      --exp_name=IGMTF_Alpha360_rolling_7y1y_YYYYMMDD \\
      run
"""
from __future__ import annotations

import sys
from pathlib import Path

_BENCHMARKS = Path(__file__).absolute().resolve().parents[1]
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

from _rolling.rolling_7y1y_common import run_rolling_cli  # noqa: E402

DIRNAME = Path(__file__).absolute().resolve().parent
DEFAULT_CONF = DIRNAME / "workflow_config_igmtf_Alpha360_rolling_7y1y.yaml"

if __name__ == "__main__":
    run_rolling_cli(DEFAULT_CONF, default_horizon=1, set_omp_threads=True)
