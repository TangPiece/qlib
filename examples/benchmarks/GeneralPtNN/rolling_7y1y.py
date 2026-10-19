# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
GeneralPtNN gru 滑动滚动：7y train + 1y valid -> 预测下一段 ~2y。

NOTE: 基于 GeneralPtNN GRU 配置；无 Alpha158/360 文件名。

Example:
    python examples/benchmarks/GeneralPtNN/rolling_7y1y.py \\
      --exp_name=GeneralPtNN_gru_rolling_7y1y_YYYYMMDD \\
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
DEFAULT_CONF = DIRNAME / "workflow_config_generalptnn_gru_rolling_7y1y.yaml"

if __name__ == "__main__":
    run_rolling_cli(DEFAULT_CONF, default_horizon=1, set_omp_threads=True)
