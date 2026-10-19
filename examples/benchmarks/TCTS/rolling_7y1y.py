# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
TCTS Alpha360 滑动滚动：7y train + 1y valid -> 预测下一段 ~2y。

NOTE: 多 label；滚动时请保持 horizon 与 yaml label 一致，勿随意改 horizon。

Example:
    python examples/benchmarks/TCTS/rolling_7y1y.py \\
      --exp_name=TCTS_Alpha360_rolling_7y1y_YYYYMMDD \\
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
DEFAULT_CONF = DIRNAME / "workflow_config_tcts_Alpha360_rolling_7y1y.yaml"

if __name__ == "__main__":
    # 保留 yaml 三 label，避免 Rolling 按 horizon 覆盖成单标签
    run_rolling_cli(DEFAULT_CONF, default_horizon=1, set_omp_threads=True, override_label=False)
