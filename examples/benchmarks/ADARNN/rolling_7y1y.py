# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
ADARNN Alpha360 滑动滚动：7y train + 1y valid -> 预测下一段 ~2y。

horizon 默认 1，与 label Ref($close,-2)/Ref($close,-1)-1 一致。

Example:
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python examples/benchmarks/ADARNN/rolling_7y1y.py \\
      --exp_name=ADARNN_Alpha360_rolling_7y1y_20261019 \\
      run

连通性检查（压低 epoch）:
    python examples/benchmarks/ADARNN/rolling_7y1y.py \\
      --exp_name=ADARNN_Alpha360_rolling_7y1y_smoke \\
      --task_ext_conf='{"model":{"kwargs":{"n_epochs":2,"early_stop":1}}}' \\
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
DEFAULT_CONF = DIRNAME / "workflow_config_adarnn_Alpha360_rolling_7y1y.yaml"

if __name__ == "__main__":
    run_rolling_cli(DEFAULT_CONF, default_horizon=1, set_omp_threads=True)
