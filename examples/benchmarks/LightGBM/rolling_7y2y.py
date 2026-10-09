# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Sliding rolling retraining: 7y train + 1y valid -> predict next ~2y.

Uses RollingGen(ROLL_SD) with per-window processor refit (no handler cache).

Example:
    python examples/benchmarks/LightGBM/rolling_7y2y.py \\
      --conf_path=examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_rolling_7y2y.yaml \\
      --exp_name=rolling_lgb_7y2y_20260930 \\
      --horizon=20 \\
      run

Prerequisite: ~/.qlib/qlib_data/cn_data must cover the yaml end_time
(use update_cn_data_daily.py to refresh data and sync end dates).
Re-runs: use a fresh --exp_name or clear conflicting mlruns experiments.
"""
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union

import fire

from qlib import auto_init
from qlib.contrib.rolling.base import Rolling
from qlib.workflow.task.gen import RollingGen, handler_mod, task_generator
from qlib.workflow.task.utils import TimeAdjuster

DIRNAME = Path(__file__).absolute().resolve().parent
DEFAULT_CONF = DIRNAME / "workflow_config_lightgbm_Alpha158_rolling_7y2y.yaml"

# Calendar anchors for ~2-year test windows (trading-day step)
STEP_START = "2017-01-01"
STEP_END = "2018-12-31"


def handler_mod_refit(task: dict, rolling_gen: RollingGen) -> None:
    """Extend handler end_time and sync fit_* to the current train segment."""
    handler_mod(task, rolling_gen)
    try:
        handler_kwargs = task["dataset"]["kwargs"]["handler"]["kwargs"]
        train_seg = task["dataset"]["kwargs"]["segments"]["train"]
        handler_kwargs["fit_start_time"] = deepcopy(train_seg[0])
        handler_kwargs["fit_end_time"] = deepcopy(train_seg[1])
    except (KeyError, TypeError):
        pass


def compute_step_2y(step_start: str = STEP_START, step_end: str = STEP_END) -> int:
    """Number of trading days from step_start to step_end (inclusive)."""
    ta = TimeAdjuster(future=True)
    start_idx = ta.align_idx(step_start, tp_type="start")
    end_idx = ta.align_idx(step_end, tp_type="end")
    step = end_idx - start_idx + 1
    if step <= 0:
        raise ValueError(f"Invalid step window {step_start} -> {step_end}: got step={step}")
    return step


class Rolling7y2y(Rolling):
    """Sliding 7y/1y/2y rolling with per-window Alpha158 processor refit."""

    def __init__(
        self,
        conf_path: Union[str, Path] = DEFAULT_CONF,
        exp_name: Optional[str] = None,
        horizon: int = 1,
        step: Optional[int] = None,
        **kwargs,
    ) -> None:
        # step=None -> compute ~2y trading days after qlib init (in get_task_list)
        super().__init__(
            conf_path=conf_path,
            exp_name=exp_name,
            horizon=horizon,
            step=step if step is not None else 0,
            **kwargs,
        )
        self._auto_step = step is None

    def basic_task(self, enable_handler_cache: Optional[bool] = False):
        # Disable handler cache so each window refits processors via fit_* times
        return super().basic_task(enable_handler_cache=False)

    def get_task_list(self) -> List[dict]:
        if self._auto_step or self.step <= 0:
            self.step = compute_step_2y()
            self.logger.info(f"Using auto step={self.step} trading days (~2y from {STEP_START} to {STEP_END})")

        task = self.basic_task()
        task_l = task_generator(
            task,
            RollingGen(
                step=self.step,
                rtype=RollingGen.ROLL_SD,
                trunc_days=self.horizon + 1,
                ds_extra_mod_func=handler_mod_refit,
            ),
        )
        for t in task_l:
            t["record"] = ["qlib.workflow.record_temp.SignalRecord"]
        self.logger.info(f"Generated {len(task_l)} sliding rolling tasks")
        for i, t in enumerate(task_l):
            segs = t["dataset"]["kwargs"]["segments"]
            self.logger.info(
                f"  window{i + 1}: train={segs['train']}, valid={segs['valid']}, test={segs['test']}"
            )
        return task_l


if __name__ == "__main__":
    auto_init()
    fire.Fire(Rolling7y2y)
