# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
共用滑动滚动：7y train + 1y valid -> 预测下一段 ~2y。

各模型目录的 rolling_7y1y.py 通过 make_rolling_cli 薄封装即可。
对齐 RollingGen(ROLL_SD) + 每窗 processor refit（无 handler cache）。
"""
from __future__ import annotations

import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Union

import fire
from ruamel.yaml import YAML

from qlib import auto_init
from qlib.contrib.rolling.base import Rolling
from qlib.workflow.task.gen import RollingGen, handler_mod, task_generator
from qlib.workflow.task.utils import TimeAdjuster

# 与 LightGBM rolling_7y1y 相同的 ~2y 交易日步长锚点
STEP_START = "2016-01-01"
STEP_END = "2017-12-31"


def handler_mod_refit(task: dict, rolling_gen: RollingGen) -> None:
    """扩展 handler end_time，并把 fit_* 同步到当前 train 段以便每窗重拟合处理器。"""
    handler_mod(task, rolling_gen)
    try:
        handler_kwargs = task["dataset"]["kwargs"]["handler"]["kwargs"]
        train_seg = task["dataset"]["kwargs"]["segments"]["train"]
        handler_kwargs["fit_start_time"] = deepcopy(train_seg[0])
        handler_kwargs["fit_end_time"] = deepcopy(train_seg[1])
    except (KeyError, TypeError):
        pass


def compute_step_2y(step_start: str = STEP_START, step_end: str = STEP_END) -> int:
    """计算 step_start~step_end（含）的交易日数量作为滚动步长。"""
    ta = TimeAdjuster(future=True)
    start_idx = ta.align_idx(step_start, tp_type="start")
    end_idx = ta.align_idx(step_end, tp_type="end")
    step = end_idx - start_idx + 1
    if step <= 0:
        raise ValueError(f"Invalid step window {step_start} -> {step_end}: got step={step}")
    return step


def load_qlib_init(conf_path: Path) -> dict:
    """从滚动 yaml 读取 qlib_init（Rolling 基类会忽略该段，需手动 auto_init）。"""
    with conf_path.open("r") as f:
        yaml = YAML(typ="safe", pure=True)
        conf = yaml.load(f) or {}
    return dict(conf.get("qlib_init") or {})


def resolve_conf_path_from_argv(default_conf: Path, env_key: str = "ROLLING_CONF") -> Path:
    """从 argv / 环境变量解析 --conf_path，供 auto_init 读取 qlib_init。"""
    conf_path = Path(os.environ.get(env_key, default_conf))
    argv = sys.argv[1:]
    for i, arg in enumerate(argv):
        if arg.startswith("--conf_path="):
            conf_path = Path(arg.split("=", 1)[1])
            break
        if arg in ("--conf_path", "conf_path") and i + 1 < len(argv):
            conf_path = Path(argv[i + 1])
            break
    return conf_path


class Rolling7y1y(Rolling):
    """7y train / 1y valid 滑动滚动（~2y step），每窗重拟合 handler 处理器。"""

    def __init__(
        self,
        conf_path: Union[str, Path],
        exp_name: Optional[str] = None,
        horizon: int = 1,
        step: Optional[int] = None,
        *,
        # TCTS 等多 label 模型需保留 yaml 原 label，避免 Rolling 按 horizon 覆盖成单标签
        override_label: bool = True,
        **kwargs,
    ) -> None:
        # step=None -> qlib init 后在 get_task_list 里按日历算 ~2y
        super().__init__(
            conf_path=conf_path,
            exp_name=exp_name,
            horizon=horizon,
            step=step if step is not None else 0,
            **kwargs,
        )
        self._auto_step = step is None
        self._override_label = override_label

    def basic_task(self, enable_handler_cache: Optional[bool] = False):
        # 关闭 handler cache，保证每窗按 fit_* 重拟合 RobustZScoreNorm 等
        task = super().basic_task(enable_handler_cache=False)
        if not self._override_label:
            # 从原始 conf 恢复 handler label（父类已按 horizon 改写）
            try:
                raw = self._raw_conf()["task"]["dataset"]["kwargs"]["handler"]["kwargs"].get("label")
                if raw is not None:
                    task["dataset"]["kwargs"]["handler"]["kwargs"]["label"] = deepcopy(raw)
                    self.logger.info("Preserved handler label from yaml (override_label=False)")
            except (KeyError, TypeError):
                pass
        return task

    def get_task_list(self) -> List[dict]:
        if self._auto_step or self.step <= 0:
            self.step = compute_step_2y()
            self.logger.info(
                f"Using auto step={self.step} trading days (~2y from {STEP_START} to {STEP_END})"
            )

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


def make_rolling_cli(
    default_conf: Union[str, Path],
    *,
    default_horizon: int = 1,
    env_key: str = "ROLLING_CONF",
    set_omp_threads: bool = True,
    override_label: bool = True,
):
    """
    返回供 fire.Fire 使用的 Rolling7y1y 子类，并完成 qlib auto_init。

    @param default_conf: 默认滚动 yaml 路径
    @param default_horizon: 默认预测 horizon（与 label 对齐）
    @param env_key: 覆盖 conf_path 的环境变量名
    @param set_omp_threads: 是否锁定 OMP/MKL 单线程（macOS 防 SIGSEGV）
    @param override_label: False 时保留 yaml 多 label（如 TCTS）
    """
    default_conf = Path(default_conf)

    class _CliRolling(Rolling7y1y):
        """带默认 conf/horizon 的滚动 CLI 入口。"""

        def __init__(
            self,
            conf_path: Union[str, Path] = default_conf,
            exp_name: Optional[str] = None,
            horizon: int = default_horizon,
            step: Optional[int] = None,
            **kwargs,
        ) -> None:
            super().__init__(
                conf_path=conf_path,
                exp_name=exp_name,
                horizon=horizon,
                step=step,
                override_label=override_label,
                **kwargs,
            )

    if set_omp_threads:
        # macOS 上 joblib fork 后多线程 OpenMP 易 SIGSEGV；训练前锁定单线程
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    conf_path = resolve_conf_path_from_argv(default_conf, env_key=env_key)
    auto_init(**load_qlib_init(conf_path))
    return _CliRolling


def run_rolling_cli(
    default_conf: Union[str, Path],
    *,
    default_horizon: int = 1,
    env_key: str = "ROLLING_CONF",
    set_omp_threads: bool = True,
    override_label: bool = True,
) -> None:
    """各模型 rolling_7y1y.py 的 __main__ 入口：auto_init + fire。"""
    cli_cls = make_rolling_cli(
        default_conf,
        default_horizon=default_horizon,
        env_key=env_key,
        set_omp_threads=set_omp_threads,
        override_label=override_label,
    )
    fire.Fire(cli_cls)
