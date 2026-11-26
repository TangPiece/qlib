# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
from pathlib import Path

from qlib.contrib.data.loader import Alpha158DL, Alpha360DL
from .l2_daily import get_l2_feature_config
from ...data.dataset.handler import DataHandlerLP
from ...data.dataset.processor import Processor
from ...utils import get_callable_kwargs
from ...data.dataset import processor as processor_module
from inspect import getfullargspec


def _factor_lab_ab_feature_sets_dir() -> Path:
    """LightGBM 特征 JSON 目录：examples/benchmarks/LightGBM/feature_sets。"""
    # handler.py → data → contrib → qlib(package) → qlib(repo)
    return (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "benchmarks"
        / "LightGBM"
        / "feature_sets"
    )


def _factor_lab_ab_default_json() -> Path:
    """默认特征清单：feature_sets/factor_lab_ab_113.json。"""
    return _factor_lab_ab_feature_sets_dir() / "factor_lab_ab_113.json"


def _resolve_factor_lab_feature_path(feature_list_path) -> Path:
    """解析特征 JSON。cwd 相对路径找不到时，回退到 feature_sets。"""
    raw = Path(feature_list_path).expanduser()
    candidates = [raw.resolve() if raw.is_absolute() else (Path.cwd() / raw).resolve()]
    feature_dir = _factor_lab_ab_feature_sets_dir()
    # 仓库根目录或 examples/ 下启动时，相对路径常绑错 cwd
    candidates.append((feature_dir / raw.name).resolve())
    candidates.append((feature_dir / raw).resolve())
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "FactorLabAB feature json not found: %s (tried %s)"
        % (feature_list_path, [str(p) for p in candidates])
    )


def check_transform_proc(proc_l, fit_start_time, fit_end_time):
    new_l = []
    for p in proc_l:
        if not isinstance(p, Processor):
            klass, pkwargs = get_callable_kwargs(p, processor_module)
            args = getfullargspec(klass).args
            if "fit_start_time" in args and "fit_end_time" in args:
                assert (
                    fit_start_time is not None and fit_end_time is not None
                ), "Make sure `fit_start_time` and `fit_end_time` are not None."
                pkwargs.update(
                    {
                        "fit_start_time": fit_start_time,
                        "fit_end_time": fit_end_time,
                    }
                )
            proc_config = {"class": klass.__name__, "kwargs": pkwargs}
            if isinstance(p, dict) and "module_path" in p:
                proc_config["module_path"] = p["module_path"]
            new_l.append(proc_config)
        else:
            new_l.append(p)
    return new_l


_DEFAULT_LEARN_PROCESSORS = [
    {"class": "DropnaLabel"},
    {"class": "CSZScoreNorm", "kwargs": {"fields_group": "label"}},
]
_DEFAULT_INFER_PROCESSORS = [
    {"class": "ProcessInf", "kwargs": {}},
    {"class": "ZScoreNorm", "kwargs": {}},
    {"class": "Fillna", "kwargs": {}},
]


class Alpha360(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=_DEFAULT_INFER_PROCESSORS,
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": Alpha360DL.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }

        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            learn_processors=learn_processors,
            infer_processors=infer_processors,
            **kwargs,
        )

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha360vwap(Alpha360):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]


class Alpha158(DataHandlerLP):
    def __init__(
        self,
        instruments="csi500",
        start_time=None,
        end_time=None,
        freq="day",
        infer_processors=[],
        learn_processors=_DEFAULT_LEARN_PROCESSORS,
        fit_start_time=None,
        fit_end_time=None,
        process_type=DataHandlerLP.PTYPE_A,
        filter_pipe=None,
        inst_processors=None,
        **kwargs,
    ):
        infer_processors = check_transform_proc(infer_processors, fit_start_time, fit_end_time)
        learn_processors = check_transform_proc(learn_processors, fit_start_time, fit_end_time)

        data_loader = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": self.get_feature_config(),
                    "label": kwargs.pop("label", self.get_label_config()),
                },
                "filter_pipe": filter_pipe,
                "freq": freq,
                "inst_processors": inst_processors,
            },
        }
        super().__init__(
            instruments=instruments,
            start_time=start_time,
            end_time=end_time,
            data_loader=data_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
            process_type=process_type,
            **kwargs,
        )

    def get_feature_config(self):
        conf = {
            "kbar": {},
            "price": {
                "windows": [0],
                "feature": ["OPEN", "HIGH", "LOW", "VWAP"],
            },
            "rolling": {},
        }
        return Alpha158DL.get_feature_config(conf)

    def get_label_config(self):
        return ["Ref($close, -2)/Ref($close, -1) - 1"], ["LABEL0"]


class Alpha158vwap(Alpha158):
    def get_label_config(self):
        return ["Ref($vwap, -2)/Ref($vwap, -1) - 1"], ["LABEL0"]


class L2Daily(Alpha158):
    """只用 100 个预计算 L2 日频因子，标签与 Alpha158 相同。"""

    def get_feature_config(self):
        return get_l2_feature_config()


class MOM12_1(Alpha158):
    """仅 MOM12-1 单因子；继承 Alpha158 以支持 rolling 注入的 label kwargs。"""

    def get_feature_config(self):
        # 与 alphalens qlib_bridge.MOMENTUM_FACTORS["MOM12-1"] 一致
        return ["Ref($close, 21) / Ref($close, 252) - 1"], ["MOM12-1"]


class Alpha158L2Daily(Alpha158):
    """Alpha158 的 158 列再拼 100 个 L2 因子；标签只保留一份。"""

    def get_feature_config(self):
        fields, names = super().get_feature_config()
        l2_fields, l2_names = get_l2_feature_config()
        return fields + l2_fields, names + l2_names


class FactorLabAB(Alpha158):
    """Factor Lab 评级 A+B 精选因子（默认 113 列）；标签与 Alpha158 相同。

    特征来自 LightGBM/feature_sets JSON（fields/names）；可用 kwargs
    ``feature_list_path`` 覆盖路径。须在 ``super().__init__`` 前记下路径，
    因为 ``Alpha158`` 会在初始化时调用 ``get_feature_config()``。
    """

    def __init__(self, feature_list_path=None, **kwargs):
        # 须在 super().__init__ 前记下路径，Alpha158 初始化时会调 get_feature_config
        self._feature_list_path = (
            _resolve_factor_lab_feature_path(feature_list_path)
            if feature_list_path
            else _factor_lab_ab_default_json()
        )
        super().__init__(**kwargs)

    def get_feature_config(self):
        path = getattr(self, "_feature_list_path", None) or _factor_lab_ab_default_json()
        data = json.loads(Path(path).read_text())
        fields = list(data["fields"])
        names = list(data["names"])
        if len(fields) != len(names) or not fields:
            raise ValueError("invalid FactorLabAB feature json: %s" % path)
        return fields, names


class FundCn(Alpha158):
    """仅 A 股财务因子 fund_cn（默认 MVP 子集）；标签与 Alpha158 相同。"""

    def get_feature_config(self):
        # 延迟导入，避免 handler 模块加载时拉起 fund_cn 全依赖
        from qlib.contrib.data.fund_cn import get_fund_cn_feature_config

        return get_fund_cn_feature_config(mvp_only=True)


class MixFour(Alpha158):
    """Alpha158 + Alpha360 + L2Daily + FundCn 拼接；列名加前缀防撞名。"""

    def get_feature_config(self):
        from qlib.contrib.data.fund_cn import get_fund_cn_feature_config

        f158, n158 = super().get_feature_config()
        f360, n360 = Alpha360DL.get_feature_config()
        fl2, nl2 = get_l2_feature_config()
        ff, nf = get_fund_cn_feature_config(mvp_only=True)
        fields = list(f158) + list(f360) + list(fl2) + list(ff)
        names = (
            ["a158_%s" % n for n in n158]
            + ["a360_%s" % n for n in n360]
            + ["l2_%s" % n for n in nl2]
            + ["fund_%s" % n for n in nf]
        )
        return fields, names
