# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""fund_cn 本地 staging / 日宽表 / 质检路径，默认落在 qlib/data。"""

from __future__ import annotations

import os
from pathlib import Path

from .catalog import FACTOR_SET_VERSION, FACTOR_VERSION


def qlib_root() -> Path:
    """qlib 仓库根：contrib/data/fund_cn/paths_local.py 往上四级。"""
    return Path(__file__).resolve().parents[4]


def default_staging_root() -> Path:
    raw = os.getenv("FUND_CN_STAGING_ROOT", "").strip()
    path = Path(raw) if raw else qlib_root() / "data" / "fundamental_staging" / "fund_cn"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_factor_root() -> Path:
    raw = os.getenv("FUND_CN_FACTOR_ROOT", "").strip()
    path = Path(raw) if raw else qlib_root() / "data" / "fundamental_staging" / "factors" / "fund_cn"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_qa_dir() -> Path:
    path = qlib_root() / "examples" / "benchmarks" / "LightGBM" / "eval_output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def markers_dir(root: Path | None = None) -> Path:
    path = (root or default_staging_root()) / "_markers"
    path.mkdir(parents=True, exist_ok=True)
    return path


def statements_dir(root: Path | None = None) -> Path:
    path = (root or default_staging_root()) / "statements"
    path.mkdir(parents=True, exist_ok=True)
    return path


def announce_dir(root: Path | None = None) -> Path:
    path = (root or default_staging_root()) / "announce"
    path.mkdir(parents=True, exist_ok=True)
    return path


def share_capital_dir(root: Path | None = None) -> Path:
    path = (root or default_staging_root()) / "share_capital"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pit_dir(root: Path | None = None) -> Path:
    path = (root or default_staging_root()) / "pit"
    path.mkdir(parents=True, exist_ok=True)
    return path


def pit_metrics_path(root: Path | None = None) -> Path:
    return pit_dir(root) / "metrics.parquet"


def version_segment(factor_version: int = FACTOR_VERSION) -> str:
    return f"v{int(factor_version)}"


def local_factor_day_path(root: Path, trading_date: str, *, factor_version: int = FACTOR_VERSION) -> Path:
    """YYYY-MM-DD 或 YYYYMMDD → v1/YYYY/YYYYMM/YYYYMMDD.parquet。"""
    day = str(trading_date).replace("-", "")
    folder = root / version_segment(factor_version) / day[:4] / day[:6]
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{day}.parquet"


def factor_set_segment() -> str:
    return FACTOR_SET_VERSION
