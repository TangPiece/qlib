# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""flow_monitor 本地 staging / 计算结果路径。"""

from __future__ import annotations

import os
from pathlib import Path


def qlib_root() -> Path:
    """qlib 仓库根：contrib/data/flow_monitor/paths_local.py 往上四级。"""
    return Path(__file__).resolve().parents[4]


def default_staging_root() -> Path:
    raw = os.getenv("FLOW_MONITOR_STAGING_ROOT", "").strip()
    path = Path(raw) if raw else qlib_root() / "data" / "flow_staging"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir(root: Path | None = None) -> Path:
    path = (root or default_staging_root()) / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def panel_dir(root: Path | None = None) -> Path:
    path = (root or default_staging_root()) / "panels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def diag_dir(root: Path | None = None) -> Path:
    path = (root or default_staging_root()) / "diagnostics"
    path.mkdir(parents=True, exist_ok=True)
    return path


def markers_dir(root: Path | None = None) -> Path:
    path = (root or default_staging_root()) / "_markers"
    path.mkdir(parents=True, exist_ok=True)
    return path
