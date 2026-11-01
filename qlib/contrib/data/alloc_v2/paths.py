# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""alloc_v2 四层本地目录：raw / standard / factor / research。"""

from __future__ import annotations

import os
from pathlib import Path


def qlib_root() -> Path:
    """qlib 仓库根：contrib/data/alloc_v2/paths.py 往上四级。"""
    return Path(__file__).resolve().parents[4]


def default_root() -> Path:
    raw = os.getenv("ALLOC_V2_ROOT", "").strip()
    path = Path(raw) if raw else qlib_root() / "data" / "alloc_v2"
    path.mkdir(parents=True, exist_ok=True)
    return path


def layer_dir(name: str, root: Path | None = None) -> Path:
    path = (root or default_root()) / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_dir(root: Path | None = None) -> Path:
    return layer_dir("raw", root)


def standard_dir(root: Path | None = None) -> Path:
    return layer_dir("standard", root)


def factor_dir(root: Path | None = None) -> Path:
    return layer_dir("factor", root)


def research_dir(root: Path | None = None) -> Path:
    return layer_dir("research", root)


def maps_dir(root: Path | None = None) -> Path:
    return layer_dir("maps", root)


def markers_dir(root: Path | None = None) -> Path:
    return layer_dir("_markers", root)
