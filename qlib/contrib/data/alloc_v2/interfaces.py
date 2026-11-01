# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""接口实测状态：限流、失败、最近探活时间。与 registry 文档口径分开存。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .paths import markers_dir


@dataclass
class InterfaceStatus:
    """单个 adapter 的运行时状态，不回写 registry.enabled。"""

    name: str
    last_probe_at: str = ""
    last_ok_at: str = ""
    last_error: str = ""
    n_ok: int = 0
    n_fail: int = 0
    rate_limit_sec: float = 0.3
    rows: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


_STATUS: dict[str, InterfaceStatus] = {}


def status_of(name: str) -> InterfaceStatus:
    item = _STATUS.get(name)
    if item is None:
        item = InterfaceStatus(name=name)
        _STATUS[name] = item
    return item


def mark_ok(name: str, rows: int = 0) -> InterfaceStatus:
    now = datetime.now().isoformat(timespec="seconds")
    item = status_of(name)
    item.last_probe_at = now
    item.last_ok_at = now
    item.last_error = ""
    item.n_ok += 1
    item.rows += int(rows)
    return item


def mark_fail(name: str, error: BaseException | str) -> InterfaceStatus:
    now = datetime.now().isoformat(timespec="seconds")
    item = status_of(name)
    item.last_probe_at = now
    item.last_error = str(error)
    item.n_fail += 1
    return item


def dump_status(root: Path | None = None) -> Path:
    dest = markers_dir(root) / "interfaces.json"
    payload = {name: item.as_dict() for name, item in _STATUS.items()}
    dest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return dest


def reset_status_for_tests() -> None:
    _STATUS.clear()
