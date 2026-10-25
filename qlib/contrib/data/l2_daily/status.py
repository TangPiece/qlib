"""source_version 与本地 marker。不写 D1。"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .catalog import FACTOR_VERSION, GROUP_BOOK, GROUP_MICROSTRUCTURE, GROUP_ORDER, GROUP_TRADE
from .engine import source_file_path

_GROUP_FILES = {
    GROUP_TRADE: ("逐笔成交",),
    GROUP_BOOK: ("行情",),
    GROUP_ORDER: ("逐笔委托", "逐笔成交"),
    GROUP_MICROSTRUCTURE: ("行情", "逐笔成交", "逐笔委托"),
}


def source_version_for_date(
    books_root: Path,
    trade_date: str,
    factor_group: str,
    codes: list[str] | None = None,
) -> str:
    """由当日相关明细文件 size+mtime 摘要得到 source_version。"""
    files = _GROUP_FILES.get(factor_group, ("行情", "逐笔成交", "逐笔委托"))
    day = Path(books_root) / str(trade_date)
    digest = hashlib.sha256()
    digest.update(factor_group.encode("utf-8"))
    digest.update(str(trade_date).encode("utf-8"))
    if not day.is_dir():
        return digest.hexdigest()[:16]
    selected = codes
    if selected is None:
        selected = sorted(child.name for child in day.iterdir() if child.is_dir())
    for code in selected:
        for ftype in files:
            path = source_file_path(books_root, trade_date, code, ftype)
            if path is None:
                digest.update(f"missing:{code}:{ftype}".encode("utf-8"))
                continue
            stat = path.stat()
            digest.update(
                f"{code}:{ftype}:{path.suffix}:{stat.st_size}:{int(stat.st_mtime_ns)}".encode("utf-8")
            )
    return digest.hexdigest()[:16]


def should_skip(
    trade_date: str,
    factor_group: str,
    source_version: str,
    *,
    force: bool = False,
    factor_version: int = FACTOR_VERSION,
    local_status: dict | None = None,
) -> bool:
    """完成且 source_version 未变则跳过。"""
    if force:
        return False
    if not local_status:
        return False
    return (
        local_status.get("status") == "done"
        and local_status.get("source_version") == source_version
        and int(local_status.get("factor_version") or 0) == int(factor_version)
    )


def local_marker_path(output_root: Path, trade_date: str, factor_group: str) -> Path:
    return Path(output_root) / "_markers" / f"{trade_date}_{factor_group}.json"


def read_local_marker(output_root: Path, trade_date: str, factor_group: str) -> dict | None:
    path = local_marker_path(output_root, trade_date, factor_group)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def write_local_marker(
    output_root: Path,
    trade_date: str,
    factor_group: str,
    *,
    source_version: str,
    status: str,
    symbol_ok: int,
    symbol_fail: int,
    factor_version: int = FACTOR_VERSION,
) -> None:
    path = local_marker_path(output_root, trade_date, factor_group)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "trade_date": trade_date,
                "factor_group": factor_group,
                "factor_version": factor_version,
                "source_version": source_version,
                "status": status,
                "symbol_ok": symbol_ok,
                "symbol_fail": symbol_fail,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
