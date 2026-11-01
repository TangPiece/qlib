# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""把 alloc_v2 研究产物同步到 Cloudflare R2。

复用仓库 .env 的 R2_*；不上传 _markers（本机运行状态）。
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

from qlib.contrib.data.l2_daily.r2_env import load_r2_env

from .paths import default_root, markers_dir
from .store import write_marker

_log = logging.getLogger(__name__)

# 测试可注入：uploader(key, payload) -> None
_uploader: Callable[[str, bytes], None] | None = None
_client = None

UPLOAD_LAYERS = ("maps", "standard", "factor", "research")
DEFAULT_PREFIX = "research/alloc_v2"


def set_r2_uploader(uploader: Callable[[str, bytes], None] | None) -> None:
    """测试注入假上传器。"""
    global _uploader
    _uploader = uploader


def reset_r2_client_for_tests() -> None:
    global _client, _uploader
    _client = None
    _uploader = None


def r2_prefix() -> str:
    return os.getenv("ALLOC_V2_R2_PREFIX", DEFAULT_PREFIX).strip().strip("/") or DEFAULT_PREFIX


def r2_configured() -> bool:
    """四个 R2 变量都非空才算配好。会先 load 本仓库 .env。"""
    load_r2_env()
    return bool(
        os.getenv("R2_ENDPOINT_URL", "").strip()
        and os.getenv("R2_ACCESS_KEY_ID", "").strip()
        and os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        and os.getenv("R2_BUCKET", "").strip()
    )


def _bucket() -> str:
    return os.getenv("R2_BUCKET", "").strip()


def _r2_client():
    global _client
    if _client is not None:
        return _client
    import boto3
    from botocore.config import Config

    _client = boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip(),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
        region_name=os.getenv("R2_REGION", "auto").strip() or "auto",
        config=Config(signature_version="s3v4"),
    )
    return _client


def logical_key(rel_path: str) -> str:
    """相对路径 → R2 对象键。"""
    rel = rel_path.replace("\\", "/").lstrip("/")
    return f"{r2_prefix()}/{rel}"


def iter_upload_files(root: Path) -> Iterator[tuple[Path, str]]:
    """枚举可上传文件：(本地路径, 相对 root 的 posix 路径)。跳过 _markers 与 .tmp。"""
    root = Path(root)
    for layer in UPLOAD_LAYERS:
        base = root / layer
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            name = path.name
            if name.startswith(".") or name.endswith(".tmp") or ".parquet.tmp" in name:
                continue
            # 保险：路径中含 _markers 则跳过
            try:
                rel = path.relative_to(root).as_posix()
            except ValueError:
                continue
            if rel.startswith("_markers/") or "/_markers/" in f"/{rel}":
                continue
            yield path, rel


def _put(key: str, payload: bytes) -> None:
    if _uploader is not None:
        _uploader(key, payload)
        return
    _r2_client().put_object(Bucket=_bucket(), Key=key, Body=payload)


def upload_tree(
    root: Path | None = None,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    """逐文件上传 maps/standard/factor/research。失败记入 marker，不中断后续。"""
    root = root or default_root()
    load_r2_env()
    files = list(iter_upload_files(root))
    summary: dict[str, Any] = {
        "root": str(root),
        "prefix": r2_prefix(),
        "dry_run": dry_run,
        "configured": r2_configured() or _uploader is not None,
        "n_candidates": len(files),
        "uploaded": [],
        "skipped": [],
        "failed": [],
        "keys": [],
        "finished_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }

    if not files:
        summary["skipped"].append({"reason": "no_files"})
        write_marker("r2_upload", summary, root=root)
        return summary

    if not dry_run and not summary["configured"]:
        summary["failed"].append({"error": "R2 not configured (need R2_ENDPOINT_URL/ACCESS_KEY/SECRET/BUCKET)"})
        write_marker("r2_upload", summary, root=root)
        return summary

    for path, rel in files:
        key = logical_key(rel)
        if dry_run:
            summary["uploaded"].append({"path": rel, "key": key, "bytes": path.stat().st_size, "dry_run": True})
            summary["keys"].append(key)
            continue
        try:
            payload = path.read_bytes()
            _put(key, payload)
            summary["uploaded"].append({"path": rel, "key": key, "bytes": len(payload)})
            summary["keys"].append(key)
        except Exception as exc:  # noqa: BLE001 — 单文件失败不阻断
            _log.warning("alloc_v2 R2 上传失败 %s → %s: %s", rel, key, exc)
            summary["failed"].append({"path": rel, "key": key, "error": str(exc)[:500]})

    write_marker("r2_upload", summary, root=root)
    # 额外落一份失败明细到 markers，便于排查
    if summary["failed"]:
        fail_path = markers_dir(root) / "r2_upload_failed.json"
        fail_path.write_text(json.dumps(summary["failed"], ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
