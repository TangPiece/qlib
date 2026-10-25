# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""本地日宽表 parquet 与 R2 按日备份。storage_uri 只存逻辑键。"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from .catalog import FACTOR_VERSION, factor_ids, wide_columns
from .paths_local import default_factor_root, local_factor_day_path, version_segment

_log = logging.getLogger(__name__)
_uploader = None
_client = None


def set_r2_uploader(uploader) -> None:
    """测试注入：``uploader(logical_key: str, payload: bytes)``。"""
    global _uploader
    _uploader = uploader


def default_root() -> Path:
    return default_factor_root()


def r2_prefix() -> str:
    return os.getenv("R2_FUND_CN_PREFIX", "factors/fund_cn").strip().strip("/") or "factors/fund_cn"


def logical_day_key(trade_date: str, *, factor_version: int = FACTOR_VERSION) -> str:
    day = str(trade_date).replace("-", "")
    return f"{r2_prefix()}/{version_segment(factor_version)}/{day[:4]}/{day[:6]}/{day}.parquet"


def day_path(root: Path, trade_date: str, *, factor_version: int = FACTOR_VERSION) -> Path:
    return local_factor_day_path(root, trade_date, factor_version=factor_version)


def write_day(
    root: Path,
    trade_date: str,
    frame: pd.DataFrame,
    *,
    mvp_only: bool = True,
    factor_version: int = FACTOR_VERSION,
) -> Path:
    """原子写入日宽表，补齐缺失因子列。"""
    day = str(trade_date).replace("-", "")
    iso = f"{day[:4]}-{day[4:6]}-{day[6:8]}"
    panel = frame.copy()
    if "trading_date" not in panel.columns:
        panel["trading_date"] = iso
    if "instrument_key" not in panel.columns:
        raise ValueError("frame 缺少 instrument_key")
    columns = wide_columns(mvp_only=mvp_only)
    for column in columns:
        if column not in panel.columns:
            panel[column] = float("nan")
    extra = [c for c in panel.columns if c not in columns]
    ordered = columns + extra
    panel = panel[ordered]
    destination = day_path(root, day, factor_version=factor_version)
    temporary = destination.with_suffix(".parquet.tmp")
    panel.to_parquet(temporary, index=False)
    temporary.replace(destination)
    return destination


def read_day(root: Path, trade_date: str, *, factor_version: int = FACTOR_VERSION) -> pd.DataFrame | None:
    path = day_path(root, trade_date, factor_version=factor_version)
    if not path.is_file():
        return None
    return pd.read_parquet(path)


def _r2_configured() -> bool:
    return bool(
        os.getenv("R2_ENDPOINT_URL", "").strip()
        and os.getenv("R2_ACCESS_KEY_ID", "").strip()
        and os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        and _bucket()
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


def upload_day_to_r2(root: Path, trade_date: str, *, factor_version: int = FACTOR_VERSION) -> str | None:
    """上传日宽表到 R2；未配置时仍返回逻辑键。"""
    path = day_path(root, trade_date, factor_version=factor_version)
    if not path.is_file():
        return None
    key = logical_day_key(trade_date, factor_version=factor_version)
    payload = path.read_bytes()
    if _uploader is not None:
        _uploader(key, payload)
        return key
    if not _r2_configured():
        return key
    try:
        _r2_client().put_object(Bucket=_bucket(), Key=key, Body=payload)
    except Exception:
        _log.warning("fund_cn R2 上传失败 %s", trade_date, exc_info=True)
        return key
    return key


# 与 catalog 30 列对齐，防止 dump / 宽表列漂移
assert len(factor_ids()) == 30, len(factor_ids())
