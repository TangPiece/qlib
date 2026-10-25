# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""质检通过后写正式宽表并上传 R2。不写 qlib bin，不登记 Factor Lab。"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from . import storage
from .catalog import FACTOR_VERSION
from .paths_local import default_factor_root

_log = logging.getLogger(__name__)


def publish_day(
    output_root: Path | None,
    trade_date: str,
    frame: pd.DataFrame,
    *,
    upload_r2: bool = True,
    mvp_only: bool = True,
    factor_version: int = FACTOR_VERSION,
) -> dict[str, Any]:
    """覆盖正式 parquet，可选上传 R2。"""
    root = output_root or default_factor_root()
    path = storage.write_day(
        root,
        trade_date,
        frame,
        mvp_only=mvp_only,
        factor_version=factor_version,
    )
    uri = None
    if upload_r2:
        try:
            uri = storage.upload_day_to_r2(root, trade_date, factor_version=factor_version)
            if uri:
                print(f"[{trade_date}] uploaded {uri}", flush=True)
        except Exception:
            _log.warning("fund_cn 日终上传失败 %s", trade_date, exc_info=True)
            raise
    return {"path": str(path), "uri": uri, "row_count": int(len(frame))}
