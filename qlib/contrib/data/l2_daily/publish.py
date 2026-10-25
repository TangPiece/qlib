"""质检通过后写正式宽表并上传 R2。不写 qlib bin。"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from . import storage
from .catalog import FACTOR_VERSION

_log = logging.getLogger(__name__)


def publish_day(
    output_root: Path,
    trade_date: str,
    frame: pd.DataFrame,
    *,
    upload_r2: bool = True,
    factor_version: int = FACTOR_VERSION,
) -> dict[str, object]:
    """覆盖正式 parquet，可选上传 R2。"""
    path = storage.write_day(output_root, trade_date, frame, factor_version=factor_version)
    uri = None
    if upload_r2:
        try:
            uri = storage.upload_day_to_r2(output_root, trade_date, factor_version=factor_version)
            if uri:
                print(f"[{trade_date}] uploaded {uri}", flush=True)
        except Exception:
            _log.warning("l2_daily 日终上传失败 %s", trade_date, exc_info=True)
            raise
    return {"path": str(path), "uri": uri}
