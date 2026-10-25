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
    """覆盖正式 parquet，可选上传 R2。上传失败记账，不抛错。"""
    path = storage.write_day(output_root, trade_date, frame, factor_version=factor_version)
    uri = None
    r2_failed = False
    r2_error = None
    if upload_r2:
        try:
            uri = storage.upload_day_to_r2(output_root, trade_date, factor_version=factor_version)
            # 只有真正 put / 测试注入成功才打 uploaded；未配置返回 None。
            if uri:
                print(f"[{trade_date}] uploaded {uri}", flush=True)
                storage.clear_r2_failed(output_root, trade_date)
        except Exception as exc:
            # 本地已写好，失败只记账，让后续日期继续算。
            _log.warning("l2_daily 日终上传失败 %s", trade_date, exc_info=True)
            storage.mark_r2_failed(output_root, trade_date, exc, factor_version=factor_version)
            r2_failed = True
            r2_error = str(exc)
    return {"path": str(path), "uri": uri, "r2_failed": r2_failed, "r2_error": r2_error}
