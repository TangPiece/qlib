# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""将 alloc_v2 行业因子宽表写入可选 qlib day.bin（仿 dump_fund_cn）。

默认不覆盖 OHLCV；只追加 IndustryAllocationScore 等字段。
流水线本身不写 bin，评测时再跑本脚本。

Example:
    python scripts/dump_alloc_v2.py dump \\
        --qlib_dir=~/.qlib/qlib_data/cn_data
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import fire
import numpy as np
import pandas as pd
from loguru import logger

from qlib.contrib.data.alloc_v2.paths import default_root
from qlib.contrib.data.alloc_v2.store import read_factor
from qlib.utils import code_to_fname

DAY_STEM_RE = re.compile(r"^\d{8}$")
_PROTECTED = frozenset({"open", "close", "high", "low", "volume", "vwap", "factor", "change", "money"})


def factor_to_daily_wide(factors: pd.DataFrame, factor_id: str) -> pd.DataFrame:
    part = factors[factors["factor_id"] == factor_id].copy()
    if part.empty:
        return pd.DataFrame()
    part["event_date"] = pd.to_datetime(part["event_date"])
    part["date"] = part["event_date"].dt.strftime("%Y%m%d")
    return part.pivot_table(index="date", columns="entity_id", values="value", aggfunc="last")


def dump(
    qlib_dir: str = "~/.qlib/qlib_data/cn_data",
    alloc_root: str = "",
    factor_ids: Optional[str] = "IndustryAllocationScore,IndustryETFFlow20",
) -> None:
    """把指定因子写成 features/<code>/<factor>.day.bin。实体需已是 qlib 股票代码。"""
    root = Path(alloc_root) if alloc_root else default_root()
    factors = read_factor("all", root=root)
    if factors.empty:
        raise SystemExit(f"no factors under {root}")
    qdir = Path(qlib_dir).expanduser()
    feature_dir = qdir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    ids = [x.strip() for x in (factor_ids or "").split(",") if x.strip()]
    for factor_id in ids:
        if factor_id in _PROTECTED:
            logger.warning("skip protected field name {}", factor_id)
            continue
        wide = factor_to_daily_wide(factors, factor_id)
        if wide.empty:
            logger.warning("factor {} empty", factor_id)
            continue
        # 行业指数实体通常不是股票代码：仅当 entity 形如 6 位或 SH/SZ 时写入
        written = 0
        for entity in wide.columns:
            code = str(entity)
            if not (code.isdigit() and len(code) == 6) and not re.match(r"^(SH|SZ)\d{6}$", code, re.I):
                continue
            fname = code_to_fname(code if code.upper().startswith(("SH", "SZ")) else f"SH{code}")
            dest_dir = feature_dir / fname
            dest_dir.mkdir(parents=True, exist_ok=True)
            series = wide[entity].dropna().sort_index()
            arr = series.to_numpy(dtype=np.float32)
            # 简化：写连续数组；正式对齐日历请用 dump_fund_cn 同类逻辑
            (dest_dir / f"{factor_id.lower()}.day.bin").write_bytes(arr.tobytes())
            written += 1
        logger.info("factor {} written instruments={}", factor_id, written)


if __name__ == "__main__":
    fire.Fire({"dump": dump})
