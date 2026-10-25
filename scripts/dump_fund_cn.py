# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""将 fund_cn 日宽表写入现有 qlib cn_data。

只新增 roe.day.bin 等财务因子，不改 calendars / instruments，也不覆盖 OHLCV。
流水线本身不写 bin，训练/评估时再单独跑本脚本。

Example:
    python scripts/dump_fund_cn.py dump \\
        --qlib_dir=~/.qlib/qlib_data/cn_data
"""

from __future__ import annotations

import re
import shutil
import tempfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import fire
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from qlib.contrib.data.fund_cn import FUND_CN_FACTOR_IDS
from qlib.contrib.data.fund_cn.market import instrument_to_qlib
from qlib.utils import code_to_fname

DAY_STEM_RE = re.compile(r"^\d{8}$")
_DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "fundamental_staging" / "factors" / "fund_cn" / "v1"
)
_PROTECTED_FIELDS = frozenset(
    {"open", "close", "high", "low", "volume", "vwap", "factor", "change", "money"}
)


def to_qlib_symbol(instrument_key: str) -> Optional[str]:
    """CNStock:600000 / 600000.SH → SH600000。"""
    return instrument_to_qlib(instrument_key)


def iter_day_files(data_path: Path) -> List[Path]:
    """只收 YYYYMMDD.parquet。"""
    files: List[Path] = []
    for path in sorted(data_path.rglob("*.parquet")):
        if DAY_STEM_RE.fullmatch(path.stem):
            files.append(path)
    return files


def read_calendars(qlib_dir: Path, freq: str = "day") -> List[pd.Timestamp]:
    calendar_path = qlib_dir / "calendars" / f"{freq}.txt"
    if not calendar_path.exists():
        raise FileNotFoundError(f"missing calendar: {calendar_path}")
    return sorted(
        map(
            pd.Timestamp,
            pd.read_csv(calendar_path, header=None).loc[:, 0].tolist(),
        )
    )


def load_day_frame(path: str, factor_ids: Tuple[str, ...]) -> pd.DataFrame:
    """读一日宽表，映射代码/日期，因子列转 float32。"""
    raw = pd.read_parquet(path)
    missing = [fid for fid in factor_ids if fid not in raw.columns]
    for fid in missing:
        raw[fid] = np.nan
    keep = ["trading_date", "instrument_key", *factor_ids]
    df = raw.loc[:, [c for c in keep if c in raw.columns]].copy()
    df["date"] = pd.to_datetime(df["trading_date"].astype(str), errors="coerce")
    df["symbol"] = df["instrument_key"].map(to_qlib_symbol)
    df = df.dropna(subset=["date", "symbol"])
    df = df.drop(columns=["trading_date", "instrument_key"], errors="ignore")
    df[list(factor_ids)] = df[list(factor_ids)].astype("float32")
    return df


def align_to_calendar(df: pd.DataFrame, calendar: Sequence[pd.Timestamp]) -> pd.DataFrame:
    if df.empty:
        return df
    dates = pd.to_datetime(df["date"])
    start, end = dates.min(), dates.max()
    cal = [ts for ts in calendar if start <= ts <= end]
    if not cal:
        return df.iloc[0:0]
    aligned = df.drop_duplicates("date").set_index("date").reindex(cal)
    return aligned


def write_symbol_bins(
    df: pd.DataFrame,
    calendar: Sequence[pd.Timestamp],
    features_root: Path,
    symbol: str,
    fields: Sequence[str],
    freq: str = "day",
) -> int:
    """把一只股票的财务因子列写成 bin；返回写入字段数。"""
    aligned = align_to_calendar(df, calendar)
    if aligned.empty:
        logger.warning(f"{symbol} 日期不在现有日历内，跳过")
        return 0
    date_index = list(calendar).index(aligned.index.min())
    out_dir = features_root / code_to_fname(symbol).lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for field in fields:
        if field in _PROTECTED_FIELDS:
            continue
        if field not in aligned.columns:
            continue
        bin_path = out_dir / f"{field}.{freq}.bin"
        payload = np.hstack([date_index, aligned[field].to_numpy()]).astype("<f")
        payload.tofile(str(bin_path.resolve()))
        written += 1
    return written


def _dump_symbol_job(
    parquet_paths: List[str],
    calendar_values: List[str],
    features_root: str,
    symbol: str,
    fields: Tuple[str, ...],
) -> Tuple[str, int]:
    frames = [pd.read_parquet(path) for path in parquet_paths]
    df = pd.concat(frames, ignore_index=True)
    calendar = [pd.Timestamp(v) for v in calendar_values]
    n = write_symbol_bins(df, calendar, Path(features_root), symbol, fields)
    return symbol, n


class DumpFundCn:
    """fund_cn 日宽表 -> 现有 cn_data 的财务因子 day.bin。"""

    def __init__(
        self,
        data_path: str = "",
        qlib_dir: str = "~/.qlib/qlib_data/cn_data",
        max_workers: int = 8,
        dry_run: bool = False,
        symbols: str = "",
        limit_symbols: int = None,
        freq: str = "day",
    ):
        resolved = Path(data_path).expanduser() if str(data_path).strip() else _DEFAULT_DATA_PATH
        self.data_path = resolved.resolve()
        self.qlib_dir = Path(qlib_dir).expanduser().resolve()
        self.max_workers = max(1, int(max_workers))
        self.dry_run = dry_run
        self.symbol_filter = {s.strip().upper() for s in symbols.split(",") if s.strip()}
        self.limit_symbols = limit_symbols
        self.freq = freq
        self.factor_ids = FUND_CN_FACTOR_IDS
        self.features_dir = self.qlib_dir / "features"

    def dump(self) -> None:
        day_files = iter_day_files(self.data_path)
        if not day_files:
            raise FileNotFoundError(f"no YYYYMMDD.parquet under {self.data_path}")
        calendar = read_calendars(self.qlib_dir, self.freq)
        logger.info(
            f"day files={len(day_files)} calendar={calendar[0].date()}~{calendar[-1].date()} "
            f"factors={len(self.factor_ids)} qlib_dir={self.qlib_dir}"
        )
        if self.dry_run:
            sample = load_day_frame(str(day_files[0]), self.factor_ids)
            dates = [p.stem for p in day_files]
            logger.info(
                f"dry-run dates={dates[0]}~{dates[-1]} files={len(day_files)} "
                f"first_rows={len(sample)}"
            )
            return

        temp_root = Path(tempfile.mkdtemp(prefix="fund_cn_dump_"))
        try:
            symbol_files = self._materialize_by_year(day_files, temp_root)
            if not symbol_files:
                raise RuntimeError("no symbols materialized from fund_cn")
            self._dump_bins(symbol_files, calendar)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def _materialize_by_year(self, day_files: Sequence[Path], temp_root: Path) -> dict:
        by_year = defaultdict(list)
        for path in day_files:
            by_year[path.stem[:4]].append(path)
        symbol_files: dict = defaultdict(list)
        for year, files in sorted(by_year.items()):
            year_dir = temp_root / year
            year_dir.mkdir(parents=True, exist_ok=True)
            frames = [load_day_frame(str(p), self.factor_ids) for p in files]
            if not frames:
                continue
            panel = pd.concat(frames, ignore_index=True)
            if self.symbol_filter:
                panel = panel[panel["symbol"].isin(self.symbol_filter)]
            grouped = panel.groupby("symbol", sort=False)
            symbols = list(grouped.groups)
            if self.limit_symbols is not None:
                symbols = symbols[: int(self.limit_symbols)]
            for symbol in tqdm(symbols, desc=f"write {year}"):
                g = grouped.get_group(symbol)
                out = year_dir / f"{symbol}.parquet"
                g.to_parquet(out, index=False)
                symbol_files[symbol].append(str(out))
        return symbol_files

    def _dump_bins(self, symbol_files: dict, calendar: Sequence[pd.Timestamp]) -> None:
        calendar_values = [ts.strftime("%Y-%m-%d") for ts in calendar]
        jobs = [
            (paths, calendar_values, str(self.features_dir), symbol, self.factor_ids)
            for symbol, paths in sorted(symbol_files.items())
        ]
        if self.max_workers == 1:
            for job in tqdm(jobs, desc="dump bins"):
                _dump_symbol_job(*job)
            return
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = [pool.submit(_dump_symbol_job, *job) for job in jobs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="dump bins"):
                fut.result()


def dump(
    data_path: str = "",
    qlib_dir: str = "~/.qlib/qlib_data/cn_data",
    max_workers: int = 8,
    dry_run: bool = False,
    symbols: str = "",
    limit_symbols: int = None,
    freq: str = "day",
) -> None:
    """CLI 入口：未传 data_path 时用 qlib 财务宽表根。"""
    DumpFundCn(
        data_path=data_path,
        qlib_dir=qlib_dir,
        max_workers=max_workers,
        dry_run=dry_run,
        symbols=symbols,
        limit_symbols=limit_symbols,
        freq=freq,
    ).dump()


if __name__ == "__main__":
    fire.Fire({"dump": dump})
