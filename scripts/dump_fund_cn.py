# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""将 fund_cn 日宽表写入独立 cn_data_fund，并可合回训练用 cn_data。

只新增 roe.day.bin 等财务因子，不改 calendars / instruments，也不覆盖 OHLCV。
持久化默认写到 ~/.qlib/qlib_data/cn_data_fund（与社区 cn_data 解耦，避免整包更新覆盖）；
日历仍读自 cn_data；训练时再把财务因子合入 cn_data/features。

Example:
    # 默认：日历 cn_data，bin 持久化 cn_data_fund，并 merge 进 cn_data
    python scripts/dump_fund_cn.py dump
    python scripts/dump_fund_cn.py dump \\
        --qlib_dir=~/.qlib/qlib_data/cn_data_fund \\
        --calendar_dir=~/.qlib/qlib_data/cn_data \\
        --merge_into=~/.qlib/qlib_data/cn_data
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
    Path(__file__).resolve().parents[1]
    / "data"
    / "fundamental_staging"
    / "factors"
    / "fund_cn"
    / "v1"
)
# 活跃行情 / 训练 provider（只读日历，并作为 merge 目标）
_DEFAULT_CN_DATA = "~/.qlib/qlib_data/cn_data"
# 财务因子 bin 持久化根：与 cn_data 并列，避免 update 整包替换时丢失
_DEFAULT_FUND_QLIB_DIR = "~/.qlib/qlib_data/cn_data_fund"
_PROTECTED_FIELDS = frozenset(
    {"open", "close", "high", "low", "volume", "vwap", "factor", "change", "money", "amount", "adjclose"}
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
    """从指定 qlib 根读取交易日历。"""
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
    """按现有日历 reindex，缺日填 NaN。"""
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


def sync_fund_feature_bins(
    src_qlib_dir: Path,
    dst_qlib_dir: Path,
    factor_ids: Sequence[str] = FUND_CN_FACTOR_IDS,
    freq: str = "day",
) -> int:
    """把 src/features/**/{fid}.day.bin 拷到 dst/features（仅财务因子，不碰 OHLCV/L2）。

    Returns
    -------
    int
        成功拷贝的文件数。
    """
    src_features = src_qlib_dir / "features"
    dst_features = dst_qlib_dir / "features"
    if not src_features.is_dir():
        logger.warning(f"no fund features to sync: {src_features}")
        return 0
    dst_features.mkdir(parents=True, exist_ok=True)
    allow = {f"{fid}.{freq}.bin" for fid in factor_ids if fid not in _PROTECTED_FIELDS}
    copied = 0
    for src in src_features.rglob("*.*.bin"):
        if src.name not in allow:
            continue
        rel = src.relative_to(src_features)
        dst = dst_features / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    logger.info(f"synced {copied} fund_*.bin from {src_qlib_dir} -> {dst_qlib_dir}")
    return copied


def has_fund_feature_bins(
    qlib_dir: Path,
    factor_ids: Sequence[str] = FUND_CN_FACTOR_IDS,
    freq: str = "day",
) -> bool:
    """判断目录下是否已有任意财务因子 day.bin。"""
    features = qlib_dir / "features"
    if not features.is_dir():
        return False
    for fid in factor_ids:
        if fid in _PROTECTED_FIELDS:
            continue
        if next(features.rglob(f"{fid}.{freq}.bin"), None) is not None:
            return True
    return False


def _dump_symbol_job(
    parquet_paths: List[str],
    calendar_values: List[str],
    features_root: str,
    symbol: str,
    fields: Tuple[str, ...],
) -> Tuple[str, int]:
    """子进程：拼接一只股票的临时 parquet 并写 bin。"""
    frames = [pd.read_parquet(path) for path in parquet_paths]
    df = pd.concat(frames, ignore_index=True)
    calendar = [pd.Timestamp(v) for v in calendar_values]
    n = write_symbol_bins(df, calendar, Path(features_root), symbol, fields)
    return symbol, n


class DumpFundCn:
    """fund_cn 日宽表 -> cn_data_fund 持久化，并可 merge 进训练用 cn_data。"""

    def __init__(
        self,
        data_path: str = "",
        qlib_dir: str = _DEFAULT_FUND_QLIB_DIR,
        calendar_dir: str = _DEFAULT_CN_DATA,
        merge_into: str = _DEFAULT_CN_DATA,
        max_workers: int = 8,
        dry_run: bool = False,
        symbols: str = "",
        limit_symbols: int = None,
        freq: str = "day",
    ):
        """
        Parameters
        ----------
        data_path : str
            fund_cn/v1 根目录（YYYY/YYYYMM/YYYYMMDD.parquet）
        qlib_dir : str
            财务 bin 持久化目录（默认 cn_data_fund），与社区 cn_data 分离
        calendar_dir : str
            读取 calendars/day.txt 的目录（默认 cn_data）；索引必须对齐活跃行情
        merge_into : str
            额外把财务因子同步进训练目录（默认 cn_data）；传空字符串则跳过
        """
        resolved = Path(data_path).expanduser() if str(data_path).strip() else _DEFAULT_DATA_PATH
        self.data_path = resolved.resolve()
        self.qlib_dir = Path(qlib_dir).expanduser().resolve()
        self.calendar_dir = Path(calendar_dir).expanduser().resolve()
        merge_raw = str(merge_into).strip() if merge_into is not None else ""
        self.merge_into: Optional[Path] = (
            Path(merge_raw).expanduser().resolve() if merge_raw else None
        )
        self.max_workers = max(1, int(max_workers))
        self.dry_run = dry_run
        self.symbol_filter = {s.strip().upper() for s in symbols.split(",") if s.strip()}
        self.limit_symbols = limit_symbols
        self.freq = freq
        self.factor_ids = FUND_CN_FACTOR_IDS
        self.features_dir = self.qlib_dir / "features"

    def dump(self) -> None:
        """扫描日宽表并写入财务 bin，必要时合回训练目录。"""
        day_files = iter_day_files(self.data_path)
        if not day_files:
            raise FileNotFoundError(f"no YYYYMMDD.parquet under {self.data_path}")
        # 日历始终来自活跃行情目录，避免 cn_data_fund 无 calendars
        calendar = read_calendars(self.calendar_dir, self.freq)
        logger.info(
            f"day files={len(day_files)} calendar={calendar[0].date()}~{calendar[-1].date()} "
            f"factors={len(self.factor_ids)} qlib_dir={self.qlib_dir} "
            f"calendar_dir={self.calendar_dir} merge_into={self.merge_into}"
        )
        if self.dry_run:
            sample = load_day_frame(str(day_files[0]), self.factor_ids)
            dates = [p.stem for p in day_files]
            logger.info(
                f"dry-run dates={dates[0]}~{dates[-1]} files={len(day_files)} "
                f"first_rows={len(sample)}"
            )
            return

        self.features_dir.mkdir(parents=True, exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(prefix="fund_cn_dump_"))
        try:
            symbol_files = self._materialize_by_year(day_files, temp_root)
            if not symbol_files:
                raise RuntimeError("no symbols materialized from fund_cn")
            self._dump_bins(symbol_files, calendar)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        # 训练仍用单一 provider_uri=cn_data，故把持久化财务因子合回
        if self.merge_into is not None:
            if self.merge_into.resolve() == self.qlib_dir.resolve():
                logger.info("merge_into == qlib_dir；跳过二次同步")
            else:
                sync_fund_feature_bins(
                    self.qlib_dir, self.merge_into, self.factor_ids, self.freq
                )

    def _materialize_by_year(self, day_files: Sequence[Path], temp_root: Path) -> dict:
        """按年物化临时按股票 parquet，供并行写 bin。"""
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
        """并行把各股票临时表写成 day.bin。"""
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
    qlib_dir: str = _DEFAULT_FUND_QLIB_DIR,
    calendar_dir: str = _DEFAULT_CN_DATA,
    merge_into: str = _DEFAULT_CN_DATA,
    max_workers: int = 8,
    dry_run: bool = False,
    symbols: str = "",
    limit_symbols: int = None,
    freq: str = "day",
) -> None:
    """CLI 入口：默认写 cn_data_fund 并 merge 进 cn_data。"""
    DumpFundCn(
        data_path=data_path,
        qlib_dir=qlib_dir,
        calendar_dir=calendar_dir,
        merge_into=merge_into,
        max_workers=max_workers,
        dry_run=dry_run,
        symbols=symbols,
        limit_symbols=limit_symbols,
        freq=freq,
    ).dump()


if __name__ == "__main__":
    fire.Fire({"dump": dump})
