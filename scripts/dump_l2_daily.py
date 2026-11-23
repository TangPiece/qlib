# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""将 QuantDinger l2_daily 日宽表写入独立 L2 bin 目录，并可合回训练用 cn_data。

只新增 l2_*.day.bin，不改 calendars / instruments，也不覆盖 OHLCV。
日宽表是「一天一个文件、多只股票」，不能直接走 dump_bin.dump_all。

持久化默认写到 ~/.qlib/qlib_data/cn_data_l2（与社区 cn_data 解耦，避免整包更新覆盖）；
日历仍读自 cn_data；训练时再把 l2_* 合入 cn_data/features。

Example:
    python scripts/dump_l2_daily.py dump
    # 默认：日历 cn_data，bin 持久化 cn_data_l2，并 merge 进 cn_data
    # 宽表默认扫 qlib/data/level2_staging/factors/l2_daily/v1
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

from qlib.contrib.data.l2_daily import L2_DAILY_FACTOR_IDS
from qlib.utils import code_to_fname

DAY_STEM_RE = re.compile(r"^\d{8}$")
# 搬家后的宽表根：YYYY/YYYYMM/YYYYMMDD.parquet
_DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "level2_staging" / "factors" / "l2_daily" / "v1"
)
# 行情字段禁止覆盖，dump 只写 l2_*
_PROTECTED_FIELDS = frozenset(
    {"open", "close", "high", "low", "volume", "vwap", "factor", "change", "money"}
)
_EXCHANGES = frozenset({"SH", "SZ"})
# 活跃行情 / 训练 provider（只读日历，并作为 merge 目标）
_DEFAULT_CN_DATA = "~/.qlib/qlib_data/cn_data"
# L2 bin 持久化根：与 cn_data 并列，避免 update 整包替换时丢失
_DEFAULT_L2_QLIB_DIR = "~/.qlib/qlib_data/cn_data_l2"


def to_qlib_symbol(symbol: str) -> Optional[str]:
    """把宽表代码 600000.SH 转成 qlib 的 SH600000；无法识别则返回 None。"""
    text = str(symbol).strip()
    if "." not in text:
        return None
    code, exch = text.rsplit(".", 1)
    exch = exch.upper()
    if exch not in _EXCHANGES or not code.isdigit():
        return None
    return f"{exch}{code}"


def iter_day_files(data_path: Path) -> List[Path]:
    """只收 YYYYMMDD.parquet，跳过 status / 月度 compact / 其它附属文件。"""
    files: List[Path] = []
    for path in sorted(data_path.rglob("*.parquet")):
        if DAY_STEM_RE.fullmatch(path.stem):
            files.append(path)
    return files


def read_calendars(qlib_dir: Path, freq: str = "day") -> List[pd.Timestamp]:
    """读取现有 qlib 日历；bin 起始下标必须相对这份日历，不能重建。"""
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
    keep = ["trade_date", "symbol", *factor_ids]
    df = raw.loc[:, keep].copy()
    df["date"] = pd.to_datetime(df["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    df["symbol"] = df["symbol"].map(to_qlib_symbol)
    df = df.dropna(subset=["date", "symbol"])
    df = df.drop(columns=["trade_date"])
    df[list(factor_ids)] = df[list(factor_ids)].astype("float32")
    return df


def align_to_calendar(df: pd.DataFrame, calendar: Sequence[pd.Timestamp]) -> pd.DataFrame:
    """按现有日历 reindex，缺日填 NaN；与 dump_bin.data_merge_calendar 对齐。"""
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
    """把一只股票的 l2_* 列写成 bin；返回写入字段数。"""
    aligned = align_to_calendar(df, calendar)
    if aligned.empty:
        logger.warning(f"{symbol} 日期不在现有日历内，跳过")
        return 0
    date_index = list(calendar).index(aligned.index.min())
    out_dir = features_root / code_to_fname(symbol).lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for field in fields:
        if field in _PROTECTED_FIELDS or not field.startswith("l2_"):
            continue
        if field not in aligned.columns:
            continue
        bin_path = out_dir / f"{field}.{freq}.bin"
        payload = np.hstack([date_index, aligned[field].to_numpy()]).astype("<f")
        payload.tofile(str(bin_path.resolve()))
        written += 1
    return written


def sync_l2_feature_bins(src_qlib_dir: Path, dst_qlib_dir: Path) -> int:
    """把 src/features/**/l2_*.*.bin 拷到 dst/features（仅 L2，不碰 OHLCV）。

    Returns
    -------
    int
        成功拷贝的文件数。
    """
    src_features = src_qlib_dir / "features"
    dst_features = dst_qlib_dir / "features"
    if not src_features.is_dir():
        logger.warning(f"no L2 features to sync: {src_features}")
        return 0
    dst_features.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in src_features.rglob("l2_*.*.bin"):
        # 相对 features 的路径保持一致，如 sh600000/l2_trade_count.day.bin
        rel = src.relative_to(src_features)
        dst = dst_features / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    logger.info(f"synced {copied} l2_*.bin from {src_qlib_dir} -> {dst_qlib_dir}")
    return copied


def has_l2_feature_bins(qlib_dir: Path) -> bool:
    """判断目录下是否已有任意 l2_*.*.bin。"""
    features = qlib_dir / "features"
    if not features.is_dir():
        return False
    return next(features.rglob("l2_*.*.bin"), None) is not None


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


class DumpL2Daily:
    """l2_daily 日宽表 -> cn_data_l2 持久化，并可 merge 进训练用 cn_data。"""

    def __init__(
        self,
        data_path: str = "",
        qlib_dir: str = _DEFAULT_L2_QLIB_DIR,
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
            l2_daily/v1 根目录（YYYY/YYYYMM/YYYYMMDD.parquet）
        qlib_dir : str
            L2 bin 持久化目录（默认 cn_data_l2），与社区 cn_data 分离
        calendar_dir : str
            读取 calendars/day.txt 的目录（默认 cn_data）；索引必须对齐活跃行情
        merge_into : str
            额外把 l2_* 同步进训练目录（默认 cn_data）；传空字符串则跳过
        max_workers : int
            读日文件 / 写 bin 的进程数
        dry_run : bool
            只统计日期、股票、列，不写 bin
        symbols : str
            逗号分隔的 qlib 代码过滤，如 SH600000,SZ000001
        limit_symbols : int
            调试用，只转前 N 只
        freq : str
            频率后缀，默认 day
        """
        # 未显式传入时用搬家后的 qlib 宽表根
        resolved = Path(data_path).expanduser() if str(data_path).strip() else _DEFAULT_DATA_PATH
        self.data_path = resolved.resolve()
        self.qlib_dir = Path(qlib_dir).expanduser().resolve()
        self.calendar_dir = Path(calendar_dir).expanduser().resolve()
        # 空字符串表示不同步；否则展开为绝对路径
        merge_raw = str(merge_into).strip() if merge_into is not None else ""
        self.merge_into: Optional[Path] = (
            Path(merge_raw).expanduser().resolve() if merge_raw else None
        )
        self.max_workers = max(1, int(max_workers))
        self.dry_run = dry_run
        self.symbol_filter = {s.strip().upper() for s in symbols.split(",") if s.strip()}
        self.limit_symbols = limit_symbols
        self.freq = freq
        self.factor_ids = L2_DAILY_FACTOR_IDS
        self.features_dir = self.qlib_dir / "features"

    def dump(self) -> None:
        """扫描日宽表并写入 l2_* bin，必要时合回训练目录。"""
        day_files = iter_day_files(self.data_path)
        if not day_files:
            raise FileNotFoundError(f"no YYYYMMDD.parquet under {self.data_path}")
        # 日历始终来自活跃行情目录，避免 cn_data_l2 无 calendars
        calendar = read_calendars(self.calendar_dir, self.freq)
        logger.info(
            f"day files={len(day_files)} calendar={calendar[0].date()}~{calendar[-1].date()} "
            f"factors={len(self.factor_ids)} qlib_dir={self.qlib_dir} "
            f"calendar_dir={self.calendar_dir} merge_into={self.merge_into}"
        )

        if self.dry_run:
            self._dry_run(day_files)
            return

        self.features_dir.mkdir(parents=True, exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(prefix="l2_daily_dump_"))
        try:
            symbol_files = self._materialize_by_year(day_files, temp_root)
            if not symbol_files:
                raise RuntimeError("no symbols materialized from l2_daily")
            self._dump_bins(symbol_files, calendar)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
            logger.info("removed temp parquet dir")

        # 训练仍用单一 provider_uri=cn_data，故把持久化 L2 合回
        if self.merge_into is not None:
            if self.merge_into.resolve() == self.qlib_dir.resolve():
                logger.info("merge_into == qlib_dir；跳过二次同步")
            else:
                sync_l2_feature_bins(self.qlib_dir, self.merge_into)

    def _dry_run(self, day_files: Sequence[Path]) -> None:
        """扫描覆盖范围，不写任何 bin。"""
        sample = load_day_frame(str(day_files[0]), self.factor_ids)
        last = load_day_frame(str(day_files[-1]), self.factor_ids)
        dates = [p.stem for p in day_files]
        logger.info(
            f"dry-run dates={dates[0]}~{dates[-1]} files={len(day_files)} "
            f"first_rows={len(sample)} last_rows={len(last)} "
            f"sample_symbol={sample['symbol'].iloc[0] if not sample.empty else None}"
        )

    def _materialize_by_year(self, day_files: Sequence[Path], temp_root: Path) -> dict:
        """按年读入日宽表，写成 temp/YYYY/SH600000.parquet，控制峰值内存。"""
        by_year = defaultdict(list)
        for path in day_files:
            by_year[path.stem[:4]].append(path)

        symbol_files: dict = defaultdict(list)
        for year, files in sorted(by_year.items()):
            year_dir = temp_root / year
            year_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"materialize {year} files={len(files)}")
            frames = self._load_year_frames(files)
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
            del panel, frames, grouped
        return symbol_files

    def _load_year_frames(self, files: Sequence[Path]) -> List[pd.DataFrame]:
        """并行读取一年内的日宽表。"""
        paths = [str(p) for p in files]
        if self.max_workers == 1:
            return [load_day_frame(p, self.factor_ids) for p in tqdm(paths, desc="read days")]
        frames: List[pd.DataFrame] = []
        # 线程池避免 macOS spawn 再进 fire.Fire；pyarrow 读 parquet 会释 GIL
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for df in tqdm(
                pool.map(load_day_frame, paths, [self.factor_ids] * len(paths)),
                total=len(paths),
                desc="read days",
            ):
                frames.append(df)
        return frames

    def _dump_bins(self, symbol_files: dict, calendar: Sequence[pd.Timestamp]) -> None:
        """按股票写 l2_*.day.bin 到持久化目录。"""
        calendar_values = [ts.strftime("%Y-%m-%d") for ts in calendar]
        jobs = [
            (paths, calendar_values, str(self.features_dir), symbol, self.factor_ids)
            for symbol, paths in sorted(symbol_files.items())
        ]
        logger.info(f"dump bins symbols={len(jobs)} fields={len(self.factor_ids)}")
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
    qlib_dir: str = _DEFAULT_L2_QLIB_DIR,
    calendar_dir: str = _DEFAULT_CN_DATA,
    merge_into: str = _DEFAULT_CN_DATA,
    max_workers: int = 8,
    dry_run: bool = False,
    symbols: str = "",
    limit_symbols: int = None,
    freq: str = "day",
) -> None:
    """CLI 入口：实例化 DumpL2Daily 并执行。未传 data_path 时用 qlib 宽表根。"""
    DumpL2Daily(
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
