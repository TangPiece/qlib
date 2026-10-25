"""解压准备与日期发现。保证当日 CSV 目录可用。"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

from .extract import extract_archive

# 因子实际读取的三类 CSV。
DETAIL_TYPES = ("行情", "逐笔成交", "逐笔委托")
DEFAULT_DOWNLOAD_POLL_SEC = 60


def download_poll_seconds() -> int:
    """下载等待间隔，秒。LEVEL2_DOWNLOAD_POLL_SEC 可改。"""
    raw = os.getenv("LEVEL2_DOWNLOAD_POLL_SEC", str(DEFAULT_DOWNLOAD_POLL_SEC)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_DOWNLOAD_POLL_SEC


def _date_prefix(name: str) -> str | None:
    """文件名开头的 YYYYMMDD。"""
    stem = str(name).split(".")[0]
    if stem.isdigit() and len(stem) == 8:
        return stem
    return None


def _is_downloading_name(name: str) -> bool:
    """百度云/aria2/part 未完成标记。完整 .7z 不算。"""
    lower = str(name).lower()
    if lower.endswith(".7z") and lower.count(".") == 1:
        return False
    return "downloading" in lower or lower.endswith(".aria2") or lower.endswith(".part")


def discover_downloading_dates(raw_root: Path, only: list[str] | None = None) -> list[str]:
    """扫描未下完的日期文件，例如 YYYYMMDD.7z.baiduyun.p.downloading。"""
    root = Path(raw_root)
    dates: list[str] = []
    if root.is_dir():
        for path in root.iterdir():
            if not path.is_file() or not _is_downloading_name(path.name):
                continue
            day = _date_prefix(path.name)
            if day:
                dates.append(day)
    dates = sorted(set(dates))
    if only:
        wanted = set(only)
        dates = [day for day in dates if day in wanted]
    return dates


def _qlib_root() -> Path:
    """qlib 仓库根：contrib/data/l2_daily/prepare.py 往上四级。"""
    return Path(__file__).resolve().parents[4]


def default_raw_root() -> Path:
    raw = os.getenv("LEVEL2_RAW_DIR", "").strip()
    if raw:
        return Path(raw)
    return _qlib_root() / "data" / "level2_raw"


def default_output_root() -> Path:
    raw = os.getenv("LEVEL2_L2_DAILY_ROOT", "").strip()
    if raw:
        path = Path(raw)
        path.mkdir(parents=True, exist_ok=True)
        return path
    path = _qlib_root() / "data" / "level2_staging" / "factors" / "l2_daily"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_qa_dir() -> Path:
    path = _qlib_root() / "examples" / "benchmarks" / "LightGBM" / "eval_output"
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_local_dates(raw_root: Path, only: list[str] | None = None) -> list[str]:
    """扫描已解压目录 raw_root/{date}/。"""
    root = Path(raw_root)
    if not root.is_dir():
        return []
    dates = sorted(p.name for p in root.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8)
    if only:
        wanted = set(only)
        dates = [d for d in dates if d in wanted]
    return dates


def discover_7z_dates(raw_root: Path, only: list[str] | None = None) -> list[str]:
    """扫描 raw_root/*.7z。"""
    root = Path(raw_root)
    dates: list[str] = []
    if root.is_dir():
        dates = sorted(p.stem for p in root.glob("*.7z") if p.stem.isdigit() and len(p.stem) == 8)
    if only:
        wanted = set(only)
        dates = [d for d in dates if d in wanted]
    return dates


def day_has_sources(raw_root: Path, trade_date: str) -> bool:
    """有因子用到的三类 CSV 即视为已解压。"""
    day = Path(raw_root) / str(trade_date)
    if not day.is_dir():
        return False
    for child in day.iterdir():
        if not child.is_dir():
            continue
        if any((child / f"{ftype}.csv").is_file() for ftype in DETAIL_TYPES):
            return True
    return False


def day_has_csv(raw_root: Path, trade_date: str) -> bool:
    """兼容旧名：只认 CSV。"""
    return day_has_sources(raw_root, trade_date)


def ensure_day_csv(trade_date: str, raw_root: Path, *, dry_run: bool = False) -> Path:
    """保证 {raw_root}/{date}/ 下有 CSV。已有则跳过；否则解压 {date}.7z。"""
    root = Path(raw_root)
    day_dir = root / str(trade_date)
    if day_has_sources(root, trade_date):
        return day_dir
    seven_z = root / f"{trade_date}.7z"
    if not seven_z.is_file():
        raise FileNotFoundError(f"无已解压 CSV 且找不到归档: {seven_z}")
    if dry_run:
        print(f"[{trade_date}] dry-run：将解压 {seven_z.name}", flush=True)
        return day_dir
    return extract_archive(seven_z, dest_root=root)


def delete_day_sources(
    trade_date: str,
    raw_root: Path,
    *,
    keep_archive: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """回收当日原料。默认删解压目录；keep_archive=False 时连 .7z 一起删。"""
    deleted: list[str] = []
    root = Path(raw_root)
    day_dir = root / str(trade_date)
    seven_z = root / f"{trade_date}.7z"
    if day_dir.is_dir():
        if not dry_run:
            shutil.rmtree(day_dir)
        deleted.append(str(day_dir))
    # 质检失败只留压缩包，方便以后重算
    if not keep_archive and seven_z.is_file():
        if not dry_run:
            seven_z.unlink()
        deleted.append(str(seven_z))
    return deleted


def _pid_running(pid_file: Path) -> bool:
    """pid 文件存在且进程仍在。"""
    if not pid_file.is_file():
        return False
    text = pid_file.read_text(encoding="utf-8").strip()
    if not text.isdigit():
        return False
    try:
        os.kill(int(text), 0)
    except OSError:
        return False
    return True


def detect_in_progress_dates(raw_root: Path, output_root: Path | None = None) -> set[str]:
    """正在算的日期：单日 pid，或 marker 未齐 / 刚写完尚未出 QA。"""
    from .catalog import ALL_GROUPS

    out = Path(output_root) if output_root is not None else default_output_root()
    logs = _qlib_root() / "data" / "level2_staging" / "logs"
    active: set[str] = set()
    if logs.is_dir():
        for pid_file in logs.glob("l2_daily_*.pid"):
            key = pid_file.stem[len("l2_daily_") :]
            if key.isdigit() and len(key) == 8 and _pid_running(pid_file):
                active.add(key)
    all_running = _pid_running(logs / "l2_daily_all.pid")
    qa_dir = default_qa_dir()
    now = time.time()
    # 6 小时内改过的 marker 视为本轮还在跑
    recent_sec = 6 * 3600
    for day in discover_local_dates(raw_root):
        marker_paths = [out / "_markers" / f"{day}_{group}.json" for group in ALL_GROUPS]
        existing = [path for path in marker_paths if path.is_file()]
        if not existing:
            continue
        newest = max(path.stat().st_mtime for path in existing)
        recent = (now - newest) < recent_sec
        qa_path = qa_dir / f"l2_daily_day_qa_{day}.json"
        # 本轮还在写 marker / 尚未出 QA 才算进行中，隔夜残留半套 marker 不当成在跑
        if 0 < len(existing) < len(ALL_GROUPS) and recent:
            active.add(day)
        elif len(existing) == len(ALL_GROUPS) and not qa_path.is_file() and (all_running or recent):
            active.add(day)
    return active


def qa_passed(trade_date: str) -> bool:
    """当日质检 JSON 为 ok。"""
    path = default_qa_dir() / f"l2_daily_day_qa_{trade_date}.json"
    if not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return bool(payload.get("ok"))


def prune_finished_sources(
    raw_root: Path | None = None,
    output_root: Path | None = None,
    *,
    skip_dates: list[str] | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    """清已结束日的解压目录；QA 通过的连 .7z 一起删。跳过正在算的日。"""
    root = Path(raw_root) if raw_root is not None else default_raw_root()
    out = Path(output_root) if output_root is not None else default_output_root()
    skip = set(skip_dates or ())
    skip |= detect_in_progress_dates(root, out)
    deleted: list[str] = []
    kept_archive: list[str] = []
    skipped: list[str] = []
    days = sorted(set(discover_local_dates(root)) | set(discover_7z_dates(root)))
    for day in days:
        day_dir = root / day
        if day in skip:
            skipped.append(day)
            continue
        if not day_dir.is_dir():
            continue
        keep_archive = not qa_passed(day)
        removed = delete_day_sources(day, root, keep_archive=keep_archive, dry_run=dry_run)
        deleted.extend(removed)
        if keep_archive:
            kept_archive.append(day)
    return {
        "deleted": deleted,
        "kept_archive": kept_archive,
        "skipped": sorted(skip),
        "dry_run": dry_run,
    }
