#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Daily (manual) refresh of crowd-source cn_data + rolling yaml end dates.

Downloads the latest qlib_bin.tar.gz from chenditc/investment_data when local
data is behind that release, replaces ~/.qlib/qlib_data/cn_data, then syncs
**end_time** fields in rolling configs to **today's date** (local machine date),
not the calendar last trading day.

investment_data only ships a full archive (~540MB); there is no incremental
day pack. This script skips re-download when the local release marker (or
calendar end) already matches / covers the remote latest tag.

Never rewrites start_time, fit_start_time, or train/valid segment endpoints.

Usage:
    python examples/benchmarks/LightGBM/update_cn_data_daily.py
    python examples/benchmarks/LightGBM/update_cn_data_daily.py --dry-run
    python examples/benchmarks/LightGBM/update_cn_data_daily.py --skip-download
    python examples/benchmarks/LightGBM/update_cn_data_daily.py --force
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

DIRNAME = Path(__file__).absolute().resolve().parent
DEFAULT_QLIB_DIR = Path("~/.qlib/qlib_data/cn_data").expanduser()
DEFAULT_CONFIGS = [
    DIRNAME / "workflow_config_lightgbm_Alpha158_rolling_7y1y.yaml",
    # 20 日预测配置的结束日与隔日配置一起写成当天
    DIRNAME / "workflow_config_lightgbm_Alpha158_rolling_7y1y_1m.yaml",
    DIRNAME / "workflow_config_lightgbm_Alpha158_rolling_7y1y_ind_weight.yaml",
]
DOWNLOAD_URL = (
    "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
)
RELEASE_MARKER_NAME = ".investment_data_release"
_DATE_TAG_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
# 大文件下载易中断：默认重试次数与分块大小
DOWNLOAD_RETRIES = 8
DOWNLOAD_CHUNK_SIZE = 1 << 20  # 1 MiB
DOWNLOAD_RETRY_SLEEP_SEC = 3.0


def log(msg: str) -> None:
    print(msg, flush=True)


def _download_cache_path() -> Path:
    """固定缓存路径，脚本重跑时可断点续传（不放在临时目录里）。"""
    cache_dir = Path(tempfile.gettempdir()) / "qlib_bin_download"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / "qlib_bin.tar.gz"


def _probe_remote_size(url: str) -> Optional[int]:
    """HEAD 探测 Content-Length；失败则返回 None。"""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            length = resp.headers.get("Content-Length")
            if length and length.isdigit():
                return int(length)
    except Exception as exc:  # noqa: BLE001
        log(f"Warning: HEAD size probe failed ({exc})")
    return None


def _print_progress(downloaded: int, total_size: int, last_pct: List[int]) -> None:
    """按整数百分比刷新进度条，避免刷屏。"""
    if total_size <= 0:
        mb = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  downloaded {mb:.1f} MB")
        sys.stdout.flush()
        return
    pct = int(min(downloaded, total_size) * 100 / total_size)
    if pct == last_pct[0] and downloaded < total_size:
        return
    last_pct[0] = pct
    mb = downloaded / (1024 * 1024)
    total_mb = total_size / (1024 * 1024)
    sys.stdout.write(f"\r  {pct:3d}%  ({mb:.1f}/{total_mb:.1f} MB)")
    sys.stdout.flush()
    if downloaded >= total_size:
        sys.stdout.write("\n")


def download_archive(
    url: str,
    dest: Path,
    *,
    retries: int = DOWNLOAD_RETRIES,
) -> None:
    """下载大文件，支持 Range 断点续传与自动重试。

    GitHub release 约 500MB+，``urlretrieve`` 中途断连会抛 ContentTooShortError；
    本函数在同一 dest 上续传，并在多次重试后仍校验最终大小。
    """
    log(f"Downloading: {url}")
    log(f"        to: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    remote_size = _probe_remote_size(url)
    last_pct = [-1]
    attempt = 0

    while attempt < retries:
        attempt += 1
        existing = dest.stat().st_size if dest.exists() else 0

        # 本地已完整则直接复用
        if remote_size is not None and existing == remote_size:
            log(f"Cache hit: {dest} already {existing / (1024 * 1024):.1f} MB")
            return
        if remote_size is not None and existing > remote_size:
            log(f"Local file larger than remote ({existing} > {remote_size}); truncating")
            dest.unlink(missing_ok=True)
            existing = 0

        headers = {}
        mode = "wb"
        if existing > 0:
            headers["Range"] = f"bytes={existing}-"
            mode = "ab"
            log(f"Resume from {existing / (1024 * 1024):.1f} MB (attempt {attempt}/{retries})")
        else:
            log(f"Start download (attempt {attempt}/{retries})")

        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                # 不支持 Range 时服务器会返回 200 全量，需改回覆盖写
                status = getattr(resp, "status", None) or resp.getcode()
                if existing > 0 and status == 200:
                    log("Server ignored Range; restarting from scratch")
                    dest.unlink(missing_ok=True)
                    existing = 0
                    mode = "wb"
                    last_pct[0] = -1

                # 从响应头更新期望总大小
                total_size = remote_size
                if status == 206:
                    content_range = resp.headers.get("Content-Range", "")
                    m = re.search(r"/(\d+)\s*$", content_range)
                    if m:
                        total_size = int(m.group(1))
                        remote_size = total_size
                elif total_size is None:
                    cl = resp.headers.get("Content-Length")
                    if cl and cl.isdigit():
                        total_size = int(cl) if existing == 0 else existing + int(cl)
                        remote_size = total_size

                downloaded = existing
                with open(dest, mode) as out:
                    while True:
                        chunk = resp.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        _print_progress(downloaded, total_size or 0, last_pct)

            final_size = dest.stat().st_size
            if remote_size is not None and final_size < remote_size:
                raise urllib.error.ContentTooShortError(
                    f"retrieval incomplete: got only {final_size} out of {remote_size} bytes",
                    None,
                )
            if last_pct[0] < 100 and remote_size:
                _print_progress(final_size, remote_size, last_pct)
            elif last_pct[0] >= 0 and not (remote_size and final_size >= remote_size):
                sys.stdout.write("\n")
            size_mb = final_size / (1024 * 1024)
            log(f"Download complete: {size_mb:.1f} MB")
            return

        except (urllib.error.URLError, urllib.error.ContentTooShortError, TimeoutError, OSError) as exc:
            sys.stdout.write("\n")
            partial = dest.stat().st_size if dest.exists() else 0
            log(f"Download interrupted ({exc}); kept {partial / (1024 * 1024):.1f} MB partial")
            if attempt >= retries:
                raise
            sleep_s = DOWNLOAD_RETRY_SLEEP_SEC * attempt
            log(f"Retrying in {sleep_s:.0f}s ...")
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to download after {retries} attempts: {url}")


def archive_strip_components(members: Iterable[tarfile.TarInfo]) -> int:
    """Detect how many leading path components to strip so calendars/ is at root."""
    names = [m.name for m in members if m.name and not m.name.endswith("/")]
    if not names:
        raise RuntimeError("Archive is empty")

    def has_calendar_at_depth(depth: int) -> bool:
        prefix = "calendars/day.txt"
        for name in names:
            parts = Path(name).parts
            if len(parts) <= depth:
                continue
            rel = "/".join(parts[depth:])
            if rel == prefix or rel.startswith("calendars/"):
                return True
        return False

    for depth in (0, 1, 2):
        if has_calendar_at_depth(depth):
            return depth
    raise RuntimeError(
        "Cannot find calendars/ inside archive; unexpected qlib_bin layout"
    )


def extract_qlib_bin(archive: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        strip = archive_strip_components(members)
        log(f"Extracting with strip_components={strip} -> {target_dir}")

        for member in members:
            parts = Path(member.name).parts
            if len(parts) <= strip:
                continue
            member.name = "/".join(parts[strip:])
            if not member.name or member.name.startswith("/"):
                continue
            tar.extract(member, path=target_dir)


def backup_qlib_dir(qlib_dir: Path) -> Optional[Path]:
    if not qlib_dir.exists():
        log(f"No existing data at {qlib_dir}; skip backup")
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = qlib_dir.parent / f"{qlib_dir.name}.bak_{stamp}"
    log(f"Backing up: {qlib_dir} -> {bak}")
    qlib_dir.rename(bak)
    return bak


def prune_backups(qlib_dir: Path, keep: int) -> None:
    if keep < 0:
        return
    parent = qlib_dir.parent
    pattern = f"{qlib_dir.name}.bak_*"
    backups = sorted(parent.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in backups[keep:]:
        log(f"Removing old backup: {old}")
        if old.is_dir():
            shutil.rmtree(old)
        else:
            old.unlink()


def read_calendar_bounds(qlib_dir: Path) -> Tuple[str, str]:
    cal = qlib_dir / "calendars" / "day.txt"
    if not cal.exists():
        raise FileNotFoundError(
            f"Missing {cal}. Extraction may have failed; restore from backup if needed."
        )
    dates = [line.strip() for line in cal.read_text().splitlines() if line.strip()]
    if not dates:
        raise RuntimeError(f"Calendar is empty: {cal}")
    return dates[0], dates[-1]


def count_instruments(qlib_dir: Path) -> int:
    all_txt = qlib_dir / "instruments" / "all.txt"
    if not all_txt.exists():
        return 0
    return sum(1 for line in all_txt.read_text().splitlines() if line.strip())


def release_marker_path(qlib_dir: Path) -> Path:
    return qlib_dir / RELEASE_MARKER_NAME


def read_release_marker(qlib_dir: Path) -> Optional[str]:
    path = release_marker_path(qlib_dir)
    if not path.exists():
        return None
    tag = path.read_text().strip()
    return tag or None


def write_release_marker(qlib_dir: Path, release_tag: str) -> None:
    qlib_dir.mkdir(parents=True, exist_ok=True)
    release_marker_path(qlib_dir).write_text(f"{release_tag}\n")


def is_local_up_to_date(qlib_dir: Path, remote_tag: str) -> Tuple[bool, str]:
    """Return (up_to_date, reason) against remote release tag."""
    if remote_tag in ("", "latest"):
        return False, f"remote tag unresolved ({remote_tag!r}); need full download"

    marker = read_release_marker(qlib_dir)
    if marker == remote_tag:
        return True, f"release marker matches remote tag {remote_tag}"

    if not qlib_dir.exists():
        return False, f"no local data at {qlib_dir}"

    cal_path = qlib_dir / "calendars" / "day.txt"
    if not cal_path.exists():
        return False, f"missing calendar {cal_path}"

    if not _DATE_TAG_RE.match(remote_tag):
        reason = f"marker {marker!r} != remote {remote_tag!r} and tag is not YYYY-MM-DD"
        return False, reason

    try:
        _, local_end = read_calendar_bounds(qlib_dir)
    except (FileNotFoundError, RuntimeError) as exc:
        return False, f"cannot read calendar ({exc})"

    if local_end >= remote_tag:
        return (
            True,
            f"calendar end {local_end} >= remote tag {remote_tag}"
            + (f" (marker was {marker!r})" if marker else " (no marker)"),
        )

    return (
        False,
        f"calendar end {local_end} < remote tag {remote_tag}"
        + (f"; marker {marker!r}" if marker else "; no marker"),
    )


def _date_token(latest_date: str, quote: str) -> str:
    """按原文引号风格写回日期（无引号 / 单引号 / 双引号）。"""
    return f"{quote}{latest_date}{quote}" if quote else latest_date


def update_config_end_times(config_path: Path, latest_date: str, dry_run: bool) -> List[str]:
    """Sync only end dates to ``latest_date`` (typically today's date).

    Updates ``data_handler_config.end_time``, ``backtest.end_time``, and the
    **right** endpoint of ``segments.test``. Does not modify any start_time,
    fit_* dates, or train/valid segment bounds.

    兼容 LightGBM 风格（4 空格、无引号、``test: [a, b]``）与 ruamel 生成的
    滚动 yaml（2 空格、带引号、多行 list）。
    """
    text = config_path.read_text()
    changes: List[str] = []

    def _apply(label: str, pattern: str, repl, date_group: int) -> bool:
        nonlocal text
        matched = re.search(pattern, text)
        if not matched:
            changes.append(f"{label}: NOT FOUND (no change)")
            return False
        old_date = matched.group(date_group)
        new_text = re.sub(pattern, repl, text, count=1)
        if new_text != text:
            changes.append(f"{label}: {old_date} -> {latest_date}")
        else:
            changes.append(f"{label}: already {latest_date}")
        text = new_text
        return True

    # end_time: 可选引号；缩进兼容 2/4 空格
    _apply(
        "data_handler_config.end_time",
        r"(data_handler_config:[^\n]*\n(?:[^\n]*\n)*?[ \t]+end_time:[ \t]*)(['\"]?)(\d{4}-\d{2}-\d{2})\2",
        lambda m: f"{m.group(1)}{_date_token(latest_date, m.group(2))}",
        3,
    )
    _apply(
        "backtest.end_time",
        r"([ \t]+backtest:\n(?:[^\n]*\n)*?[ \t]+end_time:[ \t]*)(['\"]?)(\d{4}-\d{2}-\d{2})\2",
        lambda m: f"{m.group(1)}{_date_token(latest_date, m.group(2))}",
        3,
    )

    # segments.test 右端：先试紧凑 [a, b]，再试多行 list
    if not _apply(
        "segments.test right endpoint",
        r"(test:\s*\[[^\],]+,\s*)(['\"]?)(\d{4}-\d{2}-\d{2})\2(\s*\])",
        lambda m: f"{m.group(1)}{_date_token(latest_date, m.group(2))}{m.group(4)}",
        3,
    ):
        # 去掉上一条 NOT FOUND，改用 list 形式
        if changes and changes[-1].startswith("segments.test right endpoint: NOT FOUND"):
            changes.pop()
        _apply(
            "segments.test right endpoint",
            # group: 1=prefix+first line+second dash, 2/3=first date, 4/5=second date (quote backref \4)
            r"([ \t]*test:\s*\n[ \t]*-[ \t]*(['\"]?)(\d{4}-\d{2}-\d{2})\2\s*\n[ \t]*-[ \t]*)(['\"]?)(\d{4}-\d{2}-\d{2})\4",
            lambda m: f"{m.group(1)}{_date_token(latest_date, m.group(4))}",
            5,
        )

    if dry_run:
        log("[dry-run] config changes:")
        for c in changes:
            log(f"  {c}")
        return changes

    config_path.write_text(text)
    log(f"Updated config: {config_path}")
    for c in changes:
        log(f"  {c}")
    return changes


def resolve_latest_release_tag() -> str:
    """Best-effort: follow redirects to get tag from Location / final URL."""
    req = urllib.request.Request(
        "https://github.com/chenditc/investment_data/releases/latest",
        method="HEAD",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            url = resp.geturl()
        # .../releases/tag/YYYY-MM-DD
        m = re.search(r"/releases/tag/([^/?#]+)", url)
        if m:
            return m.group(1)
    except Exception as exc:  # noqa: BLE001
        log(f"Warning: could not resolve release tag ({exc})")
    return "latest"


def run(
    qlib_dir: Path,
    configs: Sequence[Path],
    skip_download: bool,
    dry_run: bool,
    keep_backups: int,
    force: bool = False,
) -> None:
    qlib_dir = qlib_dir.expanduser().resolve()
    config_paths = [c.expanduser().resolve() for c in configs]

    release_tag = "local"
    did_download = False

    if skip_download:
        log("Skipping download (--skip-download); using existing local data")
        marker = read_release_marker(qlib_dir)
        if marker:
            release_tag = marker
    else:
        release_tag = resolve_latest_release_tag()
        log(f"Latest release tag: {release_tag}")

        up_to_date, reason = is_local_up_to_date(qlib_dir, release_tag)
        if force:
            log(f"Force download requested (--force); ignoring freshness: {reason}")
            need_download = True
        elif up_to_date:
            log(f"Local data is up to date; skip download ({reason})")
            need_download = False
            # Backfill marker when calendar-end path skipped without one
            if not dry_run and read_release_marker(qlib_dir) != release_tag:
                write_release_marker(qlib_dir, release_tag)
                log(f"Wrote release marker: {release_marker_path(qlib_dir)} -> {release_tag}")
        else:
            log(f"Local data behind remote; will download ({reason})")
            need_download = True

        if need_download:
            if dry_run:
                log(f"[dry-run] would download {DOWNLOAD_URL}")
                log(f"[dry-run] would backup/replace {qlib_dir}")
                log(f"[dry-run] would write marker {RELEASE_MARKER_NAME}={release_tag}")
            else:
                # 使用固定缓存路径，中断后重跑可断点续传
                archive = _download_cache_path()
                download_archive(DOWNLOAD_URL, archive)
                backup_qlib_dir(qlib_dir)
                extract_qlib_bin(archive, qlib_dir)
                prune_backups(qlib_dir, keep_backups)
                write_release_marker(qlib_dir, release_tag)
                log(f"Wrote release marker: {release_marker_path(qlib_dir)} -> {release_tag}")
                # 解压成功后清理缓存，避免下次误用旧包
                try:
                    archive.unlink(missing_ok=True)
                    log(f"Removed download cache: {archive}")
                except OSError as exc:
                    log(f"Warning: could not remove cache {archive} ({exc})")
                did_download = True

    first, cal_latest = read_calendar_bounds(qlib_dir)
    n_inst = count_instruments(qlib_dir)
    yaml_end = datetime.now().date().isoformat()
    log(f"Calendar: {first} .. {cal_latest}")
    log(f"Instruments (all.txt): {n_inst}")
    log(f"Release: {release_tag}" + (" (downloaded)" if did_download else ""))
    log(f"YAML end date target (today): {yaml_end}")
    if yaml_end > cal_latest:
        log(
            f"Warning: today ({yaml_end}) is after local calendar end ({cal_latest}); "
            "features/backtest may lack the most recent days"
        )

    for config in config_paths:
        if not config.exists():
            log(f"Config missing, skip yaml update: {config}")
            continue
        update_config_end_times(config, yaml_end, dry_run=dry_run)
    log("Done.")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download latest crowd-source cn_data when behind, then sync rolling yaml "
            "end dates to today's date (never rewrites start dates)."
        )
    )
    parser.add_argument(
        "--qlib-dir",
        type=Path,
        default=DEFAULT_QLIB_DIR,
        help=f"qlib provider dir (default: {DEFAULT_QLIB_DIR})",
    )
    parser.add_argument(
        "--config",
        type=Path,
        nargs="*",
        default=None,
        help=(
            "rolling yaml(s) to sync end dates; "
            "default: both workflow_config_lightgbm_Alpha158_rolling_7y1y*.yaml"
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip data download; still sync yaml end dates to today",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full download even if local data matches latest release",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing",
    )
    parser.add_argument(
        "--keep-backups",
        type=int,
        default=2,
        help="Keep N most recent cn_data.bak_* dirs (default: 2)",
    )
    args = parser.parse_args(argv)
    configs = list(args.config) if args.config else list(DEFAULT_CONFIGS)
    run(
        qlib_dir=args.qlib_dir,
        configs=configs,
        skip_download=args.skip_download,
        dry_run=args.dry_run,
        keep_backups=args.keep_backups,
        force=args.force,
    )


if __name__ == "__main__":
    main()
