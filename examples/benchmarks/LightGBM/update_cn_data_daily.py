#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Daily (manual) refresh of crowd-source cn_data + rolling yaml end dates.

Downloads the latest qlib_bin.tar.gz from chenditc/investment_data when local
data is behind that release, replaces ~/.qlib/qlib_data/cn_data, then syncs
**end_time** fields in rolling configs to **today's date** (local machine date),
not the calendar last trading day.

After a full replace, restores custom L2 bins: prefer re-dump from
data/level2_staging into ~/.qlib/qlib_data/cn_data_l2 and merge into cn_data
(so community updates do not permanently wipe L2). Use --skip-l2-restore to
disable.

Also restores fund_cn financial bins the same way: durable store at
~/.qlib/qlib_data/cn_data_fund (from fundamental_staging or copy), then merge
into cn_data. Use --skip-fund-restore to disable.

investment_data only ships a full archive (~540MB); there is no incremental
day pack. This script skips re-download when the local release marker (or
calendar end) already matches / covers the remote latest tag.

Never rewrites start_time, fit_start_time, or train/valid segment endpoints.

Usage:
    python examples/benchmarks/LightGBM/update_cn_data_daily.py
    python examples/benchmarks/LightGBM/update_cn_data_daily.py --dry-run
    python examples/benchmarks/LightGBM/update_cn_data_daily.py --skip-download
    python examples/benchmarks/LightGBM/update_cn_data_daily.py --force
    python examples/benchmarks/LightGBM/update_cn_data_daily.py --skip-l2-restore
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

DIRNAME = Path(__file__).absolute().resolve().parent
# LightGBM -> benchmarks -> examples -> qlib 仓库根
REPO_ROOT = DIRNAME.parents[2]
DEFAULT_QLIB_DIR = Path("~/.qlib/qlib_data/cn_data").expanduser()
# L2 bin 持久化目录：与 cn_data 并列，整包替换行情时不会被抹掉
DEFAULT_L2_QLIB_DIR = Path("~/.qlib/qlib_data/cn_data_l2").expanduser()
# staging 真源：优先据此按新日历重 dump
DEFAULT_L2_STAGING = REPO_ROOT / "data" / "level2_staging" / "factors" / "l2_daily" / "v1"
# 财务因子 bin 持久化目录：与 cn_data / cn_data_l2 并列
DEFAULT_FUND_QLIB_DIR = Path("~/.qlib/qlib_data/cn_data_fund").expanduser()
DEFAULT_FUND_STAGING = (
    REPO_ROOT / "data" / "fundamental_staging" / "factors" / "fund_cn" / "v1"
)
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
# 大文件下载易中断/半挂死：优先 curl 续传；urllib 兜底并带读超时
DOWNLOAD_RETRIES = 16
DOWNLOAD_CHUNK_SIZE = 1 << 20  # 1 MiB
DOWNLOAD_RETRY_SLEEP_SEC = 2.0
DOWNLOAD_READ_TIMEOUT_SEC = 30  # 单次读超过该秒数视为卡死，触发续传
DOWNLOAD_CONNECT_TIMEOUT_SEC = 30


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
        with urllib.request.urlopen(req, timeout=DOWNLOAD_CONNECT_TIMEOUT_SEC) as resp:
            length = resp.headers.get("Content-Length")
            if length and length.isdigit():
                return int(length)
    except Exception as exc:  # noqa: BLE001
        log(f"Warning: HEAD size probe failed ({exc})")
    return None


def _print_progress(
    downloaded: int,
    total_size: int,
    state: List[float],
    *,
    force: bool = False,
) -> None:
    """刷新进度：至少每 1MB 或每 2 秒一次，并显示瞬时速度，避免「卡在 xx%」。

    state = [last_printed_bytes, last_print_time, last_bytes_for_speed, last_speed_time]
    """
    now = time.monotonic()
    last_printed, last_t, speed_bytes, speed_t = state
    if (
        not force
        and downloaded < total_size
        and (downloaded - last_printed) < DOWNLOAD_CHUNK_SIZE
        and (now - last_t) < 2.0
    ):
        return

    # 近窗口速度（B/s）
    dt = max(now - speed_t, 1e-6)
    speed = (downloaded - speed_bytes) / dt
    state[0] = float(downloaded)
    state[1] = now
    state[2] = float(downloaded)
    state[3] = now

    mb = downloaded / (1024 * 1024)
    speed_mb = speed / (1024 * 1024)
    if total_size > 0:
        pct = min(downloaded, total_size) * 100 / total_size
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  {pct:5.1f}%  ({mb:.1f}/{total_mb:.1f} MB)  {speed_mb:.2f} MB/s")
    else:
        sys.stdout.write(f"\r  downloaded {mb:.1f} MB  {speed_mb:.2f} MB/s")
    sys.stdout.flush()
    if total_size > 0 and downloaded >= total_size:
        sys.stdout.write("\n")


def _curl_available() -> bool:
    return shutil.which("curl") is not None


def _download_with_curl(url: str, dest: Path, remote_size: Optional[int]) -> None:
    """用 curl 断点续传（对 GitHub CDN / 不稳定链路更稳）。"""
    # -C - 续传；--retry 处理瞬时失败；-L 跟随 redirect；--fail 把 HTTP 错误当失败
    cmd = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "8",
        "--retry-delay",
        "2",
        "--retry-all-errors",
        "--connect-timeout",
        str(DOWNLOAD_CONNECT_TIMEOUT_SEC),
        "--speed-time",
        "30",  # 连续 30s 速度过低则中止，便于外层重跑续传
        "--speed-limit",
        "1024",  # <1 KiB/s 视为卡死
        "-C",
        "-",
        "-o",
        str(dest),
        url,
    ]
    log(f"Using curl resume download -> {dest}")
    # 让 curl 自己画进度条到 stderr
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"curl exited with code {proc.returncode}")
    final_size = dest.stat().st_size if dest.exists() else 0
    if remote_size is not None and final_size < remote_size:
        raise urllib.error.ContentTooShortError(
            f"retrieval incomplete: got only {final_size} out of {remote_size} bytes",
            None,
        )
    log(f"Download complete: {final_size / (1024 * 1024):.1f} MB")


def _download_with_urllib(
    url: str,
    dest: Path,
    remote_size: Optional[int],
    *,
    retries: int,
) -> None:
    """urllib 兜底：Range 续传 + 读超时，避免半挂死卡在某个百分比。"""
    progress_state = [0.0, time.monotonic(), 0.0, time.monotonic()]
    attempt = 0

    while attempt < retries:
        attempt += 1
        existing = dest.stat().st_size if dest.exists() else 0

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
            with urllib.request.urlopen(req, timeout=DOWNLOAD_CONNECT_TIMEOUT_SEC) as resp:
                # 连接成功后单独收紧读超时，半挂死时尽快抛错续传
                try:
                    resp.fp.raw._sock.settimeout(DOWNLOAD_READ_TIMEOUT_SEC)  # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001
                    socket.setdefaulttimeout(DOWNLOAD_READ_TIMEOUT_SEC)

                status = getattr(resp, "status", None) or resp.getcode()
                if existing > 0 and status == 200:
                    log("Server ignored Range; restarting from scratch")
                    dest.unlink(missing_ok=True)
                    existing = 0
                    mode = "wb"
                    progress_state[0] = 0.0

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
                progress_state[2] = float(downloaded)
                progress_state[3] = time.monotonic()
                with open(dest, mode) as out:
                    while True:
                        chunk = resp.read(DOWNLOAD_CHUNK_SIZE)
                        if not chunk:
                            break
                        out.write(chunk)
                        downloaded += len(chunk)
                        _print_progress(downloaded, total_size or 0, progress_state)

            final_size = dest.stat().st_size
            if remote_size is not None and final_size < remote_size:
                raise urllib.error.ContentTooShortError(
                    f"retrieval incomplete: got only {final_size} out of {remote_size} bytes",
                    None,
                )
            _print_progress(final_size, remote_size or final_size, progress_state, force=True)
            if not (remote_size and final_size >= remote_size):
                sys.stdout.write("\n")
            log(f"Download complete: {final_size / (1024 * 1024):.1f} MB")
            return

        except (
            urllib.error.URLError,
            urllib.error.ContentTooShortError,
            TimeoutError,
            socket.timeout,
            OSError,
        ) as exc:
            sys.stdout.write("\n")
            partial = dest.stat().st_size if dest.exists() else 0
            log(f"Download interrupted ({exc}); kept {partial / (1024 * 1024):.1f} MB partial")
            if attempt >= retries:
                raise
            sleep_s = DOWNLOAD_RETRY_SLEEP_SEC * min(attempt, 5)
            log(f"Retrying in {sleep_s:.0f}s ...")
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to download after {retries} attempts: {url}")


def download_archive(
    url: str,
    dest: Path,
    *,
    retries: int = DOWNLOAD_RETRIES,
) -> None:
    """下载大文件，优先 curl 断点续传，失败再回退 urllib。

    GitHub release 约 500MB+，链路抖动时易半挂死「进度停在某 %」；
    curl 的 ``--speed-time`` / ``-C -`` 对续传更稳。
    """
    log(f"Downloading: {url}")
    log(f"        to: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)

    remote_size = _probe_remote_size(url)
    existing = dest.stat().st_size if dest.exists() else 0
    if remote_size is not None and existing == remote_size:
        log(f"Cache hit: {dest} already {existing / (1024 * 1024):.1f} MB")
        return
    if remote_size is not None and existing > remote_size:
        log(f"Local file larger than remote ({existing} > {remote_size}); truncating")
        dest.unlink(missing_ok=True)

    if _curl_available():
        try:
            _download_with_curl(url, dest, remote_size)
            return
        except Exception as exc:  # noqa: BLE001
            partial = dest.stat().st_size if dest.exists() else 0
            log(f"curl download failed ({exc}); falling back to urllib (kept {partial / (1024 * 1024):.1f} MB)")

    _download_with_urllib(url, dest, remote_size, retries=retries)


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


def _load_dump_script_module(script_name: str, module_name: str) -> Any:
    """按路径加载 scripts/<script_name>.py（scripts 非包）。"""
    path = REPO_ROOT / "scripts" / f"{script_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"missing dump script: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_dump_l2_module() -> Any:
    """按路径加载 scripts/dump_l2_daily.py。"""
    return _load_dump_script_module("dump_l2_daily", "dump_l2_daily")


def _load_dump_fund_module() -> Any:
    """按路径加载 scripts/dump_fund_cn.py。"""
    return _load_dump_script_module("dump_fund_cn", "dump_fund_cn")


def restore_l2_bins(
    qlib_dir: Path,
    *,
    l2_qlib_dir: Path = DEFAULT_L2_QLIB_DIR,
    staging_dir: Path = DEFAULT_L2_STAGING,
    dry_run: bool = False,
) -> None:
    """cn_data 整包替换后恢复 L2 bin：优先 staging 重 dump，否则从 cn_data_l2 拷回。"""
    qlib_dir = qlib_dir.expanduser().resolve()
    l2_qlib_dir = l2_qlib_dir.expanduser().resolve()
    staging_dir = staging_dir.expanduser().resolve()

    try:
        dump_mod = _load_dump_l2_module()
    except Exception as exc:  # noqa: BLE001
        log(f"Warning: cannot load dump_l2_daily ({exc}); skip L2 restore")
        return

    day_files = dump_mod.iter_day_files(staging_dir) if staging_dir.is_dir() else []
    if day_files:
        log(
            f"Restoring L2 bins from staging ({len(day_files)} day files) "
            f"-> {l2_qlib_dir} + merge into {qlib_dir}"
        )
        if dry_run:
            log("[dry-run] would run DumpL2Daily.dump() with new calendar")
            return
        dump_mod.DumpL2Daily(
            data_path=str(staging_dir),
            qlib_dir=str(l2_qlib_dir),
            calendar_dir=str(qlib_dir),
            merge_into=str(qlib_dir),
        ).dump()
        log("L2 restore via staging dump done")
        return

    if dump_mod.has_l2_feature_bins(l2_qlib_dir):
        log(
            f"Warning: staging missing under {staging_dir}; "
            f"copying existing L2 bins from {l2_qlib_dir} into {qlib_dir}. "
            "Calendar may have changed — prefer re-dump from staging when available."
        )
        if dry_run:
            log("[dry-run] would sync_l2_feature_bins(cn_data_l2 -> cn_data)")
            return
        n = dump_mod.sync_l2_feature_bins(l2_qlib_dir, qlib_dir)
        log(f"L2 restore via copy done ({n} files)")
        return

    log(
        f"Warning: no L2 staging at {staging_dir} and no bins under {l2_qlib_dir}; "
        "skip L2 restore (run scripts/dump_l2_daily.py dump after factors are ready)"
    )


def restore_fund_cn_bins(
    qlib_dir: Path,
    *,
    fund_qlib_dir: Path = DEFAULT_FUND_QLIB_DIR,
    staging_dir: Path = DEFAULT_FUND_STAGING,
    dry_run: bool = False,
) -> None:
    """cn_data 整包替换后恢复财务因子 bin：优先 staging 重 dump，否则从 cn_data_fund 拷回。"""
    qlib_dir = qlib_dir.expanduser().resolve()
    fund_qlib_dir = fund_qlib_dir.expanduser().resolve()
    staging_dir = staging_dir.expanduser().resolve()

    try:
        dump_mod = _load_dump_fund_module()
    except Exception as exc:  # noqa: BLE001
        log(f"Warning: cannot load dump_fund_cn ({exc}); skip fund restore")
        return

    day_files = dump_mod.iter_day_files(staging_dir) if staging_dir.is_dir() else []
    if day_files:
        log(
            f"Restoring fund_cn bins from staging ({len(day_files)} day files) "
            f"-> {fund_qlib_dir} + merge into {qlib_dir}"
        )
        if dry_run:
            log("[dry-run] would run DumpFundCn.dump() with new calendar")
            return
        dump_mod.DumpFundCn(
            data_path=str(staging_dir),
            qlib_dir=str(fund_qlib_dir),
            calendar_dir=str(qlib_dir),
            merge_into=str(qlib_dir),
        ).dump()
        log("fund_cn restore via staging dump done")
        return

    if dump_mod.has_fund_feature_bins(fund_qlib_dir):
        log(
            f"Warning: fund staging missing under {staging_dir}; "
            f"copying existing fund bins from {fund_qlib_dir} into {qlib_dir}. "
            "Calendar may have changed — prefer re-dump from staging when available."
        )
        if dry_run:
            log("[dry-run] would sync_fund_feature_bins(cn_data_fund -> cn_data)")
            return
        n = dump_mod.sync_fund_feature_bins(fund_qlib_dir, qlib_dir)
        log(f"fund_cn restore via copy done ({n} files)")
        return

    log(
        f"Warning: no fund staging at {staging_dir} and no bins under {fund_qlib_dir}; "
        "skip fund restore (run scripts/dump_fund_cn.py dump after factors are ready)"
    )


def run(
    qlib_dir: Path,
    configs: Sequence[Path],
    skip_download: bool,
    dry_run: bool,
    keep_backups: int,
    force: bool = False,
    skip_l2_restore: bool = False,
    l2_qlib_dir: Path = DEFAULT_L2_QLIB_DIR,
    skip_fund_restore: bool = False,
    fund_qlib_dir: Path = DEFAULT_FUND_QLIB_DIR,
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
                if not skip_l2_restore:
                    restore_l2_bins(
                        qlib_dir, l2_qlib_dir=l2_qlib_dir, dry_run=True
                    )
                if not skip_fund_restore:
                    restore_fund_cn_bins(
                        qlib_dir, fund_qlib_dir=fund_qlib_dir, dry_run=True
                    )
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
                # 社区包不含自定义 L2 / fund：按新日历重 dump 或从持久化目录合回
                if not skip_l2_restore:
                    restore_l2_bins(qlib_dir, l2_qlib_dir=l2_qlib_dir, dry_run=False)
                else:
                    log("Skipping L2 restore (--skip-l2-restore)")
                if not skip_fund_restore:
                    restore_fund_cn_bins(
                        qlib_dir, fund_qlib_dir=fund_qlib_dir, dry_run=False
                    )
                else:
                    log("Skipping fund restore (--skip-fund-restore)")

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
    parser.add_argument(
        "--skip-l2-restore",
        action="store_true",
        help="After download, do not re-dump/merge L2 bins into cn_data",
    )
    parser.add_argument(
        "--l2-qlib-dir",
        type=Path,
        default=DEFAULT_L2_QLIB_DIR,
        help=f"durable L2 bin root (default: {DEFAULT_L2_QLIB_DIR})",
    )
    parser.add_argument(
        "--skip-fund-restore",
        action="store_true",
        help="After download, do not re-dump/merge fund_cn bins into cn_data",
    )
    parser.add_argument(
        "--fund-qlib-dir",
        type=Path,
        default=DEFAULT_FUND_QLIB_DIR,
        help=f"durable fund_cn bin root (default: {DEFAULT_FUND_QLIB_DIR})",
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
        skip_l2_restore=args.skip_l2_restore,
        l2_qlib_dir=args.l2_qlib_dir,
        skip_fund_restore=args.skip_fund_restore,
        fund_qlib_dir=args.fund_qlib_dir,
    )


if __name__ == "__main__":
    main()
