#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Daily (manual) refresh of crowd-source cn_data + rolling yaml end dates.

Downloads the latest qlib_bin.tar.gz from chenditc/investment_data, replaces
~/.qlib/qlib_data/cn_data, then syncs **end_time** fields in rolling configs
to calendars/day.txt last trading day.

Never rewrites start_time, fit_start_time, or train/valid segment endpoints.

Usage:
    python examples/benchmarks/LightGBM/update_cn_data_daily.py
    python examples/benchmarks/LightGBM/update_cn_data_daily.py --dry-run
    python examples/benchmarks/LightGBM/update_cn_data_daily.py --skip-download
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

DIRNAME = Path(__file__).absolute().resolve().parent
DEFAULT_QLIB_DIR = Path("~/.qlib/qlib_data/cn_data").expanduser()
DEFAULT_CONFIGS = [
    DIRNAME / "workflow_config_lightgbm_Alpha158_rolling_7y2y.yaml",
    DIRNAME / "workflow_config_lightgbm_Alpha158_rolling_7y2y_ind_weight.yaml",
]
DOWNLOAD_URL = (
    "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
)
def log(msg: str) -> None:
    print(msg, flush=True)


def download_archive(url: str, dest: Path) -> None:
    log(f"Downloading: {url}")
    log(f"        to: {dest}")

    last_pct = [-1]

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        pct = int(downloaded * 100 / total_size)
        if pct == last_pct[0] and downloaded < total_size:
            return
        last_pct[0] = pct
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  {pct:3d}%  ({mb:.1f}/{total_mb:.1f} MB)")
        sys.stdout.flush()
        if downloaded >= total_size:
            sys.stdout.write("\n")

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    size_mb = dest.stat().st_size / (1024 * 1024)
    log(f"Download complete: {size_mb:.1f} MB")


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


def update_config_end_times(config_path: Path, latest_date: str, dry_run: bool) -> List[str]:
    """Sync only end dates to ``latest_date``.

    Updates ``data_handler_config.end_time``, ``backtest.end_time``, and the
    **right** endpoint of ``segments.test``. Does not modify any start_time,
    fit_* dates, or train/valid segment bounds.
    """
    text = config_path.read_text()
    changes: List[str] = []

    replacements = [
        (
            "data_handler_config.end_time",
            r"(data_handler_config:[^\n]*\n(?:[^\n]*\n)*?    end_time: )(\d{4}-\d{2}-\d{2})",
            lambda m: f"{m.group(1)}{latest_date}",
        ),
        (
            "backtest.end_time",
            r"(    backtest:\n(?:[^\n]*\n)*?        end_time: )(\d{4}-\d{2}-\d{2})",
            lambda m: f"{m.group(1)}{latest_date}",
        ),
        (
            "segments.test right endpoint",
            r"(test:\s*\[[\d-]+,\s*)(\d{4}-\d{2}-\d{2})(\s*\])",
            lambda m: f"{m.group(1)}{latest_date}{m.group(3)}",
        ),
    ]

    for label, pattern, repl in replacements:
        matched = re.search(pattern, text)
        if not matched:
            changes.append(f"{label}: NOT FOUND (no change)")
            continue
        old = matched.group(0)
        new = repl(matched)
        if old != new:
            # show only the date part transition when possible
            old_date = matched.group(2)
            changes.append(f"{label}: {old_date} -> {latest_date}")
        else:
            changes.append(f"{label}: already {latest_date}")
        text = re.sub(pattern, repl, text, count=1)

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
) -> None:
    qlib_dir = qlib_dir.expanduser().resolve()
    config_paths = [c.expanduser().resolve() for c in configs]

    release_tag = "local"
    if not skip_download:
        release_tag = resolve_latest_release_tag()
        log(f"Latest release tag: {release_tag}")
        if dry_run:
            log(f"[dry-run] would download {DOWNLOAD_URL}")
            log(f"[dry-run] would backup/replace {qlib_dir}")
        else:
            with tempfile.TemporaryDirectory(prefix="qlib_bin_") as tmp:
                archive = Path(tmp) / "qlib_bin.tar.gz"
                download_archive(DOWNLOAD_URL, archive)
                backup_qlib_dir(qlib_dir)
                extract_qlib_bin(archive, qlib_dir)
                prune_backups(qlib_dir, keep_backups)
    else:
        log("Skipping download; using existing local data")

    first, latest = read_calendar_bounds(qlib_dir)
    n_inst = count_instruments(qlib_dir)
    log(f"Calendar: {first} .. {latest}")
    log(f"Instruments (all.txt): {n_inst}")
    log(f"Release: {release_tag}")

    for config in config_paths:
        if not config.exists():
            log(f"Config missing, skip yaml update: {config}")
            continue
        update_config_end_times(config, latest, dry_run=dry_run)
    log("Done.")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download latest crowd-source cn_data and sync rolling yaml end dates only "
            "(never rewrites start dates)."
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
            "default: both workflow_config_lightgbm_Alpha158_rolling_7y2y*.yaml"
        ),
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only refresh yaml from existing local calendar",
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
    )


if __name__ == "__main__":
    main()
