"""CLI：python -m qlib.contrib.data.l2_daily"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .catalog import ALL_GROUPS, FACTOR_VERSION
from . import prepare, storage
from .pipeline import run, run_batch

DEFAULT_WORKERS = 8


def _default_workers() -> int:
    raw = os.getenv("LEVEL2_L2_DAILY_WORKERS", str(DEFAULT_WORKERS)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return DEFAULT_WORKERS


def _parse_explicit_dates(args: argparse.Namespace) -> list[str]:
    """解析 --dates 或 --start/--end；区间只收 raw 里已有的日目录/.7z。"""
    if args.dates:
        return [item.strip() for item in args.dates.split(",") if item.strip()]
    if args.start and args.end:
        start = str(args.start)
        end = str(args.end)
        if not (start.isdigit() and end.isdigit() and len(start) == 8 and len(end) == 8):
            raise SystemExit("--start/--end 需要 YYYYMMDD")
        root = Path(args.raw_dir) if args.raw_dir else prepare.default_raw_root()
        dates: list[str] = []
        if root.is_dir():
            for child in sorted(root.iterdir()):
                name = child.stem if child.is_file() else child.name
                if name.isdigit() and len(name) == 8 and start <= name <= end:
                    if child.is_dir() or (child.is_file() and child.suffix == ".7z"):
                        dates.append(name)
        return sorted(set(dates))
    return []


def main(argv: list[str] | None = None) -> int:
    """与旧 QuantDinger CLI 对齐；不调用 dump bin。"""
    parser = argparse.ArgumentParser(description="L2 百因子日频：解压(可选)→CSV 直算→质检→本地/R2。不写 bin。")
    parser.add_argument("command", nargs="?", default="", help="all | reupload | prune-sources | YYYYMMDD")
    parser.add_argument("--dates", default="", help="逗号分隔交易日 YYYYMMDD")
    parser.add_argument("--start", default="", help="起始交易日 YYYYMMDD")
    parser.add_argument("--end", default="", help="结束交易日 YYYYMMDD")
    parser.add_argument("--raw-dir", default="", help="明细根（CSV/.7z）")
    parser.add_argument("--parquet-dir", default="", help="兼容旧名：同 --raw-dir")
    parser.add_argument("--output", default="", help="因子输出根")
    parser.add_argument("--groups", default="", help=f"逗号分隔组名，默认全部：{','.join(ALL_GROUPS)}")
    parser.add_argument("--force", action="store_true", help="忽略完成标记重算")
    parser.add_argument("--no-r2", action="store_true", help="不上传 R2")
    parser.add_argument("--no-compact", action="store_true", help="不跑月 compact")
    parser.add_argument("--no-seed", action="store_true", help="兼容旧开关，D1 已去掉")
    parser.add_argument("--no-delete", action="store_true", help="算完不回收解压目录/.7z")
    parser.add_argument("--delete-sources", action="store_true", help="兼容旧开关：默认就会回收")
    parser.add_argument("--keep-old-r2", action="store_true", help="reupload 时保留 R2 旧键")
    parser.add_argument("--local-only", action="store_true", help="只处理已解压目录")
    parser.add_argument("--7z-only", action="store_true", dest="z7_only", help="只处理 .7z")
    parser.add_argument("--dry-run", action="store_true", help="只调度/解压预览，不算因子")
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--factor-version", type=int, default=FACTOR_VERSION)
    args = parser.parse_args(argv)
    workers = max(1, int(args.workers))

    groups = [g.strip() for g in args.groups.split(",") if g.strip()] or None
    if groups:
        unknown = set(groups) - set(ALL_GROUPS)
        if unknown:
            raise SystemExit(f"未知 factor_group: {sorted(unknown)}")

    raw = Path(args.raw_dir or args.parquet_dir) if (args.raw_dir or args.parquet_dir) else prepare.default_raw_root()
    output = Path(args.output) if args.output else None
    delete_sources = not args.no_delete

    if args.command == "prune-sources":
        summary = prepare.prune_finished_sources(
            raw,
            output,
            dry_run=args.dry_run,
        )
        print(summary, flush=True)
        return 0

    if args.command == "reupload":
        dates = _parse_explicit_dates(args)
        summary = storage.reupload_local_days(
            output,
            dates=dates or None,
            factor_version=args.factor_version,
            delete_old_r2=not args.keep_old_r2,
            dry_run=args.dry_run,
        )
        print(summary, flush=True)
        return 1 if summary.get("missing") else 0

    batch_mode = args.command == "all" or (not args.command and not args.dates and not (args.start and args.end))

    if args.command and args.command not in ("all", "reupload", "prune-sources"):
        if not (args.command.isdigit() and len(args.command) == 8):
            raise SystemExit(f"未知 command: {args.command}（需要 all、reupload、prune-sources 或 YYYYMMDD）")
        dates = [args.command]
        batch_mode = False
    else:
        dates = _parse_explicit_dates(args)

    if batch_mode or args.command == "all" or (not dates and not args.command):
        print(f"l2_daily batch raw={raw} force={args.force} workers={workers}", flush=True)
        summary = run_batch(
            raw_root=raw,
            output_root=output,
            groups=groups,
            dates=dates or None,
            force=args.force,
            factor_version=args.factor_version,
            upload_r2=not args.no_r2,
            compact_months=not args.no_compact,
            delete_sources=delete_sources,
            dry_run=args.dry_run,
            local_only=args.local_only,
            z7_only=args.z7_only,
            workers=workers,
        )
    else:
        if not dates:
            raise SystemExit("需要 all、YYYYMMDD、--dates 或 --start/--end")
        for day in dates:
            prepare.ensure_day_csv(day, raw, dry_run=args.dry_run)
        if args.dry_run:
            print({"dates": dates, "dry_run": True}, flush=True)
            return 0
        print(
            f"l2_daily dates={len(dates)} groups={groups or list(ALL_GROUPS)} "
            f"books={raw} force={args.force} workers={workers}",
            flush=True,
        )
        summary = run(
            dates,
            books_root=raw,
            output_root=output,
            groups=groups,
            force=args.force,
            factor_version=args.factor_version,
            upload_r2=not args.no_r2,
            compact_months=not args.no_compact,
            workers=workers,
            delete_sources=delete_sources,
        )

    print(summary, flush=True)
    failed = int(summary.get("failed") or 0)
    processed = int(summary.get("processed") or 0)
    return 1 if failed > 0 and processed == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
