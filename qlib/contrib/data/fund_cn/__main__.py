# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLI：python -m qlib.contrib.data.fund_cn"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .catalog import coverage_matrix, schema_hash
from .coverage import CoverageFailed
from .market import calendar_dates, default_instruments, load_market_file, load_unadjusted_close
from .paths_local import default_staging_root
from .pipeline import run_compute_publish, run_coverage, run_pipeline, run_pit, run_stage
from .qa import DayQAFailed
from .staging import read_statement_rows


def _parse_dates(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def _parse_instruments(text: str) -> list[str]:
    return [k.strip() for k in text.split(",") if k.strip()]


def _resolve_instruments(args: argparse.Namespace, root: Path) -> list[str]:
    raw = getattr(args, "instruments", "") or ""
    if raw:
        return _parse_instruments(raw)
    staged = read_statement_rows(root)
    if not staged.empty and "instrument_key" in staged.columns:
        return sorted(staged["instrument_key"].astype(str).unique())
    qlib_dir = getattr(args, "qlib_dir", "") or None
    return default_instruments(qlib_dir=qlib_dir)


def _resolve_dates(args: argparse.Namespace) -> list[str]:
    if getattr(args, "dates", ""):
        return _parse_dates(args.dates)
    start = getattr(args, "start", "") or ""
    end = getattr(args, "end", "") or ""
    if start and end:
        qlib_dir = getattr(args, "qlib_dir", "") or None
        return calendar_dates(start, end, qlib_dir=qlib_dir)
    return []


def _resolve_market(args: argparse.Namespace, keys: list[str], dates: list[str]):
    market_file = getattr(args, "market_file", "") or ""
    if market_file:
        return load_market_file(market_file)
    qlib_dir = getattr(args, "qlib_dir", "") or None
    if not keys or not dates:
        return None
    return load_unadjusted_close(keys, dates, qlib_dir=qlib_dir)


def _dump(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    """与 L2 CLI 对齐：catalog / stage / coverage / pit / compute / all。无 register。"""
    parser = argparse.ArgumentParser(description="A 股 fund_cn_basic@v1：计算、质检、本地/R2。不写 bin。")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_catalog = sub.add_parser("catalog", help="打印因子目录")
    p_catalog.add_argument("--json", action="store_true")

    def _common(p: argparse.ArgumentParser, *, need_keys: bool = False, need_dates: bool = False) -> None:
        p.add_argument("--root", default="", help="staging 根")
        p.add_argument("--instruments", default="", help="逗号分隔 CNStock:xxxxxx；可省")
        p.add_argument("--dates", default="", help="逗号分隔 YYYY-MM-DD")
        p.add_argument("--start", default="", help="起始交易日")
        p.add_argument("--end", default="", help="结束交易日")
        p.add_argument("--qlib-dir", default="", help="qlib cn_data 根")
        p.add_argument("--market-file", default="", help="未复权行情 CSV/Parquet")
        p.add_argument("--full", action="store_true", help="算 compute_ok 的非 MVP 因子")
        p.add_argument("--no-r2", action="store_true", help="不上传 R2")
        p.add_argument("--allow-uncovered", action="store_true", help="coverage 失败仍继续（仅诊断）")
        if need_keys:
            pass
        if need_dates:
            pass

    p_stage = sub.add_parser("stage", help="现拉 AKShare/巨潮并写入 staging")
    _common(p_stage)

    p_cov = sub.add_parser("coverage", help="覆盖率报告")
    _common(p_cov)

    p_pit = sub.add_parser("pit", help="staging → 本地 PIT parquet")
    _common(p_pit)

    p_compute = sub.add_parser("compute", help="日频宽表：质检后写 parquet/R2")
    _common(p_compute)
    p_compute.add_argument("--output", default="", help="因子输出根")

    p_all = sub.add_parser("all", help="stage → coverage → pit → compute")
    _common(p_all)
    p_all.add_argument("--output", default="", help="因子输出根")

    args = parser.parse_args(argv)

    if args.cmd == "catalog":
        matrix = coverage_matrix()
        if args.json:
            return _dump({"schema_hash": schema_hash(), "factors": matrix})
        print(f"factor_set schema_hash={schema_hash()} count={len(matrix)}")
        for row in matrix:
            print(
                f"{row['factor_id']:28} mvp={row['mvp']} "
                f"RAW={row['RAW_OK']} COMPUTE={row['COMPUTE_OK']} PIT={row['PIT_OK']} "
                f"policy={row['information_policy']}"
            )
        return 0

    root = Path(args.root) if getattr(args, "root", "") else default_staging_root()

    try:
        if args.cmd == "stage":
            keys = _resolve_instruments(args, root)
            if not keys:
                print("error: 需要 --instruments 或 staging 已有票", file=sys.stderr)
                return 2
            from .adapters.share_capital import CninfoShareCapitalAdapter

            start = date.fromisoformat(args.start) if args.start else None
            end = date.fromisoformat(args.end) if args.end else None
            info = run_stage(
                keys,
                root=root,
                share_capital_adapter=CninfoShareCapitalAdapter(),
                start=start,
                end=end,
            )
            return _dump(info)

        if args.cmd == "coverage":
            return _dump(run_coverage(root, require=not args.allow_uncovered))

        if args.cmd == "pit":
            if not args.allow_uncovered:
                run_coverage(root, require=True)
            return _dump(run_pit(root))

        if args.cmd in ("compute", "all"):
            keys = _resolve_instruments(args, root)
            dates = _resolve_dates(args)
            if not keys:
                print("error: 需要 --instruments 或 staging 已有票", file=sys.stderr)
                return 2
            if not dates:
                print("error: 需要 --dates 或 --start/--end", file=sys.stderr)
                return 2
            market = _resolve_market(args, keys, dates)
            factor_root = Path(args.output) if getattr(args, "output", "") else None
            if args.cmd == "all":
                from .adapters.share_capital import CninfoShareCapitalAdapter

                result = run_pipeline(
                    keys,
                    dates,
                    market=market,
                    share_capital_adapter=CninfoShareCapitalAdapter(),
                    root=root,
                    factor_root=factor_root,
                    mvp_only=not args.full,
                    require_coverage=not args.allow_uncovered,
                    upload=not args.no_r2,
                )
            else:
                if not args.allow_uncovered:
                    run_coverage(root, require=True)
                result = run_compute_publish(
                    dates,
                    keys,
                    market,
                    root=root,
                    factor_root=factor_root,
                    mvp_only=not args.full,
                    upload=not args.no_r2,
                )
            return _dump(result)
    except CoverageFailed as exc:
        print(json.dumps({"error": "coverage_failed", "failed": exc.failed}, ensure_ascii=False, indent=2))
        return 3
    except DayQAFailed as exc:
        print(json.dumps({"error": "day_qa_failed", **exc.payload}, ensure_ascii=False, indent=2, default=str))
        return 4

    return 1


if __name__ == "__main__":
    sys.exit(main())
