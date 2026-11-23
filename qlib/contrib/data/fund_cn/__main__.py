# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLI：python -m qlib.contrib.data.fund_cn

默认 ``--source csmar``：国泰安科目为主，EM NOTICE 补公告，免费源抽样验证。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .catalog import coverage_matrix, schema_hash
from .coverage import CoverageFailed
from .market import AdjustFactorRequiredError, calendar_dates, default_instruments, load_market_file, load_unadjusted_close
from .paths_local import default_staging_root
from .pipeline import run_compute_publish, run_coverage, run_pipeline, run_pit, run_stage
from .qa import DayQAFailed
from .source_policy import resolve_policy
from .staging import read_marker, read_statement_rows
from .verify import VerifyFailed


def _parse_dates(text: str) -> list[str]:
    return [p.strip() for p in text.split(",") if p.strip()]


def _parse_instruments(text: str) -> list[str]:
    return [k.strip() for k in text.split(",") if k.strip()]


def _resolve_instruments(args: argparse.Namespace, root: Path) -> list[str]:
    file_path = getattr(args, "instruments_file", "") or ""
    if file_path:
        text = Path(file_path).expanduser().read_text(encoding="utf-8")
        return [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    raw = getattr(args, "instruments", "") or ""
    if raw:
        return _parse_instruments(raw)
    staged = read_statement_rows(root)
    if not staged.empty and "instrument_key" in staged.columns:
        return sorted(staged["instrument_key"].astype(str).unique())
    qlib_dir = getattr(args, "qlib_dir", "") or None
    return default_instruments(qlib_dir=qlib_dir)


def _trade_bounds(args: argparse.Namespace) -> tuple[str, str]:
    """交易日窗口：--trade-start/end 优先，否则回退 --start/--end。"""
    trade_start = getattr(args, "trade_start", "") or ""
    trade_end = getattr(args, "trade_end", "") or ""
    start = trade_start or (getattr(args, "start", "") or "")
    end = trade_end or (getattr(args, "end", "") or "")
    return start, end


def _report_bounds(args: argparse.Namespace) -> tuple[date | None, date | None]:
    """报告期窗口：--report-start/end；未设时不强制（stage 可用 trade 兼容旧调用）。"""
    rs = getattr(args, "report_start", "") or ""
    re = getattr(args, "report_end", "") or ""
    return (
        date.fromisoformat(rs) if rs else None,
        date.fromisoformat(re) if re else None,
    )


def _resolve_dates(args: argparse.Namespace) -> list[str]:
    if getattr(args, "dates", ""):
        return _parse_dates(args.dates)
    start, end = _trade_bounds(args)
    if start and end:
        qlib_dir = getattr(args, "qlib_dir", "") or None
        return calendar_dates(start, end, qlib_dir=qlib_dir)
    return []


def _resolve_market(args: argparse.Namespace, keys: list[str], dates: list[str], *, require_factor: bool):
    market_file = getattr(args, "market_file", "") or ""
    if market_file:
        return load_market_file(market_file)
    qlib_dir = getattr(args, "qlib_dir", "") or None
    if not keys or not dates:
        return None
    return load_unadjusted_close(keys, dates, qlib_dir=qlib_dir, require_factor=require_factor)


def _dump(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def _policy_from_args(args: argparse.Namespace):
    policy = resolve_policy(source=getattr(args, "source", "csmar") or "csmar")
    if getattr(args, "skip_verify", False):
        from dataclasses import replace

        policy = replace(policy, verify_with_free_sources=False)
    if getattr(args, "use_cninfo_shares", False):
        from dataclasses import replace

        policy = replace(policy, use_cninfo_share_events=True, prefer_report_shares=False)
    return policy


def main(argv: list[str] | None = None) -> int:
    """catalog / stage / coverage / pit / compute / all。默认国泰安主源。"""
    parser = argparse.ArgumentParser(
        description="A 股 fund_cn_basic@v1：国泰安为主 + 免费源验证。不写 bin。"
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_catalog = sub.add_parser("catalog", help="打印因子目录")
    p_catalog.add_argument("--json", action="store_true")

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", default="", help="staging 根")
        p.add_argument("--instruments", default="", help="逗号分隔 CNStock:xxxxxx；可省")
        p.add_argument("--instruments-file", default="", help="每行一个 CNStock:xxxxxx")
        p.add_argument("--dates", default="", help="逗号分隔 YYYY-MM-DD")
        p.add_argument(
            "--start",
            default="",
            help="兼容旧参数：未设 --trade-start 时作交易日起始；stage 亦可作报告期下界",
        )
        p.add_argument(
            "--end",
            default="",
            help="兼容旧参数：未设 --trade-end 时作交易日结束；stage 亦可作报告期上界",
        )
        p.add_argument("--trade-start", default="", help="交易日起始 YYYY-MM-DD（compute/all）")
        p.add_argument("--trade-end", default="", help="交易日结束 YYYY-MM-DD（compute/all）")
        p.add_argument("--report-start", default="", help="报告期下界 YYYY-MM-DD（stage）")
        p.add_argument("--report-end", default="", help="报告期上界 YYYY-MM-DD（stage）")
        p.add_argument("--qlib-dir", default="", help="qlib cn_data 根")
        p.add_argument("--market-file", default="", help="未复权行情 CSV/Parquet")
        p.add_argument("--full", action="store_true", help="算 compute_ok 的非 MVP 因子")
        p.add_argument("--no-r2", action="store_true", help="不上传 R2")
        p.add_argument("--allow-uncovered", action="store_true", help="coverage 失败仍继续（仅诊断）")
        p.add_argument(
            "--source",
            default="csmar",
            help="科目主源：csmar（默认）| akshare（诊断）",
        )
        p.add_argument("--csmar-path", default="", help="国泰安季度合并.csv 路径")
        p.add_argument(
            "--skip-verify",
            action="store_true",
            help="跳过免费源交叉验证（正式出数请勿使用）",
        )
        p.add_argument(
            "--use-cninfo-shares",
            action="store_true",
            help="允许巨潮股本事件改估值主路径（默认关闭，仅验证）",
        )
        p.add_argument(
            "--resume-from",
            default="",
            help="compute 从该交易年起跑（YYYY）；已完整年份默认跳过",
        )
        p.add_argument(
            "--no-resume",
            action="store_true",
            help="禁用断点续跑，强制重算已有年",
        )

    p_stage = sub.add_parser("stage", help="国泰安科目 + EM 公告补洞 → staging")
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
    policy = _policy_from_args(args)
    csmar_path = getattr(args, "csmar_path", "") or None
    report_start, report_end = _report_bounds(args)

    try:
        if args.cmd == "stage":
            keys = _resolve_instruments(args, root)
            if not keys:
                print("error: 需要 --instruments 或 staging 已有票", file=sys.stderr)
                return 2
            # stage：报告期优先；未设 report_* 时用 --start/--end 兼容旧调用
            start = report_start
            end = report_end
            if start is None and args.start:
                start = date.fromisoformat(args.start)
            if end is None and args.end:
                end = date.fromisoformat(args.end)
            share_adapter = None
            if policy.use_cninfo_share_events:
                from .adapters.share_capital import CninfoShareCapitalAdapter

                share_adapter = CninfoShareCapitalAdapter()
            info = run_stage(
                keys,
                root=root,
                share_capital_adapter=share_adapter,
                start=start,
                end=end,
                report_start=report_start,
                report_end=report_end,
                policy=policy,
                csmar_path=csmar_path,
                skip_verify=bool(getattr(args, "skip_verify", False)),
            )
            return _dump(info)

        if args.cmd == "coverage":
            verify = read_marker("verify_free_source", root=root)
            return _dump(run_coverage(root, require=not args.allow_uncovered, verify_payload=verify))

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
                print(
                    "error: 需要 --dates 或 --trade-start/--trade-end（或兼容 --start/--end）",
                    file=sys.stderr,
                )
                return 2
            qlib_dir = getattr(args, "qlib_dir", "") or None
            market_file = getattr(args, "market_file", "") or ""
            use_lazy_market = (not market_file) and (len(keys) > 200 or len(dates) > 400)
            if use_lazy_market:
                market = None
            else:
                market = _resolve_market(
                    args, keys, dates, require_factor=policy.require_adjust_factor
                )
            factor_root = Path(args.output) if getattr(args, "output", "") else None
            resume_from = getattr(args, "resume_from", "") or None
            resume = not bool(getattr(args, "no_resume", False))
            if args.cmd == "all":
                share_adapter = None
                if policy.use_cninfo_share_events:
                    from .adapters.share_capital import CninfoShareCapitalAdapter

                    share_adapter = CninfoShareCapitalAdapter()
                result = run_pipeline(
                    keys,
                    dates,
                    market=market,
                    share_capital_adapter=share_adapter,
                    root=root,
                    factor_root=factor_root,
                    mvp_only=not args.full,
                    require_coverage=not args.allow_uncovered,
                    upload=not args.no_r2,
                    policy=policy,
                    csmar_path=csmar_path,
                    skip_verify=bool(getattr(args, "skip_verify", False)),
                    qlib_dir=qlib_dir,
                    report_start=report_start,
                    report_end=report_end,
                    resume=resume,
                    resume_from=resume_from,
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
                    prefer_report_shares=policy.prefer_report_shares,
                    qlib_dir=qlib_dir,
                    resume=resume,
                    resume_from=resume_from,
                )
            return _dump(result)
    except CoverageFailed as exc:
        print(json.dumps({"error": "coverage_failed", "failed": exc.failed}, ensure_ascii=False, indent=2))
        return 3
    except DayQAFailed as exc:
        print(json.dumps({"error": "day_qa_failed", **exc.payload}, ensure_ascii=False, indent=2, default=str))
        return 4
    except VerifyFailed as exc:
        print(json.dumps({"error": "verify_failed", **exc.payload}, ensure_ascii=False, indent=2, default=str))
        return 5
    except AdjustFactorRequiredError as exc:
        print(json.dumps({"error": "adjust_factor_required", "message": str(exc)}, ensure_ascii=False, indent=2))
        return 6

    return 1


if __name__ == "__main__":
    sys.exit(main())
