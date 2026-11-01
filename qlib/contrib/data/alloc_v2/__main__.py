# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLI：python -m qlib.contrib.data.alloc_v2"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .paths import default_root
from .pipeline import (
    UpdateError,
    init_maps,
    run_backtest,
    run_factor,
    run_fetch,
    run_pipeline,
    run_signal,
    run_standard,
    run_update,
)
from .r2_upload import upload_tree
from .registry import catalog_rows
from .schema import SERIES_SET


def _dump(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def _parse_day(text: str) -> date | None:
    if not text:
        return None
    return date.fromisoformat(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A 股资金配置研究 V2：fetch/standard/factor/signal。")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_catalog = sub.add_parser("catalog", help="打印数据集目录")
    p_catalog.add_argument("--json", action="store_true")
    p_catalog.add_argument("--include-disabled", action="store_true")

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", default="", help="数据根，默认 data/alloc_v2")
        p.add_argument("--start", default="", help="起始日期 YYYY-MM-DD")
        p.add_argument("--end", default="", help="结束日期 YYYY-MM-DD")

    _common(sub.add_parser("maps", help="写出映射表空结构/种子"))
    _common(sub.add_parser("fetch", help="拉取公开数据写 standard（整表覆盖）"))
    _common(sub.add_parser("standard", help="重跑标准层质量标记"))
    _common(sub.add_parser("factor", help="计算因子"))
    _common(sub.add_parser("signal", help="计算 S1–S4"))
    _common(sub.add_parser("backtest", help="IC/分组/持有期记录"))
    _common(sub.add_parser("all", help="maps → fetch → factor → signal → backtest"))

    p_update = sub.add_parser("update", help="从本地最近日期增量拉取并合并，再重算因子/信号")
    p_update.add_argument("--root", default="", help="数据根，默认 data/alloc_v2")
    p_update.add_argument("--overlap-days", type=int, default=7, help="相对最近日向前重叠自然日，默认 7")
    p_update.add_argument("--end", default="", help="结束日期 YYYY-MM-DD，默认今天")

    p_upload = sub.add_parser("upload", help="同步 maps/standard/factor/research 到 R2")
    p_upload.add_argument("--root", default="", help="数据根，默认 data/alloc_v2")
    p_upload.add_argument("--dry-run", action="store_true", help="只列将上传的键，不 put")

    args = parser.parse_args(argv)

    if args.cmd == "catalog":
        rows = catalog_rows(include_disabled=args.include_disabled)
        if args.json:
            return _dump({"series_set": SERIES_SET, "count": len(rows), "datasets": rows})
        print(f"{SERIES_SET} count={len(rows)}")
        for row in rows:
            flag = "on " if row["enabled"] else "off"
            print(
                f"{row['dataset']:28} {row['layer']} {row['evidence_class']:8} "
                f"{row['freq']:7} {flag} {row['display_name']}"
            )
        return 0

    root = Path(args.root) if getattr(args, "root", "") else default_root()
    start = _parse_day(getattr(args, "start", ""))
    end = _parse_day(getattr(args, "end", ""))

    if args.cmd == "maps":
        return _dump(init_maps(root))
    if args.cmd == "fetch":
        return _dump(run_fetch(start=start, end=end, root=root))
    if args.cmd == "standard":
        return _dump(run_standard(root=root))
    if args.cmd == "factor":
        return _dump(run_factor(root=root))
    if args.cmd == "signal":
        return _dump(run_signal(root=root))
    if args.cmd == "backtest":
        return _dump(run_backtest(root=root))
    if args.cmd == "all":
        return _dump(run_pipeline(start=start, end=end, root=root))
    if args.cmd == "update":
        try:
            return _dump(run_update(root=root, overlap_days=int(args.overlap_days), end=end))
        except UpdateError as exc:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False, indent=2))
            return 2
    if args.cmd == "upload":
        summary = upload_tree(root=root, dry_run=bool(args.dry_run))
        code = _dump(summary)
        if summary.get("failed"):
            return 2
        if not summary.get("dry_run") and not summary.get("configured"):
            return 2
        return code
    return 1


if __name__ == "__main__":
    sys.exit(main())
