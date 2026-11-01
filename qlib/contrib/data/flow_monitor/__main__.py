# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLI：python -m qlib.contrib.data.flow_monitor"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from .catalog import SERIES_SET, catalog_rows
from .compute import run_compute
from .paths_local import default_staging_root
from .pipeline import run_fetch, run_pipeline


def _dump(payload: dict) -> int:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def _parse_day(text: str) -> date | None:
    if not text:
        return None
    return date.fromisoformat(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="全球到 A 股资金流向监测：fetch/compute，不写 bin。")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_catalog = sub.add_parser("catalog", help="打印序列目录")
    p_catalog.add_argument("--json", action="store_true")
    p_catalog.add_argument("--include-disabled", action="store_true")

    def _common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", default="", help="staging 根，默认 data/flow_staging")
        p.add_argument("--start", default="", help="起始日期 YYYY-MM-DD")
        p.add_argument("--end", default="", help="结束日期 YYYY-MM-DD")

    _common(sub.add_parser("fetch", help="拉取公开数据并写 raw/long.parquet"))
    _common(sub.add_parser("compute", help="由 staging 计算诊断面板"))
    _common(sub.add_parser("all", help="fetch → compute"))

    args = parser.parse_args(argv)

    if args.cmd == "catalog":
        rows = catalog_rows(include_disabled=args.include_disabled)
        if args.json:
            return _dump({"series_set": SERIES_SET, "count": len(rows), "series": rows})
        print(f"{SERIES_SET} count={len(rows)}")
        for row in rows:
            flag = "on " if row["enabled"] else "off"
            print(
                f"{row['series_id']:32} {row['evidence_grade']} {row['freq']:6} "
                f"L{row['layer']} {flag} {row['display_name']}"
            )
        return 0

    root = Path(args.root) if args.root else default_staging_root()
    start = _parse_day(getattr(args, "start", ""))
    end = _parse_day(getattr(args, "end", ""))

    if args.cmd == "fetch":
        return _dump(run_fetch(start=start, end=end, root=root))
    if args.cmd == "compute":
        return _dump(run_compute(root=root))
    if args.cmd == "all":
        return _dump(run_pipeline(start=start, end=end, root=root))
    return 1


if __name__ == "__main__":
    sys.exit(main())
