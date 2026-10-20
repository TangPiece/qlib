#!/usr/bin/env python3
"""按代码和日期查询复权收盘、现价，以及该日的申万一级行业。

持仓页给账本日补数时调用。qlib 放在这个子进程里初始化，控制台本身不依赖 qlib。

stdin: {"keys":[{"instrument":"SH600000","date":"2026-10-19"}]}
stdout: 一行 JSON {"quotes":[{"instrument","date","price","raw_price","industry_code","industry_name"}]}
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import qlib
from qlib.constant import REG_CN

# 与 predict_live_signal 同一份本地行情
_PROVIDER = "~/.qlib/qlib_data/cn_data"


def main() -> None:
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    payload = json.loads(sys.stdin.read() or "{}")
    raw_keys = payload.get("keys") or []
    keys = []
    seen = set()
    for item in raw_keys:
        if not isinstance(item, dict):
            continue
        code = str(item.get("instrument") or "").strip().upper()
        day = str(item.get("date") or "").strip()
        if not code or not day or (code, day) in seen:
            continue
        seen.add((code, day))
        keys.append((code, day))

    # 延迟导入：export 模块会拉 qlib.data，须在 init 前只导入函数
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import export_positions_view as exp  # noqa: E402

    quotes = []
    if keys:
        qlib.init(provider_uri=_PROVIDER, region=REG_CN)
        prices = exp._fetch_adj_raw_prices(keys)
        # 行业表缺失时留空，页面不要把所有股票标成「未知」
        intervals = exp.load_industry_intervals(exp.DEFAULT_INDUSTRY_MAP)
        for code, day in keys:
            adj, raw = prices.get((code, day), (None, None))
            ind_code, ind_name = (None, None)
            if intervals is not None:
                ind_code, ind_name = exp.lookup_industry(intervals, code, day)
            quotes.append(
                {
                    "instrument": code,
                    "date": day,
                    "price": adj,
                    "raw_price": raw,
                    "industry_code": ind_code or "",
                    "industry_name": ind_name or "",
                }
            )
    json.dump({"quotes": quotes}, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
