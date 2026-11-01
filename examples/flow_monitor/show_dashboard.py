# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""读取 flow_staging 并打印五个问题的监测表；供 notebook 或命令行使用。"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qlib.contrib.data.flow_monitor.paths_local import default_staging_root
from qlib.contrib.data.flow_monitor.report import dashboard_payload


def _print_table(title: str, frame: pd.DataFrame, *, note: str = "") -> None:
    print("\n" + "=" * 80)
    print(title)
    if note:
        print(note)
    if frame is None or frame.empty:
        print("（无数据：请先运行 ./scripts/run_flow_monitor.sh all）")
        return
    with pd.option_context("display.max_rows", 80, "display.max_columns", 20, "display.width", 160):
        print(frame.to_string(index=False))


def render(root: Path | None = None) -> dict:
    """组装仪表盘表并打印。返回 payload 供 notebook 继续画图。"""
    payload = dashboard_payload(root or default_staging_root())
    _print_table(
        "Q1/Q2 钱从哪来、到哪去（全球 + 中国大类）",
        payload["q_source_dest"],
        note="L1/L2 才接近流量；L3 只是价格或活跃度。",
    )
    _print_table(
        "C  A 股行业 / 市值风格",
        payload["q_industry"],
        note="行业涨跌与成交额是 L3；对应 ETF 份额才是 L1。",
    )
    _print_table(
        "Q3 谁在买（按投资者 + asof）",
        payload["q_who"],
        note="月/季数据看 asof，不要当成当日实时。",
    )
    _print_table(
        "Q4 可能的驱动（并列事实，不是因果结论）",
        payload["q_why"],
    )
    _print_table(
        "Q5 是否真实：价格 vs 流量",
        payload["q_real"],
        note="PRICE_DRIVEN=涨了但流量没有跟上；ACTIVITY_ONLY=只有成交额。",
    )
    _print_table(
        "并发配置变化证据矩阵",
        payload["allocation"],
        note="不是已证实的资金从 X 流向 Y。",
    )
    missing = payload["unavailable"]
    if not missing.empty:
        _print_table("未启用或本次未取到的序列", missing[["series_id", "display_name", "status", "notes"]])
    return payload


if __name__ == "__main__":
    render()
