# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""alloc_v2 绘图辅助：只读 staging，matplotlib。价格/活跃度 ≠ 流入。"""

from __future__ import annotations

from typing import Iterable, Sequence

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_FONT_READY = False


def configure_cjk_font() -> str | None:
    """优先用本机中文字体，避免标题方框。返回选中的字体名。"""
    global _FONT_READY
    candidates = (
        "PingFang SC",
        "Heiti SC",
        "STHeiti",
        "Songti SC",
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        "SimHei",
        "Microsoft YaHei",
    )
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    chosen = next((name for name in candidates if name in available), None)
    if chosen:
        matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
        matplotlib.rcParams["axes.unicode_minus"] = False
    _FONT_READY = True
    return chosen


def _empty_msg(ax, title: str, msg: str = "无数据") -> None:
    ax.set_title(title)
    ax.text(0.5, 0.5, msg, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def series_from_standard(
    standard: pd.DataFrame,
    *,
    dataset: str | None = None,
    entity_id: str,
    field: str,
) -> pd.Series:
    """从标准层抽出一条时间序列（按 event_date）。"""
    if standard is None or standard.empty:
        return pd.Series(dtype="float64")
    part = standard.copy()
    if dataset is not None:
        part = part[part["dataset"] == dataset]
    part = part[(part["entity_id"] == entity_id) & (part["field"] == field)]
    if part.empty:
        return pd.Series(dtype="float64")
    part = part.copy()
    part["event_date"] = pd.to_datetime(part["event_date"])
    return part.sort_values("event_date").set_index("event_date")["value"].astype(float)


def series_from_factor(factors: pd.DataFrame, *, factor_id: str, entity_id: str) -> pd.Series:
    if factors is None or factors.empty:
        return pd.Series(dtype="float64")
    part = factors[(factors["factor_id"] == factor_id) & (factors["entity_id"] == entity_id)].copy()
    if part.empty:
        return pd.Series(dtype="float64")
    part["event_date"] = pd.to_datetime(part["event_date"])
    return part.sort_values("event_date").set_index("event_date")["value"].astype(float)


def plot_normalized(
    series_map: dict[str, pd.Series],
    *,
    title: str,
    ylabel: str = "归一化（起点=1）",
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """多序列归一化折线；起点为 0 的跳过。"""
    fig, ax = plt.subplots(figsize=figsize)
    plotted = 0
    for label, series in series_map.items():
        clean = series.dropna().sort_index()
        if clean.empty or float(clean.iloc[0]) == 0.0:
            continue
        ax.plot(clean.index, clean / clean.iloc[0], label=label)
        plotted += 1
    if not plotted:
        _empty_msg(ax, title)
    else:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
    fig.tight_layout()
    return fig


def plot_levels(
    series_map: dict[str, pd.Series],
    *,
    title: str,
    ylabel: str = "水平值",
    figsize: tuple[float, float] = (10, 4),
) -> plt.Figure:
    """多序列原始水平折线（不归一化）。"""
    fig, ax = plt.subplots(figsize=figsize)
    plotted = 0
    for label, series in series_map.items():
        clean = series.dropna().sort_index()
        if clean.empty:
            continue
        ax.plot(clean.index, clean, label=label)
        plotted += 1
    if not plotted:
        _empty_msg(ax, title)
    else:
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_bar(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    ylabel: str = "",
    figsize: tuple[float, float] = (10, 4),
    highlight_failed: Sequence[bool] | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    if not labels:
        _empty_msg(ax, title)
        fig.tight_layout()
        return fig
    x = np.arange(len(labels))
    colors = None
    if highlight_failed is not None and len(highlight_failed) == len(labels):
        colors = ["#c44e52" if failed else "#4c72b0" for failed in highlight_failed]
    ax.bar(x, list(values), color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels), rotation=30, ha="right", fontsize=8)
    ax.set_title(title)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    fig.tight_layout()
    return fig


def latest_cross_section(factors: pd.DataFrame, factor_id: str, *, top_n: int = 15) -> pd.Series:
    """取某因子最新截面，按绝对值排序取 Top-N。"""
    if factors is None or factors.empty:
        return pd.Series(dtype="float64")
    part = factors[factors["factor_id"] == factor_id].copy()
    if part.empty:
        return pd.Series(dtype="float64")
    part["event_date"] = pd.to_datetime(part["event_date"])
    last_day = part["event_date"].max()
    snap = part[part["event_date"] == last_day].set_index("entity_id")["value"].astype(float)
    snap = snap.dropna().sort_values(key=lambda s: s.abs(), ascending=False)
    return snap.head(top_n)


def plot_heatmap_pivot(
    factors: pd.DataFrame,
    factor_id: str,
    *,
    title: str,
    entities: Iterable[str] | None = None,
    last_n_days: int = 60,
    figsize: tuple[float, float] = (12, 5),
) -> plt.Figure:
    """近 N 日因子热力；实体过多时只取最新截面 Top 实体。"""
    fig, ax = plt.subplots(figsize=figsize)
    if factors is None or factors.empty:
        _empty_msg(ax, title)
        fig.tight_layout()
        return fig
    part = factors[factors["factor_id"] == factor_id].copy()
    if part.empty:
        _empty_msg(ax, title, f"无因子 {factor_id}")
        fig.tight_layout()
        return fig
    part["event_date"] = pd.to_datetime(part["event_date"])
    if entities is None:
        top = latest_cross_section(factors, factor_id, top_n=12)
        entities = list(top.index)
    entity_list = list(entities)
    part = part[part["entity_id"].isin(entity_list)]
    if part.empty:
        _empty_msg(ax, title)
        fig.tight_layout()
        return fig
    max_day = part["event_date"].max()
    min_day = max_day - pd.Timedelta(days=int(last_n_days))
    part = part[part["event_date"] >= min_day]
    pivot = part.pivot_table(index="entity_id", columns="event_date", values="value", aggfunc="last")
    if pivot.empty:
        _empty_msg(ax, title)
        fig.tight_layout()
        return fig
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="RdBu_r")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(list(pivot.index), fontsize=8)
    # 日期刻度稀疏显示
    cols = list(pivot.columns)
    step = max(1, len(cols) // 8)
    ticks = list(range(0, len(cols), step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([cols[i].strftime("%Y-%m-%d") for i in ticks], rotation=30, ha="right", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    return fig


def plot_etf_three_metrics(
    factors: pd.DataFrame,
    etf_code: str,
    *,
    title: str | None = None,
    figsize: tuple[float, float] = (10, 6),
) -> plt.Figure:
    """三类指标并列：份额、原始份额变化、估算配置。"""
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    specs = (
        ("RawShares", "份额（direct）", axes[0]),
        ("Raw_Share_Change", "原始份额变化（direct）", axes[1]),
        ("Estimated_Allocation", "估算配置 ΔShares×NAV（estimate）", axes[2]),
    )
    any_plot = False
    for factor_id, ylabel, ax in specs:
        series = series_from_factor(factors, factor_id=factor_id, entity_id=etf_code).dropna()
        if series.empty:
            _empty_msg(ax, ylabel, f"{etf_code} 无 {factor_id}")
            continue
        ax.plot(series.index, series)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.axhline(0.0, color="gray", linewidth=0.6) if factor_id != "RawShares" else None
        any_plot = True
    fig.suptitle(title or f"ETF {etf_code}：三类指标（份额变化 ≠ 净流入）")
    if not any_plot:
        pass
    fig.tight_layout()
    return fig


def plot_s1_components(s1: pd.DataFrame | dict, *, title: str = "S1 市场组件") -> plt.Figure:
    """S1 components 布尔条。"""
    fig, ax = plt.subplots(figsize=(6, 3))
    if isinstance(s1, pd.DataFrame):
        if s1.empty:
            _empty_msg(ax, title)
            fig.tight_layout()
            return fig
        row = s1.iloc[0].to_dict()
    else:
        row = dict(s1 or {})
    comps = row.get("components") or {}
    if isinstance(comps, str):
        try:
            import ast

            comps = ast.literal_eval(comps)
        except Exception:  # noqa: BLE001
            comps = {}
    if not comps:
        _empty_msg(ax, title, f"S1 value={row.get('value')}")
        fig.tight_layout()
        return fig
    labels = list(comps.keys())
    values = [1.0 if bool(comps[k]) else 0.0 for k in labels]
    ax.bar(labels, values, color=["#55a868" if v else "#c44e52" for v in values])
    ax.set_ylim(0, 1.2)
    ax.set_title(f"{title}（总信号={row.get('value')}）")
    ax.set_ylabel("通过=1")
    fig.tight_layout()
    return fig


def plot_s2_pass_bar(s2: pd.DataFrame, *, title: str = "S2 行业信号") -> plt.Figure:
    if s2 is None or s2.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        _empty_msg(ax, title)
        fig.tight_layout()
        return fig
    frame = s2.copy()
    frame = frame.sort_values(["value", "entity_id"], ascending=[False, True])
    return plot_bar(
        list(frame["entity_id"].astype(str)),
        list(frame["value"].astype(float)),
        title=title,
        ylabel="通过=1",
        figsize=(10, 3.5),
    )


def plot_backtest_bars(backtest: pd.DataFrame, *, metric: str = "rank_ic_mean_h20", title: str | None = None) -> plt.Figure:
    """回测条形：失败因子标红。"""
    if backtest is None or backtest.empty:
        fig, ax = plt.subplots(figsize=(8, 3))
        _empty_msg(ax, title or metric)
        fig.tight_layout()
        return fig
    frame = backtest.copy()
    if metric not in frame.columns:
        # 退化：展示 status
        failed = (frame.get("status", pd.Series(dtype=str)) == "failed").tolist()
        values = [0.0 if f else 1.0 for f in failed]
        return plot_bar(
            list(frame["factor_id"].astype(str)),
            values,
            title=title or "回测状态（1=有结果）",
            highlight_failed=failed,
        )
    values = frame[metric].astype(float).fillna(0.0).tolist()
    failed = (frame.get("status", pd.Series(["ok"] * len(frame))) == "failed").tolist()
    return plot_bar(
        list(frame["factor_id"].astype(str)),
        values,
        title=title or metric,
        ylabel=metric,
        highlight_failed=failed,
    )
