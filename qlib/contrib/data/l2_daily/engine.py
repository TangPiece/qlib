"""按日加载明细并计算指定因子组。只读 CSV。"""
from __future__ import annotations

import logging
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd

from .catalog import (
    GROUP_BOOK,
    GROUP_MICROSTRUCTURE,
    GROUP_ORDER,
    GROUP_TRADE,
    factor_ids_for_group,
)
from .csv_io import read_csv_bytes
from .groups import (
    calc_book_factors,
    calc_microstructure_factors,
    calc_order_factors,
    calc_trade_factors,
)
from .util import canonical_symbol, is_equity

_log = logging.getLogger(__name__)
_FILE_TYPES = ("行情", "逐笔成交", "逐笔委托")
_PROGRESS_EVERY = 100
_GROUP_SOURCES = {
    GROUP_TRADE: ("逐笔成交",),
    GROUP_BOOK: ("行情",),
    GROUP_ORDER: ("逐笔委托", "逐笔成交"),
    GROUP_MICROSTRUCTURE: ("行情", "逐笔成交", "逐笔委托"),
}


def _usable_csv(path: Path) -> bool:
    """存在且非 0 字节才当源文件。万得停牌占位常是空行情.csv。"""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def resolve_code_dir(day_dir: Path, code: str) -> str | None:
    """按大小写不敏感找到真实目录名；没有则 None。"""
    if not day_dir.is_dir():
        return None
    target = str(code).lower()
    for child in day_dir.iterdir():
        if child.is_dir() and child.name.lower() == target:
            return child.name
    return None


def source_file_path(root: Path, date: str, code: str, ftype: str) -> Path | None:
    """只认非空 CSV；不存在或 0 字节返回 None。大小写不对时回退真实目录。"""
    day_dir = Path(root) / str(date)
    csv_path = day_dir / str(code) / f"{ftype}.csv"
    if _usable_csv(csv_path):
        return csv_path
    actual = resolve_code_dir(day_dir, code)
    if actual is None or actual == str(code):
        return None
    csv_path = day_dir / actual / f"{ftype}.csv"
    return csv_path if _usable_csv(csv_path) else None


def list_codes(books_root: Path, trade_date: str) -> list[str]:
    """列出当日股票代码（目录真名，供读文件）。"""
    day = Path(books_root) / str(trade_date)
    if not day.is_dir():
        return []
    return sorted(child.name for child in day.iterdir() if child.is_dir() and is_equity(child.name))


def _needed_ftypes(factor_groups: list[str]) -> tuple[str, ...]:
    needed: list[str] = []
    seen: set[str] = set()
    for group in factor_groups:
        for ftype in _GROUP_SOURCES.get(group, _FILE_TYPES):
            if ftype not in seen:
                seen.add(ftype)
                needed.append(ftype)
    return tuple(needed)


def load_symbol_frames(
    books_root: Path,
    trade_date: str,
    code: str,
    ftypes: tuple[str, ...] | list[str],
) -> dict[str, pd.DataFrame | None]:
    """按类型读明细；缺文件或空 CSV 为 None，避免把停牌误当成空表。"""
    frames: dict[str, pd.DataFrame | None] = {}
    for ftype in ftypes:
        path = source_file_path(books_root, trade_date, code, ftype)
        if path is None:
            frames[ftype] = None
            continue
        try:
            frames[ftype] = _read_path(path, ftype)
        except pd.errors.EmptyDataError:
            # 0 字节或无列表头：当缺源，不让一只坏行情打死成交。
            frames[ftype] = None
    return frames


def _calc_group_from_frames(
    frames: dict[str, pd.DataFrame | None],
    code: str,
    factor_group: str,
) -> dict[str, Any] | None:
    """用已读入的帧算一组。缺必要源返回 None。"""
    if factor_group == GROUP_TRADE:
        trades = frames.get("逐笔成交")
        if trades is None:
            return None
        return calc_trade_factors(trades, code)
    if factor_group == GROUP_BOOK:
        snap = frames.get("行情")
        if snap is None:
            return None
        return calc_book_factors(snap)
    if factor_group == GROUP_ORDER:
        orders = frames.get("逐笔委托")
        trades = frames.get("逐笔成交")
        empty_orders = orders is None or orders.empty
        empty_trades = trades is None or trades.empty
        if empty_orders and empty_trades:
            return None
        return calc_order_factors(
            orders if orders is not None else pd.DataFrame(),
            trades if trades is not None else pd.DataFrame(),
            code,
        )
    if factor_group == GROUP_MICROSTRUCTURE:
        snap = frames.get("行情")
        trades = frames.get("逐笔成交")
        orders = frames.get("逐笔委托")
        if all(frame is None or frame.empty for frame in (snap, trades, orders)):
            return None
        return calc_microstructure_factors(
            snap if snap is not None else pd.DataFrame(),
            trades if trades is not None else pd.DataFrame(),
            orders if orders is not None else pd.DataFrame(),
            code,
        )
    raise ValueError(f"unknown factor_group: {factor_group}")


def calc_groups_for_symbol(
    books_root: Path,
    trade_date: str,
    code: str,
    factor_groups: list[str],
) -> dict[str, dict[str, Any] | None]:
    """一只股票一次读齐所需明细，再算多个组。"""
    frames = load_symbol_frames(books_root, trade_date, code, _needed_ftypes(factor_groups))
    out: dict[str, dict[str, Any] | None] = {}
    for factor_group in factor_groups:
        try:
            row = _calc_group_from_frames(frames, code, factor_group)
        except Exception:
            _log.warning("l2_daily 计算失败 %s %s %s", trade_date, code, factor_group, exc_info=True)
            row = None
        if row is not None:
            row["trade_date"] = str(trade_date)
            row["symbol"] = str(code)
        out[factor_group] = row
    return out


def calc_group_for_symbol(
    books_root: Path,
    trade_date: str,
    code: str,
    factor_group: str,
) -> dict[str, Any] | None:
    """计算一只股票一个组；缺关键输入返回 None。质检抽样走这条。"""
    return calc_groups_for_symbol(books_root, trade_date, code, [factor_group]).get(factor_group)


def calc_groups_for_symbol_worker(
    trade_date: str,
    code: str,
    books_root: str,
    factor_groups: list[str],
) -> dict[str, dict[str, Any] | None]:
    """子进程入口：一股多组。"""
    return calc_groups_for_symbol(Path(books_root), trade_date, code, factor_groups)


def calc_group_for_symbol_worker(
    trade_date: str,
    code: str,
    books_root: str,
    factor_group: str,
) -> dict[str, Any] | None:
    """子进程入口：一股一组。"""
    return calc_group_for_symbol(Path(books_root), trade_date, code, factor_group)


def _emit_symbol_progress(trade_date: str, factor_group: str, done: int, total: int) -> None:
    if total <= 0:
        return
    if done == total or done % _PROGRESS_EVERY == 0:
        print(f"[{trade_date}] {factor_group} 已完成 {done}/{total}", flush=True)


def _group_all_nan(result: dict[str, Any], columns: list[str]) -> bool:
    """组内有效因子全是 NaN 视为失败，避免再标 done 后删源。"""
    values = []
    for column in columns:
        raw = result.get(column, float("nan"))
        try:
            number = float(raw)
        except (TypeError, ValueError):
            return False
        values.append(number)
    if not values:
        return True
    return all(math.isnan(v) for v in values)


def _required_sources_missing(
    books_root: Path,
    trade_date: str,
    code: str,
    factor_group: str,
) -> bool:
    """缺该组必要明细。停牌只有空行情、没有逐笔成交，不算计算失败。"""
    if factor_group == GROUP_TRADE:
        return source_file_path(books_root, trade_date, code, "逐笔成交") is None
    if factor_group == GROUP_ORDER:
        return (
            source_file_path(books_root, trade_date, code, "逐笔委托") is None
            and source_file_path(books_root, trade_date, code, "逐笔成交") is None
        )
    if factor_group == GROUP_BOOK:
        return source_file_path(books_root, trade_date, code, "行情") is None
    if factor_group == GROUP_MICROSTRUCTURE:
        return all(
            source_file_path(books_root, trade_date, code, name) is None
            for name in ("行情", "逐笔成交", "逐笔委托")
        )
    return False


def _append_symbol_result(
    *,
    trade_date: str,
    code: str,
    factor_group: str,
    result: dict[str, Any] | None,
    columns: list[str],
    rows: list[dict[str, Any]],
    status_rows: list[dict[str, Any]],
    books_root: Path | None = None,
) -> tuple[int, int]:
    """把单股结果写入 rows/status_rows，返回 (ok_delta, fail_delta)。

    缺必要源文件记 skipped，不抬失败率，避免 2017 年停牌把整组打成 partial。
    宽表/账本 symbol 用规范化代码，读文件仍用传入的目录真名。
    """
    symbol = canonical_symbol(code)
    if result is None or _group_all_nan(result, columns):
        if (
            result is None
            and books_root is not None
            and _required_sources_missing(books_root, trade_date, code, factor_group)
        ):
            status_rows.append(
                {
                    "symbol": symbol,
                    "factor_group": factor_group,
                    "status": "skipped",
                    "source_version": "",
                }
            )
            return 0, 0
        status_rows.append(
            {
                "symbol": symbol,
                "factor_group": factor_group,
                "status": "failed",
                "source_version": "",
            }
        )
        return 0, 1
    slim = {"trade_date": trade_date, "symbol": symbol}
    for column in columns:
        slim[column] = result.get(column, float("nan"))
    rows.append(slim)
    status_rows.append(
        {
            "symbol": symbol,
            "factor_group": factor_group,
            "status": "done",
            "source_version": "",
        }
    )
    return 1, 0


def _empty_group_buckets(
    factor_groups: list[str],
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int, int]]:
    """每组一个 (rows, status_rows, columns, ok, fail)。"""
    buckets: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int, int]] = {}
    for group in factor_groups:
        buckets[group] = ([], [], factor_ids_for_group(group), 0, 0)
    return buckets


def _absorb_symbol_groups(
    *,
    trade_date: str,
    code: str,
    results: dict[str, dict[str, Any] | None],
    buckets: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int, int]],
    books_root: Path,
) -> None:
    """把一只股票的多组结果写入各桶。"""
    for group, (rows, status_rows, columns, ok, fail) in list(buckets.items()):
        d_ok, d_fail = _append_symbol_result(
            trade_date=trade_date,
            code=code,
            factor_group=group,
            result=results.get(group),
            columns=columns,
            rows=rows,
            status_rows=status_rows,
            books_root=books_root,
        )
        buckets[group] = (rows, status_rows, columns, ok + d_ok, fail + d_fail)


def calc_groups_day(
    books_root: Path,
    trade_date: str,
    factor_groups: list[str],
    *,
    codes: list[str] | None = None,
    workers: int = 1,
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]]:
    """一天内按股票一次读齐，多组同算。返回 {group: (rows, status_rows, ok, fail)}。"""
    selected = codes if codes is not None else list_codes(books_root, trade_date)
    groups = list(factor_groups)
    buckets = _empty_group_buckets(groups)
    total = len(selected)
    pool_workers = max(1, int(workers))

    def _run_serial() -> None:
        for done, code in enumerate(selected, start=1):
            results = calc_groups_for_symbol(books_root, trade_date, code, groups)
            _absorb_symbol_groups(
                trade_date=trade_date,
                code=code,
                results=results,
                buckets=buckets,
                books_root=books_root,
            )
            _emit_symbol_progress(trade_date, "factors", done, total)

    if pool_workers <= 1 or total <= 1:
        _run_serial()
    else:
        root_str = str(books_root)
        try:
            pool = ProcessPoolExecutor(max_workers=min(pool_workers, total))
        except PermissionError:
            print(f"[{trade_date}] 进程池不可用，回退串行 factors", flush=True)
            _run_serial()
        else:
            with pool:
                futures = [
                    pool.submit(calc_groups_for_symbol_worker, trade_date, code, root_str, groups)
                    for code in selected
                ]
                for done, (code, future) in enumerate(zip(selected, futures), start=1):
                    try:
                        results = future.result()
                    except Exception:
                        _log.warning(
                            "l2_daily 并行计算异常 %s %s",
                            trade_date,
                            code,
                            exc_info=True,
                        )
                        results = {group: None for group in groups}
                    _absorb_symbol_groups(
                        trade_date=trade_date,
                        code=code,
                        results=results,
                        buckets=buckets,
                        books_root=books_root,
                    )
                    _emit_symbol_progress(trade_date, "factors", done, total)

    out: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]] = {}
    for group, (rows, status_rows, _columns, ok, fail) in buckets.items():
        rows.sort(key=lambda item: str(item.get("symbol") or ""))
        status_rows.sort(key=lambda item: str(item.get("symbol") or ""))
        out[group] = (rows, status_rows, ok, fail)
    return out


def calc_group_day(
    books_root: Path,
    trade_date: str,
    factor_group: str,
    *,
    codes: list[str] | None = None,
    workers: int = 1,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    """返回 (factor_rows, status_rows, ok, fail)。"""
    return calc_groups_day(
        books_root,
        trade_date,
        [factor_group],
        codes=codes,
        workers=workers,
    )[factor_group]


def _read_path(path: Path, ftype: str) -> pd.DataFrame:
    """读本地 CSV 明细。"""
    return read_csv_bytes(path.read_bytes(), ftype)


def _read_local(root: Path, date: str, code: str, ftype: str) -> pd.DataFrame:
    """只读本地 CSV。"""
    path = source_file_path(root, date, code, ftype)
    if path is None:
        raise FileNotFoundError(Path(root) / str(date) / str(code) / f"{ftype}.csv")
    return _read_path(path, ftype)
