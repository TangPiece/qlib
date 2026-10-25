"""按日编排：解压 → 四组计算 → 质检 → parquet + R2 → 下一日。不写 bin。"""
from __future__ import annotations

import logging
from pathlib import Path

from . import prepare, publish, qa, status, storage
from .catalog import ALL_GROUPS, FACTOR_VERSION, factor_ids_for_group
from .engine import calc_group_day, list_codes

_log = logging.getLogger(__name__)


def _recycle_day_sources(
    trade_date: str,
    books_root: Path,
    *,
    delete_sources: bool,
    keep_archive: bool,
    summary: dict[str, object],
) -> None:
    """一天结束后回收原料：失败只删解压目录，通过连 7z 一起删。"""
    if not delete_sources:
        return
    removed = prepare.delete_day_sources(trade_date, books_root, keep_archive=keep_archive)
    existing = list(summary.get("deleted") or [])
    existing.extend(removed)
    summary["deleted"] = existing
    if removed:
        kind = "解压目录" if keep_archive else "目录和 7z"
        print(f"[{trade_date}] 已回收{kind} {removed}", flush=True)


def compute_day_group(
    trade_date: str,
    factor_group: str,
    *,
    books_root: Path,
    output_root: Path,
    force: bool = False,
    factor_version: int = FACTOR_VERSION,
    workers: int = 1,
) -> dict[str, object]:
    """计算一组，只返回结果，不写正式宽表。"""
    codes = list_codes(books_root, trade_date)
    source_version = status.source_version_for_date(books_root, trade_date, factor_group, codes=codes)
    local = status.read_local_marker(output_root, trade_date, factor_group)
    if status.should_skip(
        trade_date,
        factor_group,
        source_version,
        force=force,
        factor_version=factor_version,
        local_status=local,
    ):
        return {"status": "skipped", "trade_date": trade_date, "factor_group": factor_group}

    if not codes:
        return {
            "status": "skipped",
            "trade_date": trade_date,
            "factor_group": factor_group,
            "error": "no symbols",
        }

    rows, status_rows, ok, fail = calc_group_day(
        books_root,
        trade_date,
        factor_group,
        codes=codes,
        workers=max(1, int(workers)),
    )
    for item in status_rows:
        item["source_version"] = source_version
    fail_rate = fail / max(1, ok + fail)
    if fail_rate >= qa.GROUP_FAIL_NULL:
        day_status = "failed"
    elif fail == 0 and ok > 0:
        day_status = "done"
    elif ok > 0:
        day_status = "partial"
    else:
        day_status = "failed"
    return {
        "status": day_status,
        "trade_date": trade_date,
        "factor_group": factor_group,
        "symbol_ok": ok,
        "symbol_fail": fail,
        "rows": rows,
        "status_rows": status_rows,
        "source_version": source_version,
    }


def run_one_day(
    trade_date: str,
    *,
    books_root: Path,
    output_root: Path,
    groups: list[str] | None = None,
    force: bool = False,
    factor_version: int = FACTOR_VERSION,
    upload_r2: bool = True,
    compact_months: bool = True,
    workers: int = 1,
    delete_sources: bool = True,
) -> dict[str, object]:
    """一日四组算完后质检，通过才写盘/上传。不写 qlib bin。"""
    selected = list(groups) if groups else list(ALL_GROUPS)
    existing = storage.read_day(output_root, trade_date, factor_version=factor_version)
    merged = existing
    summary = {
        "trade_date": trade_date,
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "status": "done",
        "deleted": [],
    }

    for factor_group in selected:
        result = compute_day_group(
            trade_date,
            factor_group,
            books_root=books_root,
            output_root=output_root,
            force=force,
            factor_version=factor_version,
            workers=workers,
        )
        state = str(result.get("status"))
        if state == "skipped":
            summary["skipped"] = int(summary["skipped"]) + 1
            continue
        if state == "failed":
            summary["failed"] = int(summary["failed"]) + 1
            summary["status"] = "failed"
            summary["error"] = f"{factor_group} failed"
            print(f"[{trade_date}] {factor_group} 失败，停止当日发布", flush=True)
            _recycle_day_sources(
                trade_date, books_root, delete_sources=delete_sources, keep_archive=True, summary=summary
            )
            return summary
        summary["processed"] = int(summary["processed"]) + 1
        merged = storage.merge_group_columns(
            merged,
            list(result.get("rows") or []),
            factor_ids_for_group(factor_group),
            trade_date,
        )
        storage.write_status(
            output_root,
            trade_date,
            list(result.get("status_rows") or []),
            factor_version=factor_version,
        )
        status.write_local_marker(
            output_root,
            trade_date,
            factor_group,
            source_version=str(result.get("source_version") or ""),
            status=state,
            symbol_ok=int(result.get("symbol_ok") or 0),
            symbol_fail=int(result.get("symbol_fail") or 0),
            factor_version=factor_version,
        )

    if merged is None or merged.empty:
        summary["failed"] = int(summary["failed"]) + 1
        summary["status"] = "failed"
        summary["error"] = "empty wide"
        _recycle_day_sources(
            trade_date, books_root, delete_sources=delete_sources, keep_archive=True, summary=summary
        )
        return summary

    report = qa.check_day(day=trade_date, wide=merged, raw_root=books_root, old_wide=existing)
    if not report.get("ok"):
        summary["failed"] = int(summary["failed"]) + 1
        summary["status"] = "failed"
        summary["error"] = "; ".join(report.get("errors") or ["qa failed"])
        print(f"[{trade_date}] 质检失败: {summary['error']}", flush=True)
        _recycle_day_sources(
            trade_date, books_root, delete_sources=delete_sources, keep_archive=True, summary=summary
        )
        return summary

    publish.publish_day(
        output_root,
        trade_date,
        merged,
        upload_r2=upload_r2,
        factor_version=factor_version,
    )
    if compact_months:
        try:
            storage.compact_month(output_root, trade_date[:6], factor_version=factor_version, delete_days=False)
        except Exception:
            _log.warning("月 compact 失败 %s", trade_date[:6], exc_info=True)
    print(f"[{trade_date}] 质检通过，已写宽表", flush=True)
    _recycle_day_sources(
        trade_date, books_root, delete_sources=delete_sources, keep_archive=False, summary=summary
    )
    return summary


def run(
    dates: list[str],
    *,
    books_root: Path | None = None,
    parquet_root: Path | None = None,
    output_root: Path | None = None,
    groups: list[str] | None = None,
    force: bool = False,
    factor_version: int = FACTOR_VERSION,
    upload_r2: bool = True,
    seed_defs: bool = False,
    compact_months: bool = True,
    workers: int = 1,
    delete_sources: bool = True,
) -> dict[str, object]:
    """显式日期列表：按日串行，失败即停。"""
    del seed_defs  # D1 不迁
    root = Path(output_root) if output_root is not None else storage.default_root()
    root.mkdir(parents=True, exist_ok=True)
    books = Path(books_root if books_root is not None else parquet_root or prepare.default_raw_root())
    ordered = sorted({str(d) for d in dates if str(d).isdigit() and len(str(d)) == 8})
    summary = {
        "dates": ordered,
        "groups": groups or list(ALL_GROUPS),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "rows": 0,
        "deleted": [],
    }
    for trade_date in ordered:
        day = run_one_day(
            trade_date,
            books_root=books,
            output_root=root,
            groups=groups,
            force=force,
            factor_version=factor_version,
            upload_r2=upload_r2,
            compact_months=compact_months,
            workers=workers,
            delete_sources=delete_sources,
        )
        summary["processed"] = int(summary["processed"]) + int(day.get("processed") or 0)
        summary["skipped"] = int(summary["skipped"]) + int(day.get("skipped") or 0)
        summary["failed"] = int(summary["failed"]) + int(day.get("failed") or 0)
        summary["deleted"].extend(day.get("deleted") or [])  # type: ignore[union-attr]
        if int(day.get("failed") or 0) > 0:
            print(f"[{trade_date}] 当日失败，停止后续", flush=True)
            break
    return summary


def _discover_batch_dates(
    root: Path,
    *,
    only: list[str] | None,
    local_only: bool,
    z7_only: bool,
) -> tuple[list[str], list[str], list[str]]:
    local_dates = [] if z7_only else prepare.discover_local_dates(root, only)
    z7_dates = [] if local_only else prepare.discover_7z_dates(root, only)
    # 显式 only 时缺失日仍入 planned，ensure 时再报错
    planned = sorted(set(only) if only else (set(local_dates) | set(z7_dates)))
    return local_dates, z7_dates, planned


def run_batch(
    *,
    raw_root: Path | None = None,
    output_root: Path | None = None,
    groups: list[str] | None = None,
    dates: list[str] | None = None,
    force: bool = False,
    factor_version: int = FACTOR_VERSION,
    upload_r2: bool = True,
    seed_defs: bool = False,
    compact_months: bool = True,
    delete_sources: bool = True,
    dry_run: bool = False,
    local_only: bool = False,
    z7_only: bool = False,
    workers: int = 1,
) -> dict[str, object]:
    """扫描 raw：已解压优先，再逐日解压；质检失败停跑且不续批。默认回收解压目录。"""
    del seed_defs
    root = Path(raw_root) if raw_root is not None else prepare.default_raw_root()
    out = Path(output_root) if output_root is not None else storage.default_root()
    only = list(dates) if dates else None
    allow_rescan = only is None
    summary: dict[str, object] = {
        "dates": [],
        "groups": list(groups) if groups else list(ALL_GROUPS),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "rows": 0,
        "deleted": [],
    }
    seen: set[str] = set()
    wave = 0
    stop_batch = False

    while True:
        local_dates, z7_dates, discovered = _discover_batch_dates(
            root, only=only, local_only=local_only, z7_only=z7_only
        )
        planned = [d for d in discovered if d not in seen]
        if not planned:
            if wave == 0:
                print("无可处理日期", flush=True)
                summary["failed"] = 1
            else:
                print(f"l2_daily batch 无新日期，结束（waves={wave}）", flush=True)
            break

        wave += 1
        print(
            f"l2_daily batch wave={wave} raw={root} local={len(local_dates)} "
            f"7z={len(z7_dates)} planned={len(planned)} workers={workers} "
            f"delete={delete_sources} dry_run={dry_run}",
            flush=True,
        )
        if local_dates:
            print(f"  阶段A local CSV: {local_dates}", flush=True)
        if z7_dates:
            print(f"  阶段B 7z 串行: {z7_dates}", flush=True)

        for trade_date in planned:
            seen.add(trade_date)
            try:
                prepare.ensure_day_csv(trade_date, root, dry_run=dry_run)
            except Exception as exc:
                _log.exception("准备 CSV 失败 %s", trade_date)
                print(f"[{trade_date}] 准备失败: {exc}，停止后续", flush=True)
                summary["failed"] = int(summary["failed"]) + 1
                if delete_sources:
                    removed = prepare.delete_day_sources(trade_date, root, keep_archive=True)
                    summary["deleted"].extend(removed)  # type: ignore[union-attr]
                stop_batch = True
                break

            if dry_run:
                codes = list_codes(root, trade_date)
                print(f"[{trade_date}] dry-run symbols={len(codes)}", flush=True)
                summary["dates"].append(trade_date)  # type: ignore[union-attr]
                summary["skipped"] = int(summary["skipped"]) + 1
                continue

            day = run_one_day(
                trade_date,
                books_root=root,
                output_root=out,
                groups=groups,
                force=force,
                factor_version=factor_version,
                upload_r2=upload_r2,
                compact_months=compact_months,
                workers=workers,
                delete_sources=delete_sources,
            )
            summary["dates"].append(trade_date)  # type: ignore[union-attr]
            summary["processed"] = int(summary["processed"]) + int(day.get("processed") or 0)
            summary["skipped"] = int(summary["skipped"]) + int(day.get("skipped") or 0)
            summary["failed"] = int(summary["failed"]) + int(day.get("failed") or 0)
            summary["deleted"].extend(day.get("deleted") or [])  # type: ignore[union-attr]
            if int(day.get("failed") or 0) > 0:
                print(f"[{trade_date}] 当日失败，停止后续解压", flush=True)
                stop_batch = True
                break

        if stop_batch or not allow_rescan:
            break

    return summary
