"""按日编排：解压 → CSV 直算（按股票四组同算）→ 质检 → 宽表 + R2 → 回收。不写 bin。"""
from __future__ import annotations

import logging
import time
from pathlib import Path

from . import prepare, publish, qa, status, storage
from .catalog import ALL_GROUPS, FACTOR_VERSION, factor_ids_for_group
from .engine import calc_group_day, calc_groups_day, list_codes

_log = logging.getLogger(__name__)
# 单日失败 kind 对应的中文，批次结束按类打印。
_DAY_FAIL_LABELS = {"qa": "质检", "extract": "解压", "compute": "计算"}


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


def _record_day_failure(
    *,
    trade_date: str,
    books_root: Path,
    output_root: Path,
    delete_sources: bool,
    summary: dict[str, object],
    error: str,
    kind: str,
) -> dict[str, object]:
    """单日失败：记账、删解压留 7z，不阻断后续日期。"""
    fail_kind = kind if kind in storage.DAY_FAIL_KINDS else "qa"
    label = _DAY_FAIL_LABELS.get(fail_kind, fail_kind)
    summary["failed"] = int(summary["failed"]) + 1
    summary["status"] = "failed"
    summary["error"] = error
    summary["fail_kind"] = fail_kind
    summary["qa_failed"] = fail_kind == "qa"
    print(f"[{trade_date}] {label}失败: {error}", flush=True)
    storage.mark_day_failed(output_root, trade_date, error, kind=fail_kind)
    _recycle_day_sources(
        trade_date, books_root, delete_sources=delete_sources, keep_archive=True, summary=summary
    )
    return summary


def _record_qa_failure(
    *,
    trade_date: str,
    books_root: Path,
    output_root: Path,
    delete_sources: bool,
    summary: dict[str, object],
    error: str,
) -> dict[str, object]:
    """质检失败：走统一日失败路径。"""
    return _record_day_failure(
        trade_date=trade_date,
        books_root=books_root,
        output_root=output_root,
        delete_sources=delete_sources,
        summary=summary,
        error=error,
        kind="qa",
    )


def _day_fail_kind(day: dict[str, object]) -> str:
    """从当日结果读失败类别；旧字段 qa_failed 仍当质检。"""
    kind = str(day.get("fail_kind") or "").strip()
    if kind in storage.DAY_FAIL_KINDS:
        return kind
    if day.get("qa_failed"):
        return "qa"
    if int(day.get("failed") or 0) > 0:
        return "compute"
    return ""


def _empty_batch_summary(groups: list[str] | None) -> dict[str, object]:
    """批次汇总：三类日失败分开记，便于结束时按 kind 打印。"""
    return {
        "dates": [],
        "groups": list(groups) if groups else list(ALL_GROUPS),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "rows": 0,
        "deleted": [],
        "r2_failed": [],
        "qa_failed": [],
        "extract_failed": [],
        "compute_failed": [],
    }


def _collect_day_result(summary: dict[str, object], trade_date: str, day: dict[str, object]) -> None:
    """把一日结果并进批次汇总；失败只记账，不停止后续。"""
    summary["processed"] = int(summary["processed"]) + int(day.get("processed") or 0)
    summary["skipped"] = int(summary["skipped"]) + int(day.get("skipped") or 0)
    summary["failed"] = int(summary["failed"]) + int(day.get("failed") or 0)
    summary["deleted"].extend(day.get("deleted") or [])  # type: ignore[union-attr]
    summary["r2_failed"].extend(day.get("r2_failed") or [])  # type: ignore[union-attr]
    kind = _day_fail_kind(day)
    if not kind:
        return
    key = f"{kind}_failed"
    bucket = list(summary.get(key) or [])
    bucket.append(trade_date)
    summary[key] = bucket
    label = _DAY_FAIL_LABELS.get(kind, kind)
    print(f"[{trade_date}] {label}失败已记账，继续后续", flush=True)


def _print_day_failures(summary: dict[str, object], root: Path) -> None:
    """批次结束按 kind 打印失败日期和账本路径。"""
    if summary.get("r2_failed"):
        print(f"R2 上传失败 {summary['r2_failed']}，可 reupload --failed", flush=True)
    for kind in storage.DAY_FAIL_KINDS:
        days = summary.get(f"{kind}_failed") or []
        if not days:
            continue
        label = _DAY_FAIL_LABELS.get(kind, kind)
        print(f"{label}失败 {days}，账本 {storage.qa_failed_path(root)}", flush=True)


def _safe_run_one_day(
    trade_date: str,
    *,
    books_root: Path,
    output_root: Path,
    delete_sources: bool,
    **kwargs,
) -> dict[str, object]:
    """包一层：未捕获计算异常记 compute，不把整批打挂。"""
    try:
        return run_one_day(
            trade_date,
            books_root=books_root,
            output_root=output_root,
            delete_sources=delete_sources,
            **kwargs,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        _log.exception("计算当日失败 %s", trade_date)
        summary = {
            "trade_date": trade_date,
            "processed": 0,
            "skipped": 0,
            "failed": 0,
            "status": "failed",
            "deleted": [],
            "r2_failed": [],
            "qa_failed": False,
            "fail_kind": "compute",
        }
        return _record_day_failure(
            trade_date=trade_date,
            books_root=books_root,
            output_root=output_root,
            delete_sources=delete_sources,
            summary=summary,
            error=str(exc),
            kind="compute",
        )


def _group_compute_status(ok: int, fail: int) -> str:
    """一组的日状态：失败率过高或零成功为 failed。"""
    fail_rate = fail / max(1, ok + fail)
    if fail_rate >= qa.GROUP_FAIL_NULL:
        return "failed"
    if fail == 0 and ok > 0:
        return "done"
    if ok > 0:
        return "partial"
    return "failed"


def _write_group_result(
    *,
    trade_date: str,
    factor_group: str,
    result: dict[str, object],
    output_root: Path,
    factor_version: int,
    merged: object,
) -> object:
    """把一组成功结果并进宽表并写 marker。"""
    merged = storage.merge_group_columns(
        merged,  # type: ignore[arg-type]
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
        status=str(result.get("status") or ""),
        symbol_ok=int(result.get("symbol_ok") or 0),
        symbol_fail=int(result.get("symbol_fail") or 0),
        factor_version=factor_version,
    )
    return merged


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
    day_status = _group_compute_status(ok, fail)
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
    """一日按股票四组同算 CSV；质检通过才写盘/上传。"""
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
        "r2_failed": [],
        "qa_failed": False,
        "fail_kind": "",
    }

    codes = list_codes(books_root, trade_date)
    pending: list[str] = []
    versions: dict[str, str] = {}
    for factor_group in selected:
        source_version = status.source_version_for_date(
            books_root, trade_date, factor_group, codes=codes
        )
        versions[factor_group] = source_version
        local = status.read_local_marker(output_root, trade_date, factor_group)
        if status.should_skip(
            trade_date,
            factor_group,
            source_version,
            force=force,
            factor_version=factor_version,
            local_status=local,
        ):
            summary["skipped"] = int(summary["skipped"]) + 1
            continue
        if not codes:
            summary["skipped"] = int(summary["skipped"]) + 1
            continue
        pending.append(factor_group)

    if pending:
        by_group = calc_groups_day(
            books_root,
            trade_date,
            pending,
            codes=codes,
            workers=max(1, int(workers)),
        )
        failed_groups: list[str] = []
        for factor_group in pending:
            rows, status_rows, ok, fail = by_group[factor_group]
            source_version = versions[factor_group]
            for item in status_rows:
                item["source_version"] = source_version
            state = _group_compute_status(ok, fail)
            result = {
                "status": state,
                "trade_date": trade_date,
                "factor_group": factor_group,
                "symbol_ok": ok,
                "symbol_fail": fail,
                "rows": rows,
                "status_rows": status_rows,
                "source_version": source_version,
            }
            if state == "failed":
                print(f"[{trade_date}] {factor_group} 失败，停止当日发布", flush=True)
                failed_groups.append(factor_group)
                continue
            summary["processed"] = int(summary["processed"]) + 1
            merged = _write_group_result(
                trade_date=trade_date,
                factor_group=factor_group,
                result=result,
                output_root=output_root,
                factor_version=factor_version,
                merged=merged,
            )
        if failed_groups:
            return _record_day_failure(
                trade_date=trade_date,
                books_root=books_root,
                output_root=output_root,
                delete_sources=delete_sources,
                summary=summary,
                error=f"{','.join(failed_groups)} failed",
                kind="compute",
            )

    # 停牌占位不进宽表，也不计入空值率；缺源过多则视为坏档。
    miss_rate = qa.trade_source_miss_rate(books_root, trade_date, codes)
    if miss_rate >= qa.TRADE_SOURCE_MISS_MAX:
        return _record_qa_failure(
            trade_date=trade_date,
            books_root=books_root,
            output_root=output_root,
            delete_sources=delete_sources,
            summary=summary,
            error=f"逐笔成交缺源率 {miss_rate:.3f} ≥ {qa.TRADE_SOURCE_MISS_MAX}",
        )
    if merged is not None and not merged.empty:
        merged = qa.drop_rows_without_trade_source(merged, books_root, trade_date)

    if merged is None or merged.empty:
        return _record_qa_failure(
            trade_date=trade_date,
            books_root=books_root,
            output_root=output_root,
            delete_sources=delete_sources,
            summary=summary,
            error="empty wide",
        )

    report = qa.check_day(day=trade_date, wide=merged, raw_root=books_root, old_wide=existing)
    if not report.get("ok"):
        return _record_qa_failure(
            trade_date=trade_date,
            books_root=books_root,
            output_root=output_root,
            delete_sources=delete_sources,
            summary=summary,
            error="; ".join(report.get("errors") or ["qa failed"]),
        )

    published = publish.publish_day(
        output_root,
        trade_date,
        merged,
        upload_r2=upload_r2,
        factor_version=factor_version,
    )
    # R2 失败不算当日计算失败，只记下来后面补传。
    if published.get("r2_failed"):
        print(f"[{trade_date}] R2 上传失败，已标记，后续继续", flush=True)
        summary["r2_failed"] = [trade_date]
        summary["r2_error"] = published.get("r2_error")
    if compact_months:
        try:
            storage.compact_month(output_root, trade_date[:6], factor_version=factor_version, delete_days=False)
        except Exception:
            _log.warning("月 compact 失败 %s", trade_date[:6], exc_info=True)
    print(f"[{trade_date}] 质检通过，已写宽表", flush=True)
    storage.clear_day_failed(output_root, trade_date)
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
    """显式日期列表：按日串行。单日失败记账后继续。"""
    del seed_defs  # D1 不迁
    root = Path(output_root) if output_root is not None else storage.default_root()
    root.mkdir(parents=True, exist_ok=True)
    books = Path(books_root if books_root is not None else parquet_root or prepare.default_raw_root())
    ordered = sorted({str(d) for d in dates if str(d).isdigit() and len(str(d)) == 8})
    summary = _empty_batch_summary(groups)
    summary["dates"] = ordered
    for trade_date in ordered:
        day = _safe_run_one_day(
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
        _collect_day_result(summary, trade_date, day)
    _print_day_failures(summary, root)
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
    """扫描 raw：已解压优先，再逐日解压。单日失败记账后续跑；下载中则等待续扫。"""
    del seed_defs
    root = Path(raw_root) if raw_root is not None else prepare.default_raw_root()
    out = Path(output_root) if output_root is not None else storage.default_root()
    only = list(dates) if dates else None
    allow_rescan = only is None
    wait_downloads = allow_rescan and not dry_run and not local_only
    summary = _empty_batch_summary(groups)
    seen: set[str] = set()
    wave = 0

    while True:
        local_dates, z7_dates, discovered = _discover_batch_dates(
            root, only=only, local_only=local_only, z7_only=z7_only
        )
        day_skip = set() if force else set(storage.list_day_failed(out))
        planned = [d for d in discovered if d not in seen and d not in day_skip]
        if day_skip:
            skipped_now = [d for d in discovered if d in day_skip and d not in seen]
            if skipped_now:
                print(f"  跳过日失败账本: {skipped_now}", flush=True)
                seen.update(skipped_now)
        if not planned:
            downloading = prepare.discover_downloading_dates(root, only) if wait_downloads else []
            if downloading:
                poll = prepare.download_poll_seconds()
                print(f"等待下载完成 {downloading}，{poll}s 后重扫", flush=True)
                time.sleep(poll)
                continue
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
            print(f"  阶段A local 明细: {local_dates}", flush=True)
        if z7_dates:
            print(f"  阶段B 7z 串行: {z7_dates}", flush=True)

        for trade_date in planned:
            seen.add(trade_date)
            try:
                prepare.ensure_day_csv(trade_date, root, dry_run=dry_run)
            except Exception as exc:
                _log.exception("准备明细失败 %s", trade_date)
                day = _record_day_failure(
                    trade_date=trade_date,
                    books_root=root,
                    output_root=out,
                    delete_sources=delete_sources,
                    summary={
                        "trade_date": trade_date,
                        "processed": 0,
                        "skipped": 0,
                        "failed": 0,
                        "status": "failed",
                        "deleted": [],
                        "r2_failed": [],
                        "qa_failed": False,
                        "fail_kind": "extract",
                    },
                    error=str(exc),
                    kind="extract",
                )
                summary["dates"].append(trade_date)  # type: ignore[union-attr]
                _collect_day_result(summary, trade_date, day)
                continue

            if dry_run:
                codes = list_codes(root, trade_date)
                print(f"[{trade_date}] dry-run symbols={len(codes)}", flush=True)
                summary["dates"].append(trade_date)  # type: ignore[union-attr]
                summary["skipped"] = int(summary["skipped"]) + 1
                continue

            day = _safe_run_one_day(
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
            _collect_day_result(summary, trade_date, day)

        if not allow_rescan:
            break

    _print_day_failures(summary, out)
    return summary
