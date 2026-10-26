"""本地日/月 Parquet 与 R2 按日备份。storage_uri 只存逻辑键。"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .catalog import FACTOR_VERSION, factor_ids, wide_columns
from .r2_env import load_r2_env
from .util import canonical_symbol

_log = logging.getLogger(__name__)
_uploader = None
_client = None
# 上传失败 / 日失败账本，与 _markers 并列，不进 R2。
R2_FAILED_NAME = "_r2_failed.json"
# 文件名沿用 _qa_failed.json，条目用 kind 区分质检/解压/计算。
QA_FAILED_NAME = "_qa_failed.json"
DAY_FAIL_KINDS = ("qa", "extract", "compute")


def set_r2_uploader(uploader) -> None:
    """测试注入：``uploader(logical_key: str, payload: bytes)``。"""
    global _uploader
    _uploader = uploader


def default_root() -> Path:
    from .prepare import default_output_root

    return default_output_root()


def r2_prefix() -> str:
    return os.getenv("R2_L2_DAILY_PREFIX", "factors/l2_daily").strip().strip("/") or "factors/l2_daily"


def version_segment(factor_version: int = FACTOR_VERSION) -> str:
    """路径版本段：v1（非 version=1）。"""
    return f"v{int(factor_version)}"


def logical_day_key(trade_date: str, *, factor_version: int = FACTOR_VERSION) -> str:
    day = str(trade_date)
    return f"{r2_prefix()}/{version_segment(factor_version)}/{day[:4]}/{day[:6]}/{day}.parquet"


def logical_status_key(trade_date: str, *, factor_version: int = FACTOR_VERSION) -> str:
    day = str(trade_date)
    return (
        f"{r2_prefix()}/{version_segment(factor_version)}/"
        f"{day[:4]}/{day[:6]}/{day}.status.parquet"
    )


def month_dir(root: Path, trade_date: str, *, factor_version: int = FACTOR_VERSION) -> Path:
    day = str(trade_date)
    path = root / version_segment(factor_version) / day[:4] / day[:6]
    path.mkdir(parents=True, exist_ok=True)
    return path


def day_path(root: Path, trade_date: str, *, factor_version: int = FACTOR_VERSION) -> Path:
    return month_dir(root, trade_date, factor_version=factor_version) / f"{trade_date}.parquet"


def status_path(root: Path, trade_date: str, *, factor_version: int = FACTOR_VERSION) -> Path:
    return month_dir(root, trade_date, factor_version=factor_version) / f"{trade_date}.status.parquet"


def month_part_path(root: Path, yyyymm: str, *, factor_version: int = FACTOR_VERSION) -> Path:
    return root / version_segment(factor_version) / yyyymm[:4] / yyyymm / "part-000.parquet"


def _coalesce_series(series: pd.Series):
    """同一 key 多行时取最后一个非空，避免旧 .sh 空行盖住新值。"""
    valid = series.dropna()
    if valid.empty:
        return series.iloc[-1]
    return valid.iloc[-1]


def canonicalize_panel(frame: pd.DataFrame | None, keys: list[str] | None = None) -> pd.DataFrame:
    """symbol 统一成 .SH/.SZ，并把大小写重复行折成一行。"""
    if frame is None or frame.empty or "symbol" not in frame.columns:
        return frame if frame is not None else pd.DataFrame()
    out = frame.copy()
    out["symbol"] = [canonical_symbol(item) if pd.notna(item) else item for item in out["symbol"]]
    if keys is None:
        keys = ["trade_date", "symbol"] if "trade_date" in out.columns else ["symbol"]
    keys = [key for key in keys if key in out.columns]
    if not keys or not out.duplicated(subset=keys).any():
        return out
    grouped = out.groupby(keys, as_index=False, sort=False)
    return grouped.agg({col: _coalesce_series for col in out.columns if col not in keys})


def read_day(root: Path, trade_date: str, *, factor_version: int = FACTOR_VERSION) -> pd.DataFrame | None:
    path = day_path(root, trade_date, factor_version=factor_version)
    if not path.is_file():
        return None
    return canonicalize_panel(pd.read_parquet(path))


def write_day(
    root: Path,
    trade_date: str,
    frame: pd.DataFrame,
    *,
    factor_version: int = FACTOR_VERSION,
) -> Path:
    """原子写入日宽表，补齐缺失因子列。"""
    columns = wide_columns()
    panel = frame.copy()
    if "trade_date" not in panel.columns:
        panel["trade_date"] = str(trade_date)
    if "symbol" not in panel.columns:
        raise ValueError("frame 缺少 symbol")
    panel = canonicalize_panel(panel)
    for column in columns:
        if column not in panel.columns:
            panel[column] = float("nan")
    panel = panel[columns]
    destination = day_path(root, trade_date, factor_version=factor_version)
    temporary = destination.with_suffix(".parquet.tmp")
    panel.to_parquet(temporary, index=False)
    temporary.replace(destination)
    return destination


def merge_group_columns(
    existing: pd.DataFrame | None,
    rows: list[dict],
    group_columns: list[str],
    trade_date: str,
) -> pd.DataFrame:
    """按规范化 symbol 合并一组新列到日宽表。"""
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        if existing is not None:
            return canonicalize_panel(existing)
        empty = pd.DataFrame(columns=wide_columns())
        return empty
    incoming["trade_date"] = str(trade_date)
    keep = ["trade_date", "symbol", *group_columns]
    for column in keep:
        if column not in incoming.columns and column not in ("trade_date", "symbol"):
            incoming[column] = float("nan")
    incoming = incoming[[c for c in keep if c in incoming.columns]]
    incoming = canonicalize_panel(incoming)

    if existing is None or existing.empty:
        base = incoming
    else:
        existing = canonicalize_panel(existing)
        base = existing.drop(columns=[c for c in group_columns if c in existing.columns], errors="ignore")
        base = base.merge(incoming, on=["trade_date", "symbol"], how="outer")
    for column in wide_columns():
        if column not in base.columns:
            base[column] = float("nan")
    return canonicalize_panel(base[wide_columns()])


def write_status(
    root: Path,
    trade_date: str,
    status_rows: list[dict],
    *,
    factor_version: int = FACTOR_VERSION,
) -> Path:
    """写入/合并逐股×组状态。"""
    path = status_path(root, trade_date, factor_version=factor_version)
    status_keys = ["symbol", "factor_group"]
    frame = canonicalize_panel(pd.DataFrame(status_rows), keys=status_keys)
    if path.is_file() and not frame.empty:
        old = canonicalize_panel(pd.read_parquet(path), keys=status_keys)
        # 同 symbol+group 覆盖
        old = old.set_index(status_keys)
        frame = frame.set_index(status_keys)
        old.update(frame)
        combined = pd.concat([old, frame[~frame.index.isin(old.index)]])
        frame = canonicalize_panel(combined.reset_index(), keys=status_keys)
    elif frame.empty and path.is_file():
        return path
    temporary = path.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)
    return path


def upload_day_to_r2(root: Path, trade_date: str, *, factor_version: int = FACTOR_VERSION) -> str | None:
    """上传日宽表到 R2，返回逻辑键。未配置返回 None；put 失败抛错。"""
    load_r2_env()
    path = day_path(root, trade_date, factor_version=factor_version)
    if not path.is_file():
        return None
    key = logical_day_key(trade_date, factor_version=factor_version)
    payload = path.read_bytes()
    if _uploader is not None:
        _uploader(key, payload)
        return key
    if not _r2_configured():
        return None
    try:
        _r2_client().put_object(Bucket=_bucket(), Key=key, Body=payload)
    except Exception:
        _log.warning("l2_daily R2 上传失败 %s", trade_date, exc_info=True)
        raise
    return key


def r2_failed_path(root: Path) -> Path:
    """上传失败账本路径：{output}/_r2_failed.json。"""
    return Path(root) / R2_FAILED_NAME


def _read_r2_failed(root: Path) -> dict[str, object]:
    """读账本；坏文件当空。"""
    path = r2_failed_path(root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_r2_failed(root: Path, payload: dict[str, object]) -> None:
    """原子写账本。"""
    path = r2_failed_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def mark_r2_failed(
    root: Path,
    trade_date: str,
    error: object,
    *,
    factor_version: int = FACTOR_VERSION,
) -> None:
    """记下某日上传失败，供之后 reupload --failed。"""
    payload = _read_r2_failed(root)
    payload[str(trade_date)] = {
        "error": str(error)[:500],
        "key": logical_day_key(trade_date, factor_version=factor_version),
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    _write_r2_failed(root, payload)


def clear_r2_failed(root: Path, trade_date: str) -> None:
    """上传成功后清掉该日失败记录。"""
    payload = _read_r2_failed(root)
    if str(trade_date) not in payload:
        return
    payload.pop(str(trade_date), None)
    if payload:
        _write_r2_failed(root, payload)
        return
    r2_failed_path(root).unlink(missing_ok=True)


def list_r2_failed(root: Path) -> list[str]:
    """账本里的失败交易日，升序。"""
    return sorted(day for day in _read_r2_failed(root) if str(day).isdigit() and len(str(day)) == 8)


def qa_failed_path(root: Path) -> Path:
    """日失败账本：{output}/_qa_failed.json（含 qa/extract/compute）。"""
    return Path(root) / QA_FAILED_NAME


def _normalize_fail_kind(kind: object) -> str:
    """缺 kind 的旧记录当质检失败。"""
    value = str(kind or "qa").strip().lower()
    return value if value in DAY_FAIL_KINDS else "qa"


def _day_fail_kind(item: object) -> str:
    """从账本条目读 kind；非 dict 或缺失都当 qa。"""
    if isinstance(item, dict):
        return _normalize_fail_kind(item.get("kind"))
    return "qa"


def _read_qa_failed(root: Path) -> dict[str, object]:
    """读日失败账本；坏文件当空。"""
    path = qa_failed_path(root)
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _write_qa_failed(root: Path, payload: dict[str, object]) -> None:
    """原子写日失败账本。"""
    path = qa_failed_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def mark_day_failed(root: Path, trade_date: str, error: object, *, kind: str = "qa") -> None:
    """记下某日失败（qa/extract/compute），供之后统一处理。"""
    payload = _read_qa_failed(root)
    payload[str(trade_date)] = {
        "error": str(error)[:500],
        "kind": _normalize_fail_kind(kind),
        "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    _write_qa_failed(root, payload)


def mark_qa_failed(root: Path, trade_date: str, error: object) -> None:
    """兼容旧名：记下质检失败。"""
    mark_day_failed(root, trade_date, error, kind="qa")


def clear_day_failed(root: Path, trade_date: str) -> None:
    """发布成功后清掉该日失败记录。"""
    payload = _read_qa_failed(root)
    if str(trade_date) not in payload:
        return
    payload.pop(str(trade_date), None)
    if payload:
        _write_qa_failed(root, payload)
        return
    qa_failed_path(root).unlink(missing_ok=True)


def clear_qa_failed(root: Path, trade_date: str) -> None:
    """兼容旧名：清掉该日失败记录。"""
    clear_day_failed(root, trade_date)


def list_day_failed(root: Path, *, kind: str | None = None) -> list[str]:
    """日失败账本里的交易日，升序；kind 可筛 qa/extract/compute。"""
    wanted = None if kind is None else _normalize_fail_kind(kind)
    dates: list[str] = []
    for day, item in _read_qa_failed(root).items():
        if not (str(day).isdigit() and len(str(day)) == 8):
            continue
        if wanted is not None and _day_fail_kind(item) != wanted:
            continue
        dates.append(str(day))
    return sorted(dates)


def list_qa_failed(root: Path) -> list[str]:
    """账本里 kind=qa（含旧记录）的交易日。"""
    return list_day_failed(root, kind="qa")


def _day_parquet_files(directory: Path) -> list[Path]:
    """YYYYMMDD.parquet，排除 status / part。"""
    out: list[Path] = []
    for path in directory.glob("????????.parquet"):
        stem = path.stem
        if len(stem) == 8 and stem.isdigit():
            out.append(path)
    return sorted(out)


def compact_month(
    root: Path,
    yyyymm: str,
    *,
    factor_version: int = FACTOR_VERSION,
    delete_days: bool = False,
) -> Path | None:
    """合并当月全部 YYYYMMDD.parquet 为 part-000.parquet。"""
    directory = root / version_segment(factor_version) / yyyymm[:4] / yyyymm
    if not directory.is_dir():
        return None
    parts = _day_parquet_files(directory)
    if not parts:
        return None
    frames = [pd.read_parquet(path) for path in parts]
    panel = pd.concat(frames, ignore_index=True)
    for column in wide_columns():
        if column not in panel.columns:
            panel[column] = float("nan")
    panel = panel[wide_columns()]
    destination = month_part_path(root, yyyymm, factor_version=factor_version)
    temporary = destination.with_suffix(".parquet.tmp")
    panel.to_parquet(temporary, index=False)
    temporary.replace(destination)
    if delete_days:
        for path in parts:
            path.unlink(missing_ok=True)
    return destination


def list_local_day_files(root: Path, yyyymm: str, *, factor_version: int = FACTOR_VERSION) -> list[str]:
    directory = root / version_segment(factor_version) / yyyymm[:4] / yyyymm
    if not directory.is_dir():
        return []
    return [path.stem for path in _day_parquet_files(directory)]


def _legacy_version_dirs(root: Path, factor_version: int) -> list[Path]:
    """旧版目录：version=N 以及仍带 Hive 子路径的同名根。"""
    out: list[Path] = []
    for name in (f"version={factor_version}", version_segment(factor_version)):
        path = Path(root) / name
        if path.is_dir():
            out.append(path)
    return out


def legacy_logical_day_keys(trade_date: str, *, factor_version: int = FACTOR_VERSION) -> list[str]:
    """历史上可能出现过的日宽表 R2 键（不含当前 vN 新键）。"""
    day = str(trade_date)
    prefix = r2_prefix()
    return [
        f"{prefix}/version={factor_version}/year={day[:4]}/month={day[:6]}/date={day}.parquet",
        f"{prefix}/version={factor_version}/{day[:4]}/{day[:6]}/{day}.parquet",
    ]


def legacy_day_candidates(root: Path, trade_date: str, *, factor_version: int = FACTOR_VERSION) -> list[Path]:
    day = str(trade_date)
    return [
        Path(root) / f"version={factor_version}" / day[:4] / day[:6] / f"{day}.parquet",
        Path(root)
        / f"version={factor_version}"
        / f"year={day[:4]}"
        / f"month={day[:6]}"
        / f"date={day}.parquet",
        Path(root)
        / version_segment(factor_version)
        / f"year={day[:4]}"
        / f"month={day[:6]}"
        / f"date={day}.parquet",
    ]


def discover_local_day_dates(root: Path, *, factor_version: int = FACTOR_VERSION) -> list[str]:
    """扫描 vN / version=N 下新旧布局日宽表日期。"""
    dates: set[str] = set()
    for base in _legacy_version_dirs(root, factor_version):
        for path in base.rglob("????????.parquet"):
            if path.name.endswith(".status.parquet") or path.name.startswith("part-"):
                continue
            stem = path.stem
            if len(stem) == 8 and stem.isdigit():
                dates.add(stem)
        for path in base.rglob("date=????????.parquet"):
            if path.name.endswith(".status.parquet"):
                continue
            stem = path.name[len("date=") : -len(".parquet")]
            if len(stem) == 8 and stem.isdigit():
                dates.add(stem)
    return sorted(dates)


def migrate_hive_layout(root: Path, *, factor_version: int = FACTOR_VERSION) -> dict[str, object]:
    """把本地旧路径（version= / year= / date=）迁到 vN/YYYY/YYYYMM/YYYYMMDD。"""
    moved: list[str] = []
    skipped: list[str] = []
    sources = _legacy_version_dirs(root, factor_version)
    # 若只有 version=N，也要扫它
    version_eq = Path(root) / f"version={factor_version}"
    if version_eq.is_dir() and version_eq not in sources:
        sources.append(version_eq)

    for base in sources:
        hive_day_files = list(base.rglob("date=????????.parquet")) + list(
            base.rglob("date=????????.status.parquet")
        )
        # 已是 YYYYMMDD 但挂在 version=N 下
        plain_day_files = [
            p
            for p in base.rglob("????????.parquet")
            if len(p.stem) == 8 and p.stem.isdigit() and not p.name.endswith(".status.parquet")
        ]
        plain_status_files = list(base.rglob("????????.status.parquet"))

        for src in sorted(set(hive_day_files + plain_day_files + plain_status_files)):
            name = src.name
            is_status = name.endswith(".status.parquet")
            if name.startswith("date="):
                day = (
                    name[len("date=") : -len(".status.parquet")]
                    if is_status
                    else name[len("date=") : -len(".parquet")]
                )
            else:
                day = name[: -len(".status.parquet")] if is_status else src.stem
            if not (len(day) == 8 and day.isdigit()):
                continue
            dest = (
                status_path(root, day, factor_version=factor_version)
                if is_status
                else day_path(root, day, factor_version=factor_version)
            )
            if dest.resolve() == src.resolve():
                continue
            if dest.is_file():
                skipped.append(str(src))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved.append(f"{src} -> {dest}")

        for src in sorted(base.rglob("part-000.parquet")):
            parent = src.parent
            # version=N/YYYY/YYYYMM 或 year=/month=
            if parent.name.startswith("month="):
                yyyymm = parent.name[len("month=") :]
            elif len(parent.name) == 6 and parent.name.isdigit():
                yyyymm = parent.name
            else:
                continue
            if not (len(yyyymm) == 6 and yyyymm.isdigit()):
                continue
            dest = month_part_path(root, yyyymm, factor_version=factor_version)
            if dest.resolve() == src.resolve():
                continue
            if dest.is_file():
                skipped.append(str(src))
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            moved.append(f"{src} -> {dest}")

    # 清理空的旧 version= 目录树
    old_root = Path(root) / f"version={factor_version}"
    if old_root.is_dir():
        for dirpath, dirnames, filenames in os.walk(old_root, topdown=False):
            path = Path(dirpath)
            try:
                if not any(path.iterdir()):
                    path.rmdir()
            except OSError:
                pass

    return {"moved": moved, "skipped": skipped}


def reupload_local_days(
    root: Path | None = None,
    *,
    dates: list[str] | None = None,
    factor_version: int = FACTOR_VERSION,
    delete_old_r2: bool = True,
    dry_run: bool = False,
) -> dict[str, object]:
    """迁移本地旧布局后，按 vN/... 新键重传日宽表；可选删除 R2 旧键。"""
    out_root = Path(root) if root is not None else default_root()
    migrate = {"moved": [], "skipped": []}
    if not dry_run:
        migrate = migrate_hive_layout(out_root, factor_version=factor_version)
    else:
        print(f"dry-run migrate root={out_root}", flush=True)

    selected = dates if dates is not None else discover_local_day_dates(out_root, factor_version=factor_version)
    uploaded: list[str] = []
    missing: list[str] = []
    deleted_old: list[str] = []
    r2_failed: list[str] = []

    for trade_date in selected:
        new_path = day_path(out_root, trade_date, factor_version=factor_version)
        path = new_path if new_path.is_file() else None
        if path is None:
            for candidate in legacy_day_candidates(out_root, trade_date, factor_version=factor_version):
                if candidate.is_file():
                    path = candidate
                    break
        key = logical_day_key(trade_date, factor_version=factor_version)
        if path is None or not path.is_file():
            missing.append(trade_date)
            continue
        if dry_run:
            print(f"[{trade_date}] dry-run upload {path} -> {key}", flush=True)
            uploaded.append(key)
            continue
        if path.resolve() != new_path.resolve():
            new_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(new_path))
        # 单日 put 失败记账并继续后面的天，不阻断补传。
        try:
            uri = upload_day_to_r2(out_root, trade_date, factor_version=factor_version)
        except Exception as exc:
            _log.warning("l2_daily 补传失败 %s", trade_date, exc_info=True)
            mark_r2_failed(out_root, trade_date, exc, factor_version=factor_version)
            r2_failed.append(trade_date)
            print(f"[{trade_date}] R2 上传失败，已标记，后续继续", flush=True)
            continue
        if uri:
            clear_r2_failed(out_root, trade_date)
            uploaded.append(uri)
            print(f"[{trade_date}] uploaded {uri}", flush=True)
        else:
            missing.append(trade_date)

        if delete_old_r2 and _r2_configured():
            for old_key in legacy_logical_day_keys(trade_date, factor_version=factor_version):
                try:
                    _r2_client().delete_object(Bucket=_bucket(), Key=old_key)
                    deleted_old.append(old_key)
                    print(f"[{trade_date}] deleted old {old_key}", flush=True)
                except Exception:
                    _log.warning("删除旧 R2 键失败 %s", old_key, exc_info=True)

    return {
        "root": str(out_root),
        "migrate": migrate,
        "dates": selected,
        "uploaded": uploaded,
        "missing": missing,
        "r2_failed": r2_failed,
        "deleted_old": deleted_old,
        "dry_run": dry_run,
    }


def r2_configured() -> bool:
    """四个 R2 变量都非空才算配好。会先尝试 load 本仓库 .env。"""
    load_r2_env()
    return _r2_configured()


def _r2_configured() -> bool:
    return bool(
        os.getenv("R2_ENDPOINT_URL", "").strip()
        and os.getenv("R2_ACCESS_KEY_ID", "").strip()
        and os.getenv("R2_SECRET_ACCESS_KEY", "").strip()
        and _bucket()
    )


def _bucket() -> str:
    return os.getenv("R2_BUCKET", "").strip()


def _r2_client():
    global _client
    if _client is not None:
        return _client
    import boto3
    from botocore.config import Config

    _client = boto3.client(
        "s3",
        endpoint_url=os.getenv("R2_ENDPOINT_URL", "").strip(),
        aws_access_key_id=os.getenv("R2_ACCESS_KEY_ID", "").strip(),
        aws_secret_access_key=os.getenv("R2_SECRET_ACCESS_KEY", "").strip(),
        region_name=os.getenv("R2_REGION", "auto").strip() or "auto",
        config=Config(signature_version="s3v4"),
    )
    return _client


# 供测试断言列数
assert len(factor_ids()) == 100
