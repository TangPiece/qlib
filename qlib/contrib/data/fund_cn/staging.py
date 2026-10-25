"""Local staging for standardized statements + announce calendar (raw SSOT before PIT)."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from .adapters.base import AnnounceRow, ShareCapitalRow, StatementFieldRow, rows_as_dicts
from .paths_local import (
    announce_dir,
    default_staging_root,
    markers_dir,
    share_capital_dir,
    statements_dir,
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(type(value))


def write_statement_rows(
    rows: Sequence[StatementFieldRow],
    *,
    root: Path | None = None,
    part_name: str = "statements.parquet",
) -> Path:
    dest_dir = statements_dir(root)
    dest = dest_dir / part_name
    frame = pd.DataFrame(rows_as_dicts(rows))
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "instrument_key",
                "report_period_end",
                "report_period_start",
                "fiscal_year",
                "fiscal_quarter",
                "statement_type",
                "field_code",
                "value",
                "currency",
                "source",
                "source_record_id",
                "revision",
            ]
        )
    temporary = dest.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(dest)
    return dest


def read_statement_rows(root: Path | None = None, part_name: str = "statements.parquet") -> pd.DataFrame:
    path = statements_dir(root) / part_name
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


def write_announce_rows(
    rows: Sequence[AnnounceRow],
    *,
    root: Path | None = None,
    part_name: str = "announce.parquet",
) -> Path:
    dest_dir = announce_dir(root)
    dest = dest_dir / part_name
    frame = pd.DataFrame(rows_as_dicts(rows))
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "instrument_key",
                "report_period_end",
                "publish_time",
                "available_time",
                "revision",
                "is_restatement",
                "source",
                "source_record_id",
            ]
        )
    temporary = dest.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(dest)
    return dest


def read_announce_rows(root: Path | None = None, part_name: str = "announce.parquet") -> pd.DataFrame:
    path = announce_dir(root) / part_name
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_parquet(path)


SHARE_CAPITAL_COLUMNS = [
    "instrument_key",
    "effective_date",
    "announce_time",
    "total_shares",
    "source",
    "source_record_id",
]


def write_share_capital_rows(
    rows: Sequence[ShareCapitalRow],
    *,
    root: Path | None = None,
    part_name: str = "share_capital.parquet",
) -> Path:
    dest = share_capital_dir(root) / part_name
    frame = pd.DataFrame(rows_as_dicts(rows))
    if frame.empty:
        frame = pd.DataFrame(columns=SHARE_CAPITAL_COLUMNS)
    temporary = dest.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(dest)
    return dest


def read_share_capital_rows(
    root: Path | None = None, part_name: str = "share_capital.parquet"
) -> pd.DataFrame:
    path = share_capital_dir(root) / part_name
    if not path.is_file():
        return pd.DataFrame(columns=SHARE_CAPITAL_COLUMNS)
    return pd.read_parquet(path)


def write_marker(
    job: str,
    payload: dict[str, Any],
    *,
    root: Path | None = None,
) -> Path:
    path = markers_dir(root) / f"{job}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, default=_json_default, indent=2), encoding="utf-8")
    return path


def read_marker(job: str, *, root: Path | None = None) -> dict[str, Any] | None:
    path = markers_dir(root) / f"{job}.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def staging_source_version(root: Path | None = None) -> str:
    import hashlib

    base = root or default_staging_root()
    digest = hashlib.sha256()
    for rel in (
        "statements/statements.parquet",
        "announce/announce.parquet",
        "share_capital/share_capital.parquet",
    ):
        path = base / rel
        if not path.is_file():
            digest.update(f"missing:{rel}".encode())
            continue
        stat = path.stat()
        digest.update(f"{rel}:{stat.st_size}:{int(stat.st_mtime_ns)}".encode())
    return digest.hexdigest()[:16]
