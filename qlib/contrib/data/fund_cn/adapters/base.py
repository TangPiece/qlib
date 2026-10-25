"""Source adapter SPI — unify rows into pit_fundamental / staging shapes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class StatementFieldRow:
    instrument_key: str
    report_period_end: date
    report_period_start: date | None
    fiscal_year: int
    fiscal_quarter: int
    statement_type: str  # balance | income | cashflow | indicator
    field_code: str
    value: float
    currency: str = "CNY"
    source: str = ""
    source_record_id: str = ""
    revision: int = 0


@dataclass(frozen=True)
class AnnounceRow:
    instrument_key: str
    report_period_end: date
    publish_time: datetime
    available_time: datetime
    revision: int = 0
    is_restatement: bool = False
    source: str = ""
    source_record_id: str = ""


@dataclass(frozen=True)
class ShareCapitalRow:
    """Total share capital change event (总股本, in shares — not 万股)."""

    instrument_key: str
    effective_date: date
    announce_time: datetime
    total_shares: float
    source: str = ""
    source_record_id: str = ""


class StatementAdapter(Protocol):
    def fetch_statements(
        self,
        instrument_keys: Sequence[str],
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[StatementFieldRow]:
        ...


class AnnounceAdapter(Protocol):
    def fetch_announcements(
        self,
        instrument_keys: Sequence[str],
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[AnnounceRow]:
        ...


class ShareCapitalAdapter(Protocol):
    def fetch_share_capital(
        self,
        instrument_keys: Sequence[str],
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[ShareCapitalRow]:
        ...


def instrument_to_a_code(instrument_key: str) -> str:
    """CNStock:000001 → 000001."""
    text = str(instrument_key or "").strip()
    if ":" in text:
        return text.split(":", 1)[1].zfill(6)
    return text.zfill(6)


def a_code_to_instrument(code: str, *, market: str = "CNStock") -> str:
    return f"{market}:{str(code).zfill(6)}"


def rows_as_dicts(rows: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        if hasattr(row, "__dataclass_fields__"):
            out.append({k: getattr(row, k) for k in row.__dataclass_fields__})
        elif isinstance(row, dict):
            out.append(dict(row))
        else:
            raise TypeError(type(row))
    return out
