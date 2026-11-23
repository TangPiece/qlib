# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""国泰安季度合并宽表 → StatementFieldRow（科目主源）。

只读本地 CSV/xlsx 映射出的宽表，不拉外网。滤 ``报表类型编码=A``（合并报表）。
差错更正行：有「差错更正披露日期」则 revision=1；公告日历由 EM NOTICE 补洞，
本适配器可附带 industry 映射供 non_financial 掩码使用。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from ..source_policy import default_csmar_quarterly_path
from .announce import as_utc
from .akshare_adapter import merge_interest_bearing
from .base import (
    AnnounceRow,
    StatementFieldRow,
    a_code_to_instrument,
    instrument_to_a_code,
)

_log = logging.getLogger(__name__)

# field_code → 国泰安中文列（按优先级）
_FIELD_COLUMNS: dict[str, tuple[str, ...]] = {
    # 银行等金融业「营业收入」常为空，回退「营业总收入」
    "revenue": ("营业收入", "营业总收入"),
    "operating_cost": ("营业成本",),
    "net_profit_parent": ("归属于母公司所有者的净利润",),
    "net_profit_deducted": ("归属于上市公司股东的扣除非经常性损益的净利润",),
    "operating_profit": ("营业利润",),
    "selling_expense": ("销售费用",),
    "admin_expense": ("管理费用",),
    "rd_expense": ("研发费用",),
    "financial_expense": ("财务费用",),
    "cash_from_operations": ("经营活动产生的现金流量净额",),
    "capex": ("购建固定资产、无形资产和其他长期资产支付的现金",),
    "total_assets": ("资产总计",),
    "total_liabilities": ("负债合计",),
    "shareholder_equity": ("归属于母公司所有者权益合计",),
    "current_assets": ("流动资产合计",),
    "current_liabilities": ("流动负债合计",),
    "accounts_receivable": ("应收账款净额", "应收账款"),
    "inventory": ("存货净额", "存货"),
    "short_term_debt": ("短期借款",),
    "long_term_debt": ("长期借款",),
    "bonds_payable": ("应付债券",),
    "shares_outstanding": ("实收资本(或股本)",),
}

_FIELD_STATEMENT_TYPE: dict[str, str] = {
    "revenue": "income",
    "operating_cost": "income",
    "net_profit_parent": "income",
    "net_profit_deducted": "income",
    "operating_profit": "income",
    "selling_expense": "income",
    "admin_expense": "income",
    "rd_expense": "income",
    "financial_expense": "income",
    "cash_from_operations": "cashflow",
    "capex": "cashflow",
    "total_assets": "balance",
    "total_liabilities": "balance",
    "shareholder_equity": "balance",
    "current_assets": "balance",
    "current_liabilities": "balance",
    "accounts_receivable": "balance",
    "inventory": "balance",
    "short_term_debt": "balance",
    "long_term_debt": "balance",
    "bonds_payable": "balance",
    "shares_outstanding": "balance",
}

_ID_COLS = (
    "证券代码",
    "季度日期",
    "报表类型编码",
    "是否发生差错更正",
    "差错更正披露日期",
    "行业代码",
    "行业名称",
    "行业代码1",
)


@dataclass(frozen=True)
class IndustryMapRow:
    """instrument → 行业代码（用于 non_financial 掩码）。"""

    instrument_key: str
    industry_code: str
    industry_name: str = ""
    as_of: date | None = None


def _fiscal_quarter(period_end: date) -> int:
    return (period_end.month - 1) // 3 + 1


def _parse_date(value: object) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _float(value: object) -> float | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(out) else out


def _is_truthy_correct(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "y", "yes", "true", "是", "t"}


def _usecols(available: Iterable[str]) -> list[str]:
    cols = set(available)
    wanted = [c for c in _ID_COLS if c in cols]
    for candidates in _FIELD_COLUMNS.values():
        for col in candidates:
            if col in cols and col not in wanted:
                wanted.append(col)
    return wanted


def dataframe_to_csmar_rows(
    frame: pd.DataFrame,
    *,
    source: str = "csmar",
    instrument_keys: Sequence[str] | None = None,
    start: date | None = None,
    end: date | None = None,
) -> tuple[list[StatementFieldRow], list[AnnounceRow], list[IndustryMapRow]]:
    """宽表 → 科目长表 + 差错更正公告（若有披露日）+ 行业映射。"""
    if frame is None or frame.empty:
        return [], [], []
    work = frame.copy()
    if "报表类型编码" in work.columns:
        work = work[work["报表类型编码"].astype(str).str.upper().str.strip() == "A"]
    if work.empty:
        return [], [], []

    wanted_codes: set[str] | None = None
    if instrument_keys:
        wanted_codes = {instrument_to_a_code(k) for k in instrument_keys}

    statements: list[StatementFieldRow] = []
    corrects: list[AnnounceRow] = []
    industries: dict[str, IndustryMapRow] = {}

    for _, raw in work.iterrows():
        code = str(raw.get("证券代码") or "").zfill(6)
        if not code.isdigit():
            continue
        if wanted_codes is not None and code not in wanted_codes:
            continue
        period_end = _parse_date(raw.get("季度日期"))
        if period_end is None:
            continue
        if start is not None and period_end < start:
            continue
        if end is not None and period_end > end:
            continue

        instrument = a_code_to_instrument(code)
        is_correct = _is_truthy_correct(raw.get("是否发生差错更正"))
        correct_date = _parse_date(raw.get("差错更正披露日期"))
        revision = 1 if (is_correct or correct_date is not None) else 0

        industry_code = str(raw.get("行业代码") or raw.get("行业代码1") or "").strip()
        industry_name = str(raw.get("行业名称") or raw.get("行业名称1") or "").strip()
        if industry_code and instrument not in industries:
            industries[instrument] = IndustryMapRow(
                instrument_key=instrument,
                industry_code=industry_code,
                industry_name=industry_name,
                as_of=period_end,
            )

        if revision == 1 and correct_date is not None:
            at = as_utc(correct_date)
            corrects.append(
                AnnounceRow(
                    instrument_key=instrument,
                    report_period_end=period_end,
                    publish_time=at,
                    available_time=at,
                    revision=1,
                    is_restatement=True,
                    source="csmar_correct_declare",
                    source_record_id=f"{instrument}:{period_end.isoformat()}:r1",
                )
            )

        for field_code, candidates in _FIELD_COLUMNS.items():
            num = None
            for col in candidates:
                if col not in work.columns:
                    continue
                num = _float(raw.get(col))
                if num is not None:
                    break
            if num is None:
                continue
            statement_type = _FIELD_STATEMENT_TYPE[field_code]
            statements.append(
                StatementFieldRow(
                    instrument_key=instrument,
                    report_period_end=period_end,
                    report_period_start=None,
                    fiscal_year=period_end.year,
                    fiscal_quarter=_fiscal_quarter(period_end),
                    statement_type=statement_type,
                    field_code=field_code,
                    value=num,
                    source=source,
                    source_record_id=(
                        f"{instrument}:{statement_type}:{period_end.isoformat()}"
                        f":{field_code}:r{revision}"
                    ),
                    revision=revision,
                )
            )

    return merge_interest_bearing(statements), corrects, list(industries.values())


class CsmarStatementAdapter:
    """本地国泰安季度合并 → 科目主源；不提供 EM 科目。"""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        source: str = "csmar",
        chunksize: int = 50_000,
    ) -> None:
        self.path = Path(path) if path else default_csmar_quarterly_path()
        self.source = source
        self.chunksize = int(chunksize)
        self.last_correct_announces: list[AnnounceRow] = []
        self.last_industry_rows: list[IndustryMapRow] = []

    def _iter_chunks(self, usecols: list[str] | None) -> Iterable[pd.DataFrame]:
        if not self.path.is_file():
            raise FileNotFoundError(f"国泰安季度合并不存在: {self.path}")
        suffix = self.path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(self.path, usecols=usecols)
            yield frame
            return
        reader = pd.read_csv(
            self.path,
            encoding="utf-8-sig",
            usecols=usecols,
            chunksize=self.chunksize,
            low_memory=False,
        )
        yield from reader

    def fetch_statements(
        self,
        instrument_keys: Sequence[str],
        *,
        start: date | None = None,
        end: date | None = None,
    ) -> list[StatementFieldRow]:
        header = pd.read_csv(self.path, encoding="utf-8-sig", nrows=0)
        usecols = _usecols(header.columns)
        all_stmts: list[StatementFieldRow] = []
        all_corrects: list[AnnounceRow] = []
        industries: dict[str, IndustryMapRow] = {}
        # 季度合并.csv 按证券代码升序；超过最大目标代码后可提前结束全表扫描
        max_code = None
        if instrument_keys:
            codes = [instrument_to_a_code(k) for k in instrument_keys]
            max_code = max(codes) if codes else None
        for chunk in self._iter_chunks(usecols):
            if max_code is not None and "证券代码" in chunk.columns and not chunk.empty:
                chunk_codes = chunk["证券代码"].astype(str).str.zfill(6)
                if chunk_codes.min() > max_code:
                    break
            stmts, corrects, inds = dataframe_to_csmar_rows(
                chunk,
                source=self.source,
                instrument_keys=instrument_keys,
                start=start,
                end=end,
            )
            all_stmts.extend(stmts)
            all_corrects.extend(corrects)
            for row in inds:
                industries[row.instrument_key] = row
        self.last_correct_announces = all_corrects
        self.last_industry_rows = list(industries.values())
        _log.info(
            "csmar statements=%s corrects=%s industries=%s path=%s",
            len(all_stmts),
            len(all_corrects),
            len(industries),
            self.path,
        )
        return all_stmts

    def fetch_correct_announcements(self) -> list[AnnounceRow]:
        """最近一次 fetch 得到的差错更正披露公告。"""
        return list(self.last_correct_announces)

    def fetch_industry_map(self) -> list[IndustryMapRow]:
        return list(self.last_industry_rows)
