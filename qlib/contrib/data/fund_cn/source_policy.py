# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""fund_cn 定源策略：国泰安科目为主，免费源仅必要补洞 + 抽样验证。

硬规则：
- statements 生产值只来自 CSMAR，禁止 EM 科目写回覆盖
- 公告日国泰安缺失 → 用东财 NOTICE_DATE 补日历（非科目主源）
- 股本以国泰安报告期末为主；巨潮事件默认不改主路径
- --allow-uncovered 仅诊断，正式出数关闭
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Literal

from .paths_local import qlib_root

SourceMode = Literal["csmar", "akshare"]

# Coverage / PIT 默认只统计该报告期及以后（避免 1990s 无 EM 公告拖垮挂载率）
COVERAGE_REPORT_MIN_DATE = date(2005, 1, 1)
# 质量标签：EM 单快照无法重建更正前首发值
RESTATEMENT_UNRECOVERABLE = "restatement_unrecoverable"


class StatementSource(str, Enum):
    """科目主源枚举。"""

    CSMAR = "csmar"
    AKSHARE = "akshare"


@dataclass(frozen=True)
class FundCnSourcePolicy:
    """生产定源配置（默认国泰安为主）。"""

    statement_source: StatementSource = StatementSource.CSMAR
    # 公告：免费源必要补洞
    announce_from_em_notice: bool = True
    # 巨潮股本：默认不改估值主路径，仅验证
    use_cninfo_share_events: bool = False
    prefer_report_shares: bool = True
    # 行情：必须能还原未复权
    require_adjust_factor: bool = True
    # 交叉验证（只对重叠报告期；分批抽样）
    verify_with_free_sources: bool = True
    max_verify_rel_error: float = 0.05
    max_verify_mismatch_rate: float = 0.15
    verify_sample_size: int = 20
    verify_min_period: date = COVERAGE_REPORT_MIN_DATE
    # Coverage 报告期下界
    coverage_report_min: date = COVERAGE_REPORT_MIN_DATE


DEFAULT_POLICY = FundCnSourcePolicy()


def default_csmar_quarterly_path() -> Path:
    """默认国泰安季度合并宽表路径。"""
    return qlib_root() / "data" / "A股上市公司财务数据" / "季度合并.csv"


def resolve_policy(*, source: str = "csmar") -> FundCnSourcePolicy:
    """CLI ``--source`` → 策略对象。"""
    mode = (source or "csmar").strip().lower()
    if mode in {"csmar", "gtja", "国泰安"}:
        return FundCnSourcePolicy(statement_source=StatementSource.CSMAR)
    if mode in {"akshare", "em", "free"}:
        # 诊断/兼容路径：全免费源（科目+公告）
        return FundCnSourcePolicy(
            statement_source=StatementSource.AKSHARE,
            prefer_report_shares=False,
            use_cninfo_share_events=True,
            verify_with_free_sources=False,
        )
    raise ValueError(f"未知 --source={source!r}，仅支持 csmar / akshare")


# 证监会门类 J = 金融业 → non_financial 因子应置空
FINANCIAL_INDUSTRY_PREFIXES: tuple[str, ...] = ("J",)


def is_financial_industry(industry_code: str | None) -> bool:
    """行业代码是否金融业（CSRC 门类 J*）。"""
    text = str(industry_code or "").strip().upper()
    if not text:
        return False
    return any(text.startswith(prefix) for prefix in FINANCIAL_INDUSTRY_PREFIXES)
