# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""数据集登记：来源等级、频率、主键、时区、修订规则与实测状态。

文档口径与实测状态分开：enabled 是接口是否尝试拉取，
probe_status 是最近一次实测结果，二者不得互相覆盖。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

from .schema import SERIES_SET, SERIES_SET_VERSION

ProbeStatus = Literal["untested", "ok", "failed", "disabled"]
DatasetRole = Literal["core", "reference_only", "disabled"]


@dataclass(frozen=True)
class DatasetDef:
    """一条可拉取/可计算数据集的登记信息。"""

    dataset: str
    display_name: str
    layer: str
    source_grade: str
    freq: str
    primary_key: tuple[str, ...]
    timezone: str
    adapter: str
    source: str
    evidence_class: str
    enabled: bool = True
    role: DatasetRole = "core"
    revision_rule: str = "overwrite_same_key"
    publish_lag: str = "T+0"
    probe_status: ProbeStatus = "untested"
    notes: str = ""
    unit: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primary_key"] = list(self.primary_key)
        payload["series_set"] = SERIES_SET
        payload["series_set_version"] = SERIES_SET_VERSION
        return payload


def _d(
    dataset: str,
    display_name: str,
    layer: str,
    *,
    source_grade: str,
    freq: str,
    adapter: str,
    source: str,
    evidence_class: str,
    enabled: bool = True,
    role: DatasetRole = "core",
    publish_lag: str = "T+0",
    notes: str = "",
    unit: str = "",
    primary_key: tuple[str, ...] = ("entity_id", "event_date", "field"),
    timezone: str = "Asia/Shanghai",
    probe_status: ProbeStatus = "untested",
) -> DatasetDef:
    return DatasetDef(
        dataset=dataset,
        display_name=display_name,
        layer=layer,
        source_grade=source_grade,
        freq=freq,
        primary_key=primary_key,
        timezone=timezone,
        adapter=adapter,
        source=source,
        evidence_class=evidence_class,
        enabled=enabled,
        role=role,
        publish_lag=publish_lag,
        probe_status=probe_status,
        notes=notes,
        unit=unit,
    )


# 北向净流入不得作为核心因子；仅作历史对照。
_HSGT_NOTE = (
    "沪深港通披露口径曾调整。北向成交额 ≠ 净买入；持股市值含价格。"
    "净买额只作 reference_only，不进 S1–S4 核心因子。"
)

# 无稳定免费接口：保持登记，避免空图冒充有数据。
DATASETS: tuple[DatasetDef, ...] = (
    # ----- B 全球 / 中国跨资产 -----
    _d("idx_global", "全球与中国指数", "B", source_grade="B", freq="day", adapter="global_macro", source="akshare.sina", evidence_class="proxy", notes="价格，不是资金流入。", unit="index"),
    _d("yield_cn_us", "中美国债收益率", "B", source_grade="B", freq="day", adapter="global_macro", source="akshare.bond_zh_us_rate", evidence_class="proxy", notes="收益率是定价，不是债券申购。", unit="pct"),
    _d("fx_usdcny", "美元兑人民币中间价", "B", source_grade="B", freq="day", adapter="global_macro", source="akshare.currency_boc_safe", evidence_class="proxy", unit="cny"),
    _d("gold_sge", "上海金基准价", "B", source_grade="B", freq="day", adapter="global_macro", source="akshare.spot_golden_benchmark_sge", evidence_class="proxy", unit="cny/g"),
    _d("gold_gld", "SPDR黄金ETF持仓", "B", source_grade="B", freq="day", adapter="global_macro", source="akshare.macro_cons_gold", evidence_class="direct", publish_lag="T+1", notes="吨位增减接近投资需求；市值含金价。", unit="tonne"),
    _d("cn_macro", "中国货币与利率", "B", source_grade="B", freq="month", adapter="china", source="akshare.macro_china", evidence_class="proxy", publish_lag="T+1m", notes="月频环境，不是证券市场净流入。"),
    _d(
        "hsgt_turnover",
        "沪深股通成交额",
        "B",
        source_grade="B",
        freq="day",
        adapter="china",
        source="akshare.stock_hsgt_hist_em",
        evidence_class="proxy",
        notes=_HSGT_NOTE + " 核心用法是成交额与活跃度，不是净流入。",
        unit="cny",
    ),
    _d(
        "hsgt_hold",
        "沪深股通持股市值",
        "B",
        source_grade="B",
        freq="day",
        adapter="china",
        source="akshare.stock_hsgt_hist_em",
        evidence_class="estimate",
        notes=_HSGT_NOTE + " 市值变化需价格调整后才能当配置估计。",
        unit="cny",
    ),
    _d(
        "hsgt_net",
        "沪深股通成交净买额",
        "B",
        source_grade="C",
        freq="day",
        adapter="china",
        source="akshare.stock_hsgt_hist_em",
        evidence_class="estimate",
        role="reference_only",
        notes=_HSGT_NOTE,
        unit="cny_yi",
    ),
    _d("qdii_scale", "QDII规模", "B", source_grade="C", freq="quarter", adapter="none", source="amac/eastmoney", evidence_class="estimate", enabled=False, role="disabled", notes="协会/东财若无稳定接口则 unavailable。"),
    _d("ici_mmf", "ICI美国货币基金", "B", source_grade="A", freq="week", adapter="none", source="ici.org", evidence_class="direct", enabled=False, role="disabled", notes="无稳定免费接口。"),
    _d("epfr_fund", "EPFR基金流", "B", source_grade="A", freq="week", adapter="none", source="epfr", evidence_class="direct", enabled=False, role="disabled", notes="不接 EPFR。"),
    _d("cftc_cot", "CFTC持仓", "B", source_grade="A", freq="week", adapter="none", source="cftc", evidence_class="direct", enabled=False, role="disabled"),
    _d("fed_z1", "美联储Z.1", "B", source_grade="A", freq="quarter", adapter="none", source="federalreserve", evidence_class="direct", enabled=False, role="disabled"),
    _d("dxy", "美元指数", "B", source_grade="C", freq="day", adapter="none", source="eastmoney.push", evidence_class="proxy", enabled=False, role="disabled", notes="东财 push 常断，保持关闭。"),
    _d("crb", "CRB商品指数", "B", source_grade="C", freq="day", adapter="none", source="eastmoney.push", evidence_class="proxy", enabled=False, role="disabled"),
    # ----- C ETF / 行业 -----
    _d("etf_bar", "ETF行情", "C", source_grade="B", freq="day", adapter="etf", source="akshare.fund_etf_hist_sina", evidence_class="proxy", notes="收盘/成交额是活跃度，不是净流入。"),
    _d("etf_nav", "ETF净值", "C", source_grade="B", freq="day", adapter="etf", source="akshare.fund_etf_fund_info_em", evidence_class="direct", publish_lag="T+1"),
    _d("etf_shares", "ETF份额", "C", source_grade="A", freq="day", adapter="etf", source="akshare.fund_etf_scale_sse/szse", evidence_class="direct", publish_lag="T+1", notes="份额变化 ≠ 现金净流入。"),
    _d("etf_status", "ETF申赎与跨境状态", "C", source_grade="C", freq="day", adapter="etf", source="manual/exchange", evidence_class="direct", notes="缺公告则标记，不臆造。"),
    _d("sw_industry", "申万一级行情", "C", source_grade="B", freq="day", adapter="industry", source="akshare.index_hist_sw", evidence_class="proxy", notes="指数涨跌 ≠ 行业净流入。"),
    _d("industry_member", "行业成分与状态", "C", source_grade="B", freq="day", adapter="industry", source="manual/sw_map", evidence_class="direct", notes="停牌/新股/ST 规则写死；缺成分则 breadth 标 unavailable。"),
    _d("index_rebalance", "指数调仓事件", "C", source_grade="C", freq="event", adapter="none", source="announcement", evidence_class="direct", enabled=False, notes="先留字段骨架，缺公告则标记，避免把被动调仓当主动选股。"),
    # ----- D 机构 / 个股 -----
    _d("fund_holding", "公募重仓", "D", source_grade="B", freq="quarter", adapter="institution", source="akshare.fund_portfolio", evidence_class="direct", publish_lag="T+1q", notes="季报 ≠ 实时持仓。"),
    _d("fund_scale_snapshot", "开放式基金规模快照", "D", source_grade="C", freq="day", adapter="institution", source="akshare.fund_scale_open_sina", evidence_class="estimate", publish_lag="快照", notes="时点快照，不是净申购时间序列。"),
    _d("lhb", "龙虎榜机构席位", "D", source_grade="C", freq="event", adapter="institution", source="akshare.stock_lhb", evidence_class="proxy", notes="机构席位 ≠ 已确认公募/保险；无数据保持 NA，不填 0。"),
    _d("block_trade", "大宗交易", "D", source_grade="C", freq="day", adapter="institution", source="akshare.stock_dzjy", evidence_class="proxy", notes="折溢价与金额占比，不预设多空。"),
    _d("margin", "融资余额", "D", source_grade="B", freq="day", adapter="china", source="akshare.macro_china_market_margin", evidence_class="proxy", notes="融资负债变化 ≠ 全市场净流入。"),
    _d("holder_event", "股东与增减持公告", "D", source_grade="C", freq="event", adapter="institution", source="akshare.stock_hold_change", evidence_class="direct", notes="能解析多少算多少；主体口径见 entity_map。"),
    _d("private_hold", "私募/理财/自营持仓", "D", source_grade="D", freq="quarter", adapter="none", source="unavailable", evidence_class="estimate", enabled=False, role="disabled", notes="无完整持仓则只做主体登记。"),
)


def dataset_by_id() -> dict[str, DatasetDef]:
    return {row.dataset: row for row in DATASETS}


def enabled_datasets(*, include_disabled: bool = False) -> list[DatasetDef]:
    if include_disabled:
        return list(DATASETS)
    return [row for row in DATASETS if row.enabled]


def catalog_rows(*, include_disabled: bool = True) -> list[dict[str, Any]]:
    return [row.as_dict() for row in enabled_datasets(include_disabled=include_disabled)]


def datasets_for_adapter(adapter: str) -> list[DatasetDef]:
    return [row for row in DATASETS if row.adapter == adapter and row.enabled]


def core_datasets() -> list[DatasetDef]:
    """信号与因子只用 core；reference_only / disabled 不进 S1–S4。"""
    return [row for row in DATASETS if row.enabled and row.role == "core"]
