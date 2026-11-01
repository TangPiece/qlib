# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""资金流向监测序列目录：口径、频率、证据等级与免责说明。

证据等级不能混用：
- L1 直接流量：份额/净申购/跨境净成交
- L2 持仓变化：必须先做价格（及汇率）调整才能当流量估计
- L3 市场代理：涨跌、成交额、融资、开户，只能描述活跃度
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal

SERIES_SET = "flow_monitor@v1"
SERIES_SET_CODE = "flow_monitor"
SERIES_SET_VERSION = "v1"
SERIES_VERSION = 1

EvidenceGrade = Literal["L1", "L2", "L3"]
Frequency = Literal["day", "week", "month", "quarter"]
AssetClass = Literal["equity", "bond", "gold", "commodity", "cash", "fx", "other"]


@dataclass(frozen=True)
class SeriesDef:
    """一条可监测序列的元数据；笔记本按这些字段决定怎么解读。"""

    series_id: str
    display_name: str
    asset_class: AssetClass
    region: str
    investor: str
    evidence_grade: EvidenceGrade
    freq: Frequency
    unit: str
    adapter: str
    source: str
    enabled: bool = True
    pair_series_id: str = ""
    publish_lag: str = "T+0"
    notes: str = ""
    layer: str = "A"  # A 全球 / B 中国市场 / C 行业风格 / D 投资者

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _s(
    series_id: str,
    display_name: str,
    asset_class: AssetClass,
    region: str,
    *,
    grade: EvidenceGrade,
    freq: Frequency,
    unit: str,
    adapter: str,
    source: str,
    investor: str = "mixed",
    enabled: bool = True,
    pair_series_id: str = "",
    publish_lag: str = "T+0",
    notes: str = "",
    layer: str = "A",
) -> SeriesDef:
    return SeriesDef(
        series_id=series_id,
        display_name=display_name,
        asset_class=asset_class,
        region=region,
        investor=investor,
        evidence_grade=grade,
        freq=freq,
        unit=unit,
        adapter=adapter,
        source=source,
        enabled=enabled,
        pair_series_id=pair_series_id,
        publish_lag=publish_lag,
        notes=notes,
        layer=layer,
    )


# 沪深港通披露规则已变：历史高频北向净买额不能当作当前完整实时外资。
_HSGT_NOTE = (
    "沪深港通披露口径曾调整，历史北向净买额不能当作今日完整实时外资流量；"
    "持股市值变化需剔除价格后才能估计买入。"
)

# 代表性 A 股 ETF：价格/成交额为 L3，份额为 L1。
_CN_ETFS: tuple[tuple[str, str, AssetClass, str, str], ...] = (
    ("510300", "沪深300ETF", "equity", "cn", "broad"),
    ("510050", "上证50ETF", "equity", "cn", "large"),
    ("510500", "中证500ETF", "equity", "cn", "mid"),
    ("512100", "中证1000ETF", "equity", "cn", "small"),
    ("159915", "创业板ETF", "equity", "cn", "growth"),
    ("588000", "科创50ETF", "equity", "cn", "growth"),
    ("518880", "黄金ETF", "gold", "cn", "gold"),
    ("511260", "十年国债ETF", "bond", "cn", "bond"),
    ("512480", "半导体ETF", "equity", "cn", "semiconductor"),
    ("516160", "新能源ETF", "equity", "cn", "new_energy"),
    ("512800", "银行ETF", "equity", "cn", "bank"),
    ("512880", "证券ETF", "equity", "cn", "broker"),
    ("159928", "消费ETF", "equity", "cn", "consumer"),
    ("515080", "中证红利ETF", "equity", "cn", "dividend"),
    ("510880", "红利ETF", "equity", "cn", "dividend"),
)


def _etf_series() -> list[SeriesDef]:
    """为每只代表性 ETF 登记价格、成交额、份额三条序列。"""
    rows: list[SeriesDef] = []
    for code, name, asset, region, style in _CN_ETFS:
        layer = "C" if style not in {"broad", "gold", "bond"} else ("B" if asset != "equity" or style == "broad" else "C")
        if style in {"broad", "large", "mid", "small", "growth"}:
            layer = "B" if style == "broad" else "C"
        rows.append(
            _s(
                f"etf_{code}_px",
                f"{name}收盘",
                asset,
                region,
                grade="L3",
                freq="day",
                unit="price",
                adapter="etf_cn",
                source="akshare.fund_etf_hist_sina",
                pair_series_id=f"etf_{code}_shares",
                notes=f"价格；风格={style}。上涨不等于净申购。",
                layer=layer,
            )
        )
        rows.append(
            _s(
                f"etf_{code}_amount",
                f"{name}成交额",
                asset,
                region,
                grade="L3",
                freq="day",
                unit="cny",
                adapter="etf_cn",
                source="akshare.fund_etf_hist_sina",
                notes="成交额是双边活跃度，不是净流入。",
                layer=layer,
            )
        )
        rows.append(
            _s(
                f"etf_{code}_shares",
                f"{name}份额",
                asset,
                region,
                grade="L1",
                freq="day",
                unit="share",
                adapter="etf_cn",
                source="akshare.fund_etf_scale_sse/szse",
                pair_series_id=f"etf_{code}_px",
                publish_lag="T+1",
                notes="份额增减接近净申购，但仍可能含现金申购/实物申购，不等于当日等额买股。",
                layer=layer,
            )
        )
    return rows


def _industry_series() -> list[SeriesDef]:
    """东财行业板块涨跌与成交；名称以东财板块名为准，不是申万指数。"""
    boards = (
        "电子",
        "半导体",
        "通信设备",
        "电网设备",
        "汽车整车",
        "有色金属",
        "通用设备",
        "化学制药",
        "银行",
        "证券",
        "食品饮料",
        "房地产开发",
    )
    rows: list[SeriesDef] = []
    for name in boards:
        slug = {
            "电子": "electronics",
            "半导体": "semiconductor",
            "通信设备": "telecom",
            "电网设备": "grid",
            "汽车整车": "auto",
            "有色金属": "nmetal",
            "通用设备": "machinery",
            "化学制药": "pharma",
            "银行": "bank",
            "证券": "broker",
            "食品饮料": "food",
            "房地产开发": "property",
        }[name]
        rows.append(
            _s(
                f"ind_{slug}_px",
                f"{name}板块收盘",
                "equity",
                "cn",
                grade="L3",
                freq="day",
                unit="price",
                adapter="industry_cn",
                source="akshare.index_hist_sw",
                notes="申万一级近似映射；指数上涨不等于行业净流入。",
                layer="C",
            )
        )
        rows.append(
            _s(
                f"ind_{slug}_amount",
                f"{name}板块成交额",
                "equity",
                "cn",
                grade="L3",
                freq="day",
                unit="cny",
                adapter="industry_cn",
                source="akshare.index_hist_sw",
                notes="成交额上升只说明交易活跃。",
                layer="C",
            )
        )
    return rows


SERIES_DEFINITIONS: tuple[SeriesDef, ...] = (
    # ----- A. 全球价格 / 利率 / 汇率（L3）-----
    _s("px_spx", "标普500", "equity", "us", grade="L3", freq="day", unit="index", adapter="global_macro", source="akshare.index_us_stock_sina", notes="价格，不是资金流入。"),
    _s("px_ndx", "纳斯达克综合", "equity", "us", grade="L3", freq="day", unit="index", adapter="global_macro", source="akshare.index_us_stock_sina", notes="东财 NDX 不通时用纳斯达克综合作 L3 代理；价格≠流入。"),
    _s("px_djia", "道琼斯", "equity", "us", grade="L3", freq="day", unit="index", adapter="global_macro", source="akshare.index_us_stock_sina"),
    _s("px_csi300", "沪深300", "equity", "cn", grade="L3", freq="day", unit="index", adapter="global_macro", source="akshare.stock_zh_index_daily", layer="B"),
    _s("px_sse", "上证指数", "equity", "cn", grade="L3", freq="day", unit="index", adapter="global_macro", source="akshare.stock_zh_index_daily", layer="B"),
    _s("px_hsi", "恒生指数", "equity", "hk", grade="L3", freq="day", unit="index", adapter="global_macro", source="akshare.stock_hk_index_daily_sina", layer="B"),
    _s("px_n225", "日经225", "equity", "jp", grade="L3", freq="day", unit="index", adapter="global_macro", source="akshare.index_global_hist_sina"),
    _s("px_sx5e", "欧洲斯托克50", "equity", "eu", grade="L3", freq="day", unit="index", adapter="global_macro", source="akshare.index_global_hist_sina"),
    _s("px_sensex", "印度SENSEX", "equity", "in", grade="L3", freq="day", unit="index", adapter="global_macro", source="akshare.index_global_hist_sina"),
    _s(
        "px_dxy",
        "美元指数",
        "fx",
        "us",
        grade="L3",
        freq="day",
        unit="index",
        adapter="global_macro",
        source="akshare.index_global_hist_em",
        enabled=False,
        notes="依赖东财 push，当前网络常断连；暂关闭。",
    ),
    _s(
        "px_crb",
        "CRB商品指数",
        "commodity",
        "global",
        grade="L3",
        freq="day",
        unit="index",
        adapter="global_macro",
        source="akshare.index_global_hist_em",
        enabled=False,
        notes="依赖东财 push，当前网络常断连；暂关闭。",
    ),
    _s("px_gold_sge", "上海金基准价", "gold", "cn", grade="L3", freq="day", unit="cny/g", adapter="global_macro", source="akshare.spot_golden_benchmark_sge", layer="B"),
    _s("fx_usdcny", "美元兑人民币中间价", "fx", "cn", grade="L3", freq="day", unit="cny", adapter="global_macro", source="akshare.currency_boc_safe", layer="B"),
    _s("yld_us_2y", "美国国债收益率2年", "bond", "us", grade="L3", freq="day", unit="pct", adapter="global_macro", source="akshare.bond_zh_us_rate", notes="收益率变动是定价，不是债券基金申购。"),
    _s("yld_us_10y", "美国国债收益率10年", "bond", "us", grade="L3", freq="day", unit="pct", adapter="global_macro", source="akshare.bond_zh_us_rate"),
    _s("yld_cn_2y", "中国国债收益率2年", "bond", "cn", grade="L3", freq="day", unit="pct", adapter="global_macro", source="akshare.bond_zh_us_rate", layer="B"),
    _s("yld_cn_10y", "中国国债收益率10年", "bond", "cn", grade="L3", freq="day", unit="pct", adapter="global_macro", source="akshare.bond_zh_us_rate", layer="B"),
    # L1：SPDR 黄金 ETF 持仓增减
    _s(
        "gold_gld_tonnes",
        "SPDR黄金ETF持仓",
        "gold",
        "global",
        grade="L1",
        freq="day",
        unit="tonne",
        adapter="global_macro",
        source="akshare.macro_cons_gold",
        pair_series_id="px_gold_sge",
        publish_lag="T+1",
        notes="持仓吨位变化接近黄金 ETF 投资需求；总价值变化含金价，不能直接当流入。",
    ),
    _s(
        "gold_gld_change",
        "SPDR黄金ETF增减持",
        "gold",
        "global",
        grade="L1",
        freq="day",
        unit="tonne",
        adapter="global_macro",
        source="akshare.macro_cons_gold",
        notes="直接公布的增减持，优先于用市值倒推。",
    ),
    # 免费官方但本期无稳定接口：先登记为关闭，避免空图冒充有数据
    _s(
        "ici_mmf_aum",
        "ICI美国货币基金规模",
        "cash",
        "us",
        grade="L1",
        freq="week",
        unit="usd",
        adapter="none",
        source="ici.org",
        enabled=False,
        publish_lag="T+1w",
        notes="ICI 周报暂无稳定自动接口，现金层改看利率与国内货基规模。",
    ),
    _s(
        "us_etf_tlt_shares",
        "TLT份额",
        "bond",
        "us",
        grade="L1",
        freq="day",
        unit="share",
        adapter="none",
        source="yahoo",
        enabled=False,
        notes="美债 ETF 份额需独立源，本期未接入。",
    ),
    # ----- B/D. 中国市场资金结构 -----
    _s(
        "hsgt_north_net",
        "北向成交净买额",
        "equity",
        "cn",
        grade="L1",
        freq="day",
        unit="cny_yi",
        adapter="china_flow",
        source="akshare.stock_hsgt_hist_em",
        investor="foreign",
        publish_lag="滞后/口径变更",
        notes=_HSGT_NOTE,
        layer="D",
    ),
    _s(
        "hsgt_south_net",
        "南向成交净买额",
        "equity",
        "hk",
        grade="L1",
        freq="day",
        unit="cny_yi",
        adapter="china_flow",
        source="akshare.stock_hsgt_hist_em",
        investor="cn_onshore",
        publish_lag="滞后/口径变更",
        notes=_HSGT_NOTE,
        layer="B",
    ),
    _s(
        "hsgt_north_hold_mcap",
        "北向持股市值",
        "equity",
        "cn",
        grade="L2",
        freq="day",
        unit="cny",
        adapter="china_flow",
        source="akshare.stock_hsgt_hist_em",
        investor="foreign",
        pair_series_id="px_csi300",
        notes="市值含价格；估计买入请用价格调整公式。" + _HSGT_NOTE,
        layer="D",
    ),
    _s(
        "margin_sh_balance",
        "上海融资余额",
        "equity",
        "cn",
        grade="L3",
        freq="day",
        unit="cny",
        adapter="china_flow",
        source="akshare.macro_china_market_margin_sh",
        investor="retail_leverage",
        notes="融资负债变化，不等于全市场净流入相同金额。",
        layer="D",
    ),
    _s(
        "margin_sz_balance",
        "深圳融资余额",
        "equity",
        "cn",
        grade="L3",
        freq="day",
        unit="cny",
        adapter="china_flow",
        source="akshare.macro_china_market_margin_sz",
        investor="retail_leverage",
        notes="融资负债变化，不等于全市场净流入相同金额。",
        layer="D",
    ),
    _s(
        "account_new",
        "新增投资者数量",
        "equity",
        "cn",
        grade="L3",
        freq="week",
        unit="count",
        adapter="china_flow",
        source="akshare.stock_account_statistics_em",
        investor="retail",
        publish_lag="T+1w",
        notes="周频活跃度，不是净买入。",
        layer="D",
    ),
    _s(
        "insurance_income",
        "原保险保费收入",
        "other",
        "cn",
        grade="L2",
        freq="month",
        unit="cny",
        adapter="china_flow",
        source="akshare.macro_china_insurance_income",
        investor="insurance",
        publish_lag="T+1m",
        notes="月频保费，不是股票配置；保险资金运用余额另需银保监口径。",
        layer="D",
    ),
    _s(
        "pboc_gold_fx",
        "央行黄金与外汇储备",
        "gold",
        "cn",
        grade="L2",
        freq="month",
        unit="usd",
        adapter="china_flow",
        source="akshare.macro_china_foreign_exchange_gold",
        investor="central_bank",
        publish_lag="T+1m",
        notes="官方储备存量，含估值，需结合公布口径解读。",
        layer="D",
    ),
    _s(
        "cn_m2",
        "中国M2",
        "cash",
        "cn",
        grade="L3",
        freq="month",
        unit="cny",
        adapter="china_flow",
        source="akshare.macro_china_money_supply",
        publish_lag="T+1m",
        notes="货币环境，不是证券市场净流入。",
        layer="B",
    ),
    _s(
        "cn_lpr_1y",
        "LPR一年",
        "cash",
        "cn",
        grade="L3",
        freq="month",
        unit="pct",
        adapter="china_flow",
        source="akshare.macro_china_lpr",
        notes="政策利率，用于解释配置动机。",
        layer="B",
    ),
    _s(
        "fund_scale_equity_snapshot",
        "股票型开放式基金规模快照",
        "equity",
        "cn",
        grade="L2",
        freq="day",
        unit="cny_yi",
        adapter="china_flow",
        source="akshare.fund_scale_open_sina",
        investor="mutual_fund",
        publish_lag="快照",
        notes="个基规模加总的时点快照，不是时间序列净申购；含净值涨跌。",
        layer="D",
    ),
    _s(
        "fund_scale_bond_snapshot",
        "债券型开放式基金规模快照",
        "bond",
        "cn",
        grade="L2",
        freq="day",
        unit="cny_yi",
        adapter="china_flow",
        source="akshare.fund_scale_open_sina",
        investor="mutual_fund",
        publish_lag="快照",
        notes="时点快照，不是净申购时间序列。",
        layer="D",
    ),
    _s(
        "fund_scale_money_snapshot",
        "货币型开放式基金规模快照",
        "cash",
        "cn",
        grade="L2",
        freq="day",
        unit="cny_yi",
        adapter="china_flow",
        source="akshare.fund_scale_open_sina",
        investor="mutual_fund",
        publish_lag="快照",
        notes="现金层代理；快照含收益结转。",
        layer="D",
    ),
    _s(
        "institute_hold_snapshot",
        "机构持股家数快照",
        "equity",
        "cn",
        grade="L2",
        freq="quarter",
        unit="count",
        adapter="china_flow",
        source="akshare.stock_institute_hold",
        investor="institution",
        publish_lag="T+1q",
        notes="季报披露滞后，不能判断当日资金。",
        layer="D",
    ),
    # ----- C. 市值风格指数 -----
    _s("style_sz50_px", "上证50", "equity", "cn", grade="L3", freq="day", unit="index", adapter="industry_cn", source="akshare.index_zh_a_hist", layer="C"),
    _s("style_csi300_px", "沪深300", "equity", "cn", grade="L3", freq="day", unit="index", adapter="industry_cn", source="akshare.index_zh_a_hist", layer="C"),
    _s("style_csi500_px", "中证500", "equity", "cn", grade="L3", freq="day", unit="index", adapter="industry_cn", source="akshare.index_zh_a_hist", layer="C"),
    _s("style_csi1000_px", "中证1000", "equity", "cn", grade="L3", freq="day", unit="index", adapter="industry_cn", source="akshare.index_zh_a_hist", layer="C"),
    _s("style_chinext_px", "创业板指", "equity", "cn", grade="L3", freq="day", unit="index", adapter="industry_cn", source="akshare.index_zh_a_hist", layer="C"),
    _s("style_star50_px", "科创50", "equity", "cn", grade="L3", freq="day", unit="index", adapter="industry_cn", source="akshare.index_zh_a_hist", layer="C"),
    _s("style_dividend_px", "上证红利", "equity", "cn", grade="L3", freq="day", unit="index", adapter="industry_cn", source="akshare.index_zh_a_hist", layer="C"),
    *_etf_series(),
    *_industry_series(),
)


def series_by_id() -> dict[str, SeriesDef]:
    return {row.series_id: row for row in SERIES_DEFINITIONS}


def enabled_series(*, include_disabled: bool = False) -> list[SeriesDef]:
    if include_disabled:
        return list(SERIES_DEFINITIONS)
    return [row for row in SERIES_DEFINITIONS if row.enabled]


def series_ids(*, include_disabled: bool = False) -> list[str]:
    return [row.series_id for row in enabled_series(include_disabled=include_disabled)]


def catalog_rows(*, include_disabled: bool = True) -> list[dict[str, Any]]:
    return [row.as_dict() for row in enabled_series(include_disabled=include_disabled)]


def series_for_adapter(adapter: str) -> list[SeriesDef]:
    return [row for row in SERIES_DEFINITIONS if row.adapter == adapter and row.enabled]


def iter_pairs() -> Iterable[tuple[SeriesDef, SeriesDef]]:
    """持仓/份额序列与配对价格，供价格调整净流量估计。"""
    lookup = series_by_id()
    for row in SERIES_DEFINITIONS:
        if not row.pair_series_id:
            continue
        pair = lookup.get(row.pair_series_id)
        if pair is None:
            continue
        yield row, pair
