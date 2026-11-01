# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""中文显示名与图表读图说明。内部字段仍用英文 id。"""

from __future__ import annotations

from .adapters.industry import SW_L1

# 指数 / 宏观实体
ENTITY_NAMES: dict[str, str] = {
    "SPX": "标普500",
    "IXIC": "纳斯达克综指",
    "DJI": "道琼斯",
    "CSI300": "沪深300",
    "SSE": "上证指数",
    "HSI": "恒生指数",
    "N225": "日经225",
    "SX5E": "欧洲斯托克50",
    "SENSEX": "印度SENSEX",
    "SGE_GOLD": "上海金",
    "GLD": "SPDR黄金ETF",
    "USDCNY": "美元兑人民币",
    "US2Y": "美债2年收益率",
    "US10Y": "美债10年收益率",
    "CN2Y": "中债2年收益率",
    "CN10Y": "中债10年收益率",
    "NORTH": "北向",
    "SOUTH": "南向",
    "M2": "中国M2",
    "LPR1Y": "LPR一年",
    "SH": "上海",
    "SZ": "深圳",
    "MARKET": "全市场",
    "NONE": "无",
}

# 常用图例复合键
LEGEND_NAMES: dict[str, str] = {
    "NORTH_amount": "北向成交额",
    "NORTH_hold": "北向持股市值",
    "NORTH_net_buy": "北向净买额（仅对照）",
    "SOUTH_amount": "南向成交额",
}

FACTOR_NAMES: dict[str, str] = {
    "IndustryAllocationScore": "行业配置综合分",
    "IndustryETFFlow20": "行业ETF估算配置(20日)",
    "IndustryETFFlow20_pct120": "行业ETF流120日分位",
    "IndustryETFFlow20_pct250": "行业ETF流250日分位",
    "Industry_Turnover_Ratio": "行业成交额占比",
    "Industry_Turnover_Rel20": "行业成交相对20日",
    "Industry_Turnover_Rel60": "行业成交相对60日",
    "Industry_RelRet20": "行业相对收益20日",
    "RawShares": "ETF份额",
    "Raw_Share_Change": "原始份额变化",
    "Estimated_Allocation": "估算配置",
    "ArbitrageRisk": "套利风险标记",
    "LHB_Institution_Ratio": "龙虎榜机构占比",
    "LHB_Institution_Ratio_d5": "龙虎榜机构占比(5日衰减)",
    "HSGT_HoldMcapChg": "北向持股市值变化",
    "HSGT_Turnover": "北向成交额",
    "Close": "收盘价",
    "NAV": "净值",
    "Amount": "成交额",
}

S1_COMPONENT_NAMES: dict[str, str] = {
    "strength": "宽基强度",
    "above_ma": "站上均线",
    "active": "成交活跃",
}

ETF_NAMES: dict[str, str] = {
    "510050": "上证50ETF",
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "512100": "中证1000ETF",
    "159915": "创业板ETF",
    "588000": "科创50ETF",
}

# dashboard 每张图的读图说明
CHART_CAPTIONS: dict[str, str] = {
    "global_index": (
        "【读图】比较全球/中国主要股指与上海金的相对走势（起点=1）。\n"
        "曲线上行表示相对起点上涨更多；彼此背离只说明相对强弱。\n"
        "注意：这是价格代理，不是资金净流入，也不能证明跨市场资金转移。"
    ),
    "yield_cn_us": (
        "【读图】中美国债收益率水平（未归一化）。\n"
        "收益率上行通常对应利率收紧/定价更贵；可用来解释风险资产环境。\n"
        "注意：收益率变动不是债券基金申购量。"
    ),
    "north_core": (
        "【读图】北向成交额与持股市值的相对变化（起点=1）。\n"
        "成交升=通道更活跃；持股市值升可能含股价上涨，不一定等于多买。\n"
        "注意：成交额≠净买入；这是核心观察图，优先于净买额。"
    ),
    "north_net_ref": (
        "【读图】北向成交净买额，仅作历史对照。\n"
        "披露口径曾调整，不能当作当前完整实时外资流量。\n"
        "注意：本序列不作 S1–S4 核心因子（reference_only）。"
    ),
    "industry_score_bar": (
        "【读图】行业配置综合分最新截面 Top（估算）。\n"
        "分高=相对其它行业更「热/强」的综合分（当前等权组件）。\n"
        "注意：不是净流入金额，也不是买卖指令；权重需样本外再估。"
    ),
    "industry_score_heat": (
        "【读图】行业配置综合分近60日热力。\n"
        "颜色持续偏强=主题有持续性；突然变色=切换或回撤。\n"
        "注意：仍是估算/代理合成，不能单独当作资金流证据。"
    ),
    "industry_turnover": (
        "【读图】各行业成交额占全市场比例（代理）。\n"
        "占比升高=交易更集中于此行业。\n"
        "注意：成交活跃≠净申购或净流入。"
    ),
    "etf_three_510300": (
        "【读图】沪深300ETF 三类指标：份额、份额变化、估算配置。\n"
        "份额升接近净申购方向；估算配置=份额变化×净值，是估算不是现金转账。\n"
        "注意：价格大涨而份额下降时，更像涨价驱动，不要写成「资金流入」。"
    ),
    "etf_three_510050": (
        "【读图】上证50ETF 三类指标：份额、份额变化、估算配置。\n"
        "读法同沪深300ETF：并列看份额与价格是否背离。\n"
        "注意：估算配置必须当作 estimate，不是已证实净流入。"
    ),
    "etf_arb": (
        "【读图】套利/折溢价风险标记快照。\n"
        "折溢价过大、换手异常、份额与价格背离时，份额变化噪音更大。\n"
        "注意：有风险标记时，降低对「份额=流入」的信任。"
    ),
    "s1": (
        "【读图】S1 市场层组件：绿=通过，红=未通过。\n"
        "需宽基强度、均线、成交活跃等条件组合判断；总信号为 0 表示市场层未放行。\n"
        "注意：S1 未过时，即使有行业过 S2，S4 三层共振也不成立；不是买卖信号。"
    ),
    "s2": (
        "【读图】S2 行业是否进入观察名单（1=过）。\n"
        "通过表示行业层条件满足，可进入观察，不等于建议买入。\n"
        "注意：须结合 S1；S1=0 时此处亮灯只作观察。"
    ),
    "s4": (
        "【读图】S4 为市场+行业(+个股确认)同时成立。\n"
        "entity 为 MARKET/NONE 且值为 0，表示共振未成立。\n"
        "注意：这是研究信号，不是实盘下单指令。"
    ),
    "backtest_ic": (
        "【读图】因子与未来约20日收益的截面 RankIC 均值。\n"
        "接近 0 表示这段样本几乎没有稳定预测力；红柱=该因子计算失败/缺失。\n"
        "注意：失败≠很差的 IC，而是没有结果；不能把监测看板当成已验证 alpha。"
    ),
    "backtest_ls": (
        "【读图】高分组减低分组的多空收益均值（已扣粗成本）。\n"
        "净值为负或接近 0，说明当前参数下难有稳定超额。\n"
        "注意：失败结果也会留档；不要只看最优组合。"
    ),
}

REQUIRED_DASHBOARD_CAPTION_IDS: tuple[str, ...] = (
    "global_index",
    "yield_cn_us",
    "north_core",
    "north_net_ref",
    "industry_score_bar",
    "industry_score_heat",
    "industry_turnover",
    "etf_three_510300",
    "etf_three_510050",
    "etf_arb",
    "s1",
    "s2",
    "s4",
    "backtest_ic",
    "backtest_ls",
)


def display_name(key: str | None) -> str:
    """把内部 id 转成中文显示名；未知则原样返回。"""
    if key is None:
        return ""
    text = str(key).strip()
    if not text:
        return text
    if text in LEGEND_NAMES:
        return LEGEND_NAMES[text]
    if text in ENTITY_NAMES:
        return ENTITY_NAMES[text]
    if text in FACTOR_NAMES:
        return FACTOR_NAMES[text]
    if text in S1_COMPONENT_NAMES:
        return S1_COMPONENT_NAMES[text]
    if text in ETF_NAMES:
        return f"{ETF_NAMES[text]}({text})"
    if text in SW_L1:
        return SW_L1[text]
    # 复合键 NORTH_amount 等
    if "_" in text:
        head, *rest = text.split("_", 1)
        if head in ENTITY_NAMES and rest:
            tail = rest[0]
            tail_map = {"amount": "成交额", "hold": "持股市值", "net_buy": "净买额（仅对照）"}
            if tail in tail_map:
                return f"{ENTITY_NAMES[head]}{tail_map[tail]}"
    return text


def relabel_map(series_map: dict[str, object]) -> dict[str, object]:
    """把 dict 的键换成中文显示名（用于图例）。"""
    return {display_name(k): v for k, v in series_map.items()}


def print_caption(chart_id: str) -> None:
    """在 notebook 里 plt.show() 后打印读图说明。"""
    text = CHART_CAPTIONS.get(chart_id)
    if not text:
        print(f"【读图】（未配置说明：{chart_id}）")
        return
    print(text)
