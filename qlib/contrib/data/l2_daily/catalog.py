"""100 个 L2 日频因子定义（与 D1 l2_factor_definitions 种子一致）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

FACTOR_VERSION = 1

GROUP_TRADE = "trade"
GROUP_BOOK = "book"
GROUP_ORDER = "order"
GROUP_MICROSTRUCTURE = "microstructure"
ALL_GROUPS: tuple[str, ...] = (
    GROUP_TRADE,
    GROUP_BOOK,
    GROUP_ORDER,
    GROUP_MICROSTRUCTURE,
)


@dataclass(frozen=True)
class FactorDef:
    factor_id: str
    factor_name: str
    display_name: str
    category: str
    description: str
    formula: str
    input_fields: str
    factor_group: str


def _f(
    factor_id: str,
    display_name: str,
    category: str,
    group: str,
    formula: str,
    input_fields: str,
    description: str = "",
) -> FactorDef:
    return FactorDef(
        factor_id=factor_id,
        factor_name=factor_id,
        display_name=display_name,
        category=category,
        description=description or display_name,
        formula=formula,
        input_fields=input_fields,
        factor_group=group,
    )


FACTOR_DEFINITIONS: tuple[FactorDef, ...] = (
    # --- trade: 1-16 成交基础 ---
    _f("l2_trade_count", "成交笔数", "trade_basic", GROUP_TRADE, "count(trades)", "逐笔成交"),
    _f("l2_trade_volume", "成交量", "trade_basic", GROUP_TRADE, "sum(qty)", "逐笔成交"),
    _f("l2_trade_amount", "成交额", "trade_basic", GROUP_TRADE, "sum(price*qty/scale)", "逐笔成交"),
    _f("l2_trade_volume_mean", "平均成交量", "trade_basic", GROUP_TRADE, "mean(qty)", "逐笔成交"),
    _f("l2_trade_volume_std", "成交量标准差", "trade_basic", GROUP_TRADE, "std(qty)", "逐笔成交"),
    _f("l2_trade_volume_max", "最大成交量", "trade_basic", GROUP_TRADE, "max(qty)", "逐笔成交"),
    _f("l2_trade_amount_mean", "平均成交额", "trade_basic", GROUP_TRADE, "mean(amount)", "逐笔成交"),
    _f("l2_trade_amount_std", "成交额标准差", "trade_basic", GROUP_TRADE, "std(amount)", "逐笔成交"),
    _f("l2_trade_amount_max", "最大成交额", "trade_basic", GROUP_TRADE, "max(amount)", "逐笔成交"),
    _f("l2_trade_amount_median", "成交额中位数", "trade_basic", GROUP_TRADE, "median(amount)", "逐笔成交"),
    _f("l2_trade_amount_skew", "成交额偏度", "trade_basic", GROUP_TRADE, "skew(amount)", "逐笔成交"),
    _f("l2_trade_amount_kurt", "成交额峰度", "trade_basic", GROUP_TRADE, "kurt(amount)", "逐笔成交"),
    _f("l2_trade_price_mean", "成交价均值", "trade_basic", GROUP_TRADE, "mean(price)", "逐笔成交"),
    _f("l2_trade_price_std", "成交价标准差", "trade_basic", GROUP_TRADE, "std(price)", "逐笔成交"),
    _f("l2_vwap", "VWAP", "trade_basic", GROUP_TRADE, "sum(amount)/sum(qty)", "逐笔成交"),
    _f(
        "l2_vwap_deviation",
        "VWAP 偏离",
        "trade_basic",
        GROUP_TRADE,
        "(last_price-vwap)/vwap",
        "逐笔成交",
    ),
    # --- trade: 17-30 方向 ---
    _f("l2_active_buy_volume", "主动买入成交量", "trade_direction", GROUP_TRADE, "sum(qty|B) [09:30-14:57]", "逐笔成交"),
    _f("l2_active_sell_volume", "主动卖出成交量", "trade_direction", GROUP_TRADE, "sum(qty|S) [09:30-14:57]", "逐笔成交"),
    _f("l2_active_buy_amount", "主动买入成交额", "trade_direction", GROUP_TRADE, "sum(amt|B) [09:30-14:57]", "逐笔成交"),
    _f("l2_active_sell_amount", "主动卖出成交额", "trade_direction", GROUP_TRADE, "sum(amt|S) [09:30-14:57]", "逐笔成交"),
    _f("l2_active_buy_count", "主动买入笔数", "trade_direction", GROUP_TRADE, "count(B) [09:30-14:57]", "逐笔成交"),
    _f("l2_active_sell_count", "主动卖出笔数", "trade_direction", GROUP_TRADE, "count(S) [09:30-14:57]", "逐笔成交"),
    _f(
        "l2_active_buy_amount_ratio",
        "主动买入成交额占比",
        "trade_direction",
        GROUP_TRADE,
        "buy_amt/total_amt [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_active_sell_amount_ratio",
        "主动卖出成交额占比",
        "trade_direction",
        GROUP_TRADE,
        "sell_amt/total_amt [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_trade_amount_imbalance",
        "主动买卖成交额差",
        "trade_direction",
        GROUP_TRADE,
        "buy_amt-sell_amt [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_trade_volume_imbalance",
        "主动买卖成交量差",
        "trade_direction",
        GROUP_TRADE,
        "buy_vol-sell_vol [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_trade_amount_imbalance_ratio",
        "成交额不平衡率",
        "trade_direction",
        GROUP_TRADE,
        "(buy-sell)/(buy+sell) [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_trade_volume_imbalance_ratio",
        "成交量不平衡率",
        "trade_direction",
        GROUP_TRADE,
        "(buy-sell)/(buy+sell) [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_trade_count_imbalance",
        "主动买卖笔数差",
        "trade_direction",
        GROUP_TRADE,
        "buy_n-sell_n [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_trade_count_imbalance_ratio",
        "主动买卖笔数不平衡率",
        "trade_direction",
        GROUP_TRADE,
        "(buy_n-sell_n)/(buy_n+sell_n) [09:30-14:57]",
        "逐笔成交",
    ),
    # --- trade: 31-42 大单 ---
    _f(
        "l2_large_trade_count",
        "大单成交笔数",
        "trade_large",
        GROUP_TRADE,
        "count(trade | buy or sell order total fills >= 1e6 yuan)",
        "逐笔成交",
        "按叫买/叫卖序号累计成交额判断买卖双方是否大单，任一方是大单即计入，每笔成交只计一次",
    ),
    _f("l2_large_trade_volume", "大单成交量", "trade_large", GROUP_TRADE, "sum(qty | large trade)", "逐笔成交"),
    _f("l2_large_trade_amount", "大单成交额", "trade_large", GROUP_TRADE, "sum(amount | large trade)", "逐笔成交"),
    _f(
        "l2_large_trade_amount_ratio",
        "大单成交额占比",
        "trade_large",
        GROUP_TRADE,
        "large_amt/total_amt",
        "逐笔成交",
    ),
    _f(
        "l2_large_trade_volume_ratio",
        "大单成交量占比",
        "trade_large",
        GROUP_TRADE,
        "large_vol/total_vol",
        "逐笔成交",
    ),
    _f(
        "l2_large_active_buy_amount",
        "大单主动买入额",
        "trade_large",
        GROUP_TRADE,
        "sum(amount | BS=B and buy order large) [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_large_active_sell_amount",
        "大单主动卖出额",
        "trade_large",
        GROUP_TRADE,
        "sum(amount | BS=S and sell order large) [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_large_net_buy_amount",
        "大单净买入额",
        "trade_large",
        GROUP_TRADE,
        "large_active_buy-large_active_sell",
        "逐笔成交",
    ),
    _f(
        "l2_large_net_buy_amount_ratio",
        "大单净买入额占比",
        "trade_large",
        GROUP_TRADE,
        "large_net_buy/continuous_amount",
        "逐笔成交",
    ),
    _f(
        "l2_max_trade_amount_ratio",
        "最大单笔成交额占比",
        "trade_large",
        GROUP_TRADE,
        "max(amt)/total_amt",
        "逐笔成交",
    ),
    _f(
        "l2_top5_trade_amount_ratio",
        "前 5 大成交额占比",
        "trade_large",
        GROUP_TRADE,
        "sum(top5 amt)/total_amt",
        "逐笔成交",
    ),
    _f(
        "l2_trade_amount_hhi",
        "成交额 HHI",
        "trade_large",
        GROUP_TRADE,
        "sum((amt/total)^2)",
        "逐笔成交",
    ),
    # --- book: 43-64 ---
    _f("l2_mid_price_mean", "买卖中间价均值", "book_obi", GROUP_BOOK, "mean((bid1+ask1)/2)", "行情"),
    _f("l2_mid_price_std", "买卖中间价标准差", "book_obi", GROUP_BOOK, "std(mid)", "行情"),
    _f("l2_spread_mean", "平均买卖价差", "book_obi", GROUP_BOOK, "mean(ask1-bid1)", "行情"),
    _f("l2_spread_std", "价差标准差", "book_obi", GROUP_BOOK, "std(ask1-bid1)", "行情"),
    _f(
        "l2_relative_spread_mean",
        "平均相对价差",
        "book_obi",
        GROUP_BOOK,
        "mean((ask-bid)/mid)",
        "行情",
    ),
    _f(
        "l2_relative_spread_std",
        "相对价差标准差",
        "book_obi",
        GROUP_BOOK,
        "std((ask-bid)/mid)",
        "行情",
    ),
    _f("l2_bid1_volume", "买一挂单量", "book_obi", GROUP_BOOK, "mean(bid1_qty)", "行情"),
    _f("l2_ask1_volume", "卖一挂单量", "book_obi", GROUP_BOOK, "mean(ask1_qty)", "行情"),
    _f("l2_bid_depth5_mean", "买五档深度均值", "book_obi", GROUP_BOOK, "mean(sum bid1-5)", "行情"),
    _f("l2_ask_depth5_mean", "卖五档深度均值", "book_obi", GROUP_BOOK, "mean(sum ask1-5)", "行情"),
    _f("l2_bid_depth5_std", "买五档深度标准差", "book_obi", GROUP_BOOK, "std(sum bid1-5)", "行情"),
    _f("l2_ask_depth5_std", "卖五档深度标准差", "book_obi", GROUP_BOOK, "std(sum ask1-5)", "行情"),
    _f(
        "l2_depth5_diff",
        "五档深度差",
        "book_obi",
        GROUP_BOOK,
        "mean(bid5-ask5)",
        "行情",
    ),
    _f(
        "l2_depth5_ratio",
        "五档买侧深度占比",
        "book_obi",
        GROUP_BOOK,
        "mean(bid5/(bid5+ask5) | bid1>0 and ask1>0) [09:30-14:57]",
        "行情",
        "买侧占五档总深度比例；与 OBI=(bid-ask)/(bid+ask) 互补而非重复",
    ),
    _f("l2_obi5_mean", "五档 OBI 均值", "book_obi", GROUP_BOOK, "mean(obi5 | bid1>0 and ask1>0) [09:30-14:57]", "行情"),
    _f("l2_obi5_std", "五档 OBI 标准差", "book_obi", GROUP_BOOK, "std(obi5 | bid1>0 and ask1>0) [09:30-14:57]", "行情"),
    _f("l2_obi5_max", "五档 OBI 最大值", "book_obi", GROUP_BOOK, "max(obi5 | bid1>0 and ask1>0) [09:30-14:57]", "行情"),
    _f("l2_obi5_min", "五档 OBI 最小值", "book_obi", GROUP_BOOK, "min(obi5 | bid1>0 and ask1>0) [09:30-14:57]", "行情"),
    _f("l2_obi5_close", "五档 OBI 最后值", "book_obi", GROUP_BOOK, "last(obi5 | bid1>0 and ask1>0) [09:30-14:57]", "行情"),
    _f("l2_obi10_mean", "十档 OBI 均值", "book_obi", GROUP_BOOK, "mean(obi10 | bid1>0 and ask1>0) [09:30-14:57]", "行情"),
    _f(
        "l2_microprice_deviation_mean",
        "微观价格偏离均值",
        "book_obi",
        GROUP_BOOK,
        "mean((micro-mid)/mid)",
        "行情",
    ),
    _f(
        "l2_microprice_deviation_std",
        "微观价格偏离标准差",
        "book_obi",
        GROUP_BOOK,
        "std((micro-mid)/mid)",
        "行情",
    ),
    # --- order: 65-78 ---
    _f(
        "l2_bid_order_volume",
        "买单委托量",
        "order_cancel",
        GROUP_ORDER,
        "sum(original bid order qty) [09:30-14:57]",
        "逐笔委托,逐笔成交",
        "沪市原始量 = 委托流 A 记录量 + 该委托作为主动方的成交量；委托流中缺失的即时成交主动单从成交补出",
    ),
    _f(
        "l2_ask_order_volume",
        "卖单委托量",
        "order_cancel",
        GROUP_ORDER,
        "sum(original ask order qty) [09:30-14:57]",
        "逐笔委托,逐笔成交",
    ),
    _f(
        "l2_bid_order_amount",
        "买单委托金额",
        "order_cancel",
        GROUP_ORDER,
        "sum(original bid order amount) [09:30-14:57]",
        "逐笔委托,逐笔成交",
    ),
    _f(
        "l2_ask_order_amount",
        "卖单委托金额",
        "order_cancel",
        GROUP_ORDER,
        "sum(original ask order amount) [09:30-14:57]",
        "逐笔委托,逐笔成交",
    ),
    _f(
        "l2_order_volume_imbalance_ratio",
        "买卖委托量不平衡率",
        "order_cancel",
        GROUP_ORDER,
        "(b-a)/(b+a)",
        "逐笔委托,逐笔成交",
    ),
    _f(
        "l2_order_amount_imbalance_ratio",
        "买卖委托金额不平衡率",
        "order_cancel",
        GROUP_ORDER,
        "(b-a)/(b+a)",
        "逐笔委托,逐笔成交",
    ),
    _f("l2_bid_cancel_volume", "买单撤单量", "order_cancel", GROUP_ORDER, "sum(bid cancel qty | matched continuous order)", "逐笔委托,逐笔成交"),
    _f("l2_ask_cancel_volume", "卖单撤单量", "order_cancel", GROUP_ORDER, "sum(ask cancel qty | matched continuous order)", "逐笔委托,逐笔成交"),
    _f(
        "l2_bid_cancel_amount",
        "买单撤单金额",
        "order_cancel",
        GROUP_ORDER,
        "sum(cancel qty * order price | matched continuous order) [09:30-14:57]",
        "逐笔委托,逐笔成交",
        "深市撤单在逐笔成交中价格为 0，按委托号回查逐笔委托价格；任一价格无法回查时该侧撤单金额为 NaN",
    ),
    _f(
        "l2_ask_cancel_amount",
        "卖单撤单金额",
        "order_cancel",
        GROUP_ORDER,
        "sum(cancel qty * order price | matched continuous order) [09:30-14:57]",
        "逐笔委托,逐笔成交",
    ),
    _f(
        "l2_bid_cancel_rate",
        "买单撤单率",
        "order_cancel",
        GROUP_ORDER,
        "matched_cancel_qty/continuous_order_qty [09:30-14:57]",
        "逐笔委托,逐笔成交",
    ),
    _f(
        "l2_ask_cancel_rate",
        "卖单撤单率",
        "order_cancel",
        GROUP_ORDER,
        "matched_cancel_qty/continuous_order_qty [09:30-14:57]",
        "逐笔委托,逐笔成交",
    ),
    _f(
        "l2_cancel_volume_imbalance_ratio",
        "买卖撤单量不平衡率",
        "order_cancel",
        GROUP_ORDER,
        "(b-a)/(b+a)",
        "逐笔委托,逐笔成交",
    ),
    _f(
        "l2_cancel_amount_imbalance_ratio",
        "买卖撤单金额不平衡率",
        "order_cancel",
        GROUP_ORDER,
        "(b-a)/(b+a)",
        "逐笔委托,逐笔成交",
    ),
    # --- microstructure: 79-100 ---
    _f(
        "l2_amihud_illiquidity",
        "Amihud 非流动性",
        "liquidity",
        GROUP_MICROSTRUCTURE,
        "mean(|r_1m|/amount_1m) [09:30-14:57, reset at lunch, amount_1m>0]",
        "逐笔成交",
    ),
    _f("l2_book_depth_mean", "平均盘口深度", "liquidity", GROUP_MICROSTRUCTURE, "mean(bid10+ask10)", "行情"),
    _f(
        "l2_book_depth_volatility",
        "盘口深度波动率",
        "liquidity",
        GROUP_MICROSTRUCTURE,
        "std(depth)/mean(depth)",
        "行情",
    ),
    _f(
        "l2_depth_amount_ratio",
        "盘口深度相对成交额比",
        "liquidity",
        GROUP_MICROSTRUCTURE,
        "mean(depth10)/continuous_amount",
        "行情,逐笔成交",
    ),
    _f(
        "l2_trade_interval_mean",
        "成交间隔均值",
        "liquidity",
        GROUP_MICROSTRUCTURE,
        "mean(dt between distinct trade timestamps) [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_trade_interval_std",
        "成交间隔标准差",
        "liquidity",
        GROUP_MICROSTRUCTURE,
        "std(dt between distinct trade timestamps) [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_trade_arrival_rate",
        "成交到达率",
        "liquidity",
        GROUP_MICROSTRUCTURE,
        "count(trade rows)/14220s [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_intraday_return_std",
        "日内价格变化标准差",
        "liquidity",
        GROUP_MICROSTRUCTURE,
        "std(1m log return) [09:30-14:57, reset at lunch]",
        "逐笔成交",
    ),
    _f(
        "l2_intraday_max_drawdown",
        "日内最大回撤",
        "liquidity",
        GROUP_MICROSTRUCTURE,
        "min((p-cummax(p))/cummax(p)) [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_intraday_amount_cv",
        "日内成交额波动率",
        "liquidity",
        GROUP_MICROSTRUCTURE,
        "std(amount_1m)/mean(amount_1m) [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_open_amount_ratio",
        "开盘成交额占比",
        "liquidity",
        GROUP_MICROSTRUCTURE,
        "amount[09:30-10:00]/continuous_amount",
        "逐笔成交",
    ),
    _f(
        "l2_close_amount_ratio",
        "尾盘成交额占比",
        "liquidity",
        GROUP_MICROSTRUCTURE,
        "amount[14:30-14:57]/continuous_amount",
        "逐笔成交",
    ),
    _f(
        "l2_close_imbalance",
        "尾盘主动买卖不平衡率",
        "liquidity",
        GROUP_MICROSTRUCTURE,
        "(buy_amt-sell_amt)/(buy_amt+sell_amt) [14:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_close_obi",
        "尾盘盘口不平衡率",
        "liquidity",
        GROUP_MICROSTRUCTURE,
        "mean(obi5) [14:30-14:57]",
        "行情",
    ),
    _f("l2_ofi1", "一档订单流不平衡", "order_flow", GROUP_MICROSTRUCTURE, "sum(ofi1)/mean(depth1)", "行情"),
    _f("l2_ofi5", "五档订单流不平衡", "order_flow", GROUP_MICROSTRUCTURE, "sum(ofi5)/mean(depth5)", "行情"),
    _f("l2_ofi_std", "日内 OFI 标准差", "order_flow", GROUP_MICROSTRUCTURE, "std(ofi1/mean(depth1))", "行情"),
    _f(
        "l2_ofi_price_corr",
        "OFI 与价格变化相关性",
        "order_flow",
        GROUP_MICROSTRUCTURE,
        "corr(ofi, dmid)",
        "行情",
    ),
    _f(
        "l2_trade_book_imbalance_diff",
        "成交不平衡与盘口不平衡差",
        "order_flow",
        GROUP_MICROSTRUCTURE,
        "trade_amount_imbalance_ratio[09:30-14:57] - mean(obi5)",
        "逐笔成交,行情",
    ),
    _f(
        "l2_trade_price_direction_agreement",
        "成交方向与价格变化一致率",
        "order_flow",
        GROUP_MICROSTRUCTURE,
        "mean(sign(dp) == trade sign | dp != 0) [09:30-14:57]",
        "逐笔成交",
    ),
    _f(
        "l2_effective_spread",
        "有效价差",
        "order_flow",
        GROUP_MICROSTRUCTURE,
        "mean(2*|p-mid_prev|/mid_prev) [09:30-14:57]",
        "逐笔成交,行情",
        "mid_prev 取严格早于成交时间、买一卖一都为正的最近一帧快照",
    ),
    _f(
        "l2_kyle_lambda",
        "Kyle Lambda",
        "order_flow",
        GROUP_MICROSTRUCTURE,
        "OLS slope of r_1m on signed_amount_1m [09:30-14:57, reset at lunch]",
        "逐笔成交",
    ),
)

assert len(FACTOR_DEFINITIONS) == 100, len(FACTOR_DEFINITIONS)


def factor_ids() -> list[str]:
    return [item.factor_id for item in FACTOR_DEFINITIONS]


def factor_ids_for_group(group: str) -> list[str]:
    return [item.factor_id for item in FACTOR_DEFINITIONS if item.factor_group == group]


def definitions_for_groups(groups: Iterable[str] | None = None) -> list[FactorDef]:
    selected = set(groups) if groups is not None else set(ALL_GROUPS)
    return [item for item in FACTOR_DEFINITIONS if item.factor_group in selected]


def wide_columns() -> list[str]:
    return ["trade_date", "symbol", *factor_ids()]


def schema_hash() -> str:
    """列集指纹，用于数据集登记。"""
    import hashlib

    payload = "|".join(factor_ids()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]
