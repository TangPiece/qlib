# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Industry-count weighted TopkDropout strategy.

Keeps TopkDropoutStrategy sell/buy selection (topk / n_drop). Industry does not
change which names trade. Only the cash allocated to newly bought names is split
by n_i^alpha, where n_i is the industry count in the intended post-trade book
(holdings after sells union buy list).
"""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO
from qlib.backtest.position import Position
from qlib.contrib.strategy.signal_strategy import TopkDropoutStrategy
from qlib.log import get_module_logger

DIRNAME = Path(__file__).absolute().resolve()
_REPO_ROOT = DIRNAME.parents[3]
DEFAULT_INDUSTRY_MAP = (
    _REPO_ROOT
    / "examples"
    / "benchmarks"
    / "LightGBM"
    / "data"
    / "csi300_sw_l1_industry_intervals.parquet"
)


class IndustryCountWeightStrategy(TopkDropoutStrategy):
    """TopkDropout with industry-count cash split on new buys only.

    For each name i in the buy list:

        v_i = cash * risk_degree * (n_i ** alpha) / sum_{j in buy} (n_j ** alpha)

    where n_i is the number of names in the same industry within
    (holdings after sells) ∪ buy. Existing holdings are not rebalanced.
    ``alpha=0`` recovers equal-cash buys like TopkDropoutStrategy.
    """

    def __init__(
        self,
        *,
        topk,
        n_drop,
        alpha: float = 0.5,
        industry_map_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        super().__init__(topk=topk, n_drop=n_drop, **kwargs)
        self.alpha = float(alpha)
        self.logger = get_module_logger("IndustryCountWeightStrategy")
        path = Path(industry_map_path) if industry_map_path is not None else DEFAULT_INDUSTRY_MAP
        path = path.expanduser()
        if not path.is_absolute():
            cwd_cand = path.resolve()
            repo_cand = (_REPO_ROOT / path).resolve()
            path = cwd_cand if cwd_cand.exists() else repo_cand
        else:
            path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Industry interval map not found: {path}. "
                "Run examples/benchmarks/LightGBM/build_csi300_industry_map.py first."
            )
        self.industry_map_path = path
        self._intervals = self._load_intervals(path)
        self.logger.info(
            "loaded industry intervals from %s (%d instruments)",
            path,
            len(self._intervals),
        )

    @staticmethod
    def _load_intervals(path: Path) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, str]]]:
        df = pd.read_parquet(path)
        required = {"instrument", "industry_code", "start_date", "end_date"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"industry map missing columns: {sorted(missing)}")
        out: Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp, str]]] = defaultdict(list)
        for row in df.itertuples(index=False):
            inst = str(getattr(row, "instrument")).strip().upper()
            code = str(getattr(row, "industry_code")).strip()
            start = pd.Timestamp(getattr(row, "start_date")).normalize()
            end = pd.Timestamp(getattr(row, "end_date")).normalize()
            out[inst].append((start, end, code))
        for inst in out:
            out[inst].sort(key=lambda x: (x[1], x[0]))
        return dict(out)

    def lookup_industry(self, instrument: str, trade_date: pd.Timestamp) -> str:
        """PIT industry: covering interval; else latest end_date < trade_date; else sentinel."""
        trade_date = pd.Timestamp(trade_date).normalize()
        key = str(instrument).strip().upper()
        intervals = self._intervals.get(key)
        if not intervals:
            return f"__MISS__{key}"

        covering = [iv for iv in intervals if iv[0] <= trade_date <= iv[1]]
        if covering:
            covering.sort(key=lambda x: x[0])
            return covering[-1][2]

        earlier = [iv for iv in intervals if iv[1] < trade_date]
        if earlier:
            earlier.sort(key=lambda x: x[1])
            return earlier[-1][2]

        return f"__MISS__{key}"

    def _buy_cash_values(
        self,
        buy: Sequence[str],
        held_after_sell: Sequence[str],
        cash: float,
        trade_date: pd.Timestamp,
    ) -> Dict[str, float]:
        """Map each buy code -> cash notional (already includes risk_degree)."""
        if not buy:
            return {}
        budget = cash * self.risk_degree
        if budget <= 0:
            return {code: 0.0 for code in buy}

        intended = list(dict.fromkeys(list(held_after_sell) + list(buy)))
        industries = {code: self.lookup_industry(code, trade_date) for code in intended}
        counts = Counter(industries.values())

        raw = np.array([float(counts[industries[code]]) ** self.alpha for code in buy], dtype=float)
        total = float(raw.sum())
        if total <= 0:
            equal = budget / len(buy)
            return {code: equal for code in buy}
        weights = raw / total
        return {code: float(budget * w) for code, w in zip(buy, weights)}

    def generate_trade_decision(self, execute_result=None):
        # Same selection logic as TopkDropoutStrategy; only buy cash split differs.
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None:
            return TradeDecisionWO([], self)

        if self.only_tradable:

            def get_first_n(li, n, reverse=False):
                cur_n = 0
                res = []
                for si in reversed(li) if reverse else li:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(li, n):
                return get_first_n(li, n, reverse=True)

            def filter_stock(li):
                return [
                    si
                    for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]

        else:

            def get_first_n(li, n):
                return list(li)[:n]

            def get_last_n(li, n):
                return list(li)[-n:]

            def filter_stock(li):
                return li

        current_temp: Position = copy.deepcopy(self.trade_position)
        sell_order_list = []
        buy_order_list = []
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        last = pred_score.reindex(current_stock_list).sort_values(ascending=False).index

        if self.method_buy == "top":
            today = get_first_n(
                pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last),
            )
        elif self.method_buy == "random":
            topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.n_drop + self.topk - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError("This type of input is not supported")

        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

        if self.method_sell == "bottom":
            sell = last[last.isin(get_last_n(comb, self.n_drop))]
        elif self.method_sell == "random":
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:
                sell = candi
        else:
            raise NotImplementedError("This type of input is not supported")

        buy = today[: len(sell) + self.topk - len(last)]
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                continue
            if code in sell:
                time_per_step = self.trade_calendar.get_freq()
                if current_temp.get_stock_count(code, bar=time_per_step) < self.hold_thresh:
                    continue
                sell_amount = current_temp.get_stock_amount(code=code)
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,
                )
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    cash += trade_val - trade_cost

        held_after_sell = current_temp.get_stock_list()
        trade_date = pd.Timestamp(trade_start_time).normalize()
        buy_values = self._buy_cash_values(buy, held_after_sell, cash, trade_date)

        for code in buy:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                continue
            value = buy_values.get(code, 0.0)
            if value <= 0:
                continue
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time
            )
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            buy_order_list.append(buy_order)
        return TradeDecisionWO(sell_order_list + buy_order_list, self)
