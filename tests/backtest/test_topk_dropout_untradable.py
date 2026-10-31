"""退市下账与临停不锁死 TopkDropout。"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest

_LGB_DIR = Path(__file__).resolve().parents[2] / "examples" / "benchmarks" / "LightGBM"
if str(_LGB_DIR) not in sys.path:
    sys.path.insert(0, str(_LGB_DIR))

from qlib.backtest.account import WRITE_OFF_MISSING_BARS, Account
from qlib.backtest.backtest import lookback_windows
from qlib.backtest.decision import OrderDir
from qlib.backtest.position import Position
from qlib.contrib.strategy.industry_count_weight import IndustryCountWeightStrategy
from qlib.contrib.strategy.signal_strategy import (
    TopkDropoutStrategy,
    limit_trade_direction,
    make_topk_dropout_selectors,
    plan_topk_dropout,
)


class _FakeQuote:
    def __init__(self, listed, no_close=(), nan_close=(), empty_days=(), missing_by_day=None):
        self.listed = set(listed)
        self.no_close = set(no_close)
        self.nan_close = set(nan_close)
        self.empty_days = {pd.Timestamp(d).normalize() for d in empty_days}
        self.missing_by_day = {
            code: {pd.Timestamp(d).normalize() for d in days}
            for code, days in (missing_by_day or {}).items()
        }

    def get_all_stock(self):
        return self.listed

    def get_data(self, stock_id, start_time, end_time, field, method=None):
        del end_time, field, method
        day = pd.Timestamp(start_time).normalize()
        if day in self.empty_days:
            return None
        if stock_id not in self.listed:
            return None
        if stock_id in self.no_close:
            return None
        if day in self.missing_by_day.get(stock_id, set()):
            return None
        if stock_id in self.nan_close:
            return float("nan")
        return 1.0


class _FakeExchange:
    """listed=行情表；no_close=当日无 $close 行；untradable=当天不可卖。"""

    def __init__(self, listed, untradable=(), no_close=(), nan_close=(), empty_days=(), missing_by_day=None):
        self.quote = _FakeQuote(
            listed,
            no_close=no_close,
            nan_close=nan_close,
            empty_days=empty_days,
            missing_by_day=missing_by_day,
        )
        self.untradable = set(untradable)

    def is_stock_tradable(self, stock_id, start_time=None, end_time=None, direction=None):
        del start_time, end_time, direction
        return stock_id in self.quote.listed and stock_id not in self.untradable


class _FakeCalendar:
    def __init__(self, days):
        self.days = [pd.Timestamp(d).normalize() for d in days]
        self._step = len(self.days) - 1

    def get_trade_step(self):
        return self._step

    def get_step_time(self, trade_step=None, shift=0):
        if trade_step is None:
            trade_step = self._step
        idx = trade_step - shift
        if idx < 0 or idx >= len(self.days):
            raise IndexError(idx)
        t = self.days[idx]
        return t, t


def _windows(n, start="2025-08-13"):
    t0 = pd.Timestamp(start)
    return [(t0 + pd.Timedelta(i, unit="D"), t0 + pd.Timedelta(i, unit="D")) for i in range(n)]


def _held(n: int, extra=()):
    return [f"S{i:02d}" for i in range(n)] + list(extra)


def _scores(held, extras, dead_last=True):
    """held 分数递减；extras 更高以便成为买入候选；DEAD 最低。"""
    data = {}
    for i, code in enumerate(held):
        data[code] = 1.0 - i * 0.01
    for j, code in enumerate(extras):
        data[code] = 2.0 - j * 0.01
    if dead_last and "DEAD" in held:
        data["DEAD"] = -10.0
    return pd.Series(data)


def _plan(score, holdings, exchange, topk=10, n_drop=1):
    get_first_n, get_last_n, filter_stock, is_tradable = make_topk_dropout_selectors(
        exchange, None, None, only_tradable=False
    )
    return plan_topk_dropout(
        score,
        holdings,
        topk=topk,
        n_drop=n_drop,
        method_buy="top",
        method_sell="bottom",
        get_first_n=get_first_n,
        get_last_n=get_last_n,
        filter_stock=filter_stock,
        is_tradable=is_tradable,
    )


def _account(codes):
    return Account(
        init_cash=100.0,
        position_dict={c: {"amount": 10.0, "price": 10.0} for c in codes},
        port_metr_enabled=False,
    )


def test_write_off_stock_credits_last_mark():
    pos = Position(cash=100.0, position_dict={"DEAD": {"amount": 10.0, "price": 8.0}})
    cash = pos.write_off_stock("DEAD")
    assert cash == pytest.approx(80.0)
    assert "DEAD" not in pos.get_stock_list()
    assert pos.get_cash() == pytest.approx(180.0)


def test_write_off_unlisted_only_drops_missing_quote():
    acc = _account(["S00", "DEAD"])
    acc.current_position.position["DEAD"]["amount"] = 5.0
    acc.current_position.position["DEAD"]["price"] = 8.0
    exchange = _FakeExchange(listed=["S00"], untradable=())
    dropped = acc.write_off_unlisted(exchange)
    assert dropped == ["DEAD"]
    assert "DEAD" not in acc.current_position.get_stock_list()
    assert "S00" in acc.current_position.get_stock_list()
    assert acc.current_position.get_cash() == pytest.approx(140.0)


def test_write_off_keeps_nan_close_row():
    """有 $close 行且为 NaN：临停，即使连续多日也不下账。"""
    acc = _account(["S00", "HALT"])
    exchange = _FakeExchange(listed=["S00", "HALT"], nan_close=["HALT"])
    assert acc.write_off_unlisted(exchange, windows=_windows(WRITE_OFF_MISSING_BARS)) == []
    assert set(acc.current_position.get_stock_list()) == {"S00", "HALT"}
    assert math.isnan(exchange.quote.get_data("HALT", pd.Timestamp("2025-08-13"), None, "$close"))


def test_write_off_keeps_single_missing_bar():
    """单日漏行不够 missing_bars，不下账。"""
    acc = _account(["S00", "GAP"])
    day = pd.Timestamp("2025-08-13")
    exchange = _FakeExchange(listed=["S00", "GAP"], no_close=["GAP"])
    assert acc.write_off_unlisted(exchange, day, day, windows=[(day, day)]) == []
    assert "GAP" in acc.current_position.get_stock_list()


def test_write_off_drops_after_consecutive_missing_bars():
    """仍在历史行情表、连续 N 日无 $close 行才下账。"""
    acc = _account(["S00", "DEAD"])
    acc.current_position.position["DEAD"]["amount"] = 5.0
    acc.current_position.position["DEAD"]["price"] = 8.0
    exchange = _FakeExchange(listed=["S00", "DEAD"], no_close=["DEAD"])
    dropped = acc.write_off_unlisted(exchange, windows=_windows(WRITE_OFF_MISSING_BARS))
    assert dropped == ["DEAD"]
    assert "DEAD" not in acc.current_position.get_stock_list()
    assert acc.current_position.get_cash() == pytest.approx(140.0)


def test_write_off_skips_when_market_empty():
    """末日/日历超前：抽样全市场无行，整批不下账。"""
    acc = _account(["S00", "DEAD"])
    empty = pd.Timestamp("2026-10-31")
    exchange = _FakeExchange(listed=["S00", "DEAD"], empty_days=[empty])
    dropped = acc.write_off_unlisted(exchange, empty, empty, windows=[(empty, empty)])
    assert dropped == []
    assert set(acc.current_position.get_stock_list()) == {"S00", "DEAD"}


def test_lookback_windows_uses_outer_calendar():
    cal = _FakeCalendar(["2025-08-11", "2025-08-12", "2025-08-13"])
    wins = lookback_windows(cal, 5)
    assert [w[0].strftime("%Y-%m-%d") for w in wins] == ["2025-08-11", "2025-08-12", "2025-08-13"]


def test_delisted_writeoff_then_topk_resumes_one_in_one_out():
    """10 只可交易 + 1 只已离开行情表：下账后当天恢复 1 买 1 卖。"""
    held = _held(10, extra=["DEAD"])
    extras = [f"N{i:02d}" for i in range(5)]
    score = _scores(held, extras)
    listed = [c for c in held if c != "DEAD"] + extras
    exchange = _FakeExchange(listed=listed)

    acc = Account(
        init_cash=1000.0,
        position_dict={c: {"amount": 10.0, "price": 10.0} for c in held},
        port_metr_enabled=False,
    )
    dropped = acc.write_off_unlisted(exchange)
    assert dropped == ["DEAD"]
    remain = acc.current_position.get_stock_list()
    assert "DEAD" not in remain
    assert len(remain) == 10

    today, sell, buy, slot_last = _plan(score, remain, exchange)
    assert len(slot_last) == 10
    assert len(sell) == 1
    assert len(buy) == 1
    assert "DEAD" not in list(sell) and "DEAD" not in buy


def test_suspended_does_not_block_turnover():
    """10 只可交易 + 1 只临停：临停下账不发生，改卖次差可交易股并换入新票。"""
    held = _held(10, extra=["HALT"])
    extras = [f"N{i:02d}" for i in range(5)]
    score = _scores(held, extras)
    score["HALT"] = -10.0
    listed = held + extras
    exchange = _FakeExchange(listed=listed, untradable=["HALT"])

    today, sell, buy, slot_last = _plan(score, held, exchange)
    assert "HALT" not in slot_last
    assert len(slot_last) == 10
    assert list(sell) == ["S09"]
    assert "HALT" not in list(sell)
    assert buy == ["N00"]
    assert "HALT" in held


def test_is_tradable_honors_sell_direction_for_slots():
    exchange = _FakeExchange(listed=["A", "B"], untradable=["B"])
    _get_first_n, _get_last_n, _filter_stock, is_tradable = make_topk_dropout_selectors(
        exchange, None, None, only_tradable=False
    )
    assert is_tradable("A", direction=OrderDir.SELL)
    assert not is_tradable("B", direction=OrderDir.SELL)


def test_limit_trade_direction_matches_order_loop():
    """forbid_all_trade_at_limit 时选股/下单都用 direction=None。"""
    assert limit_trade_direction(True, OrderDir.SELL) is None
    assert limit_trade_direction(True, OrderDir.BUY) is None
    assert limit_trade_direction(False, OrderDir.SELL) == OrderDir.SELL
    assert limit_trade_direction(False, OrderDir.BUY) == OrderDir.BUY


class _LimitExchange(_FakeExchange):
    """涨停：direction=SELL 可卖，direction=None 不可交易（双向禁）。"""

    def is_stock_tradable(self, stock_id, start_time=None, end_time=None, direction=None):
        del start_time, end_time
        if stock_id == "LIMIT_UP":
            return direction == OrderDir.SELL
        return stock_id in self.quote.listed and stock_id not in self.untradable


def test_plan_sell_direction_none_skips_limit_up_slot():
    """与默认 forbid_all_trade_at_limit 对齐：涨停不占坑，改卖下一只。"""
    held = _held(10, extra=["LIMIT_UP"])
    extras = [f"N{i:02d}" for i in range(3)]
    score = _scores(held, extras)
    score["LIMIT_UP"] = -10.0
    exchange = _LimitExchange(listed=held + extras)
    get_first_n, get_last_n, filter_stock, is_tradable = make_topk_dropout_selectors(
        exchange, None, None, only_tradable=False
    )
    _today, sell, buy, slot_last = plan_topk_dropout(
        score,
        held,
        topk=10,
        n_drop=1,
        method_buy="top",
        method_sell="bottom",
        get_first_n=get_first_n,
        get_last_n=get_last_n,
        filter_stock=filter_stock,
        is_tradable=is_tradable,
        sell_direction=limit_trade_direction(True, OrderDir.SELL),
    )
    assert "LIMIT_UP" not in slot_last
    assert list(sell) == ["S09"]
    assert buy == ["N00"]


def test_industry_reuses_parent_selector():
    assert "_select_buy_sell" not in IndustryCountWeightStrategy.__dict__
    assert IndustryCountWeightStrategy._select_buy_sell is TopkDropoutStrategy._select_buy_sell


def test_live_topk_tuple_and_missing_pred_not_written_off():
    from predict_live_signal import topk_dropout_actions

    held = _held(10, extra=["KEEP"])
    extras = [f"N{i:02d}" for i in range(3)]
    score = _scores([c for c in held if c != "KEEP"], extras)
    buys, sells, holds, write_offs = topk_dropout_actions(
        score,
        held,
        topk=10,
        n_drop=1,
        write_offs=[],
        is_tradable=lambda code, direction=None: str(code) != "KEEP",
    )
    assert write_offs == []
    assert "KEEP" in holds
    assert "KEEP" not in buys and "KEEP" not in sells
    assert len(buys) == 1 and len(sells) == 1


def test_live_write_offs_passed_through():
    from predict_live_signal import topk_dropout_actions

    held = ["A", "DEAD"]
    score = pd.Series({"A": 0.1, "B": 0.9})
    buys, sells, holds, write_offs = topk_dropout_actions(
        score,
        held,
        topk=1,
        n_drop=1,
        write_offs=["DEAD"],
        is_tradable=lambda code, direction=None: True,
    )
    assert write_offs == ["DEAD"]
    assert "DEAD" not in holds
    assert "A" in sells or "A" in holds


def test_live_close_tradable_set_skips_nan_and_missing():
    """NaN 或无行 = 停牌，不能因为有 MultiIndex 行就算可交易。"""
    from predict_live_signal import live_close_tradable_set

    idx = pd.MultiIndex.from_tuples(
        [("A", pd.Timestamp("2026-10-28")), ("HALT", pd.Timestamp("2026-10-28"))],
        names=["instrument", "datetime"],
    )
    close_df = pd.DataFrame({"$close": [10.0, float("nan")]}, index=idx)
    assert live_close_tradable_set(close_df) == {"A"}
    assert live_close_tradable_set(None) == set()
    assert live_close_tradable_set(pd.DataFrame()) == set()


def test_live_nan_close_not_tradable_even_with_score():
    """有 pred 分数但无非空收盘：不进买卖名单，仍留在 holds。"""
    from predict_live_signal import topk_dropout_actions

    held = _held(10, extra=["HALT"])
    extras = [f"N{i:02d}" for i in range(3)]
    score = _scores(held, extras)
    score["HALT"] = -10.0
    tradable = {c for c in held + extras if c != "HALT"}
    buys, sells, holds, write_offs = topk_dropout_actions(
        score,
        held,
        topk=10,
        n_drop=1,
        write_offs=[],
        is_tradable=lambda code, direction=None: str(code) in tradable,
    )
    assert write_offs == []
    assert "HALT" in holds
    assert "HALT" not in buys and "HALT" not in sells
    assert len(buys) == 1 and len(sells) == 1


if __name__ == "__main__":
    pytest.main([__file__])
