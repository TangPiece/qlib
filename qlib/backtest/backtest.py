# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations

from typing import Dict, TYPE_CHECKING, Generator, Optional, Tuple, Union, cast

import pandas as pd

from qlib.backtest.decision import BaseTradeDecision
from qlib.backtest.report import Indicator

if TYPE_CHECKING:
    from qlib.strategy.base import BaseStrategy
    from qlib.backtest.executor import BaseExecutor

from tqdm.auto import tqdm

from ..utils.time import Freq
from .account import WRITE_OFF_MISSING_BARS


def lookback_windows(calendar, n: int = WRITE_OFF_MISSING_BARS):
    """当前步往前（含当日）最多 n 根 bar 的 [start, end]。步数不够则缩短。"""
    if calendar is None or n <= 0:
        return []
    step = calendar.get_trade_step()
    windows = []
    for shift in range(n):
        if step - shift < 0:
            break
        try:
            windows.append(calendar.get_step_time(step, shift=shift))
        except (IndexError, KeyError):
            break
    windows.reverse()
    return windows


def _write_off_unlisted(executor: "BaseExecutor") -> None:
    """在策略出单前，按外层日历窗口下账确认离开行情的持仓。"""
    account = getattr(executor, "trade_account", None)
    exchange = getattr(executor, "trade_exchange", None)
    if account is None or exchange is None:
        return
    write_off = getattr(account, "write_off_unlisted", None)
    if not callable(write_off):
        return
    start = end = None
    calendar = getattr(executor, "trade_calendar", None)
    windows = lookback_windows(calendar, WRITE_OFF_MISSING_BARS)
    if calendar is not None:
        start, end = calendar.get_step_time()
    write_off(exchange, start, end, windows=windows)

PORT_METRIC = Dict[str, Tuple[pd.DataFrame, dict]]
INDICATOR_METRIC = Dict[str, Tuple[pd.DataFrame, Indicator]]


def backtest_loop(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    trade_strategy: BaseStrategy,
    trade_executor: BaseExecutor,
) -> Tuple[PORT_METRIC, INDICATOR_METRIC]:
    """backtest function for the interaction of the outermost strategy and executor in the nested decision execution

    please refer to the docs of `collect_data_loop`

    Returns
    -------
    portfolio_dict: PORT_METRIC
        it records the trading portfolio_metrics information
    indicator_dict: INDICATOR_METRIC
        it computes the trading indicator
    """
    return_value: dict = {}
    for _decision in collect_data_loop(start_time, end_time, trade_strategy, trade_executor, return_value):
        pass

    portfolio_dict = cast(PORT_METRIC, return_value.get("portfolio_dict"))
    indicator_dict = cast(INDICATOR_METRIC, return_value.get("indicator_dict"))

    return portfolio_dict, indicator_dict


def collect_data_loop(
    start_time: Union[pd.Timestamp, str],
    end_time: Union[pd.Timestamp, str],
    trade_strategy: BaseStrategy,
    trade_executor: BaseExecutor,
    return_value: dict | None = None,
) -> Generator[BaseTradeDecision, Optional[BaseTradeDecision], None]:
    """Generator for collecting the trade decision data for rl training

    Parameters
    ----------
    start_time : Union[pd.Timestamp, str]
        closed start time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
    end_time : Union[pd.Timestamp, str]
        closed end time for backtest
        **NOTE**: This will be applied to the outmost executor's calendar.
        E.g. Executor[day](Executor[1min]), setting `end_time == 20XX0301` will include all the minutes on 20XX0301
    trade_strategy : BaseStrategy
        the outermost portfolio strategy
    trade_executor : BaseExecutor
        the outermost executor
    return_value : dict
        used for backtest_loop

    Yields
    -------
    object
        trade decision
    """
    trade_executor.reset(start_time=start_time, end_time=end_time)
    trade_strategy.reset(level_infra=trade_executor.get_level_infra())

    with tqdm(total=trade_executor.trade_calendar.get_trade_len(), desc="backtest loop") as bar:
        _execute_result = None
        while not trade_executor.finished():
            # 决策前先下账已离开行情表的持仓，避免退市股占坑锁死换手
            _write_off_unlisted(trade_executor)
            _trade_decision: BaseTradeDecision = trade_strategy.generate_trade_decision(_execute_result)
            _execute_result = yield from trade_executor.collect_data(_trade_decision, level=0)
            trade_strategy.post_exe_step(_execute_result)
            bar.update(1)
        trade_strategy.post_upper_level_exe_step()

    if return_value is not None:
        all_executors = trade_executor.get_all_executors()

        portfolio_dict: PORT_METRIC = {}
        indicator_dict: INDICATOR_METRIC = {}

        for executor in all_executors:
            key = "{}{}".format(*Freq.parse(executor.time_per_step))
            if executor.trade_account.is_port_metr_enabled():
                portfolio_dict[key] = executor.trade_account.get_portfolio_metrics()

            indicator_df = executor.trade_account.get_trade_indicator().generate_trade_indicators_dataframe()
            indicator_obj = executor.trade_account.get_trade_indicator()
            indicator_dict[key] = (indicator_df, indicator_obj)

        return_value.update({"portfolio_dict": portfolio_dict, "indicator_dict": indicator_dict})
