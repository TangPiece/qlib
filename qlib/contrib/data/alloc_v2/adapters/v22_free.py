# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""V2.2-Free 全球风险与商品价格代理：VIX/DXY/原油/白银/BTC/实际利率/美联储资产负债表。

全部为价格或官方存量序列，evidence_class=proxy/direct；不是资金净流入。
商业库存、CFTC、航运指数不在本适配器。
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from io import StringIO

import pandas as pd

from ..network import prepare_network, resolve_http_timeout
from ..schema import finite_number, first_col, in_range, parse_day
from .base import lookup, row_of, safe_call

_log = logging.getLogger(__name__)

_CBOE_VIX_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
# 国内直连常被墙：多端点各试一次，失败立即跳过，不挂 60s
_BTC_KLINE_URLS = (
    "https://data-api.binance.vision/api/v3/klines",
    "https://api.binance.com/api/v3/klines",
)


def _http_get(url: str, *, params: dict | None = None) -> bytes:
    """走 requests + 网络层短超时；失败抛异常由调用方降级。"""
    prepare_network()
    import requests

    resp = requests.get(
        url,
        params=params,
        headers={"User-Agent": "alloc_v2/v22_free"},
        timeout=resolve_http_timeout(),
    )
    resp.raise_for_status()
    return resp.content


def _http_csv(url: str) -> pd.DataFrame:
    raw = _http_get(url).decode("utf-8", errors="replace")
    return pd.read_csv(StringIO(raw))


def _shift_available(day: date, lag_days: int) -> date:
    return day + timedelta(days=max(0, lag_days))


class V22FreeAdapter:
    """免费可核验的全球环境序列。"""

    name = "v22_free"

    def fetch(self, *, start: date | None = None, end: date | None = None) -> list:
        # 国内源优先；境外源（VIX/FRED/BTC）放后且各自短超时降级
        rows: list = []
        rows.extend(self._fetch_dxy(start, end))
        rows.extend(self._fetch_oil(start, end))
        rows.extend(self._fetch_silver_sge(start, end))
        rows.extend(self._fetch_silver_etf(start, end))
        rows.extend(self._fetch_vix(start, end))
        rows.extend(self._fetch_fred_series("us_real_rate", "DFII10", "US_REAL10Y", "yield", "pct", start, end, lag=1))
        rows.extend(self._fetch_fred_series("fed_balance", "WALCL", "FED_WALCL", "assets", "usd_mn", start, end, lag=1))
        rows.extend(self._fetch_btc(start, end))
        return rows

    def _fetch_vix(self, start: date | None, end: date | None) -> list:
        spec = lookup("vix_cboe")
        try:
            frame = _http_csv(_CBOE_VIX_CSV)
        except Exception as exc:  # noqa: BLE001
            _log.warning("vix_cboe failed: %s", exc)
            return []
        date_col = first_col(frame, ("DATE", "Date", "date"))
        value_col = first_col(frame, ("CLOSE", "Close", "close"))
        if date_col is None or value_col is None:
            return []
        rows = []
        for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            rows.append(
                row_of(
                    spec,
                    "VIX",
                    day,
                    "close",
                    value,
                    available_at=_shift_available(day, 1),
                    unit="index",
                )
            )
        return rows

    def _fetch_dxy(self, start: date | None, end: date | None) -> list:
        import akshare as ak

        spec = lookup("dxy")
        frame = safe_call("index_global_hist_em:美元指数", ak.index_global_hist_em, symbol="美元指数")
        if frame.empty:
            return []
        date_col = first_col(frame, ("日期", "date"))
        value_col = first_col(frame, ("最新价", "收盘", "close"))
        if date_col is None or value_col is None:
            return []
        rows = []
        for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            rows.append(row_of(spec, "DXY", day, "close", value, unit="index"))
        return rows

    def _fetch_oil(self, start: date | None, end: date | None) -> list:
        import akshare as ak

        rows = []
        for dataset, symbol, entity in (
            ("oil_brent", "B00Y", "BRENT"),
            ("oil_wti", "CL00Y", "WTI"),
        ):
            spec = lookup(dataset)
            frame = safe_call(f"futures_global_hist_em:{symbol}", ak.futures_global_hist_em, symbol=symbol)
            if frame.empty:
                continue
            date_col = first_col(frame, ("日期", "date"))
            value_col = first_col(frame, ("最新价", "收盘", "close"))
            if date_col is None or value_col is None:
                continue
            for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
                day = parse_day(raw_day)
                value = finite_number(raw_value)
                if value is None or not in_range(day, start, end):
                    continue
                rows.append(row_of(spec, entity, day, "close", value, unit="usd/bbl"))
        return rows

    def _fetch_silver_sge(self, start: date | None, end: date | None) -> list:
        import akshare as ak

        spec = lookup("silver_sge")
        frame = safe_call("spot_silver_benchmark_sge", ak.spot_silver_benchmark_sge)
        if frame.empty:
            return []
        date_col = first_col(frame, ("交易时间", "交易日期", "日期", "date"))
        value_col = first_col(frame, ("晚盘价", "早盘价", "收盘价", "价格"))
        if date_col is None or value_col is None:
            return []
        rows = []
        for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            rows.append(row_of(spec, "SGE_SILVER", day, "close", value, unit="cny/kg"))
        return rows

    def _fetch_silver_etf(self, start: date | None, end: date | None) -> list:
        """金十白银 ETF 持仓：有界分页，与黄金同一套早停策略。"""
        from .jin10_etf import fetch_jin10_etf_holdings

        _log.info("拉取白银 ETF 持仓（有界分页，start=%s）…", start)
        spec = lookup("silver_etf")
        frame = safe_call(
            "macro_cons_silver",
            fetch_jin10_etf_holdings,
            attr_id="2",
            commodity="白银",
            start=start,
            end=end,
        )
        if frame.empty:
            return []
        date_col = first_col(frame, ("日期",))
        stock_col = first_col(frame, ("总库存",))
        change_col = first_col(frame, ("增持/减持", "增减持"))
        if date_col is None:
            return []
        rows = []
        for idx in range(len(frame)):
            day = parse_day(frame.iloc[idx][date_col])
            if not in_range(day, start, end):
                continue
            if stock_col is not None:
                stock = finite_number(frame.iloc[idx][stock_col])
                if stock is not None:
                    rows.append(
                        row_of(
                            spec,
                            "SLV",
                            day,
                            "tonnes",
                            stock,
                            available_at=_shift_available(day, 1),
                            evidence_class="direct",
                            unit="tonne",
                        )
                    )
            if change_col is not None:
                change = finite_number(frame.iloc[idx][change_col])
                if change is not None:
                    rows.append(
                        row_of(
                            spec,
                            "SLV",
                            day,
                            "change",
                            change,
                            available_at=_shift_available(day, 1),
                            evidence_class="direct",
                            unit="tonne",
                        )
                    )
        return rows

    def _fetch_btc(self, start: date | None, end: date | None) -> list:
        """BTC 日 K：多镜像短超时；连不上立即降级，不挂死 update。"""
        prepare_network()
        import requests

        spec = lookup("crypto_btc")
        end_day = end or date.today()
        start_day = start or date(2018, 1, 1)
        start_ms = int(datetime(start_day.year, start_day.month, start_day.day, tzinfo=timezone.utc).timestamp() * 1000)
        end_ms = int(
            datetime(end_day.year, end_day.month, end_day.day, 23, 59, 59, tzinfo=timezone.utc).timestamp() * 1000
        )
        timeout = resolve_http_timeout()
        base_url = None
        last_err: Exception | None = None
        # 先探测哪个镜像可达（各试一次）
        for url in _BTC_KLINE_URLS:
            try:
                probe = requests.get(
                    url,
                    params={"symbol": "BTCUSDT", "interval": "1d", "limit": 1},
                    headers={"User-Agent": "alloc_v2/v22_free"},
                    timeout=timeout,
                )
                probe.raise_for_status()
                base_url = url
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                _log.warning("crypto_btc probe %s failed: %s", url, exc)
        if base_url is None:
            _log.warning("crypto_btc unavailable (直连常被墙；可设代理后 ALLOC_V2_NO_PROXY=0): %s", last_err)
            return []

        cursor = start_ms
        rows: list = []
        max_pages = 15 if start is None else max(2, min(15, (end_day - start_day).days // 800 + 2))
        for _ in range(max_pages):
            if cursor > end_ms:
                break
            try:
                resp = requests.get(
                    base_url,
                    params={
                        "symbol": "BTCUSDT",
                        "interval": "1d",
                        "limit": 1000,
                        "startTime": cursor,
                        "endTime": end_ms,
                    },
                    headers={"User-Agent": "alloc_v2/v22_free"},
                    timeout=timeout,
                )
                resp.raise_for_status()
                payload = resp.json()
            except Exception as exc:  # noqa: BLE001
                _log.warning("crypto_btc page failed: %s", exc)
                break
            if not isinstance(payload, list) or not payload:
                break
            for item in payload:
                if not isinstance(item, (list, tuple)) or len(item) < 6:
                    continue
                day = datetime.fromtimestamp(int(item[0]) / 1000, tz=timezone.utc).date()
                if not in_range(day, start, end):
                    continue
                close = finite_number(item[4])
                volume = finite_number(item[5])
                if close is not None:
                    rows.append(row_of(spec, "BTCUSDT", day, "close", close, unit="usdt"))
                if volume is not None:
                    rows.append(row_of(spec, "BTCUSDT", day, "volume", volume, unit="btc"))
            last_open = int(payload[-1][0])
            next_cursor = last_open + 86_400_000
            if next_cursor <= cursor or len(payload) < 1000:
                break
            cursor = next_cursor
        return rows

    def _fetch_fred_series(
        self,
        dataset: str,
        series_id: str,
        entity_id: str,
        field: str,
        unit: str,
        start: date | None,
        end: date | None,
        *,
        lag: int,
    ) -> list:
        spec = lookup(dataset)
        try:
            frame = _http_csv(_FRED_CSV.format(series_id=series_id))
        except Exception as exc:  # noqa: BLE001
            _log.warning("%s FRED %s failed: %s", dataset, series_id, exc)
            return []
        date_col = first_col(frame, ("observation_date", "DATE", "date"))
        value_col = first_col(frame, (series_id, "value", "VALUE"))
        if date_col is None or value_col is None:
            # FRED 第二列通常是序列名
            cols = list(frame.columns)
            if len(cols) >= 2:
                date_col = cols[0]
                value_col = cols[1]
            else:
                return []
        rows = []
        for raw_day, raw_value in zip(frame[date_col], frame[value_col]):
            day = parse_day(raw_day)
            value = finite_number(raw_value)
            if value is None or not in_range(day, start, end):
                continue
            rows.append(
                row_of(
                    spec,
                    entity_id,
                    day,
                    field,
                    value,
                    available_at=_shift_available(day, lag),
                    unit=unit,
                )
            )
        return rows

