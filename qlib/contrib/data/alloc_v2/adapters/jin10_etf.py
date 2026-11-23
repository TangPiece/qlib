# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""金十 ETF 持仓有界拉取：按 start 早停，避免 update 时全历史分页拖死。"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

from ..network import prepare_network
from ..schema import parse_day

_log = logging.getLogger(__name__)

_URL = "https://datacenter-api.jin10.com/reports/list_v2"
_HEADERS = {
    "user-agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/107.0.0.0 Safari/537.36"
    ),
    "x-app-id": "rU6QIu7JHe2gOUeR",
    "x-csrf-token": "x-csrf-token",
    "x-version": "1.0.0",
}

# 无 start 时仍限制页数，防止金十挂起拖成数十分钟
_DEFAULT_MAX_PAGES = 120
_UPDATE_MAX_PAGES = 12  # 增量窗口通常只需最近几页


def _max_pages_for(start: date | None) -> int:
    if start is None:
        return _DEFAULT_MAX_PAGES
    # 近 60 天窗口：少翻页即可
    if start >= date.today() - timedelta(days=60):
        return _UPDATE_MAX_PAGES
    return _DEFAULT_MAX_PAGES


def fetch_jin10_etf_holdings(
    *,
    attr_id: str,
    commodity: str,
    start: date | None = None,
    end: date | None = None,
    max_pages: int | None = None,
) -> pd.DataFrame:
    """拉取金十 ETF 持仓；到达 start 之前或达到 max_pages 即停。

    attr_id: \"1\"=黄金, \"2\"=白银。失败返回空表，不抛到外层。
    """
    prepare_network()
    import requests

    pages = max_pages if max_pages is not None else _max_pages_for(start)
    params: dict[str, Any] = {
        "category": "etf",
        "attr_id": str(attr_id),
        "max_date": "",
        "_": str(int(round(time.time() * 1000))),
    }
    # 有 end 时从 end 往前翻，减少无关页
    if end is not None:
        params["max_date"] = end.isoformat()

    chunks: list[pd.DataFrame] = []
    for page in range(1, pages + 1):
        try:
            resp = requests.get(_URL, params=params, headers=_HEADERS)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:  # noqa: BLE001
            _log.warning("jin10 etf attr=%s page=%s failed: %s", attr_id, page, exc)
            break
        values = (payload or {}).get("data", {}).get("values") or []
        if not values:
            break
        temp = pd.DataFrame(values)
        if temp.empty:
            break
        chunks.append(temp)
        try:
            last_raw = temp.iat[-1, 0]
            last_day = parse_day(last_raw) or datetime.strptime(str(last_raw)[:10], "%Y-%m-%d").date()
        except Exception:  # noqa: BLE001
            break
        # 本页最旧日已早于 start：收齐后停
        if start is not None and last_day < start:
            _log.info("jin10 etf attr=%s 早停：page=%s last=%s < start=%s", attr_id, page, last_day, start)
            break
        next_max = (last_day - timedelta(days=1)).isoformat()
        params["max_date"] = next_max
        params["_"] = str(int(round(time.time() * 1000)))
        if page % 10 == 0:
            _log.info("jin10 etf attr=%s 进度 page=%s/%s last=%s", attr_id, page, pages, last_day)
    else:
        _log.warning("jin10 etf attr=%s 达页数上限 %s，可能未拉满历史", attr_id, pages)

    if not chunks:
        return pd.DataFrame()
    big = pd.concat(chunks, ignore_index=True)
    big.columns = ["日期", "总库存", "增持/减持", "总价值"][: len(big.columns)]
    big["商品"] = commodity
    big["日期"] = pd.to_datetime(big["日期"], errors="coerce").dt.date
    for col in ("总库存", "增持/减持", "总价值"):
        if col in big.columns:
            big[col] = pd.to_numeric(big[col], errors="coerce")
    big = big.dropna(subset=["日期"]).sort_values("日期").drop_duplicates(subset=["日期"], keep="last")
    if start is not None:
        big = big[big["日期"] >= start]
    if end is not None:
        big = big[big["日期"] <= end]
    big = big.reset_index(drop=True)
    _log.info("jin10 etf attr=%s rows=%s", attr_id, len(big))
    return big
