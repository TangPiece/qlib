#!/usr/bin/env python3
"""Export backtest positions to parquet + a self-contained HTML day browser.

Reads positions and analysis artifacts from an MLflow experiment recorder,
labels hold/buy/sell, attaches stock names and buy dates, and writes:

- long-form parquet (datetime as YYYY-MM-DD)
- single-file HTML with metrics cards; experiment JSON payload at file bottom

Example (from qlib repo root)::

    .venv/bin/python examples/benchmarks/LightGBM/export_positions_view.py \\
      --exp_name=rolling_lgb_7y2y_20261009
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import qlib
from qlib.config import C
from qlib.constant import REG_CN
from qlib.data import D
from qlib.workflow import R

DIRNAME = Path(__file__).absolute().resolve().parent
DEFAULT_OUT_DIR = DIRNAME / "data"
DEFAULT_EXP_NAME = "rolling_lgb_7y2y_20261009"
NAME_MAP_PATH = DEFAULT_OUT_DIR / "stock_name_map.json"
POSITIONS_ARTIFACT = "portfolio_analysis/positions_normal_1day.pkl"


def _latest_recorder(exp_name: str, recorder_id: Optional[str] = None):
    if recorder_id:
        return R.get_recorder(recorder_id=recorder_id, experiment_name=exp_name)
    exp = R.get_exp(experiment_name=exp_name, create=False)
    recorders = exp.list_recorders(rtype="list")
    if not recorders:
        raise RuntimeError(f"No recorders found for experiment {exp_name!r}")
    finished = [r for r in recorders if getattr(r, "status", None) == "FINISHED"]
    candidates = finished or list(recorders)
    candidates.sort(key=lambda r: getattr(r, "start_time", None) or "", reverse=True)
    return candidates[0]


def _to_qlib_code(code: str) -> str:
    code = str(code).strip()
    if code.startswith(("SH", "SZ", "BJ")):
        return code
    digits = "".join(ch for ch in code if ch.isdigit())
    if len(digits) != 6:
        return code
    if digits.startswith(("5", "6", "9")):
        return "SH" + digits
    return "SZ" + digits


def load_or_build_name_map(cache_path: Path = NAME_MAP_PATH) -> Dict[str, str]:
    """Load SH/SZ -> name map; try akshare if cache missing."""
    if cache_path.exists():
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                return {str(k): str(v) for k, v in data.items()}
        except (json.JSONDecodeError, OSError) as exc:
            warnings.warn(f"failed to read name map {cache_path}: {exc}")

    name_map: Dict[str, str] = {}
    try:
        import akshare as ak

        df = ak.stock_info_a_code_name()
        # columns typically: code, name
        code_col = "code" if "code" in df.columns else df.columns[0]
        name_col = "name" if "name" in df.columns else df.columns[1]
        for _, row in df.iterrows():
            qcode = _to_qlib_code(row[code_col])
            name = str(row[name_col]).strip()
            if qcode and name:
                name_map[qcode] = name
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(name_map, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote name map {cache_path} ({len(name_map)} names)")
    except Exception as exc:  # noqa: BLE001 — best-effort optional dependency
        warnings.warn(f"stock name map unavailable ({exc}); names will be blank")
    return name_map


def _stock_info(pos) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for code in pos.get_stock_list():
        info = pos.position[code]
        out[code] = {
            "amount": float(info.get("amount")) if info.get("amount") is not None else None,
            "price": float(info.get("price")) if info.get("price") is not None else None,
            "weight": float(info.get("weight")) if info.get("weight") is not None else None,
            "count_day": int(info.get("count_day")) if info.get("count_day") is not None else None,
        }
    return out


def _fetch_sell_closes(sell_keys: List[Tuple[str, str]]) -> Dict[Tuple[str, str], Optional[float]]:
    """Map (instrument, YYYY-MM-DD) -> $close on that day.

    Uses a single kernel: macOS spawn + python -c/<stdin> breaks qlib's multiprocessing.
    """
    out: Dict[Tuple[str, str], Optional[float]] = {k: None for k in sell_keys}
    if not sell_keys:
        return out
    instruments = sorted({code for code, _ in sell_keys})
    start = min(d for _, d in sell_keys)
    end = max(d for _, d in sell_keys)
    prev_kernels = C.kernels
    try:
        C.kernels = 1
        close_df = D.features(instruments, ["$close"], start_time=start, end_time=end, freq="day")
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"failed to load sell-day closes: {exc}")
        return out
    finally:
        C.kernels = prev_kernels
    if close_df is None or close_df.empty:
        return out
    # index: (instrument, datetime)
    for (code, date_str) in sell_keys:
        try:
            val = close_df.loc[(code, pd.Timestamp(date_str)), "$close"]
            if isinstance(val, pd.Series):
                val = val.iloc[0]
            out[(code, date_str)] = float(val) if pd.notna(val) else None
        except (KeyError, TypeError, ValueError, IndexError):
            out[(code, date_str)] = None
    return out


def build_day_frames(
    positions: Dict,
    name_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Return long parquet rows + per-day HTML payload (with name / buy_date / sell return)."""
    name_map = name_map or {}
    dates = sorted(positions.keys())
    rows: List[dict] = []
    days_payload: List[dict] = []
    prev_codes: set = set()
    prev_info: Dict[str, Dict[str, Any]] = {}
    buy_dates: Dict[str, str] = {}
    buy_prices: Dict[str, Optional[float]] = {}
    sell_keys: List[Tuple[str, str]] = []

    for dt in dates:
        pos = positions[dt]
        date_str = pd.Timestamp(dt).strftime("%Y-%m-%d")
        cash = float(pos.get_cash())
        account_value = float(pos.position.get("now_account_value", cash))
        cur_info = _stock_info(pos)
        cur_codes = set(cur_info)

        buys = sorted(cur_codes - prev_codes)
        sells = sorted(prev_codes - cur_codes)
        holds = sorted(cur_codes & prev_codes)

        sell_buy_dates = {c: buy_dates.get(c) for c in sells}
        sell_buy_prices = {c: buy_prices.get(c) for c in sells}
        for code in buys:
            buy_dates[code] = date_str
            buy_prices[code] = cur_info[code].get("price")
        for code in sells:
            buy_dates.pop(code, None)
            buy_prices.pop(code, None)
            sell_keys.append((code, date_str))

        def _item(
            code: str,
            info: Dict[str, Any],
            action: str,
            buy_date: Optional[str],
            *,
            buy_price: Optional[float] = None,
            sell_price: Optional[float] = None,
            return_rate: Optional[float] = None,
        ) -> dict:
            return {
                "instrument": code,
                "name": name_map.get(code) or "",
                "buy_date": buy_date,
                "amount": info.get("amount"),
                "price": info.get("price"),
                "weight": info.get("weight"),
                "count_day": info.get("count_day"),
                "action": action,
                "buy_price": buy_price,
                "sell_price": sell_price,
                "return_rate": return_rate,
            }

        day_hold = [
            _item(c, cur_info[c], "hold" if c in holds else "buy", buy_dates.get(c))
            for c in sorted(cur_codes)
        ]
        day_buy = [_item(c, cur_info[c], "buy", buy_dates.get(c)) for c in buys]
        day_sell = [
            _item(
                c,
                prev_info[c],
                "sell",
                sell_buy_dates.get(c),
                buy_price=sell_buy_prices.get(c),
            )
            for c in sells
        ]

        def _row(item: dict, action: Optional[str] = None) -> dict:
            return {
                "datetime": date_str,
                "instrument": item.get("instrument"),
                "name": item.get("name"),
                "buy_date": item.get("buy_date"),
                "amount": item.get("amount"),
                "price": item.get("price"),
                "weight": item.get("weight"),
                "count_day": item.get("count_day"),
                "cash": cash,
                "account_value": account_value,
                "action": action if action is not None else item.get("action"),
                "buy_price": item.get("buy_price"),
                "sell_price": item.get("sell_price"),
                "return_rate": item.get("return_rate"),
            }

        for item in day_hold:
            rows.append(_row(item))
        for item in day_sell:
            rows.append(_row(item, action="sell"))

        if not cur_codes and not sells:
            rows.append(
                {
                    "datetime": date_str,
                    "instrument": None,
                    "name": None,
                    "buy_date": None,
                    "amount": None,
                    "price": None,
                    "weight": None,
                    "count_day": None,
                    "cash": cash,
                    "account_value": account_value,
                    "action": None,
                    "buy_price": None,
                    "sell_price": None,
                    "return_rate": None,
                }
            )

        days_payload.append(
            {
                "date": date_str,
                "cash": cash,
                "account_value": account_value,
                "n_hold": len(cur_codes),
                "n_buy": len(buys),
                "n_sell": len(sells),
                "hold": day_hold,
                "buy": day_buy,
                "sell": day_sell,
            }
        )
        prev_codes = cur_codes
        prev_info = cur_info

    close_map = _fetch_sell_closes(sell_keys)
    for day in days_payload:
        for item in day["sell"]:
            code = item["instrument"]
            sell_price = close_map.get((code, day["date"]))
            buy_price = item.get("buy_price")
            item["sell_price"] = sell_price
            if buy_price and sell_price and buy_price != 0:
                item["return_rate"] = float(sell_price) / float(buy_price) - 1.0
            else:
                item["return_rate"] = None

    for row in rows:
        if row.get("action") != "sell":
            continue
        code = row.get("instrument")
        date_str = row.get("datetime")
        if not code or not date_str:
            continue
        sell_price = close_map.get((code, date_str))
        buy_price = row.get("buy_price")
        row["sell_price"] = sell_price
        if buy_price and sell_price and buy_price != 0:
            row["return_rate"] = float(sell_price) / float(buy_price) - 1.0
        else:
            row["return_rate"] = None

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["datetime", "action", "instrument"], kind="mergesort").reset_index(drop=True)
    return df, days_payload


def _series_ic_stats(ic: pd.Series) -> Tuple[Optional[float], Optional[float]]:
    ic = ic.dropna()
    if ic.empty:
        return None, None
    mean = float(ic.mean())
    std = float(ic.std())
    icir = float(mean / std) if std and not np.isnan(std) else None
    return mean, icir


def _port_block(analysis_df: pd.DataFrame, key: str) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    try:
        block = analysis_df.loc[key]
        for metric in ("mean", "std", "annualized_return", "information_ratio", "max_drawdown"):
            try:
                val = block.loc[metric]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                out[metric] = float(val) if pd.notna(val) else None
            except (KeyError, IndexError, TypeError, ValueError):
                out[metric] = None
    except KeyError:
        pass
    return out


def collect_metrics(recorder) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {
        "IC": None,
        "ICIR": None,
        "Rank IC": None,
        "Rank ICIR": None,
        "port": {
            "benchmark": {},
            "excess_without_cost": {},
            "excess_with_cost": {},
        },
        "indicators": {},
    }
    try:
        ic = recorder.load_object("sig_analysis/ic.pkl")
        ric = recorder.load_object("sig_analysis/ric.pkl")
        metrics["IC"], metrics["ICIR"] = _series_ic_stats(ic)
        metrics["Rank IC"], metrics["Rank ICIR"] = _series_ic_stats(ric)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"sig_analysis metrics unavailable: {exc}")

    try:
        analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
        metrics["port"]["benchmark"] = _port_block(analysis_df, "benchmark")
        # qlib may use these index labels:
        for src, dst in [
            ("excess_return_without_cost", "excess_without_cost"),
            ("excess_return_with_cost", "excess_with_cost"),
        ]:
            metrics["port"][dst] = _port_block(analysis_df, src)
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"port_analysis metrics unavailable: {exc}")

    try:
        ind = recorder.load_object("portfolio_analysis/indicator_analysis_1day.pkl")
        if isinstance(ind, pd.DataFrame):
            metrics["indicators"] = {
                str(k): (float(v) if pd.notna(v) else None)
                for k, v in ind["value"].items()
            }
    except Exception as exc:  # noqa: BLE001
        warnings.warn(f"indicator_analysis unavailable: {exc}")

    return metrics


def render_html(payload: Dict[str, Any]) -> str:
    """Build HTML shell; experiment JSON is appended at the very bottom."""
    exp_name = payload.get("exp_name", "")
    # Pretty-print for readability at file bottom; still valid JSON.
    data_json = json.dumps(payload, ensure_ascii=False, indent=2)

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Positions · {exp_name}</title>
<style>
  :root {{
    --bg: #0f1419;
    --panel: #1a222c;
    --text: #e7ecf1;
    --muted: #8b98a5;
    --border: #2c3640;
    --buy: #3dd68c;
    --sell: #f07178;
    --hold: #59c2ff;
    --accent: #e6b450;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.45;
  }}
  header {{
    padding: 1.25rem 1.5rem 0.75rem;
    border-bottom: 1px solid var(--border);
    background: linear-gradient(180deg, #18202a 0%, var(--bg) 100%);
  }}
  h1 {{
    margin: 0 0 0.35rem;
    font-size: 1.35rem;
    font-weight: 600;
    letter-spacing: 0.02em;
  }}
  .meta {{ color: var(--muted); font-size: 0.9rem; }}
  .meta strong {{ color: var(--accent); font-weight: 560; }}
  .metrics {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(110px, 1fr));
    gap: 0.6rem;
    padding: 0.85rem 1.5rem;
    border-bottom: 1px solid var(--border);
  }}
  .metric {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.55rem 0.7rem;
  }}
  .metric .label {{ color: var(--muted); font-size: 0.75rem; }}
  .metric .value {{ font-size: 1.05rem; font-weight: 600; margin-top: 0.15rem; font-variant-numeric: tabular-nums; }}
  details.more-metrics {{
    margin: 0 1.5rem 0.75rem;
    color: var(--muted);
    font-size: 0.88rem;
  }}
  details.more-metrics summary {{ cursor: pointer; color: var(--accent); }}
  details.more-metrics pre {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    overflow: auto;
    color: var(--text);
  }}
  .controls {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    align-items: center;
    padding: 1rem 1.5rem;
    border-bottom: 1px solid var(--border);
    position: sticky;
    top: 0;
    background: rgba(15, 20, 25, 0.94);
    backdrop-filter: blur(6px);
    z-index: 2;
  }}
  button, select, input[type="date"] {{
    background: var(--panel);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 0.45rem 0.75rem;
    font: inherit;
  }}
  button {{ cursor: pointer; }}
  button:hover {{ border-color: var(--accent); }}
  button:disabled {{ opacity: 0.4; cursor: not-allowed; }}
  .stats {{
    display: flex;
    gap: 1rem;
    flex-wrap: wrap;
    color: var(--muted);
    font-size: 0.9rem;
    margin-left: auto;
  }}
  .stats span b {{ color: var(--text); font-weight: 560; }}
  main {{
    padding: 1rem 1.5rem 2rem;
    display: grid;
    gap: 1rem;
    grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  }}
  section {{
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }}
  section h2 {{
    margin: 0;
    padding: 0.75rem 1rem;
    font-size: 0.95rem;
    border-bottom: 1px solid var(--border);
    display: flex;
    justify-content: space-between;
    align-items: baseline;
  }}
  section h2 .tag {{ font-size: 0.8rem; color: var(--muted); font-weight: 500; }}
  .buy h2 {{ color: var(--buy); }}
  .sell h2 {{ color: var(--sell); }}
  .hold h2 {{ color: var(--hold); }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-variant-numeric: tabular-nums;
    font-size: 0.85rem;
  }}
  th, td {{
    padding: 0.4rem 0.65rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{ color: var(--muted); font-weight: 500; }}
  tr:last-child td {{ border-bottom: none; }}
  .empty {{
    padding: 1rem;
    color: var(--muted);
    font-size: 0.9rem;
  }}
  .pill {{
    display: inline-block;
    padding: 0.1rem 0.4rem;
    border-radius: 4px;
    font-size: 0.75rem;
    margin-left: 0.35rem;
  }}
  .pill-buy {{ background: rgba(61,214,140,0.15); color: var(--buy); }}
  .pill-hold {{ background: rgba(89,194,255,0.12); color: var(--hold); }}
  .ret-pos {{ color: var(--sell); }}
  .ret-neg {{ color: var(--buy); }}
</style>
</head>
<body>
<header>
  <h1>持仓日度查看</h1>
  <div class="meta">
    实验 <strong id="expName"></strong>
    · recorder <code id="recId" style="color:var(--muted)"></code>
    · <span id="range"></span>
    · 账户 <span id="avSummary"></span>
  </div>
</header>
<div class="metrics" id="metricsRow"></div>
<details class="more-metrics">
  <summary>更多组合 / 成交指标</summary>
  <pre id="moreMetrics"></pre>
</details>
<div class="controls">
  <button id="prevBtn" type="button">← 前一日</button>
  <input id="dateInput" type="date"/>
  <select id="dateSelect"></select>
  <button id="nextBtn" type="button">后一日 →</button>
  <div class="stats">
    <span>现金 <b id="cash"></b></span>
    <span>总资产 <b id="account"></b></span>
    <span>持仓 <b id="nHold"></b></span>
    <span>买入 <b id="nBuy"></b></span>
    <span>卖出 <b id="nSell"></b></span>
  </div>
</div>
<main>
  <section class="hold">
    <h2>今日持仓 <span class="tag" id="holdTag"></span></h2>
    <div id="holdBody"></div>
  </section>
  <section class="buy">
    <h2>今日买入 <span class="tag" id="buyTag"></span></h2>
    <div id="buyBody"></div>
  </section>
  <section class="sell">
    <h2>今日卖出 <span class="tag" id="sellTag"></span></h2>
    <div id="sellBody"></div>
  </section>
</main>
<script>
(function () {{
  function boot() {{
    const el = document.getElementById("payload");
    if (!el) {{
      document.body.insertAdjacentHTML("afterbegin", "<p style='padding:1rem;color:#f07178'>缺少 #payload 数据</p>");
      return;
    }}
    const data = JSON.parse(el.textContent);
    const dates = data.days.map(d => d.date);
    let idx = 0;

    const fmt = (x, digits=4) => {{
      if (x === null || x === undefined || Number.isNaN(x)) return "—";
      if (typeof x === "number") return x.toLocaleString(undefined, {{ maximumFractionDigits: digits }});
      return String(x);
    }};
    const fmtPct = (x) => {{
      if (x === null || x === undefined || Number.isNaN(x)) return "—";
      return (x * 100).toFixed(2) + "%";
    }};
    const fmtNum = (x, digits=4) => {{
      if (x === null || x === undefined || Number.isNaN(x)) return "—";
      return Number(x).toFixed(digits);
    }};

    function table(items, {{showAction=false, showBuyDate=true, showReturn=false}} = {{}}) {{
      if (!items || !items.length) return '<div class="empty">无</div>';
      const head = '<tr><th>代码</th><th>名称</th>' +
        (showBuyDate ? '<th>买入日</th>' : '') +
        '<th>权重</th><th>数量</th><th>价格</th>' +
        (showReturn ? '<th>收益率</th>' : '') +
        (showAction ? '<th>标记</th>' : '') + '</tr>';
      const body = items.map(it => {{
        const pill = showAction
          ? (it.action === "buy"
              ? '<span class="pill pill-buy">buy</span>'
              : '<span class="pill pill-hold">hold</span>')
          : "";
        let retCell = "";
        if (showReturn) {{
          const r = it.return_rate;
          if (r === null || r === undefined || Number.isNaN(r)) {{
            retCell = '<td>—</td>';
          }} else {{
            const cls = r >= 0 ? "ret-pos" : "ret-neg";
            retCell = '<td class="' + cls + '">' + fmtPct(r) + '</td>';
          }}
        }}
        return '<tr>' +
          '<td>' + it.instrument + (showAction ? pill : '') + '</td>' +
          '<td>' + (it.name || "—") + '</td>' +
          (showBuyDate ? '<td>' + (it.buy_date || "—") + '</td>' : '') +
          '<td>' + fmtPct(it.weight) + '</td>' +
          '<td>' + fmt(it.amount, 2) + '</td>' +
          '<td>' + fmt(showReturn && it.sell_price != null ? it.sell_price : it.price, 4) + '</td>' +
          retCell +
          (showAction ? '<td>' + (it.action || '') + '</td>' : '') +
          '</tr>';
      }}).join("");
      return '<table><thead>' + head + '</thead><tbody>' + body + '</tbody></table>';
    }}

    function render() {{
      const day = data.days[idx];
      document.getElementById("dateInput").value = day.date;
      document.getElementById("dateSelect").value = day.date;
      document.getElementById("cash").textContent = fmt(day.cash, 2);
      document.getElementById("account").textContent = fmt(day.account_value, 2);
      document.getElementById("nHold").textContent = day.n_hold;
      document.getElementById("nBuy").textContent = day.n_buy;
      document.getElementById("nSell").textContent = day.n_sell;
      document.getElementById("holdTag").textContent = day.n_hold + " 只";
      document.getElementById("buyTag").textContent = day.n_buy + " 只";
      document.getElementById("sellTag").textContent = day.n_sell + " 只";
      document.getElementById("holdBody").innerHTML = table(day.hold, {{showAction: true}});
      document.getElementById("buyBody").innerHTML = table(day.buy, {{showBuyDate: false}});
      document.getElementById("sellBody").innerHTML = table(day.sell, {{showReturn: true}});
      document.getElementById("prevBtn").disabled = idx <= 0;
      document.getElementById("nextBtn").disabled = idx >= dates.length - 1;
    }}

    function gotoDate(dateStr) {{
      const i = dates.indexOf(dateStr);
      if (i >= 0) {{ idx = i; render(); return; }}
      let best = 0;
      for (let j = 0; j < dates.length; j++) {{
        if (dates[j] <= dateStr) best = j;
        else break;
      }}
      idx = best;
      render();
    }}

    function defaultIndex() {{
      const today = new Date();
      const y = today.getFullYear();
      const m = String(today.getMonth() + 1).padStart(2, "0");
      const d = String(today.getDate()).padStart(2, "0");
      const todayStr = y + "-" + m + "-" + d;
      if (!dates.length) return 0;
      if (todayStr < dates[0]) return 0;
      if (todayStr > dates[dates.length - 1]) return dates.length - 1;
      let best = 0;
      for (let j = 0; j < dates.length; j++) {{
        if (dates[j] <= todayStr) best = j;
        else break;
      }}
      return best;
    }}

    function renderMetrics() {{
      const m = data.metrics || {{}};
      const ewc = (m.port && m.port.excess_with_cost) || {{}};
      const cards = [
        ["IC", fmtNum(m.IC, 4)],
        ["ICIR", fmtNum(m.ICIR, 4)],
        ["Rank IC", fmtNum(m["Rank IC"], 4)],
        ["Rank ICIR", fmtNum(m["Rank ICIR"], 4)],
        ["超额年化(含成本)", fmtPct(ewc.annualized_return)],
        ["IR(含成本)", fmtNum(ewc.information_ratio, 4)],
        ["最大回撤(含成本)", fmtPct(ewc.max_drawdown)],
      ];
      document.getElementById("metricsRow").innerHTML = cards.map(([label, value]) =>
        '<div class="metric"><div class="label">' + label + '</div><div class="value">' + value + '</div></div>'
      ).join("");
      document.getElementById("moreMetrics").textContent = JSON.stringify({{
        port: m.port || {{}},
        indicators: m.indicators || {{}}
      }}, null, 2);
    }}

    document.getElementById("expName").textContent = data.exp_name;
    document.getElementById("recId").textContent = data.recorder_id;
    document.getElementById("range").textContent = data.start + " → " + data.end;
    document.getElementById("avSummary").textContent =
      fmt(data.first_account_value, 2) + " → " + fmt(data.last_account_value, 2);

    const sel = document.getElementById("dateSelect");
    dates.forEach(d => {{
      const opt = document.createElement("option");
      opt.value = d; opt.textContent = d;
      sel.appendChild(opt);
    }});

    document.getElementById("prevBtn").onclick = () => {{ if (idx > 0) {{ idx--; render(); }} }};
    document.getElementById("nextBtn").onclick = () => {{ if (idx < dates.length - 1) {{ idx++; render(); }} }};
    document.getElementById("dateSelect").onchange = (e) => gotoDate(e.target.value);
    document.getElementById("dateInput").onchange = (e) => gotoDate(e.target.value);
    document.addEventListener("keydown", (e) => {{
      if (e.key === "ArrowLeft") document.getElementById("prevBtn").click();
      if (e.key === "ArrowRight") document.getElementById("nextBtn").click();
    }});

    renderMetrics();
    idx = defaultIndex();
    render();
  }}

  if (document.readyState === "loading") {{
    document.addEventListener("DOMContentLoaded", boot);
  }} else {{
    boot();
  }}
}})();
</script>
<!-- payload: experiment data at file bottom -->
<script id="payload" type="application/json">
{data_json}
</script>
</body>
</html>
"""


def export(
    exp_name: str,
    recorder_id: Optional[str] = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    provider_uri: str = "~/.qlib/qlib_data/cn_data",
    name_map_path: Path = NAME_MAP_PATH,
) -> Tuple[Path, Path]:
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    recorder = _latest_recorder(exp_name, recorder_id)
    rid = recorder.info.get("id") if hasattr(recorder, "info") else getattr(recorder, "id", None)
    if rid is None:
        rid = str(recorder)
    print(f"using experiment={exp_name!r} recorder={rid}")

    name_map = load_or_build_name_map(Path(name_map_path))
    positions = recorder.load_object(POSITIONS_ARTIFACT)
    df, days = build_day_frames(positions, name_map=name_map)
    metrics = collect_metrics(recorder)

    dates = [d["date"] for d in days]
    payload = {
        "exp_name": exp_name,
        "recorder_id": str(rid),
        "start": dates[0] if dates else "",
        "end": dates[-1] if dates else "",
        "first_account_value": days[0]["account_value"] if days else None,
        "last_account_value": days[-1]["account_value"] if days else None,
        "metrics": metrics,
        "days": days,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = exp_name.replace("/", "_")
    parquet_path = out_dir / f"{safe_name}_positions_1day.parquet"
    html_path = out_dir / f"{safe_name}_positions_view.html"

    df.to_parquet(parquet_path, index=False)
    html_path.write_text(render_html(payload), encoding="utf-8")

    print(f"wrote {parquet_path} ({len(df)} rows, {df['datetime'].nunique()} days)")
    print(f"wrote {html_path}")
    return parquet_path, html_path


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Export positions parquet + HTML day browser")
    parser.add_argument("--exp_name", default=DEFAULT_EXP_NAME, help="MLflow experiment name")
    parser.add_argument("--recorder_id", default=None, help="Optional recorder id; default=latest FINISHED")
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--name_map",
        type=Path,
        default=NAME_MAP_PATH,
        help=f"Stock name map JSON (default: {NAME_MAP_PATH})",
    )
    parser.add_argument(
        "--provider_uri",
        default="~/.qlib/qlib_data/cn_data",
        help="qlib data provider uri",
    )
    args = parser.parse_args(argv)
    export(
        exp_name=args.exp_name,
        recorder_id=args.recorder_id,
        out_dir=args.out_dir,
        provider_uri=args.provider_uri,
        name_map_path=args.name_map,
    )


if __name__ == "__main__":
    main()
