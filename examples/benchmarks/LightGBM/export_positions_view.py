#!/usr/bin/env python3
"""Export backtest positions to parquet + a self-contained HTML day browser.

Reads ``portfolio_analysis/positions_normal_1day.pkl`` from an MLflow experiment
recorder, labels each stock as hold / buy / sell vs the previous trading day,
writes a long-form parquet (datetime as YYYY-MM-DD), and an HTML report with
date picker showing holdings / buys / sells.

Example (from qlib repo root)::

    .venv/bin/python examples/benchmarks/LightGBM/export_positions_view.py \\
      --exp_name=rolling_lgb_7y2y_20261009
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R

DIRNAME = Path(__file__).absolute().resolve().parent
DEFAULT_OUT_DIR = DIRNAME / "data"
DEFAULT_EXP_NAME = "rolling_lgb_7y2y_20261009"
POSITIONS_ARTIFACT = "portfolio_analysis/positions_normal_1day.pkl"


def _latest_recorder(exp_name: str, recorder_id: Optional[str] = None):
    if recorder_id:
        return R.get_recorder(recorder_id=recorder_id, experiment_name=exp_name)
    exp = R.get_exp(experiment_name=exp_name, create=False)
    recorders = exp.list_recorders(rtype="list")
    if not recorders:
        raise RuntimeError(f"No recorders found for experiment {exp_name!r}")
    # Prefer finished recorders with the newest start time.
    finished = [r for r in recorders if getattr(r, "status", None) == "FINISHED"]
    candidates = finished or list(recorders)
    candidates.sort(key=lambda r: getattr(r, "start_time", None) or "", reverse=True)
    return candidates[0]


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


def build_day_frames(positions: Dict) -> Tuple[pd.DataFrame, List[dict]]:
    """Return long parquet rows + per-day HTML payload."""
    dates = sorted(positions.keys())
    rows: List[dict] = []
    days_payload: List[dict] = []
    prev_codes: set = set()
    prev_info: Dict[str, Dict[str, Any]] = {}

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

        def _item(code: str, info: Dict[str, Any], action: str) -> dict:
            return {
                "instrument": code,
                "amount": info.get("amount"),
                "price": info.get("price"),
                "weight": info.get("weight"),
                "count_day": info.get("count_day"),
                "action": action,
            }

        day_hold = [_item(c, cur_info[c], "hold" if c in holds else "buy") for c in sorted(cur_codes)]
        day_buy = [_item(c, cur_info[c], "buy") for c in buys]
        day_sell = [_item(c, prev_info[c], "sell") for c in sells]

        for item in day_hold:
            rows.append(
                {
                    "datetime": date_str,
                    "instrument": item["instrument"],
                    "amount": item["amount"],
                    "price": item["price"],
                    "weight": item["weight"],
                    "count_day": item["count_day"],
                    "cash": cash,
                    "account_value": account_value,
                    "action": item["action"],
                }
            )
        for item in day_sell:
            rows.append(
                {
                    "datetime": date_str,
                    "instrument": item["instrument"],
                    "amount": item["amount"],
                    "price": item["price"],
                    "weight": item["weight"],
                    "count_day": item["count_day"],
                    "cash": cash,
                    "account_value": account_value,
                    "action": "sell",
                }
            )

        if not cur_codes and not sells:
            rows.append(
                {
                    "datetime": date_str,
                    "instrument": None,
                    "amount": None,
                    "price": None,
                    "weight": None,
                    "count_day": None,
                    "cash": cash,
                    "account_value": account_value,
                    "action": None,
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

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["datetime", "action", "instrument"], kind="mergesort").reset_index(drop=True)
    return df, days_payload


def render_html(exp_name: str, recorder_id: str, days: List[dict]) -> str:
    dates = [d["date"] for d in days]
    start = dates[0] if dates else ""
    end = dates[-1] if dates else ""
    first_av = days[0]["account_value"] if days else None
    last_av = days[-1]["account_value"] if days else None
    payload = {
        "exp_name": exp_name,
        "recorder_id": recorder_id,
        "start": start,
        "end": end,
        "first_account_value": first_av,
        "last_account_value": last_av,
        "days": days,
    }
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

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
    padding: 1.25rem 1.5rem 1rem;
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
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
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
    font-size: 0.88rem;
  }}
  th, td {{
    padding: 0.45rem 0.75rem;
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
<script id="payload" type="application/json">{data_json}</script>
<script>
(function () {{
  const data = JSON.parse(document.getElementById("payload").textContent);
  const byDate = Object.fromEntries(data.days.map(d => [d.date, d]));
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

  function table(items, {{showAction=false}} = {{}}) {{
    if (!items || !items.length) return '<div class="empty">无</div>';
    const head = '<tr><th>代码</th><th>权重</th><th>数量</th><th>价格</th>' +
      (showAction ? '<th>标记</th>' : '') + '</tr>';
    const body = items.map(it => {{
      const pill = showAction
        ? (it.action === "buy"
            ? '<span class="pill pill-buy">buy</span>'
            : '<span class="pill pill-hold">hold</span>')
        : "";
      return '<tr>' +
        '<td>' + it.instrument + (showAction ? pill : '') + '</td>' +
        '<td>' + fmtPct(it.weight) + '</td>' +
        '<td>' + fmt(it.amount, 2) + '</td>' +
        '<td>' + fmt(it.price, 4) + '</td>' +
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
    document.getElementById("buyBody").innerHTML = table(day.buy);
    document.getElementById("sellBody").innerHTML = table(day.sell);
    document.getElementById("prevBtn").disabled = idx <= 0;
    document.getElementById("nextBtn").disabled = idx >= dates.length - 1;
  }}

  function gotoDate(dateStr) {{
    const i = dates.indexOf(dateStr);
    if (i >= 0) {{ idx = i; render(); }}
    else if (byDate[dateStr]) {{ /* unreachable */ }}
    else {{
      // Snap to nearest available trading day on or before selection.
      let best = 0;
      for (let i = 0; i < dates.length; i++) {{
        if (dates[i] <= dateStr) best = i;
        else break;
      }}
      idx = best;
      render();
    }}
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

  render();
}})();
</script>
</body>
</html>
"""


def export(
    exp_name: str,
    recorder_id: Optional[str] = None,
    out_dir: Path = DEFAULT_OUT_DIR,
    provider_uri: str = "~/.qlib/qlib_data/cn_data",
) -> Tuple[Path, Path]:
    qlib.init(provider_uri=provider_uri, region=REG_CN)
    recorder = _latest_recorder(exp_name, recorder_id)
    rid = recorder.info.get("id") if hasattr(recorder, "info") else getattr(recorder, "id", None)
    if rid is None:
        rid = str(recorder)
    print(f"using experiment={exp_name!r} recorder={rid}")

    positions = recorder.load_object(POSITIONS_ARTIFACT)
    df, days = build_day_frames(positions)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = exp_name.replace("/", "_")
    parquet_path = out_dir / f"{safe_name}_positions_1day.parquet"
    html_path = out_dir / f"{safe_name}_positions_view.html"

    df.to_parquet(parquet_path, index=False)
    html_path.write_text(render_html(exp_name, str(rid), days), encoding="utf-8")

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
    )


if __name__ == "__main__":
    main()
