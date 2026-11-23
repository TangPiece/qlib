# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""alloc_v2 本地研究站：只读 parquet，按需 JSON，不把整表塞进浏览器。"""

from __future__ import annotations

import json
import logging
import mimetypes
from datetime import date, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
import pandas as pd

from .adapters.industry import SW_L1
from .catalog import indicator_catalog_frame
from .labels import CHART_CAPTIONS, ETF_NAMES, display_name
from .mappings import load_etf_master
from .paths import default_root, qlib_root, research_dir
from .registry import catalog_rows, core_datasets
from .store import read_factor, read_research, read_standard

_log = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8766
DEFAULT_LAST_N = 400
MAX_LAST_N = 8000
DEFAULT_ROW_LIMIT = 50
MAX_ROW_LIMIT = 200

WEB_DIR = qlib_root() / "examples" / "alloc_v2" / "web"


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, date, pd.Timestamp)):
        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        if ts.hour == 0 and ts.minute == 0 and ts.second == 0 and ts.microsecond == 0:
            return ts.date().isoformat()
        return ts.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return None if np.isnan(number) else number
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value))


def _safe(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, (datetime, date, pd.Timestamp, np.integer, np.floating, np.bool_)):
        return _json_default(value)
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _frame_records(frame: pd.DataFrame, *, columns: list[str] | None = None) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    work = frame if columns is None else frame[[c for c in columns if c in frame.columns]]
    return [{str(k): _safe(v) for k, v in row.items()} for row in work.to_dict("records")]


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


class LabStore:
    """按文件 mtime 缓存 parquet，避免每个请求重读十万行。"""

    def __init__(self, root: Path):
        self.root = Path(root)
        self._cache: dict[str, tuple[float, pd.DataFrame]] = {}

    def _load(self, key: str, loader) -> pd.DataFrame:
        path = {
            "standard": self.root / "standard" / "observations.parquet",
            "factor": self.root / "factor" / "all.parquet",
        }.get(key)
        stamp = _mtime(path) if path is not None else 0.0
        hit = self._cache.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
        frame = loader()
        self._cache[key] = (stamp, frame)
        return frame

    def standard(self) -> pd.DataFrame:
        return self._load("standard", lambda: read_standard(root=self.root))

    def factor(self) -> pd.DataFrame:
        return self._load("factor", lambda: read_factor("all", root=self.root))

    def research(self, name: str) -> pd.DataFrame:
        path = research_dir(self.root) / f"{name}.parquet"
        key = f"research:{name}"
        stamp = _mtime(path)
        hit = self._cache.get(key)
        if hit and hit[0] == stamp:
            return hit[1]
        frame = read_research(name, root=self.root)
        self._cache[key] = (stamp, frame)
        return frame

    def research_tables(self) -> list[str]:
        folder = research_dir(self.root)
        if not folder.is_dir():
            return []
        return sorted(p.stem for p in folder.glob("*.parquet"))


def _as_day_str(value: Any) -> str | None:
    parsed = _safe(value)
    if parsed is None:
        return None
    text = str(parsed)
    return text[:10] if text else None


def _tail_points(frame: pd.DataFrame, *, last_n: int) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    work = frame.copy()
    work["event_date"] = pd.to_datetime(work["event_date"], errors="coerce")
    work = work.dropna(subset=["event_date", "value"]).sort_values("event_date")
    if last_n > 0:
        work = work.tail(int(last_n))
    return [
        {"t": _as_day_str(ts), "v": _safe(val)}
        for ts, val in zip(work["event_date"], work["value"])
        if _as_day_str(ts) is not None
    ]


def _parse_int(raw: str | None, default: int, *, lo: int, hi: int) -> int:
    try:
        number = int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        number = default
    return max(lo, min(hi, number))


def meta_payload(store: LabStore) -> dict[str, Any]:
    standard = store.standard()
    factor = store.factor()
    s1 = store.research("signal_s1")
    s4 = store.research("signal_s4")
    last_day = None
    if not standard.empty and "event_date" in standard.columns:
        last_day = _as_day_str(pd.to_datetime(standard["event_date"], errors="coerce").max())
    s1_value = None
    s1_components: dict[str, Any] = {}
    if not s1.empty:
        row = s1.iloc[-1].to_dict()
        s1_value = _safe(row.get("value"))
        comps = row.get("components") or {}
        if isinstance(comps, str):
            try:
                comps = json.loads(comps.replace("'", '"'))
            except Exception:  # noqa: BLE001
                comps = {}
        if isinstance(comps, dict):
            s1_components = {str(k): bool(v) for k, v in comps.items()}
    s4_pass = []
    if not s4.empty and "value" in s4.columns:
        ok = s4[pd.to_numeric(s4["value"], errors="coerce").fillna(0) > 0]
        s4_pass = [str(x) for x in ok.get("entity_id", pd.Series(dtype=str)).astype(str).tolist()]
    registry = catalog_rows(include_disabled=True)
    enabled = [r for r in registry if r.get("enabled")]
    present = set(standard["dataset"].astype(str)) if not standard.empty else set()
    return {
        "last_event_date": last_day,
        "n_standard": int(len(standard)),
        "n_factor": int(len(factor)),
        "n_datasets_enabled": len(enabled),
        "n_datasets_present": len(present),
        "n_core": len(core_datasets()),
        "s1": s1_value,
        "s1_components": s1_components,
        "s4": s4_pass[:20],
        "s4_n": len(s4_pass),
    }


def registry_payload() -> list[dict[str, Any]]:
    frame = indicator_catalog_frame()
    return _frame_records(frame)


def index_payload(store: LabStore, *, layer: str, query: str, offset: int, limit: int) -> dict[str, Any]:
    layer = (layer or "standard").strip().lower()
    query = (query or "").strip().lower()
    rows: list[dict[str, Any]] = []
    if layer == "standard":
        frame = store.standard()
        if not frame.empty:
            grouped = (
                frame.groupby(["dataset", "entity_id", "field"], dropna=False)
                .agg(n_rows=("value", "size"), start=("event_date", "min"), end=("event_date", "max"))
                .reset_index()
            )
            for rec in grouped.to_dict("records"):
                rows.append(
                    {
                        "layer": "standard",
                        "dataset": str(rec.get("dataset", "")),
                        "entity_id": str(rec.get("entity_id", "")),
                        "field": str(rec.get("field", "")),
                        "n_rows": int(rec.get("n_rows") or 0),
                        "start": _as_day_str(rec.get("start")),
                        "end": _as_day_str(rec.get("end")),
                        "label": display_name(str(rec.get("entity_id", ""))),
                    }
                )
    elif layer == "factor":
        frame = store.factor()
        if not frame.empty:
            fid = "factor_id" if "factor_id" in frame.columns else "field"
            grouped = (
                frame.groupby([fid, "entity_id"], dropna=False)
                .agg(n_rows=("value", "size"), start=("event_date", "min"), end=("event_date", "max"))
                .reset_index()
            )
            for rec in grouped.to_dict("records"):
                factor_id = str(rec.get(fid, ""))
                rows.append(
                    {
                        "layer": "factor",
                        "dataset": "factor",
                        "factor_id": factor_id,
                        "entity_id": str(rec.get("entity_id", "")),
                        "field": factor_id,
                        "n_rows": int(rec.get("n_rows") or 0),
                        "start": _as_day_str(rec.get("start")),
                        "end": _as_day_str(rec.get("end")),
                        "label": f"{display_name(factor_id)} / {display_name(str(rec.get('entity_id', '')))}",
                    }
                )
    else:
        for name in store.research_tables():
            frame = store.research(name)
            if frame.empty:
                continue
            entity_col = "entity_id" if "entity_id" in frame.columns else None
            key_col = next((c for c in ("signal_id", "factor_id", "rule_id") if c in frame.columns), None)
            if entity_col and key_col:
                grouped = frame.groupby([entity_col, key_col], dropna=False).size().reset_index(name="n_rows")
                for rec in grouped.to_dict("records"):
                    rows.append(
                        {
                            "layer": "research",
                            "dataset": name,
                            "entity_id": str(rec.get(entity_col, "")),
                            "field": str(rec.get(key_col, "")),
                            "n_rows": int(rec.get("n_rows") or 0),
                            "label": f"{name} / {display_name(str(rec.get(entity_col, '')))}",
                        }
                    )
            else:
                rows.append(
                    {
                        "layer": "research",
                        "dataset": name,
                        "entity_id": "",
                        "field": "",
                        "n_rows": int(len(frame)),
                        "label": name,
                    }
                )
    if query:
        rows = [r for r in rows if query in json.dumps(r, ensure_ascii=False).lower()]
    total = len(rows)
    page = rows[offset : offset + limit]
    return {"layer": layer, "total": total, "offset": offset, "limit": limit, "rows": page}


def _research_time_col(frame: pd.DataFrame) -> str | None:
    """研究层时间轴优先 asof，其次 event_date。"""
    for col in ("asof", "event_date"):
        if col in frame.columns:
            return col
    return None


def series_payload(
    store: LabStore,
    *,
    dataset: str,
    entity_id: str,
    field: str,
    factor_id: str,
    last_n: int,
    layer: str = "",
) -> dict[str, Any]:
    """单序列下采样；缺数据返回空点 + note，不抛 500。"""
    note = ""
    points: list[dict[str, Any]] = []
    layer = (layer or "").strip().lower()
    research_names = set(store.research_tables())
    use_research = layer == "research" or (dataset in research_names and not factor_id)
    if factor_id or layer == "factor":
        fid = factor_id or field
        frame = store.factor()
        if frame.empty or "factor_id" not in frame.columns:
            note = "无数据"
        else:
            part = frame[frame["factor_id"].astype(str) == fid]
            if entity_id:
                part = part[part["entity_id"].astype(str) == entity_id]
            points = _tail_points(part, last_n=last_n)
            if not points:
                note = "无数据"
        return {
            "layer": "factor",
            "dataset": "factor",
            "factor_id": fid,
            "entity_id": entity_id,
            "field": fid,
            "label": f"{display_name(fid)} / {display_name(entity_id)}",
            "points": points,
            "note": note,
        }
    if use_research:
        name = dataset or "signal_s1"
        frame = store.research(name)
        if frame.empty:
            note = "无数据"
        else:
            part = frame.copy()
            if entity_id and "entity_id" in part.columns:
                part = part[part["entity_id"].astype(str) == entity_id]
            if field:
                for col in ("signal_id", "factor_id", "rule_id"):
                    if col in part.columns:
                        part = part[part[col].astype(str) == field]
                        break
            time_col = _research_time_col(part)
            if time_col is None or "value" not in part.columns:
                note = "无数据"
            else:
                work = part.rename(columns={time_col: "event_date"})
                points = _tail_points(work, last_n=last_n)
                if not points:
                    note = "无数据"
        return {
            "layer": "research",
            "dataset": dataset,
            "entity_id": entity_id,
            "field": field,
            "label": f"{dataset} / {display_name(entity_id or field)}",
            "points": points,
            "note": note,
        }
    frame = store.standard()
    if frame.empty:
        note = "无数据"
    else:
        part = frame.copy()
        if dataset:
            part = part[part["dataset"].astype(str) == dataset]
        if entity_id:
            part = part[part["entity_id"].astype(str) == entity_id]
        if field:
            part = part[part["field"].astype(str) == field]
        points = _tail_points(part, last_n=last_n)
        if not points:
            note = "无数据"
    return {
        "layer": "standard",
        "dataset": dataset,
        "entity_id": entity_id,
        "field": field,
        "label": display_name(entity_id or field),
        "points": points,
        "note": note,
    }


def rows_payload(
    store: LabStore,
    *,
    layer: str,
    dataset: str,
    entity_id: str,
    field: str,
    factor_id: str,
    offset: int,
    limit: int,
) -> dict[str, Any]:
    layer = (layer or "standard").strip().lower()
    if layer == "factor":
        frame = store.factor()
        if factor_id and not frame.empty and "factor_id" in frame.columns:
            frame = frame[frame["factor_id"].astype(str) == factor_id]
        if entity_id and not frame.empty:
            frame = frame[frame["entity_id"].astype(str) == entity_id]
    elif layer == "research":
        name = dataset or "signal_s1"
        frame = store.research(name)
        if entity_id and not frame.empty and "entity_id" in frame.columns:
            frame = frame[frame["entity_id"].astype(str) == entity_id]
        if field and not frame.empty:
            for col in ("signal_id", "factor_id", "rule_id"):
                if col in frame.columns:
                    frame = frame[frame[col].astype(str) == field]
                    break
    else:
        frame = store.standard()
        if dataset and not frame.empty:
            frame = frame[frame["dataset"].astype(str) == dataset]
        if entity_id and not frame.empty:
            frame = frame[frame["entity_id"].astype(str) == entity_id]
        if field and not frame.empty:
            frame = frame[frame["field"].astype(str) == field]
    # 最近行优先，目录页「原始行」按时间倒序分页
    if frame is not None and not frame.empty:
        time_col = next((c for c in ("event_date", "asof") if c in frame.columns), None)
        if time_col is not None:
            frame = frame.copy()
            frame["_t"] = pd.to_datetime(frame[time_col], errors="coerce")
            frame = frame.sort_values("_t", ascending=False).drop(columns=["_t"])
    total = int(len(frame)) if frame is not None and not frame.empty else 0
    page = frame.iloc[offset : offset + limit] if total else pd.DataFrame()
    return {
        "layer": layer,
        "total": total,
        "offset": offset,
        "limit": limit,
        "rows": _frame_records(page),
    }


def industry_latest_payload(store: LabStore, *, factor_id: str, top_n: int) -> dict[str, Any]:
    from .plotting import latest_cross_section

    snap = latest_cross_section(store.factor(), factor_id, top_n=top_n)
    items = [
        {"entity_id": str(k), "name": display_name(str(k)), "value": _safe(v)}
        for k, v in snap.items()
    ]
    return {"factor_id": factor_id, "items": items, "note": "" if items else "无数据"}


def industry_heatmap_payload(store: LabStore, *, factor_id: str, days: int) -> dict[str, Any]:
    frame = store.factor()
    if frame.empty or "factor_id" not in frame.columns:
        return {"factor_id": factor_id, "dates": [], "entities": [], "values": [], "note": "无数据"}
    part = frame[frame["factor_id"].astype(str) == factor_id].copy()
    if part.empty:
        return {"factor_id": factor_id, "dates": [], "entities": [], "values": [], "note": "无数据"}
    part["event_date"] = pd.to_datetime(part["event_date"], errors="coerce")
    last = part["event_date"].max()
    if pd.isna(last):
        return {"factor_id": factor_id, "dates": [], "entities": [], "values": [], "note": "无数据"}
    cut = last - pd.Timedelta(days=int(days) + 10)
    part = part[part["event_date"] >= cut]
    pivot = part.pivot_table(index="event_date", columns="entity_id", values="value", aggfunc="last")
    pivot = pivot.tail(int(days))
    # 实体过多时只保留最新截面绝对值 Top 12
    if pivot.shape[1] > 12:
        latest = pivot.iloc[-1].abs().sort_values(ascending=False).head(12).index
        pivot = pivot[latest]
    dates = [_as_day_str(x) for x in pivot.index]
    entities = [{"id": str(c), "name": display_name(str(c))} for c in pivot.columns]
    # 前端按实体为行、日期为列；pivot 是日期×实体，这里转置
    values = [[_safe(v) for v in pivot[col].to_numpy()] for col in pivot.columns]
    return {"factor_id": factor_id, "dates": dates, "entities": entities, "values": values, "note": ""}


def etf_list_payload(store: LabStore) -> dict[str, Any]:
    master = load_etf_master(root=store.root)
    items = []
    for rec in master.to_dict("records"):
        code = str(rec.get("etf_code", ""))
        items.append(
            {
                "etf_code": code,
                "name": rec.get("name") or ETF_NAMES.get(code, code),
                "theme": rec.get("theme"),
                "primary_industry": rec.get("primary_industry"),
                "industry_name": SW_L1.get(str(rec.get("primary_industry") or ""), ""),
                "is_cross_border": bool(rec.get("is_cross_border")),
                "feeder_target": rec.get("feeder_target") or "",
            }
        )
    return {"items": items}


def etf_detail_payload(store: LabStore, *, code: str, last_n: int) -> dict[str, Any]:
    series = {}
    for factor_id in ("RawShares", "Raw_Share_Change", "Estimated_Allocation", "NAV", "Close", "Amount", "ArbitrageRisk"):
        payload = series_payload(store, dataset="factor", entity_id=code, field=factor_id, factor_id=factor_id, last_n=last_n)
        series[factor_id] = payload
    return {"etf_code": code, "name": display_name(code), "series": series}


def signals_payload(store: LabStore) -> dict[str, Any]:
    s1 = store.research("signal_s1")
    s2 = store.research("signal_s2")
    s3 = store.research("signal_s3")
    s4 = store.research("signal_s4")
    s1_row = s1.iloc[-1].to_dict() if not s1.empty else {}
    comps = s1_row.get("components") or {}
    if isinstance(comps, str):
        try:
            comps = json.loads(comps.replace("'", '"'))
        except Exception:  # noqa: BLE001
            comps = {}
    def _latest_table(frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame is None or frame.empty:
            return []
        work = frame.copy()
        if "asof" in work.columns:
            work["asof"] = pd.to_datetime(work["asof"], errors="coerce")
            last = work["asof"].max()
            work = work[work["asof"] == last]
        return _frame_records(work)

    return {
        "s1": {
            "value": _safe(s1_row.get("value")),
            "components": {str(k): bool(v) for k, v in dict(comps).items()} if isinstance(comps, dict) else {},
            "notes": s1_row.get("notes") or "",
            "asof": _safe(s1_row.get("asof")),
        },
        "s2": _latest_table(s2),
        "s3": _latest_table(s3),
        "s4": _latest_table(s4),
    }


def backtest_payload(store: LabStore) -> dict[str, Any]:
    summary = store.research("backtest_factors")
    by_year = store.research("backtest_ic_by_year")
    corr = store.research("backtest_factor_corr")
    if not summary.empty and "factor_id" in summary.columns:
        summary = summary.copy()
        summary["label"] = summary["factor_id"].astype(str).map(display_name)
    return {
        "summary": _frame_records(summary),
        "by_year": _frame_records(by_year),
        "corr": _frame_records(corr),
    }


def captions_payload() -> dict[str, str]:
    return dict(CHART_CAPTIONS)


def make_handler(store: LabStore, web_dir: Path):
    """构造绑定了数据根的 HTTP handler 类。"""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            _log.info("%s - %s", self.address_string(), fmt % args)

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = unquote(parsed.path)
            qs = {k: (v[-1] if v else "") for k, v in parse_qs(parsed.query).items()}
            try:
                if path.startswith("/api/"):
                    payload = self._api(path, qs)
                    self._send_json(200, payload)
                    return
                self._send_static(path)
            except Exception as exc:  # noqa: BLE001
                _log.exception("handler failed")
                self._send_json(500, {"error": str(exc)})

        def _api(self, path: str, qs: dict[str, str]) -> dict[str, Any]:
            last_n = _parse_int(qs.get("last_n"), DEFAULT_LAST_N, lo=1, hi=MAX_LAST_N)
            offset = _parse_int(qs.get("offset"), 0, lo=0, hi=1_000_000)
            limit = _parse_int(qs.get("limit"), DEFAULT_ROW_LIMIT, lo=1, hi=MAX_ROW_LIMIT)
            if path == "/api/meta":
                return meta_payload(store)
            if path == "/api/registry":
                return {"rows": registry_payload()}
            if path == "/api/index":
                return index_payload(
                    store,
                    layer=qs.get("layer", "standard"),
                    query=qs.get("q", ""),
                    offset=offset,
                    limit=limit,
                )
            if path == "/api/series":
                return series_payload(
                    store,
                    dataset=qs.get("dataset", ""),
                    entity_id=qs.get("entity", qs.get("entity_id", "")),
                    field=qs.get("field", ""),
                    factor_id=qs.get("factor_id", ""),
                    last_n=last_n,
                    layer=qs.get("layer", ""),
                )
            if path == "/api/rows":
                return rows_payload(
                    store,
                    layer=qs.get("layer", "standard"),
                    dataset=qs.get("dataset", ""),
                    entity_id=qs.get("entity", qs.get("entity_id", "")),
                    field=qs.get("field", ""),
                    factor_id=qs.get("factor_id", ""),
                    offset=offset,
                    limit=limit,
                )
            if path == "/api/industry/latest":
                return industry_latest_payload(
                    store,
                    factor_id=qs.get("factor_id", "IndustryAllocationScore"),
                    top_n=_parse_int(qs.get("top_n"), 15, lo=3, hi=40),
                )
            if path == "/api/industry/heatmap":
                return industry_heatmap_payload(
                    store,
                    factor_id=qs.get("factor_id", "IndustryAllocationScore"),
                    days=_parse_int(qs.get("days"), 60, lo=10, hi=252),
                )
            if path == "/api/etf":
                code = qs.get("code", "")
                if not code:
                    return etf_list_payload(store)
                return etf_detail_payload(store, code=code, last_n=last_n)
            if path == "/api/signals":
                return signals_payload(store)
            if path == "/api/backtest":
                return backtest_payload(store)
            if path == "/api/captions":
                return {"captions": captions_payload()}
            return {"error": f"unknown api {path}"}

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_static(self, path: str) -> None:
            rel = "index.html" if path in {"/", "/index.html"} else path.lstrip("/")
            target = (web_dir / rel).resolve()
            try:
                target.relative_to(web_dir.resolve())
            except ValueError:
                self.send_error(403)
                return
            if not target.is_file():
                self.send_error(404)
                return
            mime = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            if target.suffix == ".js":
                mime = "application/javascript; charset=utf-8"
            elif target.suffix == ".css":
                mime = "text/css; charset=utf-8"
            elif target.suffix == ".html":
                mime = "text/html; charset=utf-8"
            data = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def serve_lab(*, root: Path | None = None, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    """前台阻塞服务；Ctrl+C 退出。"""
    data_root = Path(root) if root is not None else default_root()
    web_dir = WEB_DIR
    if not web_dir.is_dir():
        raise FileNotFoundError(f"web dir missing: {web_dir}")
    store = LabStore(data_root)
    handler = make_handler(store, web_dir)
    httpd = ThreadingHTTPServer((host, int(port)), handler)
    print(f"alloc_v2 研究站 http://{host}:{port}  数据={data_root}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止", flush=True)
    finally:
        httpd.server_close()
