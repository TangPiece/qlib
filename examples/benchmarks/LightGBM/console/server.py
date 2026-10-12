#!/usr/bin/env python3
"""Minimal local console for LightGBM rolling workflows (stdlib only).

Serves static UI + triggers existing scripts (update data / train / export).

Usage (from qlib repo root)::

    .venv/bin/python examples/benchmarks/LightGBM/console/server.py
    # open http://127.0.0.1:8765
"""
from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import threading
import uuid
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote, urlparse

CONSOLE_DIR = Path(__file__).absolute().resolve().parent
LGBM_DIR = CONSOLE_DIR.parent
REPO_ROOT = LGBM_DIR.parents[2]  # LightGBM -> benchmarks -> examples -> qlib
DATA_DIR = LGBM_DIR / "data"
JOBS_DIR = CONSOLE_DIR / ".jobs"
DEFAULT_QLIB_DIR = Path("~/.qlib/qlib_data/cn_data").expanduser()
HOST = "127.0.0.1"
PORT = 8765

CONFIG_CHOICES = {
    "7y1y": LGBM_DIR / "workflow_config_lightgbm_Alpha158_rolling_7y1y.yaml",
    "7y1y_test": LGBM_DIR / "workflow_config_lightgbm_Alpha158_rolling_7y1y_test.yaml",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_active_job_id: Optional[str] = None


def _load_update_module():
    path = LGBM_DIR / "update_cn_data_daily.py"
    spec = importlib.util.spec_from_file_location("lgbm_update_cn_data_daily", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_update_mod = _load_update_module()


def _json_response(handler: SimpleHTTPRequestHandler, code: int, payload: Any) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: SimpleHTTPRequestHandler) -> Dict[str, Any]:
    length = int(handler.headers.get("Content-Length") or 0)
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


def _resolve_config(key_or_path: Optional[str]) -> Path:
    if not key_or_path:
        return CONFIG_CHOICES["7y1y"]
    if key_or_path in CONFIG_CHOICES:
        return CONFIG_CHOICES[key_or_path]
    path = Path(key_or_path)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    else:
        path = path.resolve()
    # Only allow configs under LightGBM dir
    try:
        path.relative_to(LGBM_DIR)
    except ValueError as exc:
        raise ValueError(f"config must be under {LGBM_DIR}") from exc
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    return path


def _parse_yaml_ends(config_path: Path) -> Dict[str, Optional[str]]:
    text = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
    out: Dict[str, Optional[str]] = {
        "data_handler_end": None,
        "backtest_end": None,
        "test_end": None,
    }
    m = re.search(
        r"data_handler_config:[^\n]*\n(?:[^\n]*\n)*?    end_time: (\d{4}-\d{2}-\d{2})",
        text,
    )
    if m:
        out["data_handler_end"] = m.group(1)
    m = re.search(
        r"    backtest:\n(?:[^\n]*\n)*?        end_time: (\d{4}-\d{2}-\d{2})",
        text,
    )
    if m:
        out["backtest_end"] = m.group(1)
    m = re.search(r"test:\s*\[[\d-]+,\s*(\d{4}-\d{2}-\d{2})\s*\]", text)
    if m:
        out["test_end"] = m.group(1)
    return out


def _calendar_bounds() -> Dict[str, Optional[str]]:
    try:
        first, latest = _update_mod.read_calendar_bounds(DEFAULT_QLIB_DIR)
        return {"first": first, "latest": latest}
    except Exception as exc:  # noqa: BLE001
        return {"first": None, "latest": None, "error": str(exc)}


def _job_snapshot(job_id: str, log_tail: int = 200) -> Dict[str, Any]:
    with _jobs_lock:
        job = dict(_jobs.get(job_id) or {})
    if not job:
        return {"id": job_id, "error": "not found"}
    log_path = Path(job.get("log_path") or "")
    log_text = ""
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        log_text = "\n".join(lines[-log_tail:])
    return {
        "id": job["id"],
        "kind": job.get("kind"),
        "status": job.get("status"),
        "command": job.get("command"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "exit_code": job.get("exit_code"),
        "error": job.get("error"),
        "log": log_text,
    }


def _set_active(job_id: Optional[str]) -> None:
    global _active_job_id
    with _jobs_lock:
        _active_job_id = job_id


def _get_active() -> Optional[str]:
    with _jobs_lock:
        return _active_job_id


def _start_job(kind: str, command: List[str]) -> Dict[str, Any]:
    if _get_active():
        raise RuntimeError("another job is already running")

    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    log_path = JOBS_DIR / f"{job_id}.log"
    job = {
        "id": job_id,
        "kind": kind,
        "status": "running",
        "command": command,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "finished_at": None,
        "exit_code": None,
        "error": None,
        "log_path": str(log_path),
    }
    with _jobs_lock:
        _jobs[job_id] = job
    _set_active(job_id)

    def _runner() -> None:
        try:
            with log_path.open("w", encoding="utf-8") as logf:
                logf.write("$ " + " ".join(command) + "\n\n")
                logf.flush()
                proc = subprocess.Popen(
                    command,
                    cwd=str(REPO_ROOT),
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                code = proc.wait()
            with _jobs_lock:
                _jobs[job_id]["exit_code"] = code
                _jobs[job_id]["status"] = "finished" if code == 0 else "failed"
                _jobs[job_id]["finished_at"] = datetime.now().isoformat(timespec="seconds")
                if code != 0:
                    _jobs[job_id]["error"] = f"exit code {code}"
        except Exception as exc:  # noqa: BLE001
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(exc)
                _jobs[job_id]["finished_at"] = datetime.now().isoformat(timespec="seconds")
            try:
                with log_path.open("a", encoding="utf-8") as logf:
                    logf.write(f"\n[console] job failed: {exc}\n")
            except OSError:
                pass
        finally:
            _set_active(None)

    threading.Thread(target=_runner, daemon=True).start()
    return _job_snapshot(job_id)


def _list_results() -> List[Dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for path in sorted(DATA_DIR.glob("*_positions_view.html"), key=lambda p: p.stat().st_mtime, reverse=True):
        st = path.stat()
        name = path.name
        exp_name = name[: -len("_positions_view.html")] if name.endswith("_positions_view.html") else name
        rows.append(
            {
                "filename": name,
                "exp_name": exp_name,
                "url": f"/data/{name}",
                "mtime": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                "size_bytes": st.st_size,
                "size_mb": round(st.st_size / (1024 * 1024), 2),
            }
        )
    return rows


def _status_payload(config_key: str = "7y1y") -> Dict[str, Any]:
    config_path = _resolve_config(config_key)
    cal = _calendar_bounds()
    active = _get_active()
    return {
        "today": datetime.now().date().isoformat(),
        "calendar": cal,
        "config_key": config_key,
        "config_path": str(config_path.relative_to(REPO_ROOT)),
        "yaml_ends": _parse_yaml_ends(config_path),
        "configs": {k: str(v.relative_to(REPO_ROOT)) for k, v in CONFIG_CHOICES.items()},
        "active_job_id": active,
        "active_job": _job_snapshot(active) if active else None,
        "python": sys.executable,
        "repo_root": str(REPO_ROOT),
    }


class ConsoleHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(CONSOLE_DIR), **kwargs)

    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/api/status":
            from urllib.parse import parse_qs

            qs = parse_qs(parsed.query)
            config_key = (qs.get("config") or ["7y1y"])[0]
            try:
                _json_response(self, 200, _status_payload(config_key))
            except Exception as exc:  # noqa: BLE001
                _json_response(self, 400, {"error": str(exc)})
            return

        if path == "/api/results":
            _json_response(self, 200, {"results": _list_results()})
            return

        if path.startswith("/api/jobs/"):
            job_id = path[len("/api/jobs/") :].strip("/")
            snap = _job_snapshot(job_id)
            code = 404 if snap.get("error") == "not found" else 200
            _json_response(self, code, snap)
            return

        if path == "/" or path == "/index.html":
            return self._serve_file(CONSOLE_DIR / "index.html", "text/html; charset=utf-8")

        if path.startswith("/static/"):
            rel = path[len("/static/") :]
            target = (CONSOLE_DIR / rel).resolve()
            try:
                target.relative_to(CONSOLE_DIR)
            except ValueError:
                self.send_error(403)
                return
            return self._serve_file(target)

        if path.startswith("/data/"):
            rel = path[len("/data/") :]
            target = (DATA_DIR / rel).resolve()
            try:
                target.relative_to(DATA_DIR)
            except ValueError:
                self.send_error(403)
                return
            return self._serve_file(target)

        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            body = _read_json(self)
        except Exception as exc:  # noqa: BLE001
            _json_response(self, 400, {"error": f"invalid JSON: {exc}"})
            return

        try:
            if path == "/api/set-end-date":
                date = str(body.get("date") or "").strip()
                if not _DATE_RE.match(date):
                    raise ValueError("date must be YYYY-MM-DD")
                config_path = _resolve_config(body.get("config"))
                changes = _update_mod.update_config_end_times(config_path, date, dry_run=False)
                _json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "date": date,
                        "config": str(config_path.relative_to(REPO_ROOT)),
                        "changes": changes,
                    },
                )
                return

            if path == "/api/update-data":
                cmd = [sys.executable, str(LGBM_DIR / "update_cn_data_daily.py")]
                if body.get("skip_download"):
                    cmd.append("--skip-download")
                if body.get("force"):
                    cmd.append("--force")
                configs = body.get("configs")
                if isinstance(configs, list) and configs:
                    cmd.append("--config")
                    for c in configs:
                        cmd.append(str(_resolve_config(str(c))))
                job = _start_job("update-data", cmd)
                _json_response(self, 200, {"ok": True, "job": job})
                return

            if path == "/api/train":
                config_path = _resolve_config(body.get("config") or body.get("conf_path"))
                exp_name = str(body.get("exp_name") or "").strip()
                if not exp_name:
                    exp_name = f"rolling_lgb_7y1y_{datetime.now().strftime('%Y%m%d')}"
                horizon = int(body.get("horizon") or 1)
                conf_rel = config_path.relative_to(REPO_ROOT)
                cmd = [
                    sys.executable,
                    str(LGBM_DIR / "rolling_7y1y.py"),
                    f"--conf_path={conf_rel}",
                    f"--exp_name={exp_name}",
                    f"--horizon={horizon}",
                    "run",
                ]
                job = _start_job("train", cmd)
                _json_response(self, 200, {"ok": True, "job": job, "exp_name": exp_name})
                return

            if path == "/api/export":
                exp_name = str(body.get("exp_name") or "").strip()
                if not exp_name:
                    raise ValueError("exp_name is required")
                cmd = [
                    sys.executable,
                    str(LGBM_DIR / "export_positions_view.py"),
                    f"--exp_name={exp_name}",
                ]
                job = _start_job("export", cmd)
                _json_response(self, 200, {"ok": True, "job": job, "exp_name": exp_name})
                return

            _json_response(self, 404, {"error": f"unknown endpoint {path}"})
        except RuntimeError as exc:
            _json_response(self, 409, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            _json_response(self, 400, {"error": str(exc)})

    def _serve_file(self, path: Path, content_type: Optional[str] = None) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(404, f"File not found: {path.name}")
            return
        data = path.read_bytes()
        if content_type is None:
            if path.suffix == ".html":
                content_type = "text/html; charset=utf-8"
            elif path.suffix == ".js":
                content_type = "application/javascript; charset=utf-8"
            elif path.suffix == ".css":
                content_type = "text/css; charset=utf-8"
            elif path.suffix == ".json":
                content_type = "application/json; charset=utf-8"
            else:
                content_type = "application/octet-stream"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), ConsoleHandler)
    print(f"LightGBM console: http://{HOST}:{PORT}", flush=True)
    print(f"repo root: {REPO_ROOT}", flush=True)
    print(f"python:    {sys.executable}", flush=True)
    print("Ctrl+C to stop", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
