#!/usr/bin/env python3
"""按 manifest 串行单次训练，成功后 export 到控制台历史；支持断点续跑。

用法（仓库根）::

    .venv/bin/python examples/benchmarks/_factor_sweep/run_sweep.py --phase 1
    .venv/bin/python examples/benchmarks/_factor_sweep/run_sweep.py --only LightGBM,alpha158
    .venv/bin/python examples/benchmarks/_factor_sweep/run_sweep.py --phase 1,2,3
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DIR = Path(__file__).resolve().parent
REPO = DIR.parents[2]
MANIFEST = DIR / "manifest.json"
LOG_DIR = DIR / "logs"
EXPORT_SCRIPT = REPO / "examples" / "benchmarks" / "LightGBM" / "export_positions_view.py"
PYTHON = sys.executable


def _load_manifest() -> Dict[str, Any]:
    if not MANIFEST.exists():
        raise SystemExit(f"manifest missing: {MANIFEST}; run gen_configs.py first")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _save_manifest(data: Dict[str, Any]) -> None:
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _today() -> str:
    return datetime.now().strftime("%Y%m%d")


def _run_cmd(cmd: List[str], log_path: Path, timeout: Optional[int] = None) -> int:
    """跑子进程，stdout/stderr 写入日志；返回 exit code。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as logf:
        logf.write("CMD: " + " ".join(cmd) + "\n\n")
        logf.flush()
        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(DIR / ".mplconfig"))
        env.setdefault("PYTHONUNBUFFERED", "1")
        env.setdefault("OMP_NUM_THREADS", "1")
        proc = subprocess.run(
            cmd,
            cwd=str(REPO),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=timeout,
        )
        return int(proc.returncode)


def _tail(path: Path, n: int = 40) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _parse_only(raw: str) -> List[Tuple[Optional[str], Optional[str]]]:
    """解析 --only：`Model,factor` 用分号分隔多条；单 token 只匹配 model。"""
    filters: List[Tuple[Optional[str], Optional[str]]] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        if "," in part:
            model, factor = part.split(",", 1)
            filters.append((model.strip(), factor.strip()))
        else:
            filters.append((part, None))
    return filters


def _match_only(job: Dict[str, Any], filters: List[Tuple[Optional[str], Optional[str]]]) -> bool:
    if not filters:
        return True
    for model, factor in filters:
        if factor is None:
            if job["model"] == model or job["model"].lower() == model.lower():
                return True
            continue
        if (
            job["model"] == model or job["model"].lower() == model.lower()
        ) and job["factor_key"] == factor:
            return True
    return False


def run_one(job: Dict[str, Any], date_tag: str) -> None:
    """训练 + export；就地更新 job 状态字段。"""
    if job.get("status") == "skipped":
        return
    exp_name = f"{job['exp_name']}_{date_tag}"
    job["exp_name_run"] = exp_name
    yaml_path = REPO / job["yaml"]
    if not yaml_path.exists():
        job["status"] = "failed"
        job["error"] = f"yaml not found: {yaml_path}"
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    train_log = LOG_DIR / f"{exp_name}_train.log"
    export_log = LOG_DIR / f"{exp_name}_export.log"

    job["status"] = "running"
    job["started_at"] = datetime.now().isoformat(timespec="seconds")
    _save_manifest(_load_manifest_with_job_update(job))

    train_cmd = [
        PYTHON,
        "-m",
        "qlib.cli.run",
        str(yaml_path),
        f"--experiment_name={exp_name}",
    ]
    t0 = time.time()
    try:
        rc = _run_cmd(train_cmd, train_log)
    except subprocess.TimeoutExpired:
        job["status"] = "failed"
        job["error"] = "train timeout"
        job["elapsed_sec"] = round(time.time() - t0, 1)
        return

    job["elapsed_sec"] = round(time.time() - t0, 1)
    if rc != 0:
        job["status"] = "failed"
        job["error"] = f"train exit={rc}; tail:\n{_tail(train_log)}"
        return

    export_cmd = [
        PYTHON,
        str(EXPORT_SCRIPT),
        f"--exp_name={exp_name}",
    ]
    try:
        rc2 = _run_cmd(export_cmd, export_log)
    except subprocess.TimeoutExpired:
        job["status"] = "failed"
        job["error"] = "export timeout (train ok)"
        return

    if rc2 != 0:
        job["status"] = "failed"
        job["error"] = f"export exit={rc2}; tail:\n{_tail(export_log)}"
        return

    job["status"] = "success"
    job["error"] = ""
    job["finished_at"] = datetime.now().isoformat(timespec="seconds")


def _load_manifest_with_job_update(updated: Dict[str, Any]) -> Dict[str, Any]:
    """把单个 job 写回 manifest（按 model+factor_key 匹配）。"""
    data = _load_manifest()
    for i, j in enumerate(data["jobs"]):
        if j["model"] == updated["model"] and j["factor_key"] == updated["factor_key"]:
            data["jobs"][i] = updated
            break
    return data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Factor sweep single-shot trainer")
    p.add_argument(
        "--phase",
        default="1,2,3",
        help="逗号分隔 phase，如 1 或 1,2,3",
    )
    p.add_argument(
        "--only",
        default="",
        help="过滤：Model 或 Model,factor_key，逗号分隔多条",
    )
    p.add_argument(
        "--retry-failed",
        action="store_true",
        help="重跑 failed 任务",
    )
    p.add_argument(
        "--date-tag",
        default="",
        help="实验名日期后缀，默认今天 YYYYMMDD",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将跑的任务",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    phases = {int(x) for x in args.phase.split(",") if x.strip()}
    only_filters = _parse_only(args.only)
    date_tag = args.date_tag or _today()

    data = _load_manifest()
    selected: List[Dict[str, Any]] = []
    for job in data["jobs"]:
        if job.get("phase", 99) not in phases:
            continue
        if not _match_only(job, only_filters):
            continue
        st = job.get("status", "pending")
        if st == "skipped":
            continue
        # 已成功的永不重跑（即使带 --retry-failed）
        if st == "success":
            continue
        if st == "failed" and not args.retry_failed:
            continue
        # 续跑：打断的 running / 显式重试的 failed 回到 pending
        if st in ("failed", "running"):
            job["status"] = "pending"
            job["error"] = ""
        selected.append(job)

    print(f"selected {len(selected)} jobs (phases={sorted(phases)}, date_tag={date_tag})")
    if args.dry_run:
        for j in selected:
            print(f"  [phase {j['phase']}] {j['model']} / {j['factor_key']} -> {j['yaml']}")
        return

    _save_manifest(data)

    ok = fail = 0
    for idx, job in enumerate(selected, 1):
        print(f"\n=== [{idx}/{len(selected)}] {job['model']} / {job['factor_key']} ===")
        run_one(job, date_tag)
        data = _load_manifest_with_job_update(job)
        _save_manifest(data)
        print(f"    status={job['status']} elapsed={job.get('elapsed_sec')}s")
        if job["status"] == "success":
            ok += 1
        else:
            fail += 1
            print(f"    error: {job.get('error', '')[:500]}")

    # 汇总
    data = _load_manifest()
    counts: Dict[str, int] = {}
    for j in data["jobs"]:
        counts[j["status"]] = counts.get(j["status"], 0) + 1
    print("\n=== summary ===")
    print(counts)
    print(f"this run: success={ok} failed={fail}")


if __name__ == "__main__":
    main()
