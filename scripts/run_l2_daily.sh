#!/usr/bin/env bash
# L2 百因子：解压 → CSV 直算（按股票四组同算）→ 质检 → 本地/R2 宽表。不写 qlib bin。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
elif [[ -n "${LEVEL2_PYTHON:-}" && -x "${LEVEL2_PYTHON}" ]]; then
  PY="${LEVEL2_PYTHON}"
else
  PY="$(command -v python3)"
fi
STAGING="${ROOT}/data/level2_staging"
BATCH_KEY="all"
MODULE="qlib.contrib.data.l2_daily"

usage() {
  cat <<'USAGE'
用法:
  run_l2_daily.sh all [start 选项] | status | tail | stop
  run_l2_daily.sh DATE [start 选项] | status | tail | stop
  run_l2_daily.sh --start YYYYMMDD --end YYYYMMDD [选项]
  run_l2_daily.sh reupload --start YYYYMMDD --end YYYYMMDD
  run_l2_daily.sh reupload --failed

  all     扫描 data/level2_raw：已解压 CSV 优先，再逐日解压 .7z，
          按股票四组同算 CSV → 质检通过后才写本地/R2。
          一天结束就删解压目录；质检通过才连 .7z 一起删。--no-delete 全留。
          每波结束后立刻重扫：有尚未处理的新 .7z/日目录则续批。
          若只剩下载中文件（如 .baiduyun.p.downloading）则等待下完再续。
          单日解压/计算/质检失败都记入账本、删解压留 7z，不阻断后续。
          --force 才重跑账本里的日期。stop 可中断等待。
  DATE    单日 YYYYMMDD

选项:
  --groups trade,book,order,microstructure
  --raw-dir DIR
  --output DIR
  --workers N          按股票四组同算的进程数（默认 8 或 LEVEL2_L2_DAILY_WORKERS）
  --force
  --no-r2
  --no-compact
  --no-seed
  --no-delete
  --delete-sources
  --local-only
  --7z-only
  --dry-run
  --failed             reupload 只补传账本里失败的日期

示例:
  ./scripts/run_l2_daily.sh all
  ./scripts/run_l2_daily.sh all --workers 4
  ./scripts/run_l2_daily.sh 20180731 --force --no-delete
  ./scripts/run_l2_daily.sh prune-sources
  ./scripts/run_l2_daily.sh reupload --failed
  ./scripts/run_l2_daily.sh all status
  ./scripts/run_l2_daily.sh all tail
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

validate_date() {
  local d="$1"
  [[ "$d" =~ ^[0-9]{8}$ ]] || die "DATE 须为 YYYYMMDD，收到: $d"
}

log_path() {
  echo "${STAGING}/logs/l2_daily_${1}.log"
}

pid_path() {
  echo "${STAGING}/logs/l2_daily_${1}.pid"
}

trim_log() {
  local file="$1"
  local limit=$((32 * 1024 * 1024))
  local keep=$((256 * 1024))
  local size
  [[ -f "$file" ]] || return 0
  size="$(wc -c < "$file" | tr -d '[:space:]')"
  if (( size <= limit )); then
    return 0
  fi
  tail -c "$keep" "$file" > "${file}.tail"
  mv "${file}.tail" "$file"
}

read_pid() {
  local pf
  pf="$(pid_path "$1")"
  [[ -f "$pf" ]] || return 1
  tr -d '[:space:]' < "$pf"
}

is_running() {
  local pid="$1"
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null
}

# 列出某 pid 的直接子进程（ProcessPool worker 等）
list_children() {
  local pid="$1"
  pgrep -P "$pid" 2>/dev/null || true
}

# 向进程组发信号；start_new_session=True 时 pid 即 pgid
# 组杀失败时回退到单 pid，避免误伤无关进程
kill_tree() {
  local sig="$1"
  local pid="$2"
  kill "-${sig}" -- "-${pid}" 2>/dev/null || kill "-${sig}" -- "$pid" 2>/dev/null || true
}

# 兜底：清扫仍活着的 l2_daily 主进程及其子进程（历史孤儿）
sweep_l2_daily_orphans() {
  local pids child
  # 匹配 python -m qlib.contrib.data.l2_daily；排除本 shell 自身
  pids="$(pgrep -f 'python.*-m[[:space:]]+qlib\.contrib\.data\.l2_daily' 2>/dev/null || true)"
  [[ -n "$pids" ]] || return 0
  echo "清扫残留 l2_daily 进程: ${pids}"
  for pid in $pids; do
    [[ "$pid" == "$$" ]] && continue
    is_running "$pid" || continue
    for child in $(list_children "$pid"); do
      kill -TERM -- "$child" 2>/dev/null || true
    done
    kill_tree TERM "$pid"
  done
  sleep 1
  pids="$(pgrep -f 'python.*-m[[:space:]]+qlib\.contrib\.data\.l2_daily' 2>/dev/null || true)"
  for pid in $pids; do
    [[ "$pid" == "$$" ]] && continue
    is_running "$pid" || continue
    for child in $(list_children "$pid"); do
      kill -KILL -- "$child" 2>/dev/null || true
    done
    kill_tree KILL "$pid"
  done
}

cmd_status() {
  local key="$1"
  local pf pid children child_count
  pf="$(pid_path "$key")"
  echo "key:     ${key}"
  echo "log:     $(log_path "$key")"
  echo "pidfile: ${pf}"
  echo "raw:     ${ROOT}/data/level2_raw"
  echo "factors: ${STAGING}/factors/l2_daily"
  if pid="$(read_pid "$key" 2>/dev/null)" && is_running "$pid"; then
    echo "status:  running (pid=${pid})"
    ps -p "$pid" -o pid=,etime=,pcpu=,command= 2>/dev/null || true
    # 显示 ProcessPool 等子进程，便于核对是否多于 --workers
    children="$(list_children "$pid")"
    if [[ -n "$children" ]]; then
      child_count="$(echo "$children" | wc -w | tr -d '[:space:]')"
      echo "children:${child_count}"
      ps -p "$(echo "$children" | tr ' ' ',')" -o pid=,ppid=,etime=,pcpu=,rss=,command= 2>/dev/null || true
    else
      echo "children:0"
    fi
    return 0
  fi
  echo "status:  not running"
  # pidfile 失效时仍提示是否有孤儿 worker
  if pgrep -f 'python.*-m[[:space:]]+qlib\.contrib\.data\.l2_daily' >/dev/null 2>&1; then
    echo "orphan:  仍有 l2_daily 相关进程（可用 stop 清扫）"
    pgrep -fl 'python.*-m[[:space:]]+qlib\.contrib\.data\.l2_daily' 2>/dev/null || true
  fi
  return 1
}

cmd_tail() {
  local key="$1"
  local lf
  lf="$(log_path "$key")"
  mkdir -p "${STAGING}/logs"
  touch "$lf"
  tail -f "$lf"
}

cmd_stop() {
  local key="$1"
  local pid child
  pid="$(read_pid "$key" 2>/dev/null)" || {
    # 无 pidfile 时仍尝试清扫历史孤儿 worker
    echo "无 pid 文件: $(pid_path "$key")，尝试清扫残留 ..."
    sweep_l2_daily_orphans
    echo "已停止"
    return 0
  }
  if ! is_running "$pid"; then
    echo "进程已不在运行 (pid=${pid})，清扫残留 worker ..."
    sweep_l2_daily_orphans
    echo "已停止"
    return 0
  fi
  # start_new_session=True：对进程组发 SIGTERM，一次收掉主进程 + workers
  echo "发送 SIGTERM 到进程组 pgid=${pid} ..."
  kill_tree TERM "$pid"
  for _ in $(seq 1 30); do
    if ! is_running "$pid"; then
      # 主进程已退，再清可能落单的子进程 / 历史孤儿
      for child in $(list_children "$pid"); do
        kill -TERM -- "$child" 2>/dev/null || true
      done
      sweep_l2_daily_orphans
      echo "已停止"
      return 0
    fi
    sleep 1
  done
  echo "30s 未退出，对进程组发送 SIGKILL ..."
  kill_tree KILL "$pid"
  sleep 1
  sweep_l2_daily_orphans
  if is_running "$pid"; then
    die "进程组仍未退出，可手动 kill -9 -${pid}"
  fi
  echo "已强制停止"
}

cmd_start() {
  local key="$1"
  shift
  local py_args=()

  [[ -x "$PY" ]] || die "未找到 python: $PY"

  if pid="$(read_pid "$key" 2>/dev/null)" && is_running "$pid"; then
    die "已在运行 pid=${pid}，先 stop 或删 pid 文件"
  fi

  mkdir -p "${STAGING}/logs"
  local lf pf
  lf="$(log_path "$key")"
  pf="$(pid_path "$key")"
  trim_log "$lf"

  if [[ "$key" == "$BATCH_KEY" ]]; then
    py_args=(all "$@")
  else
    py_args=("$key" "$@")
  fi

  export ROOT PY LF="$lf" PF="$pf" MODULE="$MODULE"
  export PY_ARGS
  if ((${#py_args[@]} > 0)); then
    PY_ARGS="$(printf '%s\n' "${py_args[@]}")"
  else
    PY_ARGS=""
  fi
  PY_ARGS="$PY_ARGS" \
    "$PY" - <<'PY'
import os
import subprocess
from pathlib import Path

root = Path(os.environ["ROOT"])
py = Path(os.environ["PY"])
log = Path(os.environ["LF"])
pidf = Path(os.environ["PF"])
module = os.environ["MODULE"]
extra = [ln for ln in os.environ.get("PY_ARGS", "").splitlines() if ln]

cmd = [str(py), "-m", module, *extra]

log.parent.mkdir(parents=True, exist_ok=True)
with open(log, "a", encoding="utf-8") as out:
    p = subprocess.Popen(
        cmd,
        stdout=out,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        cwd=str(root),
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
pidf.write_text(str(p.pid) + "\n", encoding="utf-8")
print(f"started pid={p.pid}")
print(f"log={log}")
print("cmd=" + " ".join(cmd))
PY

  echo "tail:   ./scripts/run_l2_daily.sh ${key} tail"
  echo "status: ./scripts/run_l2_daily.sh ${key} status"
}

main() {
  [[ $# -ge 1 ]] || { usage; exit 1; }

  # 允许直接传 --start/--end、prune-sources、reupload 前台跑（不走后台）
  if [[ "$1" == --* || "$1" == "prune-sources" || "$1" == "reupload" ]]; then
    cd "${ROOT}"
    exec "${PY}" -m "${MODULE}" "$@"
  fi

  local key="$1"
  shift

  if [[ "$key" != "$BATCH_KEY" ]]; then
    validate_date "$key"
  fi

  local subcmd="start"
  case "${1:-}" in
    start|status|tail|stop)
      subcmd="$1"
      shift
      ;;
  esac

  case "$subcmd" in
    start)  cmd_start "$key" "$@" ;;
    status) cmd_status "$key" ;;
    tail)   cmd_tail "$key" ;;
    stop)   cmd_stop "$key" ;;
    *)      usage; exit 1 ;;
  esac
}

main "$@"
