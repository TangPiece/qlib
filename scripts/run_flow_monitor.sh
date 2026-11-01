#!/usr/bin/env bash
# flow_monitor：fetch 公开数据 → 计算诊断面板。不写 qlib bin。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
elif [[ -n "${FLOW_MONITOR_PYTHON:-}" && -x "${FLOW_MONITOR_PYTHON}" ]]; then
  PY="${FLOW_MONITOR_PYTHON}"
else
  PY="$(command -v python3)"
fi
STAGING="${ROOT}/data/flow_staging"
MODULE="qlib.contrib.data.flow_monitor"
# 东财常走不通 Clash/系统代理。默认直连；需要代理时：FLOW_MONITOR_NO_PROXY=0
export FLOW_MONITOR_NO_PROXY="${FLOW_MONITOR_NO_PROXY:-1}"

usage() {
  cat <<'USAGE'
用法:
  run_flow_monitor.sh catalog
  run_flow_monitor.sh fetch|compute|all [选项]

  all     拉公开数据 → 写 staging → 计算价格调整流量与配置证据矩阵
          不写 qlib bin。周/月/季数据不会伪装成日频。

选项:
  --start YYYY-MM-DD
  --end YYYY-MM-DD
  --root DIR
  --json
  --include-disabled

示例:
  ./scripts/run_flow_monitor.sh catalog
  ./scripts/run_flow_monitor.sh all --start 2018-01-01
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

[[ $# -ge 1 ]] || { usage; exit 1; }
[[ -x "$PY" ]] || die "未找到 python: $PY"
cd "${ROOT}"
mkdir -p "${STAGING}"
exec "${PY}" -m "${MODULE}" "$@"
