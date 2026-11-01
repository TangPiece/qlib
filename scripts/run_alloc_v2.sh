#!/usr/bin/env bash
# alloc_v2：四层资金配置研究。不写 qlib bin（dump 另见 dump_alloc_v2.py）。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
elif [[ -n "${ALLOC_V2_PYTHON:-}" && -x "${ALLOC_V2_PYTHON}" ]]; then
  PY="${ALLOC_V2_PYTHON}"
else
  PY="$(command -v python3)"
fi
STAGING="${ROOT}/data/alloc_v2"
MODULE="qlib.contrib.data.alloc_v2"
# 东财常走不通 Clash/系统代理。默认直连；需要代理时：ALLOC_V2_NO_PROXY=0
export ALLOC_V2_NO_PROXY="${ALLOC_V2_NO_PROXY:-1}"
export FLOW_MONITOR_NO_PROXY="${FLOW_MONITOR_NO_PROXY:-$ALLOC_V2_NO_PROXY}"

usage() {
  cat <<'USAGE'
用法:
  run_alloc_v2.sh catalog
  run_alloc_v2.sh maps|fetch|standard|factor|signal|backtest|all [选项]
  run_alloc_v2.sh update [--overlap-days N] [--end YYYY-MM-DD]
  run_alloc_v2.sh upload [--dry-run] [--root DIR]

  all     maps → fetch → factor → signal → backtest（首次全量）
          份额变化 ≠ 净流入；北向净买额不作核心因子。
  update  从本地 standard 最近 event_date 向前重叠 N 天增量拉取并合并，
          再重算 factor/signal/backtest。无本地数据时请先 all。
  upload  同步 maps/standard/factor/research 到 R2（不传 _markers）
          需 .env 中 R2_ENDPOINT_URL / R2_ACCESS_KEY_ID /
          R2_SECRET_ACCESS_KEY / R2_BUCKET；前缀 ALLOC_V2_R2_PREFIX
          默认 research/alloc_v2。

选项:
  --start YYYY-MM-DD
  --end YYYY-MM-DD
  --root DIR
  --overlap-days N
  --json
  --include-disabled
  --dry-run

示例:
  ./scripts/run_alloc_v2.sh catalog
  ./scripts/run_alloc_v2.sh all --start 2018-01-01
  ./scripts/run_alloc_v2.sh update
  ./scripts/run_alloc_v2.sh update --overlap-days 7
  ./scripts/run_alloc_v2.sh upload --dry-run
  ./scripts/run_alloc_v2.sh upload
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
