#!/usr/bin/env bash
# fund_cn：stage → coverage → 本地 PIT → 日频质检 → 本地/R2 Parquet。不写 qlib bin。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PY="${ROOT}/.venv/bin/python"
elif [[ -n "${FUND_CN_PYTHON:-}" && -x "${FUND_CN_PYTHON}" ]]; then
  PY="${FUND_CN_PYTHON}"
else
  PY="$(command -v python3)"
fi
STAGING="${ROOT}/data/fundamental_staging"
MODULE="qlib.contrib.data.fund_cn"

usage() {
  cat <<'USAGE'
用法:
  run_fund_cn.sh catalog
  run_fund_cn.sh stage|coverage|pit|compute|all [选项]
  run_fund_cn.sh all --start 2018-01-01 --end 2018-12-31

  all     现拉 AKShare/巨潮 → coverage 硬闸 → 本地 PIT → 按日计算
          → 质检通过后才写本地/R2。不写 qlib bin。

选项:
  --instruments CNStock:600000,...
  --dates 2018-01-02,2018-01-03
  --start YYYY-MM-DD
  --end YYYY-MM-DD
  --market-file PATH
  --qlib-dir DIR
  --root DIR
  --output DIR
  --full
  --no-r2
  --allow-uncovered

示例:
  ./scripts/run_fund_cn.sh catalog
  ./scripts/run_fund_cn.sh all --start 2018-01-01 --end 2018-01-31 --no-r2
  ./scripts/run_fund_cn.sh stage --instruments CNStock:600000
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
