#!/usr/bin/env bash
# fund_cn：国泰安科目为主 + EM 公告补洞 + 免费源验证 → PIT → 日频质检 → parquet/R2。不写 qlib bin。
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
  run_fund_cn.sh all --start 2018-01-01 --end 2018-12-31 --no-r2

  默认 --source csmar：国泰安科目为主，东财 NOTICE_DATE 补公告，
  免费源仅抽样验证（不覆盖国泰安数值）。不写 qlib bin。

选项:
  --source csmar|akshare
  --csmar-path PATH
  --skip-verify          # 正式出数勿用；默认开启免费源重叠期验证
  --use-cninfo-shares
  --instruments CNStock:600000,...
  --instruments-file PATH
  --dates 2018-01-02,2018-01-03
  --trade-start/--trade-end YYYY-MM-DD   # 交易日窗口（compute/all）
  --report-start/--report-end YYYY-MM-DD # 报告期窗口（stage）
  --start/--end YYYY-MM-DD               # 兼容旧参数
  --resume-from YYYY     # compute 从该年起；已完整年份默认跳过
  --no-resume            # 强制重算
  --market-file PATH
  --qlib-dir DIR
  --root DIR
  --output DIR
  --full
  --no-r2
  --allow-uncovered      # 仅诊断；正式路径应过 coverage 硬闸

示例:
  ./scripts/run_fund_cn.sh catalog
  ./scripts/run_fund_cn.sh stage --instruments CNStock:600000 --report-start 2005-01-01 --report-end 2018-12-31
  ./scripts/run_fund_cn.sh all --trade-start 2018-01-01 --trade-end 2018-01-31 --no-r2
  # 全市场 + 断点续跑（正式：不开 --skip-verify / --allow-uncovered）
  ./scripts/run_fund_cn.sh all \\
    --instruments-file data/fundamental_staging/fund_cn_all_instruments.txt \\
    --trade-start 2000-01-04 --trade-end 2026-11-13 --no-r2
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
