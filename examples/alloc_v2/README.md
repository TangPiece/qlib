# A 股多层次资金配置研究 V2

四层数据架构 + B/C/D 分层研究 + 波段信号。只用免费官方 / AKShare / 新浪 / 申万。

**份额变化 ≠ 净流入。北向成交额 ≠ 净买入。季报 ≠ 实时持仓。龙虎榜机构席位 ≠ 已确认公募/保险。**

## 刷新数据

首次全量：

```bash
./scripts/run_alloc_v2.sh catalog
./scripts/run_alloc_v2.sh all --start 2018-01-01
```

日常增量（从本地最近 `event_date` 向前重叠 7 天拉取并合并，再重算因子/信号）：

```bash
./scripts/run_alloc_v2.sh update
./scripts/run_alloc_v2.sh update --overlap-days 7 --end 2026-11-01
```

无本地 `standard` 时 `update` 会报错，需先跑 `all`。增量**不**自动上传 R2。

结果写在 `data/alloc_v2/`（不进 git）：

- `raw/` 原始响应
- `standard/observations.parquet` 统一标准层（含 `available_at`）
- `factor/all.parquet` 因子（`direct` / `proxy` / `estimate`）
- `research/` 信号、失效规则、IC/分组回测记录
- `maps/` ETF 主表、联接、主体、跨境状态

## 上传到 R2

复用仓库根目录 `.env` 里与 L2 相同的 `R2_*` 变量：

- `R2_ENDPOINT_URL`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `R2_BUCKET`
- 可选：`R2_REGION`（默认 `auto`）、`ALLOC_V2_R2_PREFIX`（默认 `research/alloc_v2`）

只上传 `maps/`、`standard/`、`factor/`、`research/`，**不上传** `_markers/`。

```bash
./scripts/run_alloc_v2.sh upload --dry-run
./scripts/run_alloc_v2.sh upload
```

成功后对象键形如 `research/alloc_v2/standard/observations.parquet`。结果摘要写在 `data/alloc_v2/_markers/r2_upload.json`。

## 看图（综合 Dashboard）

**看图入口：** [`dashboard.ipynb`](dashboard.ipynb)

六区：全球跨资产、北向、行业配置、ETF 三类指标、S1–S4 信号、回测条形。只读本地 staging，不打网。

**如何把各图串起来看（必读）：** [`CHART_READING.md`](CHART_READING.md)

读图顺序：环境（全球/利率）→ 北向通道 → 行业强度/成交 → ETF 份额与估算配置 → S1/S2/S4 门闩 → 回测置信度。上层不成立时，下层亮点只观察、不当进攻依据。详细交叉对照表、日常勾选清单见上文档。

回测读法：红柱/`factor_missing`=未建成；IC≈0=有结果但无稳定预测力。同时看全样本与近一年（共同日历），并核对综合分与成交占比相关是否再次贴死。上交所远端份额为周采样（`sparse_sample`），流因子近 180 日更可信；份额/NAV 衍生因子遵循 T+1 `available_at`。研究产物另含 `research/backtest_ic_by_year.parquet`、`research/backtest_factor_corr.parquet`。

## 专题报告（表为主）

每格区分 **事实 / 估算 / 代理 / 推断 / 信号**：

1. [`01_global.ipynb`](01_global.ipynb) — 全球与中国跨资产
2. [`02_industry.ipynb`](02_industry.ipynb) — A 股行业配置（含套利标记）
3. [`03_institution.ipynb`](03_institution.ipynb) — 机构行为（asof 与主体口径）
4. [`04_swing.ipynb`](04_swing.ipynb) — 波段研究 S1–S4

## 与 V1 关系

旧 [`examples/flow_monitor/`](../flow_monitor/) 保留只读兼容；主流程以本目录与 `run_alloc_v2.sh` 为准。
