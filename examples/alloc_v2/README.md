# A 股多层次资金配置研究 V2

四层数据架构 + B/C/D 分层研究 + 波段信号。只用免费官方 / AKShare / 新浪 / 申万 / CBOE / FRED / Binance 公开接口。

**份额变化 ≠ 净流入。北向成交额 ≠ 净买入。季报 ≠ 实时持仓。龙虎榜机构席位 ≠ 已确认公募/保险。商品涨 ≠ 相关股票利好。**

## V2.2-Free 边界

在 V2.1 之上增量接入：**VIX、DXY、Brent/WTI、上海银、BTC 现货、美实际利率、美联储资产负债表、中国 M1/M2/社融增量/PMI**。  
LME/COMEX 库存、CFTC、Baltic、SMM/Mysteel/Wind 等商业源仅登记为 `disabled`，见 [`V22_INDICATOR_CATALOG.md`](V22_INDICATOR_CATALOG.md)。  
完整商品产业链与四组模型对照留待授权后的 V2.2-C，不在本轮验收。

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

## 看图（本地研究站）

**看图入口：** [http://127.0.0.1:8766](http://127.0.0.1:8766)

```bash
./scripts/run_alloc_v2.sh serve
```

多页只读本地 `data/alloc_v2/`，不打外网。总览只放 KPI 与迷你图；专题页按叙事下钻；任意 `(dataset|factor_id, entity, field)` 走「数据目录」分页打开曲线和最近表格，不会把十万行一次塞进浏览器。

| 路径 | 职责 |
| --- | --- |
| `/` | 总览：环境是否配合、行业 Top5、S1/S4 |
| `/global.html` | 股指、利率、日频股债收益差、汇率、黄金 |
| `/commodity.html` | VIX/DXY/实际利率、原油/银/BTC |
| `/china.html` | 月频宏观（不伪装日频） |
| `/northbound.html` | 北向成交/持股，净买额仅对照 |
| `/industry.html` | 综合分、成交占比、热力 |
| `/etf.html` | ETF 主表与三类指标 |
| `/institution.html` | 龙虎榜、大宗、融资、公募季报 |
| `/signals.html` | S1–S4 |
| `/backtest.html` | IC / last252 / 多空 / 相关 |
| `/catalog.html` | 全量目录逃生舱 |

**如何把各图串起来看（必读）：** [`CHART_READING.md`](CHART_READING.md)

读图顺序：先总览，再进专题。环境（全球/利率/VIX·DXY·实际利率）→ 商品与 BTC 价格代理 → 北向通道 → 行业强度/成交 → ETF 份额与估算配置 → S1/S2/S4 门闩 → 回测置信度。上层不成立时，下层亮点只观察、不当进攻依据。Jupyter [`dashboard.ipynb`](dashboard.ipynb) 仍保留作离线对照。

回测读法：红柱/`factor_missing`=未建成；IC≈0=有结果但无稳定预测力。同时看全样本与近一年（共同日历），并核对综合分与成交占比相关是否再次贴死。上交所远端份额为周采样（`sparse_sample`），流因子近 180 日更可信；份额/NAV 衍生因子遵循 T+1 `available_at`。研究产物另含 `research/backtest_ic_by_year.parquet`、`research/backtest_factor_corr.parquet`。

## 专题报告（表为主）

每格区分 **事实 / 估算 / 代理 / 推断 / 信号**：

1. [`01_global.ipynb`](01_global.ipynb) — 全球与中国跨资产
2. [`02_industry.ipynb`](02_industry.ipynb) — A 股行业配置（含套利标记）
3. [`03_institution.ipynb`](03_institution.ipynb) — 机构行为（asof 与主体口径）
4. [`04_swing.ipynb`](04_swing.ipynb) — 波段研究 S1–S4

## 与 V1 关系

旧 [`examples/flow_monitor/`](../flow_monitor/) 保留只读兼容；主流程以本目录与 `run_alloc_v2.sh` 为准。
