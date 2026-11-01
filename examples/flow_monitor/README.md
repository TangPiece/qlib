# 全球到 A 股资金流向监测（V1，已由 V2 替代）

> **主流程请改用** [`../alloc_v2/`](../alloc_v2/) 与 `./scripts/run_alloc_v2.sh`。本目录保留只读兼容。

用免费公开数据回答五个问题：钱从哪来、到哪去、谁在买、为什么流、是否真实。

**价格上涨 ≠ 资金流入。持仓市值增加 ≠ 净买入。成交额 ≠ 净流入。**
跨资产矩阵只是并发配置变化的证据，不能证明「卖股票的钱买了黄金」。

## 刷新数据

```bash
./scripts/run_flow_monitor.sh catalog
./scripts/run_flow_monitor.sh all --start 2018-01-01
```

### 网络说明

- 东财 `push2his` 在部分网络会 `RemoteDisconnected`。股指/ETF 行情已改走**新浪**；行业改走**申万**；美元指数/CRB 暂关闭。
- 本机代理拦请求时，脚本默认直连（`FLOW_MONITOR_NO_PROXY=1`）。必须走代理：`FLOW_MONITOR_NO_PROXY=0`。
- 单次 HTTP 默认 15 秒超时（`FLOW_MONITOR_HTTP_TIMEOUT`）。SPDR 黄金持仓要翻多页，可能要一分钟。

结果写在 `data/flow_staging/`（不进 git）：

- `raw/long.parquet`：统一长表
- `diagnostics/changes.parquet`：窗口变化
- `diagnostics/flow_vs_price.parquet`：价格 vs 流量
- `diagnostics/allocation_evidence.parquet`：并发配置矩阵

## 看仪表盘

打开 [`dashboard.ipynb`](dashboard.ipynb)。笔记本只读 staging，不在单元格里打网。

## 证据等级

| 等级 | 含义 | 例子 |
|---|---|---|
| L1 | 直接流量 | ETF 份额、SPDR 黄金增减持、北向净买额（口径有变） |
| L2 | 持仓/规模 | 北向持股市值、公募规模快照；需价格调整 |
| L3 | 代理指标 | 指数、成交额、融资余额、新开户 |

北向数据：沪深港通披露规则已变，历史高频净买额不能当作当前完整实时外资。
