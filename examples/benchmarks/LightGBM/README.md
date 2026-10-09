# LightGBM
* 代码：[https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)
* 论文：LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
  [PDF](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

# 配置说明

`workflow_config_lightgbm_multi_freq.yaml`
- 使用多频率数据源做日频预测。

`workflow_config_lightgbm_Alpha158_rolling_7y2y.yaml`
- 滚动重训：7 年训练 + 1 年验证，再预测约 2 年窗口；首窗自 **2008-01-01** 起。
- 通过 [`rolling_7y2y.py`](rolling_7y2y.py) 运行；`end_time` 需与本地 `cn_data` 对齐（见下文「每日更新」）。

---

# 滚动训练（rolling_7y2y）

脚本：[`rolling_7y2y.py`](rolling_7y2y.py)  
默认配置：[`workflow_config_lightgbm_Alpha158_rolling_7y2y.yaml`](workflow_config_lightgbm_Alpha158_rolling_7y2y.yaml)

## 前置条件

1. 已安装本仓库 qlib，且能正常 `import qlib`
2. 本地数据 `~/.qlib/qlib_data/cn_data` 覆盖到 yaml 中的 `end_time`
3. 建议先跑一遍数据更新脚本，保证日历与配置一致：

```bash
python3 examples/benchmarks/LightGBM/update_cn_data_daily.py
```

## 执行

在 qlib 仓库根目录：

```bash
# 推荐：使用虚拟环境中的 python
.venv/bin/python examples/benchmarks/LightGBM/rolling_7y2y.py \
  --conf_path=examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_rolling_7y2y.yaml \
  --exp_name=rolling_lgb_7y2y_20261009 \
  --horizon=1 \
  run
```

说明：

- `run`：启动完整滚动训练（fire 子命令）
- `--exp_name`：MLflow / 实验名；**每次重跑请换新名字**，或先清理冲突的 `mlruns` 实验
- `--horizon`：标签预测期（交易日）；会参与 `trunc_days=horizon+1`，避免标签泄漏
- `--conf_path`：可省略，默认即上面的 rolling yaml
- `--step`：可选；不传则按约 2 年交易日数自动计算滑动步长

## 典型流程

```bash
# 1) 更新数据与 yaml 结束日
python3 examples/benchmarks/LightGBM/update_cn_data_daily.py

# 2) 滚动训练
.venv/bin/python examples/benchmarks/LightGBM/rolling_7y2y.py \
  --exp_name=rolling_lgb_7y2y_$(date +%Y%m%d) \
  --horizon=1 \
  run
```

---

# 持仓导出与 HTML 日度查看

回测结束后，可用 [`export_positions_view.py`](export_positions_view.py) 从 MLflow 实验读取
`positions_normal_1day.pkl`，导出：

1. **parquet**（`datetime` 为 `YYYY-MM-DD` 字符串）：长表含 `action ∈ {hold, buy, sell}`
2. **自包含 HTML**：按日切换查看「今日持仓 / 买入 / 卖出」

```bash
.venv/bin/python examples/benchmarks/LightGBM/export_positions_view.py \
  --exp_name=rolling_lgb_7y2y_20261009
```

可选参数：

- `--recorder_id`：指定 recorder；默认取该实验最新 `FINISHED` recorder
- `--out_dir`：输出目录（默认 `examples/benchmarks/LightGBM/data/`）

产物示例（目录已被 `.gitignore` 忽略）：

- `data/rolling_lgb_7y2y_20261009_positions_1day.parquet`
- `data/rolling_lgb_7y2y_20261009_positions_view.html`

用浏览器打开 HTML 即可；左右方向键可切换交易日。买卖标签由相邻交易日持仓集合差集推算。

---

# 每日更新：cn_data 与 rolling 结束日

本地 Qlib 数据目录：`~/.qlib/qlib_data/cn_data`  
数据来源为社区打包 bin：[chenditc/investment_data](https://github.com/chenditc/investment_data/releases/latest)

使用 [`update_cn_data_daily.py`](update_cn_data_daily.py) 刷新数据，并把两份 rolling yaml
（`..._rolling_7y2y.yaml` 与 `..._ind_weight.yaml`）中的**结束日**同步为
`calendars/day.txt` 的最新交易日。

**只改结束日**：不改写任何 `start_time`、`fit_start_time`，也不改 train/valid 区间或 `segments.test` 左端。

## 脚本做什么

1. 从 investment_data 最新 GitHub Release 下载 `qlib_bin.tar.gz`
2. 将现有 `cn_data` 备份为 `cn_data.bak_YYYYMMDD_HHMMSS`（默认保留 2 份）
3. 解压覆盖到 `~/.qlib/qlib_data/cn_data`
4. 默认同步两份 rolling yaml 中三处结束日：
   - `data_handler_config.end_time`
   - `backtest.end_time`
   - `segments.test` 右端点

## 用法

在 qlib 仓库根目录执行：

```bash
# 完整每日更新（推荐；默认同步两份 rolling yaml）
python3 examples/benchmarks/LightGBM/update_cn_data_daily.py

# 仅预览，不下载、不写盘
python3 examples/benchmarks/LightGBM/update_cn_data_daily.py --dry-run

# 不下载，仅按本地日历改写 yaml
python3 examples/benchmarks/LightGBM/update_cn_data_daily.py --skip-download
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--qlib-dir` | `~/.qlib/qlib_data/cn_data` | 数据目录 |
| `--config` | 两份 `rolling_7y2y*.yaml` | 要同步结束日的 YAML（可传多个路径） |
| `--keep-backups N` | `2` | 保留最近 N 个 `cn_data.bak_*` |
| `--skip-download` | 关闭 | 跳过下载，只同步 yaml |
| `--dry-run` | 关闭 | 只打印将要执行的操作 |

## 注意

- 压缩包约 500MB+；网络不稳定可能中途失败，重新跑一遍即可。
- 官方 Yahoo 离线包与本社区 bin 不能增量混用；请始终用本脚本（或重新解压 release 包）整包替换。
- 其它非 rolling yaml（momentum / chatgpt / zhipu 等）默认不改；如需更新，请传 `--config <路径>`。

---

# CSI300 申万一级行业 PIT 区间表（AKShare）

脚本：[`build_csi300_industry_map.py`](build_csi300_industry_map.py)

为本地 `cn_data` 中 **沪深300 历史成分** 生成申万一级行业归属区间表（含生效/失效日），供训练或归因侧按交易日做 PIT 查询。

## 前置条件

1. 已安装 `akshare`、`pyarrow`、`xlrd`（读取申万 xls）
2. 本地存在 `~/.qlib/qlib_data/cn_data/instruments/csi300.txt` 与 `calendars/day.txt`

```bash
.venv/bin/pip install akshare pyarrow xlrd
```

## 用法

在 qlib 仓库根目录：

```bash
.venv/bin/python examples/benchmarks/LightGBM/build_csi300_industry_map.py \
  --qlib-dir ~/.qlib/qlib_data/cn_data \
  --out-dir examples/benchmarks/LightGBM/data
```

可选：`--open-end-as-cal-end` 将仍有效区间的结束日写成日历末日（默认开放端为 `9999-12-31`，再与 CSI300 在册区间求交）。

## 产出

| 文件 | 说明 |
|------|------|
| `data/csi300_sw_l1_industry_intervals.parquet` | 区间表 |
| `data/csi300_sw_l1_industry_meta.json` | 覆盖率、接口名、PIT 规则与局限 |

区间字段：`instrument`, `industry_code`, `industry_name`, `start_date`, `end_date`, `standard`, `standard_version`, `source`, `asof`。

## PIT 规则与局限

- 数据源：**仅 AKShare** 主路径 `stock_industry_clf_hist_sw`（申万宏源 `StockClassifyUse_stock.xls`）；若 SSL 失败则按同一 URL 直连下载。
- 工作簿提供个股行业**计入日期**、通常无剔除日：按计入日排序，下一段计入日前一日作为上一段 `end_date`。
- 6 位申万内部行业代码按前缀映射到一级指数代码（`801xxx`）；含部分 SW2014 历史前缀。
- 再与 `csi300.txt` 在册区间及本地日历求交；**不会**用今日行业无日期回填整段历史。
- 细节见 meta 中的 `pit_inference_rule` / `limitations`。

生成物默认被仓库 `.gitignore` 忽略（`examples/benchmarks/LightGBM/data/`）。

---

# 行业入选数量加权策略

策略类：`IndustryCountWeightStrategy`（继承 `TopkDropoutStrategy`）  
配置：[`workflow_config_lightgbm_Alpha158_rolling_7y2y_ind_weight.yaml`](workflow_config_lightgbm_Alpha158_rolling_7y2y_ind_weight.yaml)

**调仓**：与 `TopkDropoutStrategy` 相同（`topk` / `n_drop`）；行业**不**改变买卖名单。  
**权重**：仅对新买入股票分配卖出后现金时，按行业入选数量加权：

\[
v_i = C\cdot R\cdot\frac{n_i^{\alpha}}{\sum_{j\in\mathrm{buy}} n_j^{\alpha}}
\]

其中 \(n_i\) 在「卖出后持仓 ∪ 新买」上按申万一级统计；已持仓不做全仓再平衡。

- \(\alpha=0\)：买入侧等权（同原 TopkDropout）  
- \(\alpha=0.5\)（默认）：温和偏向热门行业  
- \(\alpha=1\)：买入金额与行业入选数量成正比  

行业集中可能放大行业风险，建议与原 rolling yaml 对照验证。

## 前置

1. 已生成行业区间表（见上一节 `build_csi300_industry_map.py`）
2. yaml 中 `industry_map_path` 指向该 parquet（默认相对仓库根目录）

## 用法

```bash
.venv/bin/python examples/benchmarks/LightGBM/rolling_7y2y.py \
  --conf_path=examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_rolling_7y2y_ind_weight.yaml \
  --exp_name=rolling_lgb_7y2y_ind_weight_$(date +%Y%m%d) \
  --horizon=1 \
  run
```

行业查询：优先命中 `start_date ≤ trade_date ≤ end_date`；若 `trade_date` 晚于全部 `end_date`，取最近一段历史行业；更早或无数据则按单股哨兵行业计（\(n_i=1\)）。
