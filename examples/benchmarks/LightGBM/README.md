# LightGBM
* 代码：[https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)
* 论文：LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
  [PDF](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

# 配置说明

`workflow_config_lightgbm_multi_freq.yaml`
- 使用多频率数据源做日频预测。

`workflow_config_lightgbm_Alpha158_rolling_7y1y.yaml`
- 命名约定：`NyMy` = N 年训练 + M 年验证（滚动滑动步长仍约 2 年交易日，不写入文件名）。
- 滚动重训：7 年训练 + 1 年验证，再预测约 2 年窗口；首窗自 **2008-01-01** 起。
- 通过 [`rolling_7y1y.py`](rolling_7y1y.py) 运行；`end_time` 由每日更新脚本写成**当天日期**（见下文「每日更新」）；本地 `cn_data` 可能略滞后。

---

# 本地控制台（页面操作）

静态 HTML + 极简本地 HTTP 服务，用于在浏览器里改结束日、更新数据、跑滚动训练、导出持仓 HTML，并浏览历史结果。

```bash
# 在 qlib 仓库根目录，建议用 .venv
.venv/bin/python examples/benchmarks/LightGBM/console/server.py
# 浏览器打开 http://127.0.0.1:8765
```

页面会调用本目录现有脚本（`update_cn_data_daily.py` / `rolling_7y1y.py` / `export_positions_view.py`）；历史列表优先 `data/*_positions.json`，经共用页 [`console/positions_view.html`](console/positions_view.html) 加载；旧的 `*_positions_view.html` 仍可直开。同时只跑一个后台任务，日志在页面底部轮询显示。

---

# 滚动训练（rolling_7y1y）

脚本：[`rolling_7y1y.py`](rolling_7y1y.py)  
默认配置：[`workflow_config_lightgbm_Alpha158_rolling_7y1y.yaml`](workflow_config_lightgbm_Alpha158_rolling_7y1y.yaml)

## 前置条件

1. 已安装本仓库 qlib，且能正常 `import qlib`
2. 本地数据 `~/.qlib/qlib_data/cn_data` 尽量新；yaml `end_time` 可能晚于日历末日
3. 建议先跑一遍数据更新脚本（同步结束日为当天，并按需刷新数据）：

```bash
python3 examples/benchmarks/LightGBM/update_cn_data_daily.py
```

## 执行

在 qlib 仓库根目录：

```bash
# 推荐：使用虚拟环境中的 python
.venv/bin/python examples/benchmarks/LightGBM/rolling_7y1y.py \
  --conf_path=examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_rolling_7y1y.yaml \
  --exp_name=rolling_lgb_7y1y_20261009 \
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
.venv/bin/python examples/benchmarks/LightGBM/rolling_7y1y.py \
  --exp_name=rolling_lgb_7y1y_$(date +%Y%m%d) \
  --horizon=1 \
  run
```

---

# 持仓导出与日度查看

回测结束后，可用 [`export_positions_view.py`](export_positions_view.py) 从 MLflow 实验读取
持仓与分析产物，导出 **`data/{exp_name}_positions.json`**（指标 + 分日持仓/买卖 + **申万一级行业权重占比**）。  
UI 为共用页 [`console/positions_view.html`](console/positions_view.html)，**不再**为每个实验生成巨型自包含 HTML。

行业占比依赖 [`build_csi300_industry_map.py`](build_csi300_industry_map.py) 生成的
`data/csi300_sw_l1_industry_intervals.parquet`；缺失时导出仍成功，但无行业分布。

```bash
.venv/bin/python examples/benchmarks/LightGBM/export_positions_view.py \
  --exp_name=rolling_lgb_7y1y_20261009

# 需先启动控制台服务，再打开（file:// 下 fetch 无法加载本地 JSON）：
# http://127.0.0.1:8765/positions_view.html?data=/data/rolling_lgb_7y1y_20261009_positions.json
```

可选参数：

- `--recorder_id`：指定 recorder；默认取该实验最新 `FINISHED` recorder
- `--out_dir`：输出目录（默认 `examples/benchmarks/LightGBM/data/`）
- `--name_map`：股票名称缓存 JSON（默认 `data/stock_name_map.json`；缺失时尝试用 akshare 生成）
- `--industry_map`：申万一级 PIT 区间 parquet（默认 `data/csi300_sw_l1_industry_intervals.parquet`）

产物示例（目录已被 `.gitignore` 忽略）：

- `data/rolling_lgb_7y1y_20261009_positions.json`
- `data/stock_name_map.json`

默认日期为**当天**（非交易日则落到 ≤ 当天的最近交易日）。左右方向键可切换交易日。买卖标签由相邻交易日持仓集合差集推算；买入日为该票进入持仓的**回测成交日**（见下节）。查看器按日展示申万一级行业权重占比（需重新导出 JSON 后才有 `industries` 字段）。  
「复权价」为回测用的 qlib `$close`；「现价」为 `$close/$factor`（近似交易所名义收盘价）。重新导出后才有 `raw_price`。

---

# 信号日 / 成交日 / 实盘节奏

滚动配置（如 `workflow_config_lightgbm_Alpha158_rolling_7y1y.yaml`）使用 `TopkDropoutStrategy` + `deal_price: close`。qlib 在成交日 **T** 读取 **T-1** 的预测分数（`get_step_time(..., shift=1)`），再按 **T 日收盘价**撮合。持仓页上的「今日买入/卖出」是对相邻两日仓位做差，标签是 **回测成交日 T**，不是数据包到达日，也不是「刚看到这天数据才该下单」的日子。

日频 `cn_data` 的正常节奏：**要等 T 日收盘后，才有含 T 日的数据包。** 这不改变策略时点。

| 概念 | 时刻 | 需要的数据 | 做什么 |
|------|------|------------|--------|
| 信号就绪 | **T-1 收盘之后**（包已含 T-1） | T-1 及更早行情 | 算出分数 → 得到 **将在 T 日执行** 的买卖名单 |
| 实盘执行 | **T 日盘中 / 近收盘** | **不需要** T 日数据包 | 按名单下单（对齐 `deal_price: close`） |
| 回测复盘 | **T 收盘之后**（包已含 T） | T 日 `$close` 等 | 才能在 positions 页把「T 日买卖」撮合出来 |

**实盘 SOP（以执行日 = 10-12 为例）**

1. **10-11 收盘后**：更新数据，确认包已含 **10-11**；跑滚动/推理，得到基于 10-11 信号的调仓名单。
2. **10-12 盘中（近收盘更贴回测）**：按该名单买卖。**不要等 10-12 包。**
3. **10-12 收盘后**：更新含 10-12 的包；重跑用于 **核对回测** 与生成 **10-13 名单**。
4. **10-13**：执行的是「10-12 信号 → 10-13 成交」的单；**不要**把页面上「日期=10-12」的买卖当成 10-13 才下的补单。

**为何缺 T 包时页面上 T 日买卖为 0**

yaml `end_time` 常被写成「今天」，日历上可能已有 T，但本地尚无 T 的 `$close` 时，回测无法按 close 撮合，仓位结转 → 持仓页显示买入/卖出 = 0、全是 hold。这 **不等于**「策略认为当日无调仓」，只是复盘缺数。等 T 包到位后重跑，同一成交日 T 才会出现真实买卖。实盘信号在 **T-1 包就绪后**就已可定，不必等这份复盘页。

若数据源经常 **T+1 早上才发 T 日包**（比「T 收盘后当晚」更晚）：仍用 **最新已完整的信号日** 指导 **下一交易日** 执行；相对 `deal_price: close` 回测会有额外滑点，属数据源滞后，不是策略再晚一天。

## 持仓页「预测」按钮（实盘信号）

[`console/positions_view.html`](console/positions_view.html) 上的 **预测** 会调用 [`predict_live_signal.py`](predict_live_signal.py)：

1. 用 **`OnlineToolR`** 把当前实验中带 `params.pkl`+`pred.pkl` 的最新窗标记为 online，并尽量 `update_online_pred()` 续写到本地日历末日；
2. 取最新 **信号日** 分数，相对末日持仓套用与 yaml 一致的 TopkDropout（`topk=10`, `n_drop=1`）；
3. 写出 `data/{exp_name}_live_signal.json`，页面展示 **信号日 / 建议执行日** 与买卖名单。

这与下方「成交日持仓/买入/卖出」回测复盘表分开：预测结果是给 **下一交易日执行** 用的，不是等包到齐后再补做已过去的成交日。

命令行等价：

```bash
.venv/bin/python examples/benchmarks/LightGBM/predict_live_signal.py \
  --exp_name=rolling_lgb_7y1y_20261013_1013
```

说明：完整 **OnlineManager**（`first_train` / `routine` / `RollingStrategy`）与现有 `rolling_7y1y` 并行且更重，**未**接入本控制台；日常实盘信号走 OnlineToolR 增量预测即可。

---

# 每日更新：cn_data 与 rolling 结束日

本地 Qlib 数据目录：`~/.qlib/qlib_data/cn_data`  
数据来源为社区打包 bin：[chenditc/investment_data](https://github.com/chenditc/investment_data/releases/latest)

使用 [`update_cn_data_daily.py`](update_cn_data_daily.py) 刷新数据，并把两份 rolling yaml
（`..._rolling_7y1y.yaml` 与 `..._ind_weight.yaml`）中的**结束日**同步为
**运行当天的日期**（本机 `YYYY-MM-DD`），不是 `calendars/day.txt` 末日。本地日历仍用于判断数据是否落后于 GitHub release；若今天晚于日历末日会打印 Warning。

**只改结束日**：不改写任何 `start_time`、`fit_start_time`，也不改 train/valid 区间或 `segments.test` 左端。

investment_data **只提供全量** `qlib_bin.tar.gz`（约 540MB），无按日增量包。脚本会先比对远程 latest release tag 与本地
`cn_data/.investment_data_release`（或日历末日）；已最新则**跳过下载**，只同步 yaml；落后再整包替换。

## 脚本做什么

1. 解析 investment_data 最新 GitHub Release tag；若本地 marker / 日历已覆盖则跳过下载
2. 需要更新时：下载 `qlib_bin.tar.gz`，将现有 `cn_data` 备份为 `cn_data.bak_YYYYMMDD_HHMMSS`（默认保留 2 份）
3. 解压覆盖到 `~/.qlib/qlib_data/cn_data`，写入 `.investment_data_release`
4. 默认同步两份 rolling yaml 中三处结束日为**今天**：
   - `data_handler_config.end_time`
   - `backtest.end_time`
   - `segments.test` 右端点

## 用法

在 qlib 仓库根目录执行：

```bash
# 完整每日更新（推荐；已最新则跳过下载，仍同步 yaml）
python3 examples/benchmarks/LightGBM/update_cn_data_daily.py

# 仅预览，不下载、不写盘
python3 examples/benchmarks/LightGBM/update_cn_data_daily.py --dry-run

# 不下载，仍把 yaml 结束日改成今天
python3 examples/benchmarks/LightGBM/update_cn_data_daily.py --skip-download

# 强制全量重新下载（忽略已最新判定）
python3 examples/benchmarks/LightGBM/update_cn_data_daily.py --force
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--qlib-dir` | `~/.qlib/qlib_data/cn_data` | 数据目录 |
| `--config` | 两份 `rolling_7y1y*.yaml` | 要同步结束日的 YAML（可传多个路径） |
| `--keep-backups N` | `2` | 保留最近 N 个 `cn_data.bak_*` |
| `--skip-download` | 关闭 | 跳过下载，只同步 yaml |
| `--force` | 关闭 | 强制全量下载，即使本地已是 latest |
| `--dry-run` | 关闭 | 只打印将要执行的操作 |

## 注意

- 压缩包约 500MB+；已最新时默认不再重复下载。网络不稳定导致失败时，可重新跑或加 `--force`。
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
配置：[`workflow_config_lightgbm_Alpha158_rolling_7y1y_ind_weight.yaml`](workflow_config_lightgbm_Alpha158_rolling_7y1y_ind_weight.yaml)

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
.venv/bin/python examples/benchmarks/LightGBM/rolling_7y1y.py \
  --conf_path=examples/benchmarks/LightGBM/workflow_config_lightgbm_Alpha158_rolling_7y1y_ind_weight.yaml \
  --exp_name=rolling_lgb_7y1y_ind_weight_$(date +%Y%m%d) \
  --horizon=1 \
  run
```

行业查询：优先命中 `start_date ≤ trade_date ≤ end_date`；若 `trade_date` 晚于全部 `end_date`，取最近一段历史行业；更早或无数据则按单股哨兵行业计（\(n_i=1\)）。
