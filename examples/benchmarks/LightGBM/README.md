# LightGBM
* 代码：[https://github.com/microsoft/LightGBM](https://github.com/microsoft/LightGBM)
* 论文：LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
  [PDF](https://proceedings.neurips.cc/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf)

# 配置说明

`workflow_config_lightgbm_multi_freq.yaml`
- 使用多频率数据源做日频预测。

`workflow_config_lightgbm_Alpha158_rolling_7y2y.yaml`
- 滚动重训：7 年训练 + 1 年验证，再预测约 2 年窗口。
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

# 每日更新：cn_data 与 rolling 结束日

本地 Qlib 数据目录：`~/.qlib/qlib_data/cn_data`  
数据来源为社区打包 bin：[chenditc/investment_data](https://github.com/chenditc/investment_data/releases/latest)

使用 [`update_cn_data_daily.py`](update_cn_data_daily.py) 刷新数据，并把
`workflow_config_lightgbm_Alpha158_rolling_7y2y.yaml` 中的结束日同步为
`calendars/day.txt` 的最新交易日。

## 脚本做什么

1. 从 investment_data 最新 GitHub Release 下载 `qlib_bin.tar.gz`
2. 将现有 `cn_data` 备份为 `cn_data.bak_YYYYMMDD_HHMMSS`（默认保留 2 份）
3. 解压覆盖到 `~/.qlib/qlib_data/cn_data`
4. 更新 rolling yaml 中三处结束日：
   - `data_handler_config.end_time`
   - `backtest.end_time`
   - `segments.test` 右端点

## 用法

在 qlib 仓库根目录执行：

```bash
# 完整每日更新（推荐）
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
| `--config` | `workflow_config_lightgbm_Alpha158_rolling_7y2y.yaml` | 要更新的 YAML |
| `--keep-backups N` | `2` | 保留最近 N 个 `cn_data.bak_*` |
| `--skip-download` | 关闭 | 跳过下载，只同步 yaml |
| `--dry-run` | 关闭 | 只打印将要执行的操作 |

## 注意

- 压缩包约 500MB+；网络不稳定可能中途失败，重新跑一遍即可。
- 官方 Yahoo 离线包与本社区 bin 不能增量混用；请始终用本脚本（或重新解压 release 包）整包替换。
- 默认**不会**改其它 yaml（momentum / chatgpt / zhipu 等）；如需更新，请传 `--config <路径>`。
