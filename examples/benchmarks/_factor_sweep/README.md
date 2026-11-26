# Factor Sweep：全模型 × 五套因子单次训练

统一：`account=200000`，`topk=10`，`n_drop=1`。  
无 L2：`2008` 起；含 L2（`l2` / `mix`）：`2017–2026`。

```bash
# 1) 生成 yaml + manifest
.venv/bin/python examples/benchmarks/_factor_sweep/gen_configs.py

# 2) 先跑耗时少的树模型
.venv/bin/python examples/benchmarks/_factor_sweep/run_sweep.py --phase 1

# 3) 中等 / 深度
.venv/bin/python examples/benchmarks/_factor_sweep/run_sweep.py --phase 2
.venv/bin/python examples/benchmarks/_factor_sweep/run_sweep.py --phase 3

# 断点续跑 / 重试失败
.venv/bin/python examples/benchmarks/_factor_sweep/run_sweep.py --phase 1,2,3 --retry-failed
```

成功后会自动 `export_positions_view.py`，结果出现在控制台历史：`examples/console/data/runs/*_positions.json`。
