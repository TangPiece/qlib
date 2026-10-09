#!/usr/bin/env python3
"""Export Plotly graphs from an existing qrun recorder (no retrain).

Run from examples/:
  ../.venv/bin/python export_last_run_graphs.py

Opens HTML under examples/last_run_plots/.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.contrib.report import analysis_position
from qlib.workflow import R

EXPERIMENT_NAME = "rolling_lgb_7y2y_20260930"
RECORDER_ID = "5c27069aea2449e0ba47f5dda9372fb8"
OUT_DIR = Path(__file__).resolve().parent / "last_run_plots"


def _write_figs(figs, stem: str) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, fig in enumerate(figs):
        path = OUT_DIR / (f"{stem}.html" if len(figs) == 1 else f"{stem}_{i}.html")
        fig.write_html(str(path), include_plotlyjs="cdn")
        paths.append(path)
        print("wrote", path)
    return paths


def main() -> None:
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    recorder = R.get_recorder(recorder_id=RECORDER_ID, experiment_name=EXPERIMENT_NAME)
    print(recorder)

    pred_df = recorder.load_object("pred.pkl")
    label_df = recorder.load_object("label.pkl")
    report_normal_df = recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    analysis_df = recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")

    report_figs = analysis_position.report_graph(report_normal_df, show_notebook=False)
    _write_figs(list(report_figs), "report")

    risk_figs = analysis_position.risk_analysis_graph(
        analysis_df, report_normal_df, show_notebook=False
    )
    _write_figs(list(risk_figs), "risk_analysis")

    label = label_df.copy()
    label.columns = ["label"]
    pred_label = pd.concat([label, pred_df], axis=1, sort=True).reindex(label.index)
    ic_figs = analysis_position.score_ic_graph(pred_label, show_notebook=False)
    _write_figs(list(ic_figs), "score_ic")

    print("DONE", OUT_DIR)


if __name__ == "__main__":
    main()
