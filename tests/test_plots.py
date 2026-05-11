"""Tests for the plot subcommands."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from btflow.plots.confidence_kde import plot_confidence_kde
from btflow.plots.heatmap import plot_center_heatmap
from btflow.plots.iou_boxplot import plot_iou_boxplot


def test_confidence_kde_writes_png(tmp_path: Path) -> None:
    preds = tmp_path / "predictions"
    preds.mkdir()
    pd.DataFrame({"Confidence": [0.6, 0.7, 0.8]}).to_csv(preds / "predictions_2_6.csv", index=False)
    pd.DataFrame({"Confidence": [0.5, 0.55, 0.65]}).to_csv(
        preds / "predictions_3_6.csv", index=False
    )
    out = tmp_path / "conf.png"
    plot_confidence_kde(predictions_dir=preds, dataset_suffix="6", out_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_heatmap_writes_png(tmp_path: Path) -> None:
    csv = tmp_path / "matched.csv"
    pd.DataFrame(
        {
            "green_box": ["(10, 10, 20, 20)", "(40, 40, 20, 20)", "(70, 70, 20, 20)"],
        }
    ).to_csv(csv, index=False)
    out = tmp_path / "hm.png"
    plot_center_heatmap(csv_path=csv, out_path=out)
    assert out.exists() and out.stat().st_size > 0


def test_iou_boxplot_writes_png(tmp_path: Path) -> None:
    csv_a = tmp_path / "a.csv"
    csv_b = tmp_path / "b.csv"
    pd.DataFrame({"iou": [0.5, 0.6, 0.7, 0.8]}).to_csv(csv_a, index=False)
    pd.DataFrame({"iou": [0.4, 0.55, 0.65]}).to_csv(csv_b, index=False)
    out = tmp_path / "iou.png"
    plot_iou_boxplot(csvs=[(csv_a, "A"), (csv_b, "B")], out_path=out)
    assert out.exists() and out.stat().st_size > 0
