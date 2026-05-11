"""Integration test for the lag-correlation analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from btflow.lagcorr import run_lagcorr


def test_lagcorr_produces_expected_artifacts(
    synthetic_defmap_stack: Path,
    synthetic_yolo_labels: Path,
    tmp_path: Path,
) -> None:
    out = tmp_path / "lagcorr"
    csv_path = run_lagcorr(
        defmap_path=synthetic_defmap_stack,
        label_dir=synthetic_yolo_labels,
        out_dir=out,
        metric="div",
        conf_min=0.0,
    )
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert set(df.columns) >= {"frame", "lag", "conf", "value", "scale"}
    assert (out / "lag_corr_stats_Z1_all.csv").exists()
    assert (out / "lag_div_curve_Z1_all.png").exists()
    assert (out / "lag_div_curve_avg_Z1_all.png").exists()
    assert (out / "lag_corr_significant_div_Z1_all.png").exists()
