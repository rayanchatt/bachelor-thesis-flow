"""Integration test for cross-z label matching."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from btflow.match_labels import match_labels


def test_match_labels_writes_csv_with_iou_column(tmp_path: Path) -> None:
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / "frame_Z2_001.png"), img)

    green_dir = tmp_path / "green"
    green_dir.mkdir()
    (green_dir / "frame_Z2_001.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n")

    red_dir = tmp_path / "red"
    red_dir.mkdir()
    # Same normalised box at the same position -> IoU = 1.0, well above threshold.
    (red_dir / "frame_Z3_001.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n")

    out = tmp_path / "out"
    csv_path = match_labels(
        image_dir=img_dir,
        label_dir_green=green_dir,
        label_dir_red=red_dir,
        output_dir=out,
    )

    df = pd.read_csv(csv_path)
    assert len(df) == 1
    assert df.iloc[0]["iou"] > 0.99
    assert (out / "frame_Z2_001.png").exists()
