"""Regenerate sample figures from the synthetic fixtures.

Run via ``python scripts/regenerate_assets.py`` or ``make figures``.
Outputs land in ``assets/synthetic/`` so the real thesis figures in
``assets/`` (produced from actual microscopy data) are never overwritten.
The CI ``figures`` workflow runs this script weekly purely as a smoke
test that the pipeline still produces non-empty plots end to end.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from btflow.defmap import build_defmap_stack  # noqa: E402
from btflow.lagcorr import run_lagcorr  # noqa: E402
from btflow.plots.heatmap import plot_center_heatmap  # noqa: E402
from btflow.plots.iou_boxplot import plot_iou_boxplot  # noqa: E402

ASSETS = REPO_ROOT / "assets" / "synthetic"
ASSETS.mkdir(parents=True, exist_ok=True)


def _make_synthetic_zstack(directory: Path, n_frames: int = 20) -> None:
    rng = np.random.default_rng(seed=0)
    h, w = 64, 64
    yy, xx = np.mgrid[0:h, 0:w]
    for t in range(1, n_frames + 1):
        for z in (1, 2, 3):
            cx, cy = 16 + t, 32
            blob = 255 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 4.0**2))
            img = np.clip(blob + rng.normal(0, 5, size=(h, w)), 0, 255).astype(np.uint8)
            cv2.imwrite(str(directory / f"f_t{t:04d}_z{z:04d}.png"), img)


def _make_synthetic_labels(directory: Path, n_frames: int) -> None:
    for t in range(1, n_frames + 1):
        (directory / f"frame_{t:03d}.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        rgb = tmp / "rgb"
        rgb.mkdir()
        labels = tmp / "labels"
        labels.mkdir()
        defmap_out = tmp / "defmap"

        n = 20
        _make_synthetic_zstack(rgb, n_frames=n)
        _make_synthetic_labels(labels, n_frames=n)

        stacks = build_defmap_stack(rgb, defmap_out, metric="div")
        np_path = defmap_out / "defmap_stack_Z1_div.npy"
        np.save(np_path, stacks[1])

        lagcorr_out = tmp / "lagcorr"
        run_lagcorr(
            defmap_path=np_path,
            label_dir=labels,
            out_dir=lagcorr_out,
            metric="div",
            conf_min=0.0,
        )
        shutil.copy(lagcorr_out / "lag_div_curve_Z1_all.png", ASSETS / "lag_div_curve_Z1_all.png")

        rng = np.random.default_rng(0)
        centers = [(int(x), int(y)) for x, y in rng.uniform(10, 90, size=(80, 2))]
        df = pd.DataFrame({"green_box": [f"({x}, {y}, 10, 10)" for x, y in centers]})
        csv_path = tmp / "matched_synth.csv"
        df.to_csv(csv_path, index=False)
        plot_center_heatmap(csv_path=csv_path, out_path=ASSETS / "heatmap_6er.png")

        iou_csv_a = tmp / "iou_a.csv"
        iou_csv_b = tmp / "iou_b.csv"
        pd.DataFrame({"iou": rng.beta(5, 2, size=120)}).to_csv(iou_csv_a, index=False)
        pd.DataFrame({"iou": rng.beta(4, 2, size=120)}).to_csv(iou_csv_b, index=False)
        plot_iou_boxplot(
            csvs=[(iou_csv_a, "8-layer (Z1-Z2)"), (iou_csv_b, "6-layer (Z2-Z3)")],
            out_path=ASSETS / "iou_boxplot.png",
        )

    print(f"Regenerated assets in {ASSETS}")


if __name__ == "__main__":
    main()
