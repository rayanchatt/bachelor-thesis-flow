"""KDE plot of YOLO detection confidence, grouped by z-layer."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

_PRED_PAT = re.compile(r"predictions_(\d)_(\d+)\.csv")


def plot_confidence_kde(
    predictions_dir: Path,
    dataset_suffix: str,
    out_path: Path,
    title: str | None = None,
) -> Path:
    """Render a KDE plot of detection confidence over all predictions_X_<suffix>.csv files.

    Args:
        predictions_dir: Root directory that is searched recursively for
            ``predictions_<layer>_<suffix>.csv`` files (YOLOv5 output convention).
        dataset_suffix: Dataset suffix (e.g. ``"6"`` or ``"8"``) that filters files.
        out_path: Destination PNG path.
        title: Optional figure title; a sensible default is generated otherwise.

    Returns:
        ``out_path`` for chaining.
    """
    frames: list[pd.DataFrame] = []
    pattern = f"predictions_*_{dataset_suffix}.csv"
    for fn in predictions_dir.rglob(pattern):
        m = _PRED_PAT.search(fn.name)
        if not m or m.group(2) != dataset_suffix:
            continue
        layer = f"Z{m.group(1)}"
        df = pd.read_csv(fn, usecols=["Confidence"])
        df["Layer"] = layer
        frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No predictions_*_{dataset_suffix}.csv files found under {predictions_dir}."
        )

    conf = pd.concat(frames, ignore_index=True)

    plt.figure(figsize=(6, 4))
    sns.kdeplot(data=conf, x="Confidence", hue="Layer", fill=True, alpha=0.4, bw_adjust=1.1)
    plt.xlim(0, 1)
    plt.title(title or f"Confidence distribution: {dataset_suffix}-layer dataset")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "confidence-kde",
        help="KDE plot of YOLO detection confidence grouped by z-layer.",
    )
    p.add_argument("--predictions-dir", required=True, type=Path)
    p.add_argument("--dataset-suffix", required=True, help="Dataset suffix, e.g. '6' or '8'.")
    p.add_argument("--out", required=True, type=Path, help="Output PNG path.")
    p.add_argument("--title", default=None)
    p.set_defaults(_handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    plot_confidence_kde(
        predictions_dir=args.predictions_dir,
        dataset_suffix=args.dataset_suffix,
        out_path=args.out,
        title=args.title,
    )
    return 0
