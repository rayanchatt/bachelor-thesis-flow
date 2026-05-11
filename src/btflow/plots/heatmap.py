"""2D KDE heatmap of bounding-box centers from a matched_boxes CSV."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _parse_center(box_str: str) -> tuple[float, float]:
    """Parse a ``"(x, y, w, h)"`` tuple-string and return the box center."""
    x, y, w, h = ast.literal_eval(box_str)
    return x + w / 2, y + h / 2


def plot_center_heatmap(
    csv_path: Path,
    out_path: Path,
    box_column: str = "green_box",
    cmap: str = "Oranges",
    title: str | None = None,
) -> Path:
    """Render a 2D KDE heatmap of bounding-box centers.

    Args:
        csv_path: ``matched_boxes.csv`` (or compatible) file with a column of
            stringified ``(x, y, w, h)`` tuples.
        out_path: Destination PNG path.
        box_column: Column name to read box tuples from. Defaults to ``green_box``.
        cmap: Matplotlib colormap name.
        title: Optional figure title.

    Returns:
        ``out_path`` for chaining.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    centers = df[box_column].apply(_parse_center).tolist()
    centers_df = pd.DataFrame(centers, columns=["x_center", "y_center"])

    plt.figure(figsize=(7, 7))
    sns.kdeplot(
        data=centers_df,
        x="x_center",
        y="y_center",
        levels=10,
        cmap=cmap,
        fill=True,
        alpha=0.6,
        thresh=0.05,
    )
    plt.title(title or "Heatmap of bounding-box centers", fontsize=14)
    plt.xlabel("x center (pixels)")
    plt.ylabel("y center (pixels)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "heatmap",
        help="2D KDE heatmap of bounding-box centers from a matched_boxes CSV.",
    )
    p.add_argument("--csv", required=True, type=Path, dest="csv_path")
    p.add_argument("--out", required=True, type=Path, dest="out_path")
    p.add_argument("--box-column", default="green_box")
    p.add_argument("--cmap", default="Oranges")
    p.add_argument("--title", default=None)
    p.set_defaults(_handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    plot_center_heatmap(
        csv_path=args.csv_path,
        out_path=args.out_path,
        box_column=args.box_column,
        cmap=args.cmap,
        title=args.title,
    )
    return 0
