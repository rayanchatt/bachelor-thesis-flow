"""IoU distribution box plot for one or more matched_boxes CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def _load_iou(csv_path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    iou_col = next(c for c in df.columns if c.lower().startswith("iou"))
    return pd.DataFrame({"IoU": df[iou_col].astype(float).dropna(), "Dataset": label})


def plot_iou_boxplot(
    csvs: list[tuple[Path, str]],
    out_path: Path,
    palette: list[str] | None = None,
    title: str = "IoU distribution of matched bounding boxes",
) -> Path:
    """Render a boxplot comparing IoU distributions across one or more datasets.

    Args:
        csvs: List of ``(csv_path, label)`` pairs. Each CSV must expose a column
            whose name (case-insensitively) starts with ``iou``.
        out_path: Destination PNG path.
        palette: Optional list of hex colors (length must match number of datasets).
        title: Figure title.

    Returns:
        ``out_path`` for chaining.
    """
    frames = [_load_iou(p, label) for p, label in csvs]
    df = pd.concat(frames, ignore_index=True)

    plt.figure(figsize=(5.5, 4))
    sns.boxplot(
        x="Dataset",
        y="IoU",
        data=df,
        palette=palette or ["#5ab4ac", "#d8b365"],
        width=0.5,
        showfliers=False,
    )
    sns.stripplot(x="Dataset", y="IoU", data=df, color="k", alpha=0.35, size=3, jitter=0.25)
    plt.ylim(0, 1)
    plt.ylabel("IoU")
    plt.xlabel("")
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()
    return out_path


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "iou-boxplot",
        help="Box plot of IoU distributions across one or more matched_boxes CSVs.",
    )
    p.add_argument(
        "--csv",
        action="append",
        required=True,
        metavar="PATH:LABEL",
        help="Repeat for each dataset. Format 'path/to.csv:Label'.",
    )
    p.add_argument("--out", required=True, type=Path, dest="out_path")
    p.add_argument("--title", default="IoU distribution of matched bounding boxes")
    p.set_defaults(_handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    csvs: list[tuple[Path, str]] = []
    for spec in args.csv:
        if ":" not in spec:
            raise SystemExit(f"--csv expects 'path:label', got: {spec!r}")
        path, label = spec.rsplit(":", 1)
        csvs.append((Path(path), label))
    plot_iou_boxplot(csvs=csvs, out_path=args.out_path, title=args.title)
    return 0
