"""Single command-line entry point that dispatches to ``btflow`` subcommands."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from . import __version__, defmap, lagcorr, match_labels, rgb_stack
from .plots import confidence_kde, heatmap, iou_boxplot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="btflow",
        description=(
            "CPU-based pipeline for mitosis detection and divergence analysis "
            "(bachelor-thesis-flow)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    rgb_stack.register(subparsers)
    defmap.register(subparsers)
    match_labels.register(subparsers)
    lagcorr.register(subparsers)
    confidence_kde.register(subparsers)
    heatmap.register(subparsers)
    iou_boxplot.register(subparsers)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.error("No handler registered for this subcommand.")
    return int(handler(args))


if __name__ == "__main__":
    sys.exit(main())
