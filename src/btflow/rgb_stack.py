"""Build RGB frames from three z-slices per time point.

Expects, for every frame index, exactly three input files
``..._t<NNNN>_z0001.png``, ``..._z0002.png``, ``..._z0003.png`` and writes
``frame_<NNN>.png`` to the output directory with channels ``(B, G, R) = (Z1, Z2, Z3)``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .io import imread, parse_frame_filename


def build_rgb_stack(input_dir: Path, output_dir: Path) -> int:
    """Build an RGB stack from per-z grayscale frames.

    Args:
        input_dir: Directory containing ``..._t####_z000{1,2,3}.png`` files.
        output_dir: Directory the merged ``frame_###.png`` files are written to.
            Created if missing.

    Returns:
        Number of RGB frames written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[int, dict[int, Path]] = defaultdict(dict)
    for fname in input_dir.iterdir():
        if not fname.is_file():
            continue
        parsed = parse_frame_filename(fname.name)
        if parsed is None:
            continue
        t, z = parsed
        groups[t][z] = fname

    written = 0
    for frame_idx in sorted(groups):
        files = groups[frame_idx]
        if not all(z in files for z in (1, 2, 3)):
            print(f"Frame {frame_idx:03d}: z-channels missing, skipped")
            continue

        z1 = imread(files[1], cv2.IMREAD_GRAYSCALE)
        z2 = imread(files[2], cv2.IMREAD_GRAYSCALE)
        z3 = imread(files[3], cv2.IMREAD_GRAYSCALE)

        def to_u8(img: np.ndarray) -> np.ndarray:
            return img if img.dtype == np.uint8 else cv2.convertScaleAbs(img)

        z1, z2, z3 = (to_u8(z) for z in (z1, z2, z3))
        rgb = cv2.merge([z1, z2, z3])

        out_name = f"frame_{frame_idx:03d}.png"
        cv2.imwrite(str(output_dir / out_name), rgb)
        written += 1
        print(f"wrote: {out_name}")

    print(f"Done — {written} RGB frames written to {output_dir}")
    return written


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "rgb-stack",
        help="Merge per-z grayscale frames into RGB frames.",
        description=(
            "For every frame index ``t``, merge files ``..._t####_z000{1,2,3}.png`` "
            "into a single RGB image with channels (B, G, R) = (Z1, Z2, Z3)."
        ),
    )
    p.add_argument("--input-dir", required=True, type=Path, help="Directory with raw frames.")
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory the merged frame_###.png files are written to.",
    )
    p.set_defaults(_handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    build_rgb_stack(args.input_dir, args.output_dir)
    return 0
