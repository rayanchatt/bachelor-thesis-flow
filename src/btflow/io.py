"""I/O helpers: filename parsing, YOLO label loading, frame grouping."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from ._types import BBox

# Matches filenames of the form ``..._t<digits>_z<digits>.<ext>``
RAW_FRAME_PATTERN = re.compile(r".+?_t(\d+)_z(\d+)\.(png|tif|tiff|jpg|jpeg)$", re.IGNORECASE)


def imread(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """Read an image from disk, raising ``FileNotFoundError`` on failure.

    ``cv2.imread`` silently returns ``None`` for missing or unreadable files,
    which makes downstream code hard to reason about. This wrapper guarantees
    a non-``None`` ``ndarray`` return.
    """
    img = cv2.imread(str(path), flags)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def parse_frame_filename(name: str) -> tuple[int, int] | None:
    """Extract ``(t, z)`` indices from a raw frame filename, or ``None`` on no match."""
    m = RAW_FRAME_PATTERN.match(name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def group_frames_by_z_t(directory: Path) -> dict[int, dict[int, Path]]:
    """Group all raw frame files in ``directory`` into a ``{z: {t: path}}`` mapping."""
    out: dict[int, dict[int, Path]] = {}
    for f in sorted(directory.iterdir()):
        if not f.is_file():
            continue
        parsed = parse_frame_filename(f.name)
        if parsed is None:
            continue
        t, z = parsed
        out.setdefault(z, {})[t] = f
    return out


def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[BBox]:
    """Load YOLO-format label file and convert normalised xywh to pixel ``(x, y, w, h)``.

    Lines must contain at least ``cls x_center y_center w h`` in normalised coordinates.
    Confidence (if present in a 6th column) is ignored here.
    """
    if not label_path.exists():
        return []
    boxes: list[BBox] = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, x_c, y_c, w, h = map(float, parts[:5])
        x = int((x_c - w / 2) * img_w)
        y = int((y_c - h / 2) * img_h)
        bw = int(w * img_w)
        bh = int(h * img_h)
        boxes.append((x, y, bw, bh))
    return boxes
