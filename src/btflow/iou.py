"""Intersection-over-Union helpers for axis-aligned bounding boxes."""

from __future__ import annotations

from ._types import BBox


def calculate_iou(box1: BBox, box2: BBox) -> float:
    """Compute the Intersection-over-Union of two ``(x, y, w, h)`` boxes.

    Args:
        box1: First bounding box as ``(x, y, width, height)``.
        box2: Second bounding box as ``(x, y, width, height)``.

    Returns:
        IoU in ``[0, 1]``. Returns ``0.0`` when both boxes have zero area
        (no union to normalise against).
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area
