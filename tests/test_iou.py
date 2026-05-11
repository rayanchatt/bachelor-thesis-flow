"""Unit tests for the IoU helper."""

from __future__ import annotations

import math

import pytest

from btflow.iou import calculate_iou


def test_identical_boxes_iou_is_one() -> None:
    assert calculate_iou((10, 10, 20, 20), (10, 10, 20, 20)) == pytest.approx(1.0)


def test_disjoint_boxes_iou_is_zero() -> None:
    assert calculate_iou((0, 0, 5, 5), (100, 100, 5, 5)) == 0.0


def test_half_overlap_iou_is_one_third() -> None:
    # Two equal-size boxes shifted by half their width along x:
    # intersection area = 0.5 * box_area; union = 1.5 * box_area; IoU = 1/3.
    iou = calculate_iou((0, 0, 10, 10), (5, 0, 10, 10))
    assert math.isclose(iou, 1 / 3, rel_tol=1e-9)


def test_zero_area_boxes_return_zero() -> None:
    assert calculate_iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_iou_is_symmetric() -> None:
    a = (3, 7, 12, 9)
    b = (5, 6, 10, 11)
    assert calculate_iou(a, b) == pytest.approx(calculate_iou(b, a))
