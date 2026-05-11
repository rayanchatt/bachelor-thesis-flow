"""Unit tests for filename parsing and YOLO label loading."""

from __future__ import annotations

from pathlib import Path

import pytest

from btflow.io import group_frames_by_z_t, load_yolo_labels, parse_frame_filename


@pytest.mark.parametrize(
    "name, expected",
    [
        ("foo_t0001_z0002.png", (1, 2)),
        ("anything_t0123_z0007.tif", (123, 7)),
        ("prefix_t9_z9.jpg", (9, 9)),
        ("no_match.png", None),
        ("foo_t0001.png", None),
    ],
)
def test_parse_frame_filename(name: str, expected: tuple[int, int] | None) -> None:
    assert parse_frame_filename(name) == expected


def test_group_frames_by_z_t(synthetic_zstack_dir: Path) -> None:
    grouped = group_frames_by_z_t(synthetic_zstack_dir)
    assert sorted(grouped.keys()) == [1, 2, 3]
    for z, frames in grouped.items():
        assert sorted(frames.keys()) == [1, 2, 3, 4, 5], f"z={z}"


def test_load_yolo_labels_pixel_conversion(tmp_path: Path) -> None:
    label = tmp_path / "frame_001.txt"
    label.write_text("0 0.5 0.5 0.2 0.2 0.9\n")
    boxes = load_yolo_labels(label, img_w=100, img_h=100)
    assert boxes == [(40, 40, 20, 20)]


def test_load_yolo_labels_returns_empty_when_missing(tmp_path: Path) -> None:
    assert load_yolo_labels(tmp_path / "missing.txt", 10, 10) == []
