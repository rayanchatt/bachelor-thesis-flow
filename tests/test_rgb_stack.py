"""End-to-end test for the RGB stack builder."""

from __future__ import annotations

from pathlib import Path

import cv2

from btflow.rgb_stack import build_rgb_stack


def test_build_rgb_stack_writes_one_file_per_frame(
    synthetic_zstack_dir: Path, tmp_path: Path
) -> None:
    out = tmp_path / "rgb_out"
    written = build_rgb_stack(synthetic_zstack_dir, out)
    assert written == 5
    files = sorted(out.glob("frame_*.png"))
    assert len(files) == 5

    img = cv2.imread(str(files[0]))
    assert img is not None
    assert img.shape == (32, 32, 3)


def test_build_rgb_stack_skips_incomplete_frames(tmp_path: Path) -> None:
    src = tmp_path / "broken"
    src.mkdir()
    import numpy as np

    img = np.zeros((8, 8), dtype=np.uint8)
    # Only z=1 and z=2 — z=3 missing.
    cv2.imwrite(str(src / "x_t0001_z0001.png"), img)
    cv2.imwrite(str(src / "x_t0001_z0002.png"), img)

    out = tmp_path / "rgb_out"
    written = build_rgb_stack(src, out)
    assert written == 0
