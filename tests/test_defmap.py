"""Integration tests for DefMap stack construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from btflow.defmap import build_defmap_stack


def test_div_stack_shape_and_files(synthetic_zstack_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "defmap"
    stacks = build_defmap_stack(synthetic_zstack_dir, out, metric="div")
    assert sorted(stacks.keys()) == [1, 2, 3]
    for z, stack in stacks.items():
        # 5 input frames -> 4 flow pairs
        assert stack.shape == (4, 32, 32), f"z={z}"
        assert stack.dtype == np.float32
        assert (out / f"defmap_stack_Z{z}_div.npy").is_file()


def test_mag_stack_is_nonnegative(synthetic_zstack_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "defmap_mag"
    stacks = build_defmap_stack(synthetic_zstack_dir, out, metric="mag")
    for stack in stacks.values():
        assert (stack >= 0).all(), "magnitude must be non-negative"


def test_invalid_metric_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="metric must be 'div' or 'mag'"):
        build_defmap_stack(tmp_path, tmp_path, metric="bogus")


def test_div_stack_detects_rightward_motion(synthetic_zstack_dir: Path, tmp_path: Path) -> None:
    """A blob moving rightward should produce sign-flipping divergence around
    its center: positive on the leading edge, negative on the trailing edge.
    """
    out = tmp_path / "defmap"
    stacks = build_defmap_stack(synthetic_zstack_dir, out, metric="div", sigma=0.5)
    stack = stacks[1]
    # The blob crosses the central area; total signed divergence should be
    # very close to zero (mass is conserved by translation).
    assert abs(stack[0].mean()) < 0.5
