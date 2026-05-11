"""Shared pytest fixtures: synthetic frames, labels, and DefMap stacks.

All fixtures generate data in-memory (or in a temporary directory) so the
test suite never depends on the real microscopy datasets and can run in CI.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # Headless backend — must be set before any pyplot import.

from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG shared by all fixtures."""
    return np.random.default_rng(seed=42)


@pytest.fixture
def synthetic_zstack_dir(tmp_path: Path, rng: np.random.Generator) -> Path:
    """Create a tiny ``_t####_z####`` z-stack directory with 5 frames x 3 z-slices.

    A bright Gaussian blob is translated linearly across frames so that
    Farnebäck optical flow has a real signal to pick up.
    """
    d = tmp_path / "rgb"
    d.mkdir()
    h, w = 32, 32
    yy, xx = np.mgrid[0:h, 0:w]

    for t in range(1, 6):
        for z in (1, 2, 3):
            cx = 8 + t  # moves rightward
            cy = 16
            blob = 255 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0**2))
            noise = rng.normal(0, 5, size=(h, w))
            img = np.clip(blob + noise, 0, 255).astype(np.uint8)
            cv2.imwrite(str(d / f"frame_t{t:04d}_z{z:04d}.png"), img)
    return d


@pytest.fixture
def synthetic_rgb_dir(tmp_path: Path, rng: np.random.Generator) -> Path:
    """Same as ``synthetic_zstack_dir`` but written into a separate folder.

    Produced as 3-channel (BGR) PNGs already so downstream commands that
    expect colour input can be exercised.
    """
    d = tmp_path / "rgb_merged"
    d.mkdir()
    h, w = 32, 32
    yy, xx = np.mgrid[0:h, 0:w]
    for t in range(1, 6):
        for z in (1, 2, 3):
            cx = 8 + t
            cy = 16
            blob = 255 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 3.0**2))
            noise = rng.normal(0, 5, size=(h, w))
            gray = np.clip(blob + noise, 0, 255).astype(np.uint8)
            rgb = cv2.merge([gray, gray, gray])
            cv2.imwrite(str(d / f"frame_t{t:04d}_z{z:04d}.png"), rgb)
    return d


@pytest.fixture
def synthetic_yolo_labels(tmp_path: Path) -> Path:
    """Write a YOLO-format label directory with one detection per frame."""
    d = tmp_path / "labels"
    d.mkdir()
    for t in range(1, 6):
        (d / f"frame_{t:03d}.txt").write_text("0 0.5 0.5 0.2 0.2 0.9\n")
    return d


@pytest.fixture
def synthetic_defmap_stack(tmp_path: Path, rng: np.random.Generator) -> Path:
    """Save a small ``(N, H, W)`` divergence stack to disk."""
    stack = rng.normal(0, 1, size=(5, 32, 32)).astype(np.float32)
    p = tmp_path / "defmap_stack_Z1_div.npy"
    np.save(p, stack)
    return p
