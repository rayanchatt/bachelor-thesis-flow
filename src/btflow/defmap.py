"""Build per-z DefMap stacks from RGB time-lapse frames via Farnebäck optical flow.

For each z-slice in the input directory, dense optical flow is computed between
consecutive frames using ``cv2.calcOpticalFlowFarneback``. The resulting flow
field is reduced either to its **divergence** (``div``: ``∂vx/∂x + ∂vy/∂y``)
or its **magnitude** (``mag``: ``√(vx² + vy²)``) and saved as a NumPy stack.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .io import group_frames_by_z_t, imread

# Farnebäck parameters used throughout the thesis.
FLOW_KWARGS: dict[str, float | int] = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}

CMAP = "coolwarm"
VMAX_DIV = 4.0
VMAX_MAG = 4.0


def build_defmap_stack(
    rgb_dir: Path,
    out_dir: Path,
    metric: str = "div",
    sigma: float = 1.5,
    channel: int | None = None,
    crop: tuple[int, int, int, int] | None = None,
    save_png: bool = False,
) -> dict[int, np.ndarray]:
    """Build divergence or magnitude stacks per z-slice.

    Args:
        rgb_dir: Directory with raw frames named ``..._tXXXX_zXXXX.png``.
        out_dir: Output directory for ``.npy`` stacks and optional PNG previews.
        metric: ``"div"`` for divergence or ``"mag"`` for magnitude.
        sigma: Gaussian sigma for flow smoothing (``0`` disables smoothing).
        channel: Single colour channel index (0/1/2) or ``None`` for grayscale.
        crop: Optional ``(x0, y0, w, h)`` crop box in pixels.
        save_png: Whether to also write a PNG preview per frame.

    Returns:
        Mapping ``{z_slice: stack}`` where each ``stack`` has shape ``(T-1, H, W)``.
    """
    if metric not in {"div", "mag"}:
        raise ValueError(f"metric must be 'div' or 'mag', got {metric!r}")

    rgb_dir = rgb_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_dict = group_frames_by_z_t(rgb_dir)
    if not frame_dict:
        raise SystemExit("No raw frames matching the expected '_t####_z####' pattern were found.")

    z_slices = sorted(frame_dict.keys())
    t_list = sorted(next(iter(frame_dict.values())).keys())
    print(f"Raw data loaded: z-slices = {z_slices}, time steps = {len(t_list)}")

    stacks: dict[int, np.ndarray] = {}
    for z_use in z_slices:
        frames = [frame_dict[z_use][t] for t in t_list]
        stack = []

        for i in tqdm(range(len(frames) - 1), desc=f"Z{z_use}"):
            if channel is None:
                g1 = imread(frames[i], cv2.IMREAD_GRAYSCALE)
                g2 = imread(frames[i + 1], cv2.IMREAD_GRAYSCALE)
            else:
                g1 = cv2.split(imread(frames[i]))[channel]
                g2 = cv2.split(imread(frames[i + 1]))[channel]

            if crop is not None:
                x0, y0, w, h = crop
                g1 = g1[y0 : y0 + h, x0 : x0 + w]
                g2 = g2[y0 : y0 + h, x0 : x0 + w]

            # opencv-python stubs reject the generic ndarray dtype Any here even though
            # a uint8 grayscale array is valid input.
            flow = cv2.calcOpticalFlowFarneback(g1, g2, None, **FLOW_KWARGS)  # type: ignore[call-overload]
            vx, vy = flow[..., 0], flow[..., 1]

            if sigma > 0:
                vx = cv2.GaussianBlur(vx, (0, 0), sigmaX=sigma)
                vy = cv2.GaussianBlur(vy, (0, 0), sigmaX=sigma)

            if metric == "mag":
                data = np.sqrt(vx**2 + vy**2).astype(np.float32)
                clip_lo, clip_hi = 0.0, VMAX_MAG
            else:
                dvx_dx = np.gradient(vx, axis=1)
                dvy_dy = np.gradient(vy, axis=0)
                data = (dvx_dx + dvy_dy).astype(np.float32)
                clip_lo, clip_hi = -VMAX_DIV, VMAX_DIV

            stack.append(data)

            if save_png:
                shown = np.clip(data, clip_lo, clip_hi)
                fig = plt.figure(figsize=(4, 3))
                plt.imshow(shown, cmap=CMAP, vmin=clip_lo, vmax=clip_hi)
                plt.axis("off")
                plt.tight_layout(pad=0)
                png_path = out_dir / f"defmap_Z{z_use}_{i + 1:03d}.png"
                fig.savefig(png_path, dpi=120, bbox_inches="tight", pad_inches=0)
                plt.close(fig)

        stack_arr = np.stack(stack)
        out_npy = out_dir / f"defmap_stack_Z{z_use}_{metric}.npy"
        np.save(out_npy, stack_arr)
        print(f"DefMap stack Z{z_use} -> {out_npy}  shape={stack_arr.shape}")
        stacks[z_use] = stack_arr

    return stacks


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "defmap",
        help="Build DefMap stacks (divergence or magnitude) from RGB frames.",
        description="Build DefMap stack from RGB frames via Farnebäck optical flow.",
    )
    p.add_argument(
        "--rgb-dir",
        required=True,
        type=Path,
        help="Folder with raw input images '..._tXXXX_zXXXX.png'.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output folder for PNG/MP4 previews and .npy stacks.",
    )
    p.add_argument(
        "--metric",
        choices=["div", "mag"],
        default="div",
        help="'div' = divergence ∇·v  |  'mag' = magnitude ‖v‖  (default: div).",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=1.5,
        help="Gaussian sigma for flow smoothing (0 disables smoothing).",
    )
    p.add_argument(
        "--channel",
        type=int,
        choices=[0, 1, 2],
        default=None,
        help="Process only this colour channel (0=B/Z1, 1=G/Z2, 2=R/Z3). Omit for grayscale.",
    )
    p.add_argument(
        "--crop",
        type=int,
        nargs=4,
        metavar=("X0", "Y0", "W", "H"),
        default=None,
        help="Crop box in pixels (x0 y0 w h).",
    )
    p.add_argument(
        "--save-png",
        action="store_true",
        help="Save PNG previews of individual DefMaps.",
    )
    p.set_defaults(_handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    crop: tuple[int, int, int, int] | None = (
        (args.crop[0], args.crop[1], args.crop[2], args.crop[3]) if args.crop else None
    )
    build_defmap_stack(
        rgb_dir=args.rgb_dir,
        out_dir=args.out_dir,
        metric=args.metric,
        sigma=args.sigma,
        channel=args.channel,
        crop=crop,
        save_png=args.save_png,
    )
    return 0
