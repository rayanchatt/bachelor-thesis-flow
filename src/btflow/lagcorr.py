"""Lag correlation between YOLO detection confidence and DefMap values.

Unifies the original ``timelagcorrection_div.py`` and ``timelagcorrection_mag.py``
into a single command. The two differ only in (a) which DefMap stack files
they consume, (b) ROI scale set, and (c) whether magnitudes are taken as
absolute values. All of those are now CLI arguments.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

DEFAULT_LAGS: tuple[int, ...] = tuple(range(-10, 11))
DEFAULT_SCALES_DIV: tuple[float, ...] = (1.5, 2.0, 2.5)
DEFAULT_SCALES_MAG: tuple[float, ...] = (1.0, 1.5, 2.0)

_FRAME_PAT = re.compile(r"frame_(\d+)\.txt$")


def _collect_rows(
    defmaps: np.ndarray,
    label_dir: Path,
    scales: tuple[float, ...],
    lags: tuple[int, ...],
    conf_min: float,
    use_abs: bool,
) -> pd.DataFrame:
    n, h, w = defmaps.shape
    rows: list[dict[str, float | int]] = []

    for scale in scales:
        for txt in sorted(label_dir.glob("frame_*.txt")):
            m = _FRAME_PAT.search(txt.name)
            if not m:
                continue
            t = int(m.group(1))
            df = pd.read_csv(
                txt, sep=" ", header=None, names=["cls", "x_c", "y_c", "w", "h", "conf"]
            )
            for _, r in df.iterrows():
                x = int(r.x_c * w)
                y = int(r.y_c * h)
                if not (0 <= x < w and 0 <= y < h):
                    continue
                bb_w = int(r.w * w)
                bb_h = int(r.h * h)
                roi_r = int(max(bb_w, bb_h) * scale)
                if roi_r <= 0:
                    continue

                for lag in lags:
                    t_def = t + lag
                    if t_def < 1 or t_def > n:
                        continue
                    y0 = max(0, y - roi_r)
                    y1 = min(h, y + roi_r + 1)
                    x0 = max(0, x - roi_r)
                    x1 = min(w, x + roi_r + 1)
                    patch = defmaps[t_def - 1, y0:y1, x0:x1]
                    if patch.size == 0:
                        continue

                    h_p, w_p = patch.shape
                    y_grid, x_grid = np.mgrid[0:h_p, 0:w_p]
                    cy, cx = (h_p - 1) / 2, (w_p - 1) / 2
                    sigma = roi_r / 1.5
                    kernel = np.exp(-((x_grid - cx) ** 2 + (y_grid - cy) ** 2) / (2 * sigma**2))
                    kernel /= kernel.sum()
                    val = float((patch * kernel).sum())
                    if use_abs:
                        val = abs(val)
                    if r.conf <= conf_min:
                        continue
                    rows.append(
                        {
                            "frame": t,
                            "lag": lag,
                            "conf": float(r.conf),
                            "value": val,
                            "scale": scale,
                        }
                    )

    return pd.DataFrame(rows)


def _correlation_table(tbl: pd.DataFrame, lags: tuple[int, ...]) -> pd.DataFrame:
    rows_corr: list[dict[str, float | int | str]] = []
    for lag in lags:
        sub = tbl[tbl.lag == lag]
        if len(sub) < 2:
            print(f"lag {lag}: too few points")
            continue
        r, p = pearsonr(sub["value"], sub["conf"])
        rows_corr.append({"lag": lag, "r": r, "p": p, "method": "Pearson"})
    return pd.DataFrame(rows_corr).set_index("lag").reindex(list(lags))


def _plot_curve(
    tbl: pd.DataFrame,
    corr_df: pd.DataFrame,
    lags: tuple[int, ...],
    metric_label: str,
    z_tag: str,
    out_path: Path,
    group_by_scale: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    if group_by_scale:
        for scale, sub_df in tbl.groupby("scale"):
            avg = sub_df.groupby("lag")["value"].mean()
            sem = sub_df.groupby("lag")["value"].sem()
            ax.errorbar(
                avg.index,
                avg.values,
                yerr=sem.values,
                fmt="-o",
                label=f"ROI {scale}x",
                capsize=3,
            )
        ax.legend(loc="upper left", frameon=True, fancybox=True, edgecolor="gray")
    else:
        avg = tbl.groupby("lag")["value"].mean()
        sem = tbl.groupby("lag")["value"].sem()
        ax.errorbar(
            avg.index,
            avg.values,
            yerr=sem.values,
            fmt="-o",
            color="black",
            capsize=3,
            label="ROI mean",
        )
        ax.legend(loc="upper left", frameon=True, fancybox=True, edgecolor="gray")

    ax.set_title(f"Mean {metric_label} over lag ({z_tag})")
    ax.set_xlabel("Lag [frames]")
    ax.set_ylabel(f"Mean {metric_label}")
    ax.set_xticks(list(lags))

    n_events = tbl["frame"].nunique()
    min_p = corr_df["p"].min()
    p_str = f"{min_p:.1e}" if min_p < 0.001 else f"{min_p:.3f}"
    ax.text(
        0.98,
        0.95,
        f"N = {n_events}\np = {p_str}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "gray", "boxstyle": "round,pad=0.3"},
    )
    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _plot_significance(
    corr_df: pd.DataFrame, lags: tuple[int, ...], metric_label: str, out_path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(8, 2.6))
    bars = ax.bar(
        corr_df.index,
        corr_df["r"],
        color=["#d73027" if r > 0 else "#4575b4" for r in corr_df["r"]],
    )
    for bar, p_val in zip(bars, corr_df["p"], strict=False):
        if pd.notna(p_val) and p_val < 0.05:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                "*",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
    ax.set_xlabel("Lag [frames]")
    ax.set_ylabel("Correlation coefficient r")
    ax.set_xticks(list(lags))
    ax.set_title(f"Confidence vs. {metric_label} — significant lags (* p<0.05)")
    ax.axhline(0, lw=1, color="k")
    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def run_lagcorr(
    defmap_path: Path,
    label_dir: Path,
    out_dir: Path,
    metric: str = "div",
    conf_min: float = 0.5,
    lags: tuple[int, ...] = DEFAULT_LAGS,
    scales: tuple[float, ...] | None = None,
    use_abs: bool | None = None,
) -> Path:
    """Run lag correlation analysis for a single DefMap stack.

    Returns the path to the row-level CSV.
    """
    if metric not in {"div", "mag"}:
        raise ValueError(f"metric must be 'div' or 'mag', got {metric!r}")
    if scales is None:
        scales = DEFAULT_SCALES_DIV if metric == "div" else DEFAULT_SCALES_MAG
    if use_abs is None:
        use_abs = metric == "mag"

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = defmap_path.stem
    m_layer = re.search(r"_Z(\d+)", tag)
    z_tag = f"Z{m_layer.group(1)}" if m_layer else tag

    defmaps = np.load(defmap_path)
    print(f"Stack: {defmaps.shape}")
    print(f"Labels: {label_dir}")

    tbl = _collect_rows(defmaps, label_dir, scales, lags, conf_min, use_abs)
    csv_path = out_dir / f"lag_corr_table_{z_tag}_all.csv"
    tbl.to_csv(csv_path, index=False)
    print(f"-> Table: {csv_path}  (#rows = {len(tbl)})")

    corr_df = _correlation_table(tbl, lags)
    metric_label = "Divergence" if metric == "div" else "Magnitude"

    curve_path = out_dir / f"lag_{metric}_curve_{z_tag}_all.png"
    _plot_curve(tbl, corr_df, lags, metric_label, z_tag, curve_path, group_by_scale=True)
    print(f"Saved per-scale curve: {curve_path}")

    avg_path = out_dir / f"lag_{metric}_curve_avg_{z_tag}_all.png"
    _plot_curve(tbl, corr_df, lags, metric_label, z_tag, avg_path, group_by_scale=False)
    print(f"Saved averaged curve: {avg_path}")

    stats_csv = out_dir / f"lag_corr_stats_{z_tag}_all.csv"
    corr_df.to_csv(stats_csv)
    print(f"Saved correlation stats: {stats_csv}")

    sig_path = out_dir / f"lag_corr_significant_{metric}_{z_tag}_all.png"
    _plot_significance(corr_df, lags, metric_label, sig_path)
    print(f"Saved significance plot: {sig_path}")

    return csv_path


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "lagcorr",
        help="Lag correlation: YOLO confidence vs. DefMap (divergence or magnitude).",
    )
    p.add_argument(
        "--defmap",
        required=True,
        nargs="+",
        type=Path,
        help="One or more .npy files with DefMap stacks.",
    )
    p.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="Folder with YOLO frame_*.txt label files.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Output folder for CSVs and plots.",
    )
    p.add_argument(
        "--metric",
        choices=["div", "mag"],
        default="div",
        help="Which DefMap metric the stacks represent (default: div).",
    )
    p.add_argument(
        "--conf-min",
        type=float,
        default=0.5,
        help="Minimum YOLO confidence (default: 0.5).",
    )
    p.set_defaults(_handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    for defmap_path in args.defmap:
        run_lagcorr(
            defmap_path=defmap_path,
            label_dir=args.labels,
            out_dir=args.out_dir,
            metric=args.metric,
            conf_min=args.conf_min,
        )
    return 0
