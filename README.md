# bachelor-thesis-flow

*CPU-based pipeline for mitosis detection and divergence analysis*

[![CI](https://github.com/rayanchatt/bachelor-thesis-flow/actions/workflows/ci.yml/badge.svg)](https://github.com/rayanchatt/bachelor-thesis-flow/actions/workflows/ci.yml)
[![CodeQL](https://github.com/rayanchatt/bachelor-thesis-flow/actions/workflows/codeql.yml/badge.svg)](https://github.com/rayanchatt/bachelor-thesis-flow/actions/workflows/codeql.yml)
[![OpenSSF Scorecard](https://api.securityscorecards.dev/projects/github.com/rayanchatt/bachelor-thesis-flow/badge)](https://securityscorecards.dev/viewer/?uri=github.com/rayanchatt/bachelor-thesis-flow)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)

Source code accompanying the Bachelor's thesis
**"Quantification and Modelling of Tissue Growth"**
(Heinrich Heine University Düsseldorf, Institut für Biomedizinische Physik, 2025).

The pipeline is **GPU-free** and

1. detects mitosis events in brightfield time-lapse sequences using a compact **YOLOv5s** detector, and
2. quantifies their mechanical neighbourhood via **dense Farnebäck optical-flow** fields turned into divergence and magnitude maps (*DefMaps*).

> **Quick summary**
> – ~95 % mAP<sub>50</sub> on the training level
> – Significant divergence minima before and divergence maxima after mitosis events

---

## Install

```bash
pip install -e ".[dev]"
```

The package targets **Python ≥ 3.11** and exposes a single console
entry point `btflow` with subcommands.

```bash
btflow --help
```

## Usage

| Subcommand        | Purpose                                                         |
|-------------------|-----------------------------------------------------------------|
| `btflow rgb-stack`       | Merge per-z grayscale frames into RGB frames.            |
| `btflow defmap`          | Build divergence/magnitude DefMap stacks (Farnebäck).     |
| `btflow match-labels`    | Link YOLO detections from two z-layers by IoU.            |
| `btflow lagcorr`         | Lag correlation between YOLO confidence and DefMaps.      |
| `btflow confidence-kde`  | KDE plot of YOLO confidence grouped by z-layer.           |
| `btflow heatmap`         | 2D KDE heatmap of bounding-box centers.                   |
| `btflow iou-boxplot`     | IoU boxplot across one or more matched-box CSVs.          |

Each subcommand exposes `--help` with the full argument list.

Example: build divergence DefMap stacks from a folder of raw frames.

```bash
btflow defmap \
  --rgb-dir  path/to/raw_frames \
  --out-dir  path/to/output \
  --metric   div \
  --sigma    1.5
```

## Key plots

### 1. Mean divergence over lag

<p align="center">
  <img src="assets/lag_div_curve_Z1_all.png" width="75%" alt="Mean divergence over lag (Z1)">
</p>

Mean divergence in the vicinity of mitosis time points for different ROI scalings (Z₁).

### 2. IoU distribution of matched bounding boxes

<p align="center">
  <img src="assets/iou_boxplot.png" width="60%" alt="IoU distribution of matched bounding boxes">
</p>

IoU distribution of axially matched bounding boxes between z-layers (8-layer vs. 6-layer dataset).

### 3. Spatial distribution of mitosis events

<p align="center">
  <img src="assets/heatmap_6er.png" width="65%" alt="Heatmap of cell centers in the 6-layer dataset">
</p>

Density heatmap of cell centers in the 6-layer dataset (Z₂ / Z₃), revealing distinct tissue regions.

## Demo video

A short example video illustrating where mitosis events are detected:

- `assets/cell_division_detected.mp4` — **Mitosis detection** visualisation of the time-lapse sequence with YOLOv5s bounding boxes over detected cell divisions.

[▶ Watch mitosis detection](assets/cell_division_detected.mp4)

## Development

```bash
make install    # editable install + dev tools + pre-commit hooks
make lint       # ruff
make typecheck  # mypy --strict
make test       # pytest
make cov        # pytest with coverage
make figures    # regenerate synthetic sample plots under assets/synthetic/
```

Commits **must** follow [Conventional Commits](https://www.conventionalcommits.org/);
release-please uses the commit log to generate `CHANGELOG.md` and the next version tag.

## Citing

If you use this code, please cite it via the metadata in
[`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository"
button in the right-hand sidebar).

## License

Released under the [Creative Commons Attribution 4.0 International (CC-BY-4.0)](LICENSE) license.
