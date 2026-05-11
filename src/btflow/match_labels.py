"""Match YOLO bounding boxes across two z-layers via IoU."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2

from ._types import BBox
from .io import load_yolo_labels
from .iou import calculate_iou


def match_labels(
    image_dir: Path,
    label_dir_green: Path,
    label_dir_red: Path,
    output_dir: Path,
    iou_threshold: float = 0.5,
    green_layer: str = "Z2",
    red_layer: str = "Z3",
) -> Path:
    """Link YOLO detections from two z-layers by IoU and write annotated frames.

    Bounding boxes from ``label_dir_green`` are matched 1:1 against boxes from
    ``label_dir_red`` whenever IoU exceeds ``iou_threshold``. The annotated PNG
    (green=layer 1, red=layer 2 boxes) and a cumulative CSV of matches are
    written to ``output_dir``.

    Args:
        image_dir: Source PNG frames whose filenames contain ``green_layer``.
        label_dir_green: YOLO label .txt files for the green (first) z-layer.
        label_dir_red: YOLO label .txt files for the red (second) z-layer.
        output_dir: Where annotated PNGs and ``matched_boxes.csv`` are written.
        iou_threshold: Minimum IoU for a green/red pair to be considered a match.
        green_layer: Layer tag present in input filenames (default ``"Z2"``).
        red_layer: Layer tag substituted into red label filenames (default ``"Z3"``).

    Returns:
        Path to the cumulative CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "matched_boxes.csv"
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["filename", "green_box", "red_box", "iou"])
        if write_header:
            writer.writeheader()

        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() != ".png":
                continue
            print(f"Processing image: {image_path.name}")

            img = cv2.imread(str(image_path))
            if img is None:
                print(f"Image unreadable: {image_path}")
                continue
            h_img, w_img = img.shape[:2]

            label_path_green = label_dir_green / image_path.with_suffix(".txt").name
            label_path_red = label_dir_red / (
                image_path.name.replace(green_layer, red_layer).replace(".png", ".txt")
            )

            green_boxes: list[BBox] = load_yolo_labels(label_path_green, w_img, h_img)
            if not label_path_green.exists():
                print(f"No green labels found: {label_path_green}")
            red_boxes: list[BBox] = load_yolo_labels(label_path_red, w_img, h_img)
            if not label_path_red.exists():
                print(f"No red labels found: {label_path_red}")

            for gx, gy, gw, gh in green_boxes:
                for rx, ry, rw, rh in red_boxes:
                    iou = calculate_iou((gx, gy, gw, gh), (rx, ry, rw, rh))
                    if iou > iou_threshold:
                        writer.writerow(
                            {
                                "filename": image_path.name,
                                "green_box": (gx, gy, gw, gh),
                                "red_box": (rx, ry, rw, rh),
                                "iou": iou,
                            }
                        )
                        break
                cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (0, 255, 0), 2)

            for rx, ry, rw, rh in red_boxes:
                cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (0, 0, 255), 2)

            cv2.imwrite(str(output_dir / image_path.name), img)

    return csv_path


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = subparsers.add_parser(
        "match-labels",
        help="Link YOLO detections from two z-layers by IoU and write annotated frames.",
    )
    p.add_argument("--image-dir", required=True, type=Path)
    p.add_argument("--label-dir-green", required=True, type=Path)
    p.add_argument("--label-dir-red", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--green-layer", default="Z2")
    p.add_argument("--red-layer", default="Z3")
    p.set_defaults(_handler=_handle)


def _handle(args: argparse.Namespace) -> int:
    match_labels(
        image_dir=args.image_dir,
        label_dir_green=args.label_dir_green,
        label_dir_red=args.label_dir_red,
        output_dir=args.output_dir,
        iou_threshold=args.iou_threshold,
        green_layer=args.green_layer,
        red_layer=args.red_layer,
    )
    return 0
