from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_DATASET_DIR = Path(r"E:\projects\datasets\Power_box\Power_box_3long")

IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ImageJsonPair:
    image_path: Path
    json_path: Path


@dataclass(frozen=True)
class DatasetScanResult:
    pairs: list[ImageJsonPair]
    images_without_json: list[Path]


def scan_image_json_pairs(dataset_dir: Path) -> DatasetScanResult:
    """Pair every image in a directory with a same-stem JSON annotation."""
    dataset_dir = Path(dataset_dir)
    if not dataset_dir.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {dataset_dir}")

    files = [path for path in dataset_dir.iterdir() if path.is_file()]
    images = sorted(
        (path for path in files if path.suffix.casefold() in IMAGE_EXTENSIONS),
        key=lambda path: path.name.casefold(),
    )
    json_by_stem = {
        path.stem.casefold(): path
        for path in files
        if path.suffix.casefold() == ".json"
    }

    pairs: list[ImageJsonPair] = []
    images_without_json: list[Path] = []
    for image_path in images:
        json_path = json_by_stem.get(image_path.stem.casefold())
        if json_path is None:
            images_without_json.append(image_path)
            continue
        pairs.append(ImageJsonPair(image_path=image_path, json_path=json_path))

    return DatasetScanResult(
        pairs=pairs,
        images_without_json=images_without_json,
    )


def read_image(image_path: Path) -> np.ndarray:
    """Read an image from paths containing non-ASCII characters."""
    encoded = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def read_annotation(json_path: Path) -> dict[str, Any]:
    with Path(json_path).open("r", encoding="utf-8") as file:
        annotation = json.load(file)
    if not isinstance(annotation, dict):
        raise ValueError(f"JSON root must be an object: {json_path}")
    return annotation
