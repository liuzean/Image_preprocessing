from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ResultWriterConfig:
    output_folder_name: str = "results"
    run_number_width: int = 3
    max_run_number: int = 999
    save_comparison: bool = True
    save_processed: bool = True
    comparison_suffix: str = "_comparison.png"
    processed_suffix: str = "_processed.png"

    def validate(self) -> None:
        folder_path = Path(self.output_folder_name)
        if (
            not self.output_folder_name
            or folder_path.name != self.output_folder_name
            or self.output_folder_name in {".", ".."}
        ):
            raise ValueError("output_folder_name must be one directory name")
        if self.run_number_width < 1:
            raise ValueError("run_number_width must be at least 1")
        largest_number = (10**self.run_number_width) - 1
        if self.max_run_number < 1 or self.max_run_number > largest_number:
            raise ValueError(
                f"max_run_number must be between 1 and {largest_number}"
            )
        if not self.save_comparison and not self.save_processed:
            raise ValueError("at least one output type must be enabled")
        for suffix in (self.comparison_suffix, self.processed_suffix):
            if not suffix.lower().endswith(".png"):
                raise ValueError("output suffixes must use the .png extension")


def create_next_result_dir(
    dataset_dir: Path,
    config: ResultWriterConfig,
) -> Path:
    config.validate()
    output_root = Path(dataset_dir) / config.output_folder_name
    output_root.mkdir(parents=True, exist_ok=True)

    numeric_indices = [
        int(path.name)
        for path in output_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    next_index = max(numeric_indices, default=0) + 1
    if next_index > config.max_run_number:
        raise RuntimeError(
            f"No output index is available: {output_root} has reached "
            f"{config.max_run_number}"
        )

    output_dir = output_root / f"{next_index:0{config.run_number_width}d}"
    output_dir.mkdir(parents=False, exist_ok=False)
    return output_dir


def write_image(output_path: Path, image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(output_path.suffix, image)
    if not success:
        raise ValueError(f"Failed to encode image: {output_path}")
    encoded.tofile(str(output_path))


def crop_images_to_mask_bounding_rect(
    mask: np.ndarray,
    *images: np.ndarray,
    padding: int = 15,
) -> tuple[np.ndarray, ...]:
    """Crop images to the mask bounding rectangle plus padding."""
    if mask.ndim != 2:
        raise ValueError("mask must be a one-channel image")
    if not isinstance(padding, int) or padding < 0:
        raise ValueError("padding must be a non-negative integer")

    mask_uint8 = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(mask_uint8):
        raise ValueError("mask must contain at least one foreground pixel")

    image_height, image_width = mask.shape
    for image in images:
        if image.shape[:2] != (image_height, image_width):
            raise ValueError("every image must have the same size as the mask")

    x, y, width, height = cv2.boundingRect(mask_uint8)
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(image_width, x + width + padding)
    y_end = min(image_height, y + height + padding)

    return tuple(
        image[y_start:y_end, x_start:x_end].copy()
        for image in images
    )


class ResultWriter:
    """Create one numbered run directory and save final pipeline images."""

    def __init__(
        self,
        dataset_dir: Path,
        config: ResultWriterConfig | None = None,
    ) -> None:
        self.config = config or ResultWriterConfig()
        self.output_dir = create_next_result_dir(dataset_dir, self.config)

    def save_result(
        self,
        image_stem: str,
        original_image: np.ndarray,
        processed_image: np.ndarray,
    ) -> list[Path]:
        if original_image.shape != processed_image.shape:
            raise ValueError(
                "original_image and processed_image must have the same shape"
            )

        saved_paths: list[Path] = []
        if self.config.save_comparison:
            comparison = np.hstack((original_image, processed_image))
            comparison_path = (
                self.output_dir
                / f"{image_stem}{self.config.comparison_suffix}"
            )
            write_image(comparison_path, comparison)
            saved_paths.append(comparison_path)

        if self.config.save_processed:
            processed_path = (
                self.output_dir
                / f"{image_stem}{self.config.processed_suffix}"
            )
            write_image(processed_path, processed_image)
            saved_paths.append(processed_path)

        return saved_paths
