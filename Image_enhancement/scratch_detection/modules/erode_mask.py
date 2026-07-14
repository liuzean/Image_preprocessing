from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Image_enhancement.scratch_detection.io.dataset import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    read_annotation,
    read_image,
    scan_image_json_pairs,
)


KERNEL_SHAPES = {
    "rectangle": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


@dataclass(frozen=True)
class ErodeMaskConfig:
    #参数设置
    enabled: bool = True
    kernel_size: int = 31
    iterations: int = 1
    kernel_shape: str = "ellipse"
    mask_category: str | None = "Silver box"
    output_suffix: str = "_eroded_comparison.png"

    def validate(self) -> None:
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.kernel_shape not in KERNEL_SHAPES:
            supported = ", ".join(KERNEL_SHAPES)
            raise ValueError(f"kernel_shape must be one of: {supported}")
        if not self.output_suffix.lower().endswith(".png"):
            raise ValueError("output_suffix must use the .png extension")


def extract_segmentations(
    annotation: dict,
    category: str | None = "Silver box",
) -> list[np.ndarray]:
    objects = annotation.get("objects")
    if not isinstance(objects, list):
        raise ValueError('JSON must contain an "objects" list')

    segmentations: list[np.ndarray] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        if category is not None and item.get("category") != category:
            continue

        segmentation = item.get("segmentation")
        if not isinstance(segmentation, list):
            continue

        points: list[tuple[float, float]] = []
        for point in segmentation:
            if isinstance(point, (list, tuple)) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))

        if len(points) >= 3:
            segmentations.append(np.rint(points).astype(np.int32))

    if not segmentations:
        category_text = "all categories" if category is None else repr(category)
        raise ValueError(f"No valid segmentation found for {category_text}")
    return segmentations


def build_mask(
    image_shape: tuple[int, ...],
    segmentations: list[np.ndarray],
) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    clipped_segmentations: list[np.ndarray] = []
    for points in segmentations:
        clipped = points.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
        clipped_segmentations.append(clipped)

    cv2.fillPoly(mask, clipped_segmentations, 255)
    return mask


def erode_mask(mask: np.ndarray, config: ErodeMaskConfig) -> np.ndarray:
    config.validate()
    if not config.enabled:
        return mask.copy()

    kernel = cv2.getStructuringElement(
        KERNEL_SHAPES[config.kernel_shape],
        (config.kernel_size, config.kernel_size),
    )
    return cv2.erode(mask, kernel, iterations=config.iterations)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked_image = np.zeros_like(image)
    cv2.copyTo(image, mask, masked_image)
    return masked_image


def write_png(output_path: Path, image: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"Failed to encode image: {output_path}")
    encoded.tofile(str(output_path))


def create_next_output_dir(dataset_dir: Path) -> Path:
    output_root = Path(dataset_dir) / "erode_mask_results"
    output_root.mkdir(parents=True, exist_ok=True)

    numeric_indices = [
        int(path.name)
        for path in output_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    next_index = max(numeric_indices, default=0) + 1
    if next_index > 999:
        raise RuntimeError(
            f"No output index is available: {output_root} has reached 999"
        )

    output_dir = output_root / f"{next_index:03d}"
    output_dir.mkdir(parents=False, exist_ok=False)
    return output_dir


def process_dataset(
    dataset_dir: Path,
    config: ErodeMaskConfig,
) -> tuple[int, list[Path], Path]:
    config.validate()
    scan_result = scan_image_json_pairs(dataset_dir)
    output_dir = create_next_output_dir(dataset_dir)

    processed_count = 0
    for pair in scan_result.pairs:
        image = read_image(pair.image_path)
        annotation = read_annotation(pair.json_path)
        segmentations = extract_segmentations(annotation, config.mask_category)
        original_mask = build_mask(image.shape, segmentations)
        processing_mask = erode_mask(original_mask, config)

        eroded_image = apply_mask(image, processing_mask)
        comparison = np.hstack((image, eroded_image))
        output_path = output_dir / f"{pair.image_path.stem}{config.output_suffix}"
        write_png(output_path, comparison)
        processed_count += 1

    return processed_count, scan_result.images_without_json, output_dir


def main() -> None:
    dataset_dir = DEFAULT_DATASET_DIR#DEFAULT_DATASET_DIR  #修改路径，例如：dataset_dir = Path(r"E:\projects\datasets\Power_box\Power_box_3long")
    config = ErodeMaskConfig(
        enabled=True,
        kernel_size=31,
        iterations=1,
        kernel_shape="ellipse",
        mask_category="Silver box",
        output_suffix="_eroded_comparison.png",
    )

    processed_count, images_without_json, output_dir = process_dataset(
        dataset_dir,
        config,
    )
    print(f"Processed {processed_count} images. Results saved to: {output_dir}")

    if images_without_json:
        print("Skipped images without a same-name JSON file:")
        for image_path in images_without_json:
            print(f"  {image_path.name}")


if __name__ == "__main__":
    main()
