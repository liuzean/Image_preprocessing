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
from Image_enhancement.scratch_detection.io.result_writer import (  # noqa: E402
    ResultWriter,
    ResultWriterConfig,
    crop_images_to_mask_bounding_rect,
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

    def validate(self) -> None:
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.iterations < 1:
            raise ValueError("iterations must be at least 1")
        if self.kernel_shape not in KERNEL_SHAPES:
            supported = ", ".join(KERNEL_SHAPES)
            raise ValueError(f"kernel_shape must be one of: {supported}")


@dataclass(frozen=True)
class ErodeMaskPreviewConfig:
    original_contour_color_bgr: tuple[int, int, int] = (0, 255, 0)
    eroded_contour_color_bgr: tuple[int, int, int] = (0, 0, 255)
    contour_thickness: int = 5
    crop_padding: int = 15

    def validate(self) -> None:
        if self.contour_thickness < 1:
            raise ValueError("contour_thickness must be at least 1")
        if not isinstance(self.crop_padding, int) or self.crop_padding < 0:
            raise ValueError("crop_padding must be a non-negative integer")
        for color in (
            self.original_contour_color_bgr,
            self.eroded_contour_color_bgr,
        ):
            invalid_channel = any(channel < 0 or channel > 255 for channel in color)
            if len(color) != 3 or invalid_channel:
                raise ValueError("contour colors must contain three values from 0 to 255")


@dataclass(frozen=True)
class ErodeMaskResult:
    original_mask: np.ndarray
    eroded_mask: np.ndarray


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


def draw_mask_contours(
    image: np.ndarray,
    original_mask: np.ndarray,
    eroded_mask: np.ndarray,
    config: ErodeMaskPreviewConfig,
) -> np.ndarray:
    visualization = image.copy()
    original_contours, _ = cv2.findContours(
        original_mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    eroded_contours, _ = cv2.findContours(
        eroded_mask.copy(),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        visualization,
        original_contours,
        contourIdx=-1,
        color=config.original_contour_color_bgr,
        thickness=config.contour_thickness,
        lineType=cv2.LINE_AA,
    )
    cv2.drawContours(
        visualization,
        eroded_contours,
        contourIdx=-1,
        color=config.eroded_contour_color_bgr,
        thickness=config.contour_thickness,
        lineType=cv2.LINE_AA,
    )
    return visualization


def process_image(
    image: np.ndarray,
    annotation: dict,
    config: ErodeMaskConfig,
) -> ErodeMaskResult:
    config.validate()
    segmentations = extract_segmentations(annotation, config.mask_category)
    original_mask = build_mask(image.shape, segmentations)
    eroded_mask = erode_mask(original_mask, config)
    return ErodeMaskResult(
        original_mask=original_mask,
        eroded_mask=eroded_mask,
    )


def create_preview_image(
    image: np.ndarray,
    result: ErodeMaskResult,
    config: ErodeMaskPreviewConfig,
) -> np.ndarray:
    config.validate()
    eroded_image = apply_mask(image, result.eroded_mask)
    return draw_mask_contours(
        eroded_image,
        result.original_mask,
        result.eroded_mask,
        config,
    )


def process_dataset(
    dataset_dir: Path,
    config: ErodeMaskConfig,
    preview_config: ErodeMaskPreviewConfig,
    writer_config: ResultWriterConfig,
) -> tuple[int, list[Path], Path]:
    config.validate()
    preview_config.validate()
    scan_result = scan_image_json_pairs(dataset_dir)
    writer = ResultWriter(dataset_dir, writer_config)

    processed_count = 0
    for pair in scan_result.pairs:
        image = read_image(pair.image_path)
        annotation = read_annotation(pair.json_path)
        result = process_image(image, annotation, config)
        preview_image = create_preview_image(image, result, preview_config)
        cropped_image, cropped_preview = crop_images_to_mask_bounding_rect(
            result.original_mask,
            image,
            preview_image,
            padding=preview_config.crop_padding,
        )
        writer.save_result(
            pair.image_path.stem,
            cropped_image,
            cropped_preview,
        )
        processed_count += 1

    return processed_count, scan_result.images_without_json, writer.output_dir


def main() -> None:
    dataset_dir = DEFAULT_DATASET_DIR
    config = ErodeMaskConfig(
        enabled=True,
        kernel_size=31,
        iterations=1,
        kernel_shape="ellipse",
        mask_category="Silver box",
    )
    preview_config = ErodeMaskPreviewConfig(
        original_contour_color_bgr=(0, 255, 0),
        eroded_contour_color_bgr=(0, 0, 255),
        contour_thickness=5,
        crop_padding=15,
    )
    writer_config = ResultWriterConfig(
        output_folder_name="erode_mask_results",
        run_number_width=3,
        max_run_number=999,
        save_comparison=True,
        save_processed=False,
        comparison_suffix="_eroded_comparison.png",
    )

    processed_count, images_without_json, output_dir = process_dataset(
        dataset_dir,
        config,
        preview_config,
        writer_config,
    )
    print(f"Processed {processed_count} images. Results saved to: {output_dir}")

    if images_without_json:
        print("Skipped images without a same-name JSON file:")
        for image_path in images_without_json:
            print(f"  {image_path.name}")


if __name__ == "__main__":
    main()
