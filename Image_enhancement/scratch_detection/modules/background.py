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
    write_image,
)
from Image_enhancement.scratch_detection.modules.erode_mask import (  # noqa: E402
    ErodeMaskConfig,
    process_image as process_mask,
)
from Image_enhancement.scratch_detection.modules.grayscale import (  # noqa: E402
    convert_to_grayscale_float32,
)


@dataclass(frozen=True)
class BackgroundCorrectionConfig:
    enabled: bool = True
    gaussian_kernel_size: int = 151
    sigma: float = 0.0
    division_epsilon: float = 1e-6

    def validate(self) -> None:
        if self.gaussian_kernel_size < 1 or self.gaussian_kernel_size % 2 == 0:
            raise ValueError("gaussian_kernel_size must be a positive odd integer")
        if self.sigma < 0:
            raise ValueError("sigma must be greater than or equal to 0")
        if self.division_epsilon <= 0:
            raise ValueError("division_epsilon must be greater than 0")


@dataclass(frozen=True)
class BackgroundCorrectionResult:
    background_image: np.ndarray
    corrected_image: np.ndarray


@dataclass(frozen=True)
class BackgroundPreviewConfig:
    residual_offset: float = 128.0


def subtract_mask_aware_gaussian_background(
    grayscale_float32: np.ndarray,
    support_mask: np.ndarray,
    config: BackgroundCorrectionConfig,
) -> BackgroundCorrectionResult:
    """Estimate a Gaussian background without mixing pixels outside the mask."""
    config.validate()
    if grayscale_float32.ndim != 2:
        raise ValueError("grayscale_float32 must be a one-channel image")
    if support_mask.ndim != 2 or support_mask.shape != grayscale_float32.shape:
        raise ValueError("support_mask must match the grayscale image size")

    grayscale = grayscale_float32.astype(np.float32, copy=False)
    mask_pixels = support_mask > 0
    if not np.any(mask_pixels):
        raise ValueError("support_mask must contain at least one foreground pixel")

    if not config.enabled:
        return BackgroundCorrectionResult(
            background_image=np.zeros_like(grayscale),
            corrected_image=grayscale.copy(),
        )

    mask_weights = mask_pixels.astype(np.float32)
    kernel_size = (config.gaussian_kernel_size, config.gaussian_kernel_size)
    weighted_image = grayscale * mask_weights

    blurred_weighted_image = cv2.GaussianBlur(
        weighted_image,
        kernel_size,
        sigmaX=config.sigma,
        sigmaY=config.sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )
    blurred_mask_weights = cv2.GaussianBlur(
        mask_weights,
        kernel_size,
        sigmaX=config.sigma,
        sigmaY=config.sigma,
        borderType=cv2.BORDER_REFLECT_101,
    )

    background = np.zeros_like(grayscale)
    valid_denominator = blurred_mask_weights > config.division_epsilon
    np.divide(
        blurred_weighted_image,
        blurred_mask_weights,
        out=background,
        where=valid_denominator,
    )
    background[~mask_pixels] = 0.0

    corrected = np.zeros_like(grayscale)
    corrected[mask_pixels] = grayscale[mask_pixels] - background[mask_pixels]
    return BackgroundCorrectionResult(
        background_image=background,
        corrected_image=corrected,
    )


def create_background_preview(
    background_image: np.ndarray,
    display_mask: np.ndarray,
) -> np.ndarray:
    preview = np.zeros_like(background_image, dtype=np.uint8)
    mask_pixels = display_mask > 0
    preview[mask_pixels] = np.clip(
        background_image[mask_pixels],
        0,
        255,
    ).astype(np.uint8)
    return cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)


def create_corrected_preview(
    corrected_image: np.ndarray,
    display_mask: np.ndarray,
    config: BackgroundPreviewConfig,
) -> np.ndarray:
    preview = np.zeros_like(corrected_image, dtype=np.uint8)
    mask_pixels = display_mask > 0
    shifted = corrected_image[mask_pixels] + config.residual_offset
    preview[mask_pixels] = np.clip(shifted, 0, 255).astype(np.uint8)
    return cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)


def process_dataset(
    dataset_dir: Path,
    mask_config: ErodeMaskConfig,
    background_config: BackgroundCorrectionConfig,
    preview_config: BackgroundPreviewConfig,
    writer_config: ResultWriterConfig,
) -> tuple[int, list[Path], Path]:
    scan_result = scan_image_json_pairs(dataset_dir)
    writer = ResultWriter(dataset_dir, writer_config)

    processed_count = 0
    for pair in scan_result.pairs:
        original_image = read_image(pair.image_path)
        annotation = read_annotation(pair.json_path)
        mask_result = process_mask(original_image, annotation, mask_config)
        grayscale_result = convert_to_grayscale_float32(original_image)
        background_result = subtract_mask_aware_gaussian_background(
            grayscale_result.image,
            mask_result.original_mask,
            background_config,
        )

        corrected_preview = create_corrected_preview(
            background_result.corrected_image,
            mask_result.eroded_mask,
            preview_config,
        )
        writer.save_result(
            pair.image_path.stem,
            original_image,
            corrected_preview,
        )

        background_preview = create_background_preview(
            background_result.background_image,
            mask_result.eroded_mask,
        )
        write_image(
            writer.output_dir
            / f"{pair.image_path.stem}_estimated_background.png",
            background_preview,
        )
        processed_count += 1

    return processed_count, scan_result.images_without_json, writer.output_dir


def main() -> None:
    dataset_dir = DEFAULT_DATASET_DIR
    mask_config = ErodeMaskConfig(
        enabled=True,
        kernel_size=31,
        iterations=1,
        kernel_shape="ellipse",
        mask_category="Silver box",
    )
    background_config = BackgroundCorrectionConfig(
        enabled=True,
        gaussian_kernel_size=151,
        sigma=0.0,
        division_epsilon=1e-6,
    )
    preview_config = BackgroundPreviewConfig(residual_offset=128.0)
    writer_config = ResultWriterConfig(
        output_folder_name="background_results",
        run_number_width=3,
        max_run_number=999,
        save_comparison=True,
        save_processed=True,
        comparison_suffix="_background_comparison.png",
        processed_suffix="_background_corrected.png",
    )

    processed_count, images_without_json, output_dir = process_dataset(
        dataset_dir,
        mask_config,
        background_config,
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
