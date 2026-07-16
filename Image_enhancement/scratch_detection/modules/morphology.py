from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.morphology import skeletonize

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Image_enhancement.scratch_detection.io.dataset import (  # noqa: E402
    read_annotation,
    read_image,
    scan_image_json_pairs,
)
from Image_enhancement.scratch_detection.io.result_writer import (  # noqa: E402
    ResultWriter,
    ResultWriterConfig,
    crop_images_to_mask_bounding_rect,
    write_image,
)
from Image_enhancement.scratch_detection.modules.background import (  # noqa: E402
    BackgroundCorrectionConfig,
    subtract_mask_aware_gaussian_background,
)
from Image_enhancement.scratch_detection.modules.erode_mask import (  # noqa: E402
    ErodeMaskConfig,
    process_image as process_mask,
)
from Image_enhancement.scratch_detection.modules.frangi import (  # noqa: E402
    FrangiConfig,
    enhance_frangi,
)
from Image_enhancement.scratch_detection.modules.grayscale import (  # noqa: E402
    convert_to_grayscale_float32,
)
from Image_enhancement.scratch_detection.modules.threshold import (  # noqa: E402
    HysteresisThresholdConfig,
    apply_masked_hysteresis_threshold,
)


SKELETON_METHODS = {"zhang", "lee"}


@dataclass(frozen=True)
class SkeletonizationConfig:
    enabled: bool = True
    method: str = "zhang"

    def validate(self) -> None:
        if self.method not in SKELETON_METHODS:
            supported = ", ".join(sorted(SKELETON_METHODS))
            raise ValueError(f"method must be one of: {supported}")


@dataclass(frozen=True)
class MorphologyPreviewConfig:
    crop_padding: int = 15
    skeleton_preview_thickness: int = 3

    def validate(self) -> None:
        if not isinstance(self.crop_padding, int) or self.crop_padding < 0:
            raise ValueError("crop_padding must be a non-negative integer")
        if (
            not isinstance(self.skeleton_preview_thickness, int)
            or self.skeleton_preview_thickness < 1
            or self.skeleton_preview_thickness % 2 == 0
        ):
            raise ValueError(
                "skeleton_preview_thickness must be a positive odd integer"
            )


@dataclass(frozen=True)
class SkeletonizationResult:
    skeleton_image: np.ndarray
    foreground_pixel_count: int
    skeleton_pixel_count: int


def skeletonize_binary_candidates(
    binary_image: np.ndarray,
    processing_mask: np.ndarray,
    config: SkeletonizationConfig,
) -> SkeletonizationResult:
    """Reduce binary candidates to centerlines without joining components."""
    config.validate()
    if binary_image.ndim != 2:
        raise ValueError("binary_image must be a one-channel image")
    if processing_mask.ndim != 2 or processing_mask.shape != binary_image.shape:
        raise ValueError("processing_mask must match the binary image size")

    foreground = (binary_image > 0) & (processing_mask > 0)
    foreground_pixel_count = int(np.count_nonzero(foreground))

    if config.enabled:
        skeleton_pixels = skeletonize(
            foreground,
            method=config.method,
        )
    else:
        skeleton_pixels = foreground

    skeleton_image = np.zeros(binary_image.shape, dtype=np.uint8)
    skeleton_image[skeleton_pixels] = 255
    return SkeletonizationResult(
        skeleton_image=skeleton_image,
        foreground_pixel_count=foreground_pixel_count,
        skeleton_pixel_count=int(np.count_nonzero(skeleton_pixels)),
    )


def create_skeleton_preview(
    skeleton_image: np.ndarray,
    config: MorphologyPreviewConfig,
) -> np.ndarray:
    """Create a thicker display copy without changing the skeleton result."""
    config.validate()
    if skeleton_image.ndim != 2:
        raise ValueError("skeleton_image must be a one-channel image")

    preview = skeleton_image
    if config.skeleton_preview_thickness > 1:
        kernel_size = config.skeleton_preview_thickness
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        preview = cv2.dilate(skeleton_image, kernel, iterations=1)
    return cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)


def _create_parameter_record_dir(
    output_dir: Path,
    skeleton_config: SkeletonizationConfig,
    preview_config: MorphologyPreviewConfig,
) -> Path:
    skeleton_config.validate()
    preview_config.validate()
    parameter_dir = output_dir / (
        f"method={skeleton_config.method},"
        f"preview_thickness={preview_config.skeleton_preview_thickness}"
    )
    parameter_dir.mkdir(parents=False, exist_ok=False)
    return parameter_dir


def _run_standalone_dataset(
    dataset_dir: Path,
    mask_config: ErodeMaskConfig,
    background_config: BackgroundCorrectionConfig,
    frangi_config: FrangiConfig,
    threshold_config: HysteresisThresholdConfig,
    skeleton_config: SkeletonizationConfig,
    preview_config: MorphologyPreviewConfig,
    writer_config: ResultWriterConfig,
) -> tuple[int, list[Path], Path]:
    """Run the image-writing preview workflow for this module's main entry."""
    scan_result = scan_image_json_pairs(dataset_dir)
    writer = ResultWriter(dataset_dir, writer_config)
    _create_parameter_record_dir(
        writer.output_dir,
        skeleton_config,
        preview_config,
    )

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
        frangi_result = enhance_frangi(
            background_result.corrected_image,
            mask_result.eroded_mask,
            frangi_config,
        )
        threshold_result = apply_masked_hysteresis_threshold(
            frangi_result.bright_response_image,
            mask_result.eroded_mask,
            threshold_config,
        )
        skeleton_result = skeletonize_binary_candidates(
            threshold_result.binary_image,
            mask_result.eroded_mask,
            skeleton_config,
        )

        binary_preview = cv2.cvtColor(
            threshold_result.binary_image,
            cv2.COLOR_GRAY2BGR,
        )
        skeleton_image_bgr = cv2.cvtColor(
            skeleton_result.skeleton_image,
            cv2.COLOR_GRAY2BGR,
        )
        skeleton_preview = create_skeleton_preview(
            skeleton_result.skeleton_image,
            preview_config,
        )
        cropped_original, cropped_binary, cropped_skeleton = (
            crop_images_to_mask_bounding_rect(
                mask_result.original_mask,
                original_image,
                binary_preview,
                skeleton_image_bgr,
                padding=preview_config.crop_padding,
            )
        )
        (cropped_skeleton_preview,) = crop_images_to_mask_bounding_rect(
            mask_result.original_mask,
            skeleton_preview,
            padding=preview_config.crop_padding,
        )

        writer.save_result(
            pair.image_path.stem,
            cropped_original,
            cropped_skeleton,
        )
        stages = np.hstack(
            (cropped_original, cropped_binary, cropped_skeleton_preview)
        )
        write_image(
            writer.output_dir / f"{pair.image_path.stem}_morphology_stages.png",
            stages,
        )
        processed_count += 1

    return processed_count, scan_result.images_without_json, writer.output_dir


def main() -> None:
    dataset_dir = Path(r"E:\projects\datasets\Power_box\old")
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
    frangi_config = FrangiConfig(
        enabled=True,
        sigmas=(2.5, 3.0),
        alpha=0.5,
        beta=0.2,
        gamma=None,
        detect_bright_ridges=True,
        detect_dark_ridges=False,
        boundary_mode="reflect",
        constant_value=0.0,
    )
    threshold_config = HysteresisThresholdConfig(
        enabled=True,
        high_percentile=97.5,
        low_threshold_ratio=0.4,
        connectivity=8,
    )
    skeleton_config = SkeletonizationConfig(
        enabled=True,
        method="zhang",
    )
    preview_config = MorphologyPreviewConfig(
        crop_padding=15,
        skeleton_preview_thickness=3,
    )
    writer_config = ResultWriterConfig(
        output_folder_name="morphology_results",
        run_number_width=3,
        max_run_number=999,
        save_comparison=False,
        save_processed=True,
        comparison_suffix="_morphology_comparison.png",
        processed_suffix="_skeleton.png",
    )

    processed_count, images_without_json, output_dir = _run_standalone_dataset(
        dataset_dir,
        mask_config,
        background_config,
        frangi_config,
        threshold_config,
        skeleton_config,
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
