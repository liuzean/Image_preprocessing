from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import frangi

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
from Image_enhancement.scratch_detection.modules.grayscale import (  # noqa: E402
    convert_to_grayscale_float32,
)


BOUNDARY_MODES = {"constant", "reflect", "wrap", "nearest", "mirror"}


@dataclass(frozen=True)
class FrangiConfig:
    enabled: bool = True
    sigmas: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)
    alpha: float = 0.5
    beta: float = 0.5
    gamma: float | None = None
    detect_bright_ridges: bool = True
    detect_dark_ridges: bool = True
    boundary_mode: str = "reflect"
    constant_value: float = 0.0

    def validate(self) -> None:
        if not self.sigmas:
            raise ValueError("sigmas must not be empty")
        if not all(np.isfinite(sigma) and sigma > 0 for sigma in self.sigmas):
            raise ValueError("all sigmas must be finite and greater than 0")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and greater than 0")
        if not np.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("beta must be finite and greater than 0")
        if self.gamma is not None and (
            not np.isfinite(self.gamma) or self.gamma <= 0
        ):
            raise ValueError("gamma must be None or finite and greater than 0")
        if not self.detect_bright_ridges and not self.detect_dark_ridges:
            raise ValueError("at least one ridge polarity must be enabled")
        if self.boundary_mode not in BOUNDARY_MODES:
            supported = ", ".join(sorted(BOUNDARY_MODES))
            raise ValueError(f"boundary_mode must be one of: {supported}")
        if not np.isfinite(self.constant_value):
            raise ValueError("constant_value must be finite")


@dataclass(frozen=True)
class FrangiEnhancementResult:
    response_image: np.ndarray
    bright_response_image: np.ndarray
    dark_response_image: np.ndarray


@dataclass(frozen=True)
class FrangiPreviewConfig:
    lower_percentile: float = 0.0
    upper_percentile: float = 99.5
    crop_padding: int = 15

    def validate(self) -> None:
        if not 0 <= self.lower_percentile < self.upper_percentile <= 100:
            raise ValueError(
                "preview percentiles must satisfy 0 <= lower < upper <= 100"
            )
        if not isinstance(self.crop_padding, int) or self.crop_padding < 0:
            raise ValueError("crop_padding must be a non-negative integer")


def _expanded_mask_bounds(
    mask_pixels: np.ndarray,
    padding: int,
) -> tuple[int, int, int, int]:
    y_coordinates, x_coordinates = np.nonzero(mask_pixels)
    image_height, image_width = mask_pixels.shape
    x_start = max(0, int(x_coordinates.min()) - padding)
    y_start = max(0, int(y_coordinates.min()) - padding)
    x_end = min(image_width, int(x_coordinates.max()) + 1 + padding)
    y_end = min(image_height, int(y_coordinates.max()) + 1 + padding)
    return x_start, y_start, x_end, y_end


def enhance_frangi(
    input_float32: np.ndarray,
    processing_mask: np.ndarray,
    config: FrangiConfig,
) -> FrangiEnhancementResult:
    """Enhance bright and dark line-like ridges using a 2-D Frangi filter."""
    config.validate()
    if input_float32.ndim != 2:
        raise ValueError("input_float32 must be a one-channel image")
    if processing_mask.ndim != 2 or processing_mask.shape != input_float32.shape:
        raise ValueError("processing_mask must match the input image size")

    input_image = input_float32.astype(np.float32, copy=False)
    if not np.isfinite(input_image).all():
        raise ValueError("input_float32 must contain only finite values")

    mask_pixels = processing_mask > 0
    if not np.any(mask_pixels):
        raise ValueError("processing_mask must contain at least one foreground pixel")

    empty_response = np.zeros_like(input_image)
    if not config.enabled:
        skipped_response = np.zeros_like(input_image)
        skipped_response[mask_pixels] = input_image[mask_pixels]
        return FrangiEnhancementResult(
            response_image=skipped_response,
            bright_response_image=empty_response.copy(),
            dark_response_image=empty_response.copy(),
        )

    filter_padding = max(1, int(np.ceil(4.0 * max(config.sigmas))))
    x_start, y_start, x_end, y_end = _expanded_mask_bounds(
        mask_pixels,
        filter_padding,
    )
    input_roi = input_image[y_start:y_end, x_start:x_end]

    bright_roi = np.zeros_like(input_roi)
    if config.detect_bright_ridges:
        bright_roi = frangi(
            input_roi,
            sigmas=config.sigmas,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            black_ridges=False,
            mode=config.boundary_mode,
            cval=config.constant_value,
        ).astype(np.float32, copy=False)

    dark_roi = np.zeros_like(input_roi)
    if config.detect_dark_ridges:
        dark_roi = frangi(
            input_roi,
            sigmas=config.sigmas,
            alpha=config.alpha,
            beta=config.beta,
            gamma=config.gamma,
            black_ridges=True,
            mode=config.boundary_mode,
            cval=config.constant_value,
        ).astype(np.float32, copy=False)

    bright_response = np.zeros_like(input_image)
    dark_response = np.zeros_like(input_image)
    bright_response[y_start:y_end, x_start:x_end] = bright_roi
    dark_response[y_start:y_end, x_start:x_end] = dark_roi
    bright_response[~mask_pixels] = 0.0
    dark_response[~mask_pixels] = 0.0

    return FrangiEnhancementResult(
        response_image=np.maximum(bright_response, dark_response),
        bright_response_image=bright_response,
        dark_response_image=dark_response,
    )


def create_response_preview(
    response_image: np.ndarray,
    processing_mask: np.ndarray,
    config: FrangiPreviewConfig,
) -> np.ndarray:
    config.validate()
    if response_image.shape != processing_mask.shape:
        raise ValueError("processing_mask must match the response image size")

    preview = np.zeros_like(response_image, dtype=np.uint8)
    mask_pixels = processing_mask > 0
    masked_response = response_image[mask_pixels]
    if masked_response.size:
        lower_value, upper_value = np.percentile(
            masked_response,
            [config.lower_percentile, config.upper_percentile],
        )
        if upper_value > lower_value:
            normalized = (response_image - lower_value) * 255.0
            normalized /= upper_value - lower_value
            preview[mask_pixels] = np.clip(
                normalized[mask_pixels],
                0,
                255,
            ).astype(np.uint8)

    return cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)


def create_parameter_record_dir(
    output_dir: Path,
    config: FrangiConfig,
) -> Path:
    config.validate()
    sigma_text = "-".join(str(sigma) for sigma in config.sigmas)
    gamma_text = "auto" if config.gamma is None else str(config.gamma)
    parameter_dir = output_dir / (
        f"sigmas={sigma_text},alpha={config.alpha},"
        f"beta={config.beta},gamma={gamma_text}"
    )
    parameter_dir.mkdir(parents=False, exist_ok=False)
    return parameter_dir


def process_dataset(
    dataset_dir: Path,
    mask_config: ErodeMaskConfig,
    background_config: BackgroundCorrectionConfig,
    frangi_config: FrangiConfig,
    preview_config: FrangiPreviewConfig,
    writer_config: ResultWriterConfig,
) -> tuple[int, list[Path], Path]:
    scan_result = scan_image_json_pairs(dataset_dir)
    writer = ResultWriter(dataset_dir, writer_config)
    create_parameter_record_dir(writer.output_dir, frangi_config)

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

        combined_preview = create_response_preview(
            frangi_result.response_image,
            mask_result.eroded_mask,
            preview_config,
        )
        bright_preview = create_response_preview(
            frangi_result.bright_response_image,
            mask_result.eroded_mask,
            preview_config,
        )
        dark_preview = create_response_preview(
            frangi_result.dark_response_image,
            mask_result.eroded_mask,
            preview_config,
        )
        (
            cropped_original,
            cropped_combined,
            cropped_bright,
            cropped_dark,
        ) = crop_images_to_mask_bounding_rect(
            mask_result.original_mask,
            original_image,
            combined_preview,
            bright_preview,
            dark_preview,
            padding=preview_config.crop_padding,
        )
        writer.save_result(
            pair.image_path.stem,
            cropped_original,
            cropped_combined,
        )
        write_image(
            writer.output_dir / f"{pair.image_path.stem}_frangi_bright.png",
            cropped_bright,
        )
        write_image(
            writer.output_dir / f"{pair.image_path.stem}_frangi_dark.png",
            cropped_dark,
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
        sigmas=(2.5,3),   #表示检测尺度,尺度越多，适应的划痕宽度越广,sigmas越大，检测的划痕越宽，sigmas越小，检测的划痕越窄
        alpha=0.5,        #用于三维图像
        beta=0.2,         #控制“线状结构”和“圆形/块状结构”的区分强度。越小倾向于细长线状，越大，限制越小，弱划痕更容易保留，但纹理和斑点，圆形/块状结构也可能增加。
        gamma=None,        #控制对结构强度的敏感性，越小，弱划痕容易出现，但噪声和纹理也更多；越大，只保留较强结构，浅划痕可能消失；
        detect_bright_ridges=True,
        detect_dark_ridges=True,
        boundary_mode="reflect",
        constant_value=0.0,
    )
    preview_config = FrangiPreviewConfig(
        lower_percentile=0.0,
        upper_percentile=99.5,
        crop_padding=15,
    )
    writer_config = ResultWriterConfig(
        output_folder_name="frangi_results",
        run_number_width=3,
        max_run_number=999,
        save_comparison=True,
        save_processed=True,
        comparison_suffix="_frangi_comparison.png",
        processed_suffix="_frangi_response.png",
    )

    processed_count, images_without_json, output_dir = process_dataset(
        dataset_dir,
        mask_config,
        background_config,
        frangi_config,
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
