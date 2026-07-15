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


RESPONSE_MODES = {"absolute", "positive", "negative"}  #absolute:同一张图片中同时有亮划痕和暗划痕,positive:只保留亮划痕,negative:只保留暗划痕


@dataclass(frozen=True)
class MultiDirectionGaborConfig:
    enabled: bool = True
    angles_degrees: tuple[float, ...] = (
        0,
        15,
        30,
        45,
        60,
        75,
        90,
        105,
        120,
        135,
        150,
        165,
    )
    kernel_size: int = 31
    sigma: float = 3.0
    wavelength: float = 6.0
    gamma: float = 0.5
    psi: float = 0.0
    response_mode: str = "absolute"
    normalize_kernel_l2: bool = True

    def validate(self) -> None:
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if self.sigma <= 0:
            raise ValueError("sigma must be greater than 0")
        if self.wavelength <= 0:
            raise ValueError("wavelength must be greater than 0")
        if self.gamma <= 0:
            raise ValueError("gamma must be greater than 0")
        if not np.isfinite(self.psi):
            raise ValueError("psi must be finite")
        if not self.angles_degrees:
            raise ValueError("angles_degrees must not be empty")
        if not all(np.isfinite(angle) for angle in self.angles_degrees):
            raise ValueError("all Gabor angles must be finite")
        if self.response_mode not in RESPONSE_MODES:
            supported = ", ".join(sorted(RESPONSE_MODES))
            raise ValueError(f"response_mode must be one of: {supported}")


@dataclass(frozen=True)
class GaborEnhancementResult:
    response_image: np.ndarray
    signed_response_image: np.ndarray
    direction_map_degrees: np.ndarray


@dataclass(frozen=True)
class GaborPreviewConfig:
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


def _response_score(response: np.ndarray, mode: str) -> np.ndarray:
    if mode == "absolute":
        return np.abs(response)
    if mode == "positive":
        return np.maximum(response, 0.0)
    return np.maximum(-response, 0.0)


def enhance_multi_direction_gabor(
    input_float32: np.ndarray,
    processing_mask: np.ndarray,
    config: MultiDirectionGaborConfig,
) -> GaborEnhancementResult:
    """Keep the strongest Gabor response at each pixel across all angles."""
    config.validate()
    if input_float32.ndim != 2:
        raise ValueError("input_float32 must be a one-channel image")
    if processing_mask.ndim != 2 or processing_mask.shape != input_float32.shape:
        raise ValueError("processing_mask must match the input image size")

    input_image = input_float32.astype(np.float32, copy=False)
    mask_pixels = processing_mask > 0
    if not np.any(mask_pixels):
        raise ValueError("processing_mask must contain at least one foreground pixel")

    direction_map = np.full(input_image.shape, -1.0, dtype=np.float32)
    if not config.enabled:
        return GaborEnhancementResult(
            response_image=input_image.copy(),
            signed_response_image=input_image.copy(),
            direction_map_degrees=direction_map,
        )

    max_score = np.zeros_like(input_image)
    signed_response_at_max = np.zeros_like(input_image)

    for angle_degrees in config.angles_degrees:
        kernel = cv2.getGaborKernel(
            (config.kernel_size, config.kernel_size),
            config.sigma,
            np.deg2rad(angle_degrees),
            config.wavelength,
            config.gamma,
            config.psi,
            ktype=cv2.CV_32F,
        )
        kernel -= float(kernel.mean())
        if config.normalize_kernel_l2:
            kernel_norm = float(np.linalg.norm(kernel))
            if kernel_norm <= np.finfo(np.float32).eps:
                raise ValueError("Gabor kernel has zero L2 norm")
            kernel /= kernel_norm

        signed_response = cv2.filter2D(
            input_image,
            cv2.CV_32F,
            kernel,
            borderType=cv2.BORDER_REFLECT_101,
        )
        score = _response_score(signed_response, config.response_mode)
        update_pixels = mask_pixels & (score > max_score)
        max_score[update_pixels] = score[update_pixels]
        signed_response_at_max[update_pixels] = signed_response[update_pixels]
        direction_map[update_pixels] = float(angle_degrees % 180)

    return GaborEnhancementResult(
        response_image=max_score,
        signed_response_image=signed_response_at_max,
        direction_map_degrees=direction_map,
    )


def create_response_preview(
    response_image: np.ndarray,
    processing_mask: np.ndarray,
    config: GaborPreviewConfig,
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
    config: MultiDirectionGaborConfig,
) -> Path:
    config.validate()
    parameter_dir = output_dir / (
        f"kernel_size={config.kernel_size},"
        f"sigma={config.sigma},"
        f"wavelength={config.wavelength},"
        f"gamma={config.gamma}"
    )
    parameter_dir.mkdir(parents=False, exist_ok=False)
    return parameter_dir


def process_dataset(
    dataset_dir: Path,
    mask_config: ErodeMaskConfig,
    background_config: BackgroundCorrectionConfig,
    gabor_config: MultiDirectionGaborConfig,
    preview_config: GaborPreviewConfig,
    writer_config: ResultWriterConfig,
) -> tuple[int, list[Path], Path]:
    scan_result = scan_image_json_pairs(dataset_dir)
    writer = ResultWriter(dataset_dir, writer_config)
    create_parameter_record_dir(writer.output_dir, gabor_config)

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
        gabor_result = enhance_multi_direction_gabor(
            background_result.corrected_image,
            mask_result.eroded_mask,
            gabor_config,
        )
        response_preview = create_response_preview(
            gabor_result.response_image,
            mask_result.eroded_mask,
            preview_config,
        )
        cropped_original, cropped_response = crop_images_to_mask_bounding_rect(
            mask_result.original_mask,
            original_image,
            response_preview,
            padding=preview_config.crop_padding,
        )
        writer.save_result(
            pair.image_path.stem,
            cropped_original,
            cropped_response,
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
    gabor_config = MultiDirectionGaborConfig(
        enabled=True,
        angles_degrees=(0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165),
        kernel_size=31,
        sigma=5.0,
        wavelength=12.0,
        gamma=0.5,
        psi=0.0,
        response_mode="absolute",
        normalize_kernel_l2=True,
    )
    preview_config = GaborPreviewConfig(
        lower_percentile=0.0,
        upper_percentile=99.5,
        crop_padding=15,
    )
    writer_config = ResultWriterConfig(
        output_folder_name="gabor_results",
        run_number_width=3,
        max_run_number=999,
        save_comparison=True,
        save_processed=True,
        comparison_suffix="_gabor_comparison.png",
        processed_suffix="_gabor_response.png",
    )

    processed_count, images_without_json, output_dir = process_dataset(
        dataset_dir,
        mask_config,
        background_config,
        gabor_config,
        preview_config,
        writer_config,
    )
    print(f"Processed {processed_count} images. Results saved to: {output_dir}")

    if images_without_json:
        print("Skipped images without a same-name JSON file:")
        for image_path in images_without_json:
            print(f"  {image_path.name}")

#根据测试结果，增大wavelength，测了6,8,10,12,14，花纹会被明显减弱，但还是会存在，数值达到12的时候，较浅的划痕会开始变淡，但相对深一点的还是差不多。***kernel\_size的数值测了31,41,51差别都不大。sigma测了3,4,5.花纹变化不大，但5的话深的和浅的划痕会变淡。gamma测了0.5,0.4,0.30，差异都不大***，最好的参数为kernel_size=31, sigma=3.0, wavelength=10.0, gamma=0.5, psi=0.0, response_mode="absolute", normalize_kernel_l2=True


if __name__ == "__main__":
    main()
