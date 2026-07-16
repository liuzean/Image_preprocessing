from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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
    FrangiPreviewConfig,
    create_response_preview,
    enhance_frangi,
)
from Image_enhancement.scratch_detection.modules.grayscale import (  # noqa: E402
    convert_to_grayscale_float32,
)


CONNECTIVITY_VALUES = {4, 8}


@dataclass(frozen=True)
class HysteresisThresholdConfig:
    enabled: bool = True
    high_percentile: float = 99.5
    low_threshold_ratio: float = 0.4
    connectivity: int = 8

    def validate(self) -> None:
        if not 0.0 < self.high_percentile <= 100.0:
            raise ValueError("high_percentile must be in the interval (0, 100]")
        if not 0.0 < self.low_threshold_ratio <= 1.0:
            raise ValueError("low_threshold_ratio must be in the interval (0, 1]")
        if self.connectivity not in CONNECTIVITY_VALUES:
            raise ValueError("connectivity must be either 4 or 8")


@dataclass(frozen=True)
class HysteresisThresholdResult:
    binary_image: np.ndarray
    low_threshold: float
    high_threshold: float


def apply_masked_hysteresis_threshold(
    response_float32: np.ndarray,
    processing_mask: np.ndarray,
    config: HysteresisThresholdConfig,
) -> HysteresisThresholdResult:
    """Keep weak response components connected to high-confidence seeds."""
    config.validate()
    if response_float32.ndim != 2:
        raise ValueError("response_float32 must be a one-channel image")
    if processing_mask.ndim != 2 or processing_mask.shape != response_float32.shape:
        raise ValueError("processing_mask must match the response image size")

    response = response_float32.astype(np.float32, copy=False)
    if not np.isfinite(response).all():
        raise ValueError("response_float32 must contain only finite values")

    mask_pixels = processing_mask > 0
    if not np.any(mask_pixels):
        raise ValueError("processing_mask must contain at least one foreground pixel")

    if not config.enabled:
        binary = np.zeros(response.shape, dtype=np.uint8)
        binary[mask_pixels] = 255
        return HysteresisThresholdResult(
            binary_image=binary,
            low_threshold=0.0,
            high_threshold=0.0,
        )

    masked_response = response[mask_pixels]
    high_threshold = float(
        np.percentile(masked_response, config.high_percentile)
    )
    low_threshold = high_threshold * config.low_threshold_ratio

    binary = np.zeros(response.shape, dtype=np.uint8)
    if high_threshold <= np.finfo(np.float32).eps:
        return HysteresisThresholdResult(
            binary_image=binary,
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )

    weak_candidates = mask_pixels & (response >= low_threshold)
    strong_seeds = mask_pixels & (response >= high_threshold)
    component_count, component_labels = cv2.connectedComponents(
        weak_candidates.astype(np.uint8),
        connectivity=config.connectivity,
    )
    if component_count > 1 and np.any(strong_seeds):
        retained_labels = np.unique(component_labels[strong_seeds])
        retained_labels = retained_labels[retained_labels != 0]
        if retained_labels.size:
            retained_components = np.isin(component_labels, retained_labels)
            binary[retained_components] = 255

    return HysteresisThresholdResult(
        binary_image=binary,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )


def _create_parameter_record_dir(
    output_dir: Path,
    config: HysteresisThresholdConfig,
) -> Path:
    config.validate()
    parameter_dir = output_dir / (
        f"high_percentile={config.high_percentile},"
        f"low_threshold_ratio={config.low_threshold_ratio},"
        f"connectivity={config.connectivity}"
    )
    parameter_dir.mkdir(parents=False, exist_ok=False)
    return parameter_dir


def _run_standalone_dataset(
    dataset_dir: Path,
    mask_config: ErodeMaskConfig,
    background_config: BackgroundCorrectionConfig,
    frangi_config: FrangiConfig,
    threshold_config: HysteresisThresholdConfig,
    preview_config: FrangiPreviewConfig,
    writer_config: ResultWriterConfig,
) -> tuple[int, list[Path], Path]:
    """Run the image-writing preview workflow for this module's main entry."""
    scan_result = scan_image_json_pairs(dataset_dir)
    writer = ResultWriter(dataset_dir, writer_config)
    _create_parameter_record_dir(writer.output_dir, threshold_config)

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

        response_preview = create_response_preview(
            frangi_result.bright_response_image,
            mask_result.eroded_mask,
            preview_config,
        )
        binary_preview = cv2.cvtColor(
            threshold_result.binary_image,
            cv2.COLOR_GRAY2BGR,
        )
        cropped_original, cropped_response, cropped_binary = (
            crop_images_to_mask_bounding_rect(
                mask_result.original_mask,
                original_image,
                response_preview,
                binary_preview,
                padding=preview_config.crop_padding,
            )
        )

        writer.save_result(
            pair.image_path.stem,
            cropped_original,
            cropped_binary,
        )
        comparison = np.hstack(
            (cropped_original, cropped_response, cropped_binary)
        )
        write_image(
            writer.output_dir / f"{pair.image_path.stem}_threshold_stages.png",
            comparison,
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
        high_percentile=97.5,     #在 Mask 内的 Frangi bright 响应中计算高阈值。，增大：强种子更少，误检减少，但浅划痕可能没有种子。，减小：保留更多浅划痕，但银漆纹理也会增加。
        low_threshold_ratio=0.6,    #低阈值与高阈值的比例：，增大：筛选更严格，纹理减少，但划痕更容易断裂。减小：可以连接划痕的浅色部分，但会带入更多纹理。，
        connectivity=8,  #8：水平、垂直、斜向都视为连通，适合多方向划痕，4：只允许水平和垂直连通，筛选更严格，但斜划痕容易断裂。
    )
    preview_config = FrangiPreviewConfig(
        lower_percentile=0.0,
        upper_percentile=99.5,
        crop_padding=15,
    )
    writer_config = ResultWriterConfig(
        output_folder_name="threshold_results",
        run_number_width=3,
        max_run_number=999,
        save_comparison=False,
        save_processed=True,
        comparison_suffix="_threshold_comparison.png",
        processed_suffix="_threshold_binary.png",
    )

    processed_count, images_without_json, output_dir = _run_standalone_dataset(
        dataset_dir,
        mask_config,
        background_config,
        frangi_config,
        threshold_config,
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
