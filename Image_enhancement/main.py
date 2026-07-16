from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Image_enhancement.scratch_detection.io.dataset import (
    DEFAULT_DATASET_DIR,
    read_annotation,
    read_image,
    scan_image_json_pairs,
)
from Image_enhancement.scratch_detection.modules.background import (
    BackgroundCorrectionConfig,
)
from Image_enhancement.scratch_detection.modules.erode_mask import ErodeMaskConfig
from Image_enhancement.scratch_detection.modules.gabor import (
    MultiDirectionGaborConfig,
)
from Image_enhancement.scratch_detection.modules.frangi import FrangiConfig
from Image_enhancement.scratch_detection.modules.threshold import (
    HysteresisThresholdConfig,
)
from Image_enhancement.scratch_detection.pipeline import ScratchDetectionPipeline


def run_pipeline(dataset_dir: Path, pipeline: ScratchDetectionPipeline) -> int:
    scan_result = scan_image_json_pairs(dataset_dir)

    processed_count = 0
    for pair in scan_result.pairs:
        image = read_image(pair.image_path)
        annotation = read_annotation(pair.json_path)
        pipeline.run(image, annotation)
        processed_count += 1

    if scan_result.images_without_json:
        print("Skipped images without a same-name JSON file:")
        for image_path in scan_result.images_without_json:
            print(f"  {image_path.name}")

    return processed_count


def main() -> None:
    dataset_dir = DEFAULT_DATASET_DIR
    line_enhancement_method = "frangi"
    frangi_response_mode = "bright"

    erode_mask_config = ErodeMaskConfig(
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
        sigma=3.0,
        wavelength=6.0,
        gamma=0.5,
        psi=0.0,
        response_mode="absolute",
        normalize_kernel_l2=True,
    )
    frangi_config = FrangiConfig(
        enabled=True,
        sigmas=(2.5, 3.0),
        alpha=0.5,
        beta=0.2,
        gamma=None,
        detect_bright_ridges=True,
        detect_dark_ridges=True,
        boundary_mode="reflect",
        constant_value=0.0,
    )
    threshold_config = HysteresisThresholdConfig(
        enabled=True,
        high_percentile=97.5,
        low_threshold_ratio=0.4,
        connectivity=8,
    )
    pipeline = ScratchDetectionPipeline(
        erode_mask_config,
        background_config,
        gabor_config,
        frangi_config,
        threshold_config,
        line_enhancement_method=line_enhancement_method,
        frangi_response_mode=frangi_response_mode,
    )

    processed_count = run_pipeline(dataset_dir, pipeline)
    print(
        f"Pipeline processed {processed_count} images with "
        f"{line_enhancement_method}."
    )
    print("No final images were saved because later processing stages are not ready.")


if __name__ == "__main__":
    main()
