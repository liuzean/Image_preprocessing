from __future__ import annotations

import csv
import sys
from dataclasses import asdict, dataclass
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
from Image_enhancement.scratch_detection.modules.features import (  # noqa: E402
    ComponentFeatures,
    FeatureExtractionConfig,
    FeatureExtractionResult,
    extract_component_features,
)
from Image_enhancement.scratch_detection.modules.frangi import (  # noqa: E402
    FrangiConfig,
    enhance_frangi,
)
from Image_enhancement.scratch_detection.modules.grayscale import (  # noqa: E402
    convert_to_grayscale_float32,
)
from Image_enhancement.scratch_detection.modules.morphology import (  # noqa: E402
    SkeletonizationConfig,
    skeletonize_binary_candidates,
)
from Image_enhancement.scratch_detection.modules.threshold import (  # noqa: E402
    HysteresisThresholdConfig,
    apply_masked_hysteresis_threshold,
)


@dataclass(frozen=True)
class CandidateFilterConfig:
    enabled: bool = True
    minimum_path_length: float = 10.0
    long_min_path_length: float = 150.0
    short_min_longest_path_aspect_ratio: float = 5.0
    short_min_linearity: float = 0.90

    def validate(self) -> None:
        if (
            not np.isfinite(self.minimum_path_length)
            or self.minimum_path_length <= 0.0
        ):
            raise ValueError("minimum_path_length must be greater than 0")
        if (
            not np.isfinite(self.long_min_path_length)
            or self.long_min_path_length < self.minimum_path_length
        ):
            raise ValueError(
                "long_min_path_length must be at least minimum_path_length"
            )
        if (
            not np.isfinite(self.short_min_longest_path_aspect_ratio)
            or self.short_min_longest_path_aspect_ratio < 1.0
        ):
            raise ValueError(
                "short_min_longest_path_aspect_ratio must be at least 1"
            )
        if (
            not np.isfinite(self.short_min_linearity)
            or not 0.0 <= self.short_min_linearity <= 1.0
        ):
            raise ValueError("short_min_linearity must be in the interval [0, 1]")


@dataclass(frozen=True)
class CandidateFilterPreviewConfig:
    crop_padding: int = 15

    def validate(self) -> None:
        if not isinstance(self.crop_padding, int) or self.crop_padding < 0:
            raise ValueError("crop_padding must be a non-negative integer")


@dataclass(frozen=True)
class CandidateDecision:
    component_id: int
    kept: bool
    length_group: str
    reason: str
    longest_path_length_pixels: float
    skeleton_aspect_ratio: float
    longest_path_aspect_ratio: float
    linearity: float
    branch_density: float
    path_coverage: float


@dataclass(frozen=True)
class CandidateFilterResult:
    filtered_binary_image: np.ndarray
    filtered_skeleton_image: np.ndarray
    kept_component_mask: np.ndarray
    decisions: tuple[CandidateDecision, ...]


def _decide_candidate(
    component: ComponentFeatures,
    config: CandidateFilterConfig,
) -> CandidateDecision:
    path_length = component.longest_path_length_pixels
    is_long = path_length >= config.long_min_path_length
    if not config.enabled:
        kept = True
        reason = "filter_disabled"
        length_group = "long" if is_long else "short"
    elif path_length < config.minimum_path_length:
        kept = False
        reason = "path_length_below_minimum"
        length_group = "below_minimum"
    elif is_long:
        kept = True
        reason = "long_candidate_passed"
        length_group = "long"
    else:
        length_group = "short"
        if (
            component.longest_path_aspect_ratio
            < config.short_min_longest_path_aspect_ratio
        ):
            kept = False
            reason = "short_longest_path_aspect_ratio_too_low"
        elif component.linearity < config.short_min_linearity:
            kept = False
            reason = "short_linearity_too_low"
        else:
            kept = True
            reason = "short_candidate_passed"

    return CandidateDecision(
        component_id=component.component_id,
        kept=kept,
        length_group=length_group,
        reason=reason,
        longest_path_length_pixels=component.longest_path_length_pixels,
        skeleton_aspect_ratio=component.skeleton_aspect_ratio,
        longest_path_aspect_ratio=component.longest_path_aspect_ratio,
        linearity=component.linearity,
        branch_density=component.branch_density,
        path_coverage=component.path_coverage,
    )


def filter_candidates(
    binary_image: np.ndarray,
    skeleton_image: np.ndarray,
    feature_result: FeatureExtractionResult,
    config: CandidateFilterConfig,
) -> CandidateFilterResult:
    """Apply staged rules without changing the stored component features."""
    config.validate()
    if binary_image.ndim != 2 or skeleton_image.ndim != 2:
        raise ValueError("binary_image and skeleton_image must be one-channel")
    if binary_image.shape != skeleton_image.shape:
        raise ValueError("binary_image and skeleton_image must have the same size")
    if feature_result.component_labels.shape != binary_image.shape:
        raise ValueError("component_labels must match the input image size")

    decisions = tuple(
        _decide_candidate(component, config)
        for component in feature_result.features
    )
    label_count = int(feature_result.component_labels.max()) + 1
    kept_labels = np.zeros(label_count, dtype=bool)
    kept_long_labels = np.zeros(label_count, dtype=bool)
    kept_short_labels = np.zeros(label_count, dtype=bool)
    for decision in decisions:
        if decision.kept:
            kept_labels[decision.component_id] = True
            if decision.length_group == "long":
                kept_long_labels[decision.component_id] = True
            else:
                kept_short_labels[decision.component_id] = True

    kept_pixels = kept_labels[feature_result.component_labels]
    kept_long_pixels = kept_long_labels[feature_result.component_labels]
    kept_short_pixels = kept_short_labels[feature_result.component_labels]
    kept_component_mask = np.where(kept_pixels, 255, 0).astype(np.uint8)
    filtered_binary_image = np.where(
        (binary_image > 0) & kept_pixels,
        255,
        0,
    ).astype(np.uint8)
    if config.enabled:
        filtered_skeleton_pixels = (
            ((skeleton_image > 0) & kept_long_pixels)
            | ((feature_result.longest_path_mask > 0) & kept_short_pixels)
        )
    else:
        filtered_skeleton_pixels = (skeleton_image > 0) & kept_pixels
    filtered_skeleton_image = np.where(
        filtered_skeleton_pixels,
        255,
        0,
    ).astype(np.uint8)
    return CandidateFilterResult(
        filtered_binary_image=filtered_binary_image,
        filtered_skeleton_image=filtered_skeleton_image,
        kept_component_mask=kept_component_mask,
        decisions=decisions,
    )


def write_decision_csv(
    output_path: Path,
    decisions: tuple[CandidateDecision, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(CandidateDecision.__dataclass_fields__)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for decision in decisions:
            writer.writerow(asdict(decision))


def _create_parameter_record_dir(
    output_dir: Path,
    config: CandidateFilterConfig,
) -> Path:
    config.validate()
    parameter_dir = output_dir / (
        f"minimum_path={config.minimum_path_length},"
        f"long_path={config.long_min_path_length},"
        f"short_path_aspect={config.short_min_longest_path_aspect_ratio},"
        f"short_linearity={config.short_min_linearity}"
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
    feature_config: FeatureExtractionConfig,
    filter_config: CandidateFilterConfig,
    preview_config: CandidateFilterPreviewConfig,
    writer_config: ResultWriterConfig,
) -> tuple[int, list[Path], Path]:
    """Run the image-writing candidate-filter workflow for this module."""
    scan_result = scan_image_json_pairs(dataset_dir)
    writer = ResultWriter(dataset_dir, writer_config)
    _create_parameter_record_dir(writer.output_dir, filter_config)

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
        line_response_image = frangi_result.bright_response_image
        threshold_result = apply_masked_hysteresis_threshold(
            line_response_image,
            mask_result.eroded_mask,
            threshold_config,
        )
        skeleton_result = skeletonize_binary_candidates(
            threshold_result.binary_image,
            mask_result.eroded_mask,
            skeleton_config,
        )
        feature_result = extract_component_features(
            threshold_result.binary_image,
            skeleton_result.skeleton_image,
            mask_result.eroded_mask,
            line_response_image,
            threshold_result.high_threshold,
            feature_config,
        )
        filter_result = filter_candidates(
            threshold_result.binary_image,
            skeleton_result.skeleton_image,
            feature_result,
            filter_config,
        )

        before_preview = cv2.cvtColor(
            threshold_result.binary_image,
            cv2.COLOR_GRAY2BGR,
        )
        after_preview = cv2.cvtColor(
            filter_result.filtered_binary_image,
            cv2.COLOR_GRAY2BGR,
        )
        (
            cropped_original,
            cropped_before,
            cropped_after,
            cropped_skeleton,
        ) = (
            crop_images_to_mask_bounding_rect(
                mask_result.original_mask,
                original_image,
                before_preview,
                after_preview,
                filter_result.filtered_skeleton_image,
                padding=preview_config.crop_padding,
            )
        )
        writer.save_result(
            pair.image_path.stem,
            cropped_original,
            cropped_after,
        )
        comparison = np.hstack(
            (cropped_original, cropped_before, cropped_after)
        )
        write_image(
            writer.output_dir
            / f"{pair.image_path.stem}{writer_config.comparison_suffix}",
            comparison,
        )
        write_image(
            writer.output_dir
            / f"{pair.image_path.stem}_candidate_filter_skeleton.png",
            cropped_skeleton,
        )
        write_decision_csv(
            writer.output_dir
            / f"{pair.image_path.stem}_candidate_filter_decisions.csv",
            filter_result.decisions,
        )
        processed_count += 1

    return processed_count, scan_result.images_without_json, writer.output_dir


def main() -> None:
    dataset_dir = Path(r"E:\projects\datasets\Power_box\Scratch_old")
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
        sigmas=(2.0, 2.5),
        alpha=0.5,
        beta=0.3,
        gamma=None,
        detect_bright_ridges=True,
        detect_dark_ridges=False,
        boundary_mode="reflect",
        constant_value=0.0,
    )
    threshold_config = HysteresisThresholdConfig(
        enabled=True,
        high_percentile=97.0,
        low_threshold_ratio=0.4,
        connectivity=8,
    )
    skeleton_config = SkeletonizationConfig(
        enabled=True,
        method="zhang",
    )
    feature_config = FeatureExtractionConfig(
        enabled=True,
        connectivity=8,
        minimum_component_area=1,
        calculate_width_features=True,
        calculate_response_features=True,
    )
    filter_config = CandidateFilterConfig(
        enabled=True,
        minimum_path_length=10.0,                    #最短连续路径。低于该值的骨架直接排除。
        long_min_path_length=150.0,                  #长候选分界。，达到该长度后直接保留，不再检查长宽比和弯曲程度。
        short_min_longest_path_aspect_ratio=3.0,     #短候选骨架最小长宽比。
        short_min_linearity=0.30,                    #短候选骨架最小线性度。越大，筛选更严格，纹理减少，但划痕更容易断裂。减小：可以连接划痕的浅色部分，但会带入更多纹理。
    )
    preview_config = CandidateFilterPreviewConfig(
        crop_padding=15,
    )
    writer_config = ResultWriterConfig(
        output_folder_name="candidate_filter_results",
        run_number_width=3,
        max_run_number=999,
        save_comparison=False,
        save_processed=True,
        comparison_suffix="_candidate_filter_comparison.png",
        processed_suffix="_candidate_filter_binary.png",
    )

    processed_count, images_without_json, output_dir = _run_standalone_dataset(
        dataset_dir,
        mask_config,
        background_config,
        frangi_config,
        threshold_config,
        skeleton_config,
        feature_config,
        filter_config,
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
