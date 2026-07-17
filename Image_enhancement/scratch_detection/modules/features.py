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


CONNECTIVITY_VALUES = {4, 8}


@dataclass(frozen=True)
class FeatureExtractionConfig:
    enabled: bool = True
    connectivity: int = 8
    minimum_component_area: int = 1
    calculate_width_features: bool = True
    calculate_response_features: bool = True

    def validate(self) -> None:
        if self.connectivity not in CONNECTIVITY_VALUES:
            raise ValueError("connectivity must be either 4 or 8")
        if (
            not isinstance(self.minimum_component_area, int)
            or self.minimum_component_area < 1
        ):
            raise ValueError("minimum_component_area must be at least 1")


@dataclass(frozen=True)
class FeaturePreviewConfig:
    crop_padding: int = 15
    skeleton_preview_thickness: int = 3
    topology_marker_radius: int = 3

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
        if (
            not isinstance(self.topology_marker_radius, int)
            or self.topology_marker_radius < 0
        ):
            raise ValueError("topology_marker_radius must be non-negative")


@dataclass(frozen=True)
class ComponentFeatures:
    component_id: int
    area_pixels: int
    bbox_x: int
    bbox_y: int
    bbox_width: int
    bbox_height: int
    centroid_x: float
    centroid_y: float
    rotated_length_pixels: float
    rotated_width_pixels: float
    aspect_ratio: float
    skeleton_pixel_count: int
    skeleton_length_pixels: float
    endpoint_count: int
    branch_point_count: int
    orientation_degrees: float | None
    linearity: float
    mean_width_pixels: float | None
    max_width_pixels: float | None
    mean_response: float | None
    max_response: float | None
    strong_response_fraction: float | None
    endpoint_coordinates_xy: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class FeatureExtractionResult:
    features: tuple[ComponentFeatures, ...]
    component_labels: np.ndarray
    retained_component_mask: np.ndarray
    endpoint_mask: np.ndarray
    branch_point_mask: np.ndarray


def _skeleton_graph_length(skeleton_pixels: np.ndarray) -> float:
    pixel_count = int(np.count_nonzero(skeleton_pixels))
    if pixel_count == 0:
        return 0.0

    horizontal_edges = np.count_nonzero(
        skeleton_pixels[:, :-1] & skeleton_pixels[:, 1:]
    )
    vertical_edges = np.count_nonzero(
        skeleton_pixels[:-1, :] & skeleton_pixels[1:, :]
    )
    diagonal_edges = np.count_nonzero(
        skeleton_pixels[:-1, :-1] & skeleton_pixels[1:, 1:]
    )
    diagonal_edges += np.count_nonzero(
        skeleton_pixels[:-1, 1:] & skeleton_pixels[1:, :-1]
    )
    return float(
        1.0
        + horizontal_edges
        + vertical_edges
        + (np.sqrt(2.0) * diagonal_edges)
    )


def _orientation_and_linearity(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> tuple[float | None, float]:
    if x_coordinates.size < 2:
        return None, 0.0

    coordinates = np.column_stack((x_coordinates, y_coordinates)).astype(
        np.float64,
        copy=False,
    )
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / coordinates.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_value = float(eigenvalues[-1])
    minor_value = float(eigenvalues[0])
    if major_value <= np.finfo(np.float64).eps:
        return None, 0.0

    major_vector = eigenvectors[:, -1]
    orientation = float(
        np.degrees(np.arctan2(major_vector[1], major_vector[0])) % 180.0
    )
    linearity = float(
        np.clip(
            (major_value - minor_value) / major_value,
            0.0,
            1.0,
        )
    )
    return orientation, linearity


def _rotated_dimensions(
    x_coordinates: np.ndarray,
    y_coordinates: np.ndarray,
) -> tuple[float, float, float]:
    if x_coordinates.size == 1:
        return 1.0, 1.0, 1.0

    points = np.column_stack((x_coordinates, y_coordinates)).astype(np.float32)
    _, (rectangle_width, rectangle_height), _ = cv2.minAreaRect(points)
    length = float(max(rectangle_width, rectangle_height) + 1.0)
    width = float(min(rectangle_width, rectangle_height) + 1.0)
    return length, width, length / max(width, np.finfo(np.float32).eps)


def extract_component_features(
    binary_image: np.ndarray,
    skeleton_image: np.ndarray,
    processing_mask: np.ndarray,
    line_response_image: np.ndarray | None,
    strong_threshold: float | None,
    config: FeatureExtractionConfig,
) -> FeatureExtractionResult:
    """Measure binary components without filtering or joining them."""
    config.validate()
    if binary_image.ndim != 2 or skeleton_image.ndim != 2:
        raise ValueError("binary_image and skeleton_image must be one-channel")
    if binary_image.shape != skeleton_image.shape:
        raise ValueError("binary_image and skeleton_image must have the same size")
    if processing_mask.ndim != 2 or processing_mask.shape != binary_image.shape:
        raise ValueError("processing_mask must match the input image size")
    if config.calculate_response_features:
        if line_response_image is None:
            raise ValueError(
                "line_response_image is required when response features are enabled"
            )
        if line_response_image.ndim != 2 or line_response_image.shape != binary_image.shape:
            raise ValueError("line_response_image must match the input image size")
        if not np.isfinite(line_response_image).all():
            raise ValueError("line_response_image must contain only finite values")
    if strong_threshold is not None and not np.isfinite(strong_threshold):
        raise ValueError("strong_threshold must be finite or None")

    shape = binary_image.shape
    empty_labels = np.zeros(shape, dtype=np.int32)
    empty_mask = np.zeros(shape, dtype=np.uint8)
    if not config.enabled:
        return FeatureExtractionResult(
            features=(),
            component_labels=empty_labels,
            retained_component_mask=empty_mask.copy(),
            endpoint_mask=empty_mask.copy(),
            branch_point_mask=empty_mask.copy(),
        )

    foreground = (binary_image > 0) & (processing_mask > 0)
    skeleton_pixels = (skeleton_image > 0) & foreground
    foreground_uint8 = foreground.astype(np.uint8)
    component_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        foreground_uint8,
        connectivity=config.connectivity,
        ltype=cv2.CV_32S,
    )

    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(
        skeleton_pixels.astype(np.uint8),
        cv2.CV_16U,
        neighbor_kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    endpoint_pixels = skeleton_pixels & (neighbor_count == 1)
    branch_pixels = skeleton_pixels & (neighbor_count >= 3)
    distance_map = None
    if config.calculate_width_features:
        distance_map = cv2.distanceTransform(
            foreground_uint8,
            cv2.DIST_L2,
            cv2.DIST_MASK_PRECISE,
        )

    features: list[ComponentFeatures] = []
    retained_labels = np.zeros(component_count, dtype=bool)
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < config.minimum_component_area:
            continue
        retained_labels[component_id] = True

        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        local_labels = labels[y : y + height, x : x + width]
        component_pixels = local_labels == component_id
        local_y, local_x = np.nonzero(component_pixels)
        global_x = local_x + x
        global_y = local_y + y

        local_skeleton = (
            skeleton_pixels[y : y + height, x : x + width]
            & component_pixels
        )
        skeleton_y, skeleton_x = np.nonzero(local_skeleton)
        if skeleton_x.size:
            orientation_x = skeleton_x + x
            orientation_y = skeleton_y + y
        else:
            orientation_x = global_x
            orientation_y = global_y

        rotated_length, rotated_width, aspect_ratio = _rotated_dimensions(
            global_x,
            global_y,
        )
        orientation, linearity = _orientation_and_linearity(
            orientation_x,
            orientation_y,
        )

        local_endpoints = (
            endpoint_pixels[y : y + height, x : x + width]
            & component_pixels
        )
        endpoint_y, endpoint_x = np.nonzero(local_endpoints)
        endpoint_coordinates = tuple(
            (int(endpoint_x[index] + x), int(endpoint_y[index] + y))
            for index in range(endpoint_x.size)
        )
        local_branches = (
            branch_pixels[y : y + height, x : x + width]
            & component_pixels
        )

        mean_width = None
        max_width = None
        if distance_map is not None:
            width_sample_pixels = local_skeleton
            if not np.any(width_sample_pixels):
                width_sample_pixels = component_pixels
            distance_values = distance_map[y : y + height, x : x + width][
                width_sample_pixels
            ]
            width_values = np.maximum((2.0 * distance_values) - 1.0, 1.0)
            mean_width = float(np.mean(width_values))
            max_width = float(np.max(width_values))

        mean_response = None
        max_response = None
        strong_response_fraction = None
        if config.calculate_response_features:
            response_values = line_response_image[
                y : y + height,
                x : x + width,
            ][component_pixels]
            mean_response = float(np.mean(response_values))
            max_response = float(np.max(response_values))
            if strong_threshold is not None:
                strong_response_fraction = float(
                    np.count_nonzero(response_values >= strong_threshold)
                    / response_values.size
                )

        features.append(
            ComponentFeatures(
                component_id=component_id,
                area_pixels=area,
                bbox_x=x,
                bbox_y=y,
                bbox_width=width,
                bbox_height=height,
                centroid_x=float(centroids[component_id, 0]),
                centroid_y=float(centroids[component_id, 1]),
                rotated_length_pixels=rotated_length,
                rotated_width_pixels=rotated_width,
                aspect_ratio=aspect_ratio,
                skeleton_pixel_count=int(np.count_nonzero(local_skeleton)),
                skeleton_length_pixels=_skeleton_graph_length(local_skeleton),
                endpoint_count=len(endpoint_coordinates),
                branch_point_count=int(np.count_nonzero(local_branches)),
                orientation_degrees=orientation,
                linearity=linearity,
                mean_width_pixels=mean_width,
                max_width_pixels=max_width,
                mean_response=mean_response,
                max_response=max_response,
                strong_response_fraction=strong_response_fraction,
                endpoint_coordinates_xy=endpoint_coordinates,
            )
        )

    retained_pixels = retained_labels[labels]
    retained_component_mask = np.where(retained_pixels, 255, 0).astype(np.uint8)
    endpoint_mask = np.where(
        endpoint_pixels & retained_pixels,
        255,
        0,
    ).astype(np.uint8)
    branch_point_mask = np.where(
        branch_pixels & retained_pixels,
        255,
        0,
    ).astype(np.uint8)

    return FeatureExtractionResult(
        features=tuple(features),
        component_labels=labels,
        retained_component_mask=retained_component_mask,
        endpoint_mask=endpoint_mask,
        branch_point_mask=branch_point_mask,
    )


def create_feature_overlay(
    original_image: np.ndarray,
    skeleton_image: np.ndarray,
    feature_result: FeatureExtractionResult,
    config: FeaturePreviewConfig,
) -> np.ndarray:
    """Draw skeletons, endpoints, and branch pixels for visual verification."""
    config.validate()
    if original_image.ndim != 3 or original_image.shape[2] != 3:
        raise ValueError("original_image must be a BGR image")
    if skeleton_image.shape != original_image.shape[:2]:
        raise ValueError("skeleton_image must match the original image size")

    skeleton_visible = np.where(
        (skeleton_image > 0) & (feature_result.retained_component_mask > 0),
        255,
        0,
    ).astype(np.uint8)
    if config.skeleton_preview_thickness > 1:
        thickness = config.skeleton_preview_thickness
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (thickness, thickness),
        )
        skeleton_visible = cv2.dilate(skeleton_visible, kernel, iterations=1)

    endpoint_visible = feature_result.endpoint_mask
    branch_visible = feature_result.branch_point_mask
    if config.topology_marker_radius > 0:
        marker_size = (2 * config.topology_marker_radius) + 1
        marker_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (marker_size, marker_size),
        )
        endpoint_visible = cv2.dilate(
            endpoint_visible,
            marker_kernel,
            iterations=1,
        )
        branch_visible = cv2.dilate(
            branch_visible,
            marker_kernel,
            iterations=1,
        )

    overlay = original_image.copy()
    overlay[skeleton_visible > 0] = (0, 255, 0)
    overlay[branch_visible > 0] = (255, 0, 0)
    overlay[endpoint_visible > 0] = (0, 0, 255)
    return overlay


def write_feature_csv(
    output_path: Path,
    features: tuple[ComponentFeatures, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(ComponentFeatures.__dataclass_fields__)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for feature in features:
            row = asdict(feature)
            row["endpoint_coordinates_xy"] = ";".join(
                f"{x}:{y}" for x, y in feature.endpoint_coordinates_xy
            )
            writer.writerow(row)


def _create_parameter_record_dir(
    output_dir: Path,
    config: FeatureExtractionConfig,
) -> Path:
    config.validate()
    parameter_dir = output_dir / (
        f"connectivity={config.connectivity},"
        f"minimum_area={config.minimum_component_area},"
        f"width_features={config.calculate_width_features},"
        f"response_features={config.calculate_response_features}"
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
    preview_config: FeaturePreviewConfig,
    writer_config: ResultWriterConfig,
) -> tuple[int, list[Path], Path]:
    """Run the image-writing feature workflow for this module's main entry."""
    scan_result = scan_image_json_pairs(dataset_dir)
    writer = ResultWriter(dataset_dir, writer_config)
    _create_parameter_record_dir(writer.output_dir, feature_config)

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
        overlay = create_feature_overlay(
            original_image,
            skeleton_result.skeleton_image,
            feature_result,
            preview_config,
        )
        binary_preview = cv2.cvtColor(
            threshold_result.binary_image,
            cv2.COLOR_GRAY2BGR,
        )
        cropped_original, cropped_binary, cropped_overlay = (
            crop_images_to_mask_bounding_rect(
                mask_result.original_mask,
                original_image,
                binary_preview,
                overlay,
                padding=preview_config.crop_padding,
            )
        )
        writer.save_result(
            pair.image_path.stem,
            cropped_original,
            cropped_overlay,
        )
        comparison = np.hstack(
            (cropped_original, cropped_binary, cropped_overlay)
        )
        write_image(
            writer.output_dir
            / f"{pair.image_path.stem}{writer_config.comparison_suffix}",
            comparison,
        )
        write_feature_csv(
            writer.output_dir / f"{pair.image_path.stem}_features.csv",
            feature_result.features,
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
        sigmas=(2.0,2.5),
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
        minimum_component_area=5,      #面积小于该值的区域不写入特征记录。
        calculate_width_features=True,       #使用阈值二值图和距离变换计算平均、最大宽度。
        calculate_response_features=True,       #统计连通域内 Frangi/Gabor 的平均响应、最大响应和强响应比例。
    )
    preview_config = FeaturePreviewConfig(      #这个用于预览的参数，和真正的特征计算无关。
        crop_padding=15,           #Mask外接矩形向外扩展量，只影响保存图片。
        skeleton_preview_thickness=3,    #叠加图中的骨架显示宽度，不改变真实骨架。
        topology_marker_radius=3,         #端点和分支点的显示半径，不影响特征数值。
    )
    writer_config = ResultWriterConfig(
        output_folder_name="features_results",
        run_number_width=3,
        max_run_number=999,
        save_comparison=False,
        save_processed=True,
        comparison_suffix="_features_comparison.png",
        processed_suffix="_features_overlay.png",
    )

    processed_count, images_without_json, output_dir = _run_standalone_dataset(
        dataset_dir,
        mask_config,
        background_config,
        frangi_config,
        threshold_config,
        skeleton_config,
        feature_config,
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
