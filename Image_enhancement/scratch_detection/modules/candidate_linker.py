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
)
from Image_enhancement.scratch_detection.modules.candidate_filter import (  # noqa: E402
    CandidateFilterConfig,
)
from Image_enhancement.scratch_detection.modules.erode_mask import (  # noqa: E402
    ErodeMaskConfig,
)
from Image_enhancement.scratch_detection.modules.features import (  # noqa: E402
    FeatureExtractionConfig,
)
from Image_enhancement.scratch_detection.modules.frangi import (  # noqa: E402
    FrangiConfig,
)
from Image_enhancement.scratch_detection.modules.gabor import (  # noqa: E402
    MultiDirectionGaborConfig,
)
from Image_enhancement.scratch_detection.modules.morphology import (  # noqa: E402
    SkeletonizationConfig,
)
from Image_enhancement.scratch_detection.modules.threshold import (  # noqa: E402
    HysteresisThresholdConfig,
)
from Image_enhancement.scratch_detection.pipeline import (  # noqa: E402
    ScratchDetectionPipeline,
)


@dataclass(frozen=True)
class CandidateLinkerConfig:
    enabled: bool = True
    maximum_gap_distance: float = 20.0
    direction_window_radius: int = 7
    maximum_direction_difference_degrees: float = 20.0
    maximum_connection_angle_degrees: float = 25.0
    maximum_lateral_offset_pixels: float = 4.0
    minimum_gap_response_fraction: float = 0.25
    require_mutual_best_match: bool = True
    link_thickness: int = 1

    def validate(self) -> None:
        if (
            not np.isfinite(self.maximum_gap_distance)
            or self.maximum_gap_distance <= 0.0
        ):
            raise ValueError("maximum_gap_distance must be greater than 0")
        if (
            not isinstance(self.direction_window_radius, int)
            or self.direction_window_radius < 2
        ):
            raise ValueError("direction_window_radius must be at least 2")
        if (
            not np.isfinite(self.maximum_direction_difference_degrees)
            or not 0.0
            <= self.maximum_direction_difference_degrees
            <= 90.0
        ):
            raise ValueError(
                "maximum_direction_difference_degrees must be in [0, 90]"
            )
        if (
            not np.isfinite(self.maximum_connection_angle_degrees)
            or not 0.0 <= self.maximum_connection_angle_degrees <= 90.0
        ):
            raise ValueError(
                "maximum_connection_angle_degrees must be in [0, 90]"
            )
        if (
            not np.isfinite(self.maximum_lateral_offset_pixels)
            or self.maximum_lateral_offset_pixels < 0.0
        ):
            raise ValueError(
                "maximum_lateral_offset_pixels must be non-negative"
            )
        if (
            not np.isfinite(self.minimum_gap_response_fraction)
            or not 0.0 <= self.minimum_gap_response_fraction <= 1.0
        ):
            raise ValueError(
                "minimum_gap_response_fraction must be in [0, 1]"
            )
        if (
            not isinstance(self.link_thickness, int)
            or self.link_thickness < 1
        ):
            raise ValueError("link_thickness must be at least 1")


@dataclass(frozen=True)
class CandidateLinkerPreviewConfig:
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
class CandidateLink:
    first_component_id: int
    second_component_id: int
    first_endpoint_x: int
    first_endpoint_y: int
    second_endpoint_x: int
    second_endpoint_y: int
    gap_distance_pixels: float
    direction_difference_degrees: float
    first_connection_angle_degrees: float
    second_connection_angle_degrees: float
    lateral_offset_pixels: float
    gap_response_fraction: float
    score: float


@dataclass(frozen=True)
class CandidateLinkerResult:
    linked_skeleton_image: np.ndarray
    link_mask: np.ndarray
    links: tuple[CandidateLink, ...]
    endpoint_count: int
    proposal_count: int


@dataclass(frozen=True)
class _Endpoint:
    endpoint_id: int
    component_id: int
    x: int
    y: int
    tangent_x: float
    tangent_y: float


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = np.arange(size, dtype=np.int32)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, value: int) -> int:
        parent = int(self.parent[value])
        while parent != value:
            grandparent = int(self.parent[parent])
            self.parent[value] = grandparent
            value = parent
            parent = grandparent
        return value

    def union(self, first: int, second: int) -> bool:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        if self.rank[first_root] < self.rank[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        if self.rank[first_root] == self.rank[second_root]:
            self.rank[first_root] += 1
        return True


def _vector_angle_degrees(
    first_vector: np.ndarray,
    second_vector: np.ndarray,
    ignore_direction: bool,
) -> float:
    dot_product = float(np.dot(first_vector, second_vector))
    if ignore_direction:
        dot_product = abs(dot_product)
    return float(
        np.degrees(
            np.arccos(
                np.clip(dot_product, -1.0, 1.0),
            )
        )
    )


def _estimate_endpoint_tangent(
    labels: np.ndarray,
    component_id: int,
    endpoint_x: int,
    endpoint_y: int,
    radius: int,
) -> tuple[float, float] | None:
    image_height, image_width = labels.shape
    x_start = max(0, endpoint_x - radius)
    x_end = min(image_width, endpoint_x + radius + 1)
    y_start = max(0, endpoint_y - radius)
    y_end = min(image_height, endpoint_y + radius + 1)
    local_y, local_x = np.nonzero(
        labels[y_start:y_end, x_start:x_end] == component_id
    )
    if local_x.size < 2:
        return None

    global_x = local_x.astype(np.float64) + x_start
    global_y = local_y.astype(np.float64) + y_start
    squared_distance = (
        (global_x - endpoint_x) ** 2
        + (global_y - endpoint_y) ** 2
    )
    inside_window = squared_distance <= float(radius * radius)
    global_x = global_x[inside_window]
    global_y = global_y[inside_window]
    if global_x.size < 2:
        return None

    coordinates = np.column_stack((global_x, global_y))
    centered = coordinates - coordinates.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / coordinates.shape[0]
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if float(eigenvalues[-1]) <= np.finfo(np.float64).eps:
        return None

    tangent = eigenvectors[:, -1]
    outward_vector = np.array(
        [
            endpoint_x - float(np.mean(global_x)),
            endpoint_y - float(np.mean(global_y)),
        ],
        dtype=np.float64,
    )
    if float(np.dot(tangent, outward_vector)) < 0.0:
        tangent = -tangent
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= np.finfo(np.float64).eps:
        return None
    tangent /= tangent_norm
    return float(tangent[0]), float(tangent[1])


def _extract_endpoints(
    skeleton_pixels: np.ndarray,
    labels: np.ndarray,
    config: CandidateLinkerConfig,
) -> list[_Endpoint]:
    neighbor_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_kernel[1, 1] = 0
    neighbor_count = cv2.filter2D(
        skeleton_pixels.astype(np.uint8),
        cv2.CV_16U,
        neighbor_kernel,
        borderType=cv2.BORDER_CONSTANT,
    )
    endpoint_y, endpoint_x = np.nonzero(
        skeleton_pixels & (neighbor_count == 1)
    )

    endpoints: list[_Endpoint] = []
    for x, y in zip(endpoint_x, endpoint_y):
        component_id = int(labels[y, x])
        tangent = _estimate_endpoint_tangent(
            labels,
            component_id,
            int(x),
            int(y),
            config.direction_window_radius,
        )
        if tangent is None:
            continue
        endpoints.append(
            _Endpoint(
                endpoint_id=len(endpoints),
                component_id=component_id,
                x=int(x),
                y=int(y),
                tangent_x=tangent[0],
                tangent_y=tangent[1],
            )
        )
    return endpoints


def _sample_gap(
    first: _Endpoint,
    second: _Endpoint,
    line_response_image: np.ndarray,
    processing_mask: np.ndarray,
    response_threshold: float,
) -> tuple[bool, float]:
    sample_count = max(
        int(
            np.ceil(
                np.hypot(second.x - first.x, second.y - first.y)
            )
        )
        + 1,
        2,
    )
    sample_x = np.rint(
        np.linspace(first.x, second.x, sample_count)
    ).astype(np.int32)
    sample_y = np.rint(
        np.linspace(first.y, second.y, sample_count)
    ).astype(np.int32)
    sample_points = np.unique(
        np.column_stack((sample_x, sample_y)),
        axis=0,
    )
    if sample_points.shape[0] > 2:
        gap_points = sample_points[1:-1]
    else:
        gap_points = sample_points
    gap_x = gap_points[:, 0]
    gap_y = gap_points[:, 1]
    inside_mask = processing_mask[gap_y, gap_x] > 0
    if not np.all(inside_mask):
        return False, 0.0
    response_fraction = float(
        np.mean(
            line_response_image[gap_y, gap_x] >= response_threshold
        )
    )
    return True, response_fraction


def _build_link_proposal(
    first: _Endpoint,
    second: _Endpoint,
    line_response_image: np.ndarray,
    processing_mask: np.ndarray,
    response_threshold: float,
    config: CandidateLinkerConfig,
) -> CandidateLink | None:
    gap_vector = np.array(
        [second.x - first.x, second.y - first.y],
        dtype=np.float64,
    )
    gap_distance = float(np.linalg.norm(gap_vector))
    if (
        gap_distance <= np.finfo(np.float64).eps
        or gap_distance > config.maximum_gap_distance
    ):
        return None
    gap_direction = gap_vector / gap_distance
    first_tangent = np.array(
        [first.tangent_x, first.tangent_y],
        dtype=np.float64,
    )
    second_tangent = np.array(
        [second.tangent_x, second.tangent_y],
        dtype=np.float64,
    )

    direction_difference = _vector_angle_degrees(
        first_tangent,
        second_tangent,
        ignore_direction=True,
    )
    if direction_difference > config.maximum_direction_difference_degrees:
        return None
    first_connection_angle = _vector_angle_degrees(
        first_tangent,
        gap_direction,
        ignore_direction=False,
    )
    second_connection_angle = _vector_angle_degrees(
        second_tangent,
        -gap_direction,
        ignore_direction=False,
    )
    if (
        first_connection_angle > config.maximum_connection_angle_degrees
        or second_connection_angle > config.maximum_connection_angle_degrees
    ):
        return None

    first_lateral_offset = abs(
        (first_tangent[0] * gap_vector[1])
        - (first_tangent[1] * gap_vector[0])
    )
    second_gap_vector = -gap_vector
    second_lateral_offset = abs(
        (second_tangent[0] * second_gap_vector[1])
        - (second_tangent[1] * second_gap_vector[0])
    )
    lateral_offset = float(
        max(first_lateral_offset, second_lateral_offset)
    )
    if lateral_offset > config.maximum_lateral_offset_pixels:
        return None

    inside_mask, gap_response_fraction = _sample_gap(
        first,
        second,
        line_response_image,
        processing_mask,
        response_threshold,
    )
    if (
        not inside_mask
        or gap_response_fraction < config.minimum_gap_response_fraction
    ):
        return None

    direction_denominator = max(
        config.maximum_direction_difference_degrees,
        np.finfo(np.float64).eps,
    )
    connection_denominator = max(
        config.maximum_connection_angle_degrees,
        np.finfo(np.float64).eps,
    )
    lateral_denominator = max(
        config.maximum_lateral_offset_pixels,
        np.finfo(np.float64).eps,
    )
    score = float(
        (gap_distance / config.maximum_gap_distance)
        + (direction_difference / direction_denominator)
        + (
            max(first_connection_angle, second_connection_angle)
            / connection_denominator
        )
        + (lateral_offset / lateral_denominator)
        + (1.0 - gap_response_fraction)
    )
    return CandidateLink(
        first_component_id=first.component_id,
        second_component_id=second.component_id,
        first_endpoint_x=first.x,
        first_endpoint_y=first.y,
        second_endpoint_x=second.x,
        second_endpoint_y=second.y,
        gap_distance_pixels=gap_distance,
        direction_difference_degrees=direction_difference,
        first_connection_angle_degrees=first_connection_angle,
        second_connection_angle_degrees=second_connection_angle,
        lateral_offset_pixels=lateral_offset,
        gap_response_fraction=gap_response_fraction,
        score=score,
    )


def _generate_proposals(
    endpoints: list[_Endpoint],
    line_response_image: np.ndarray,
    processing_mask: np.ndarray,
    response_threshold: float,
    config: CandidateLinkerConfig,
) -> tuple[list[CandidateLink], list[tuple[int, int]]]:
    cell_size = config.maximum_gap_distance
    spatial_grid: dict[tuple[int, int], list[int]] = {}
    for endpoint in endpoints:
        cell = (
            int(np.floor(endpoint.x / cell_size)),
            int(np.floor(endpoint.y / cell_size)),
        )
        spatial_grid.setdefault(cell, []).append(endpoint.endpoint_id)

    proposals: list[CandidateLink] = []
    proposal_endpoint_ids: list[tuple[int, int]] = []
    for first in endpoints:
        cell_x = int(np.floor(first.x / cell_size))
        cell_y = int(np.floor(first.y / cell_size))
        for neighbor_cell_y in range(cell_y - 1, cell_y + 2):
            for neighbor_cell_x in range(cell_x - 1, cell_x + 2):
                for second_id in spatial_grid.get(
                    (neighbor_cell_x, neighbor_cell_y),
                    (),
                ):
                    if second_id <= first.endpoint_id:
                        continue
                    second = endpoints[second_id]
                    if first.component_id == second.component_id:
                        continue
                    proposal = _build_link_proposal(
                        first,
                        second,
                        line_response_image,
                        processing_mask,
                        response_threshold,
                        config,
                    )
                    if proposal is None:
                        continue
                    proposals.append(proposal)
                    proposal_endpoint_ids.append(
                        (first.endpoint_id, second.endpoint_id)
                    )
    return proposals, proposal_endpoint_ids


def _select_links(
    proposals: list[CandidateLink],
    proposal_endpoint_ids: list[tuple[int, int]],
    endpoint_count: int,
    component_count: int,
    require_mutual_best_match: bool,
) -> tuple[CandidateLink, ...]:
    ranked_indices = sorted(
        range(len(proposals)),
        key=lambda index: proposals[index].score,
    )
    eligible_indices = ranked_indices
    if require_mutual_best_match:
        best_proposal_by_endpoint = np.full(
            endpoint_count,
            -1,
            dtype=np.int32,
        )
        for proposal_index in ranked_indices:
            first_endpoint_id, second_endpoint_id = (
                proposal_endpoint_ids[proposal_index]
            )
            if best_proposal_by_endpoint[first_endpoint_id] < 0:
                best_proposal_by_endpoint[first_endpoint_id] = proposal_index
            if best_proposal_by_endpoint[second_endpoint_id] < 0:
                best_proposal_by_endpoint[second_endpoint_id] = proposal_index
        eligible_indices = [
            proposal_index
            for proposal_index in ranked_indices
            if (
                best_proposal_by_endpoint[
                    proposal_endpoint_ids[proposal_index][0]
                ]
                == proposal_index
                and best_proposal_by_endpoint[
                    proposal_endpoint_ids[proposal_index][1]
                ]
                == proposal_index
            )
        ]

    used_endpoints = np.zeros(endpoint_count, dtype=bool)
    disjoint_set = _DisjointSet(component_count)
    selected_links: list[CandidateLink] = []
    for proposal_index in eligible_indices:
        first_endpoint_id, second_endpoint_id = (
            proposal_endpoint_ids[proposal_index]
        )
        if used_endpoints[first_endpoint_id] or used_endpoints[second_endpoint_id]:
            continue
        proposal = proposals[proposal_index]
        if not disjoint_set.union(
            proposal.first_component_id,
            proposal.second_component_id,
        ):
            continue
        used_endpoints[first_endpoint_id] = True
        used_endpoints[second_endpoint_id] = True
        selected_links.append(proposal)
    return tuple(selected_links)


def link_candidate_segments(
    skeleton_image: np.ndarray,
    line_response_image: np.ndarray,
    processing_mask: np.ndarray,
    response_threshold: float,
    config: CandidateLinkerConfig,
) -> CandidateLinkerResult:
    """Connect compatible endpoints without changing existing skeleton pixels."""
    config.validate()
    if skeleton_image.ndim != 2:
        raise ValueError("skeleton_image must be one-channel")
    if (
        line_response_image.ndim != 2
        or line_response_image.shape != skeleton_image.shape
    ):
        raise ValueError("line_response_image must match skeleton_image")
    if (
        processing_mask.ndim != 2
        or processing_mask.shape != skeleton_image.shape
    ):
        raise ValueError("processing_mask must match skeleton_image")
    if not np.isfinite(line_response_image).all():
        raise ValueError("line_response_image must contain only finite values")
    if not np.isfinite(response_threshold):
        raise ValueError("response_threshold must be finite")

    skeleton_pixels = (skeleton_image > 0) & (processing_mask > 0)
    base_skeleton = np.where(skeleton_pixels, 255, 0).astype(np.uint8)
    empty_mask = np.zeros(skeleton_image.shape, dtype=np.uint8)
    if not config.enabled or not np.any(skeleton_pixels):
        return CandidateLinkerResult(
            linked_skeleton_image=base_skeleton,
            link_mask=empty_mask,
            links=(),
            endpoint_count=0,
            proposal_count=0,
        )

    component_count, labels = cv2.connectedComponents(
        skeleton_pixels.astype(np.uint8),
        connectivity=8,
        ltype=cv2.CV_32S,
    )
    endpoints = _extract_endpoints(skeleton_pixels, labels, config)
    proposals, proposal_endpoint_ids = _generate_proposals(
        endpoints,
        line_response_image,
        processing_mask,
        response_threshold,
        config,
    )
    selected_links = _select_links(
        proposals,
        proposal_endpoint_ids,
        len(endpoints),
        component_count,
        config.require_mutual_best_match,
    )

    link_mask = empty_mask.copy()
    for link in selected_links:
        cv2.line(
            link_mask,
            (link.first_endpoint_x, link.first_endpoint_y),
            (link.second_endpoint_x, link.second_endpoint_y),
            255,
            thickness=config.link_thickness,
            lineType=cv2.LINE_8,
        )
    link_mask[processing_mask == 0] = 0
    linked_skeleton_image = cv2.bitwise_or(base_skeleton, link_mask)
    return CandidateLinkerResult(
        linked_skeleton_image=linked_skeleton_image,
        link_mask=link_mask,
        links=selected_links,
        endpoint_count=len(endpoints),
        proposal_count=len(proposals),
    )


def create_linker_preview(
    skeleton_image: np.ndarray,
    linker_result: CandidateLinkerResult,
    config: CandidateLinkerPreviewConfig,
) -> tuple[np.ndarray, np.ndarray]:
    config.validate()
    if skeleton_image.shape != linker_result.linked_skeleton_image.shape:
        raise ValueError("skeleton_image must match linker result size")

    before = np.where(skeleton_image > 0, 255, 0).astype(np.uint8)
    after = linker_result.linked_skeleton_image
    link_mask = linker_result.link_mask
    if config.skeleton_preview_thickness > 1:
        kernel_size = config.skeleton_preview_thickness
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (kernel_size, kernel_size),
        )
        before = cv2.dilate(before, kernel, iterations=1)
        after = cv2.dilate(after, kernel, iterations=1)
        link_mask = cv2.dilate(link_mask, kernel, iterations=1)

    before_preview = cv2.cvtColor(before, cv2.COLOR_GRAY2BGR)
    after_preview = cv2.cvtColor(after, cv2.COLOR_GRAY2BGR)
    after_preview[link_mask > 0] = (0, 0, 255)
    return before_preview, after_preview


def write_link_csv(
    output_path: Path,
    links: tuple[CandidateLink, ...],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    field_names = list(CandidateLink.__dataclass_fields__)
    with output_path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for link in links:
            writer.writerow(asdict(link))


def _create_parameter_record_dir(
    output_dir: Path,
    config: CandidateLinkerConfig,
) -> Path:
    config.validate()
    parameter_dir = output_dir / (
        f"gap={config.maximum_gap_distance},"
        f"window={config.direction_window_radius},"
        f"direction={config.maximum_direction_difference_degrees},"
        f"connection={config.maximum_connection_angle_degrees},"
        f"lateral={config.maximum_lateral_offset_pixels},"
        f"response={config.minimum_gap_response_fraction},"
        f"mutual={int(config.require_mutual_best_match)}"
    )
    parameter_dir.mkdir(parents=False, exist_ok=False)
    return parameter_dir


def _run_standalone_dataset(
    dataset_dir: Path,
    pipeline: ScratchDetectionPipeline,
    linker_config: CandidateLinkerConfig,
    preview_config: CandidateLinkerPreviewConfig,
    writer_config: ResultWriterConfig,
) -> tuple[int, list[Path], Path]:
    scan_result = scan_image_json_pairs(dataset_dir)
    writer = ResultWriter(dataset_dir, writer_config)
    _create_parameter_record_dir(writer.output_dir, linker_config)

    processed_count = 0
    for pair in scan_result.pairs:
        original_image = read_image(pair.image_path)
        annotation = read_annotation(pair.json_path)
        pipeline_result = pipeline.run(original_image, annotation)
        before_skeleton = (
            pipeline_result.candidate_filter_result.filtered_skeleton_image
        )
        linker_result = link_candidate_segments(
            before_skeleton,
            pipeline_result.line_response_image,
            pipeline_result.processing_mask,
            pipeline_result.threshold_low_value,
            linker_config,
        )
        before_preview, after_preview = create_linker_preview(
            before_skeleton,
            linker_result,
            preview_config,
        )
        (
            cropped_original,
            cropped_before,
            cropped_after,
            cropped_link_mask,
        ) = crop_images_to_mask_bounding_rect(
            pipeline_result.original_mask,
            original_image,
            before_preview,
            after_preview,
            linker_result.link_mask,
            padding=preview_config.crop_padding,
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
            / f"{pair.image_path.stem}_candidate_link_mask.png",
            cropped_link_mask,
        )
        write_link_csv(
            writer.output_dir
            / f"{pair.image_path.stem}_candidate_links.csv",
            linker_result.links,
        )
        processed_count += 1

    return processed_count, scan_result.images_without_json, writer.output_dir


def main() -> None:
    dataset_dir = Path(r"E:\projects\datasets\Power_box\Scratch_old")
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
    candidate_filter_config = CandidateFilterConfig(
        enabled=True,
        minimum_path_length=10.0,
        long_min_path_length=150.0,
        short_min_longest_path_aspect_ratio=3.0,
        short_min_linearity=0.30,
    )
    pipeline = ScratchDetectionPipeline(
        erode_mask_config,
        background_config,
        gabor_config,
        frangi_config,
        threshold_config,
        skeleton_config,
        feature_config,
        candidate_filter_config,
        line_enhancement_method="frangi",
        frangi_response_mode="bright",
    )
    linker_config = CandidateLinkerConfig(
        enabled=True,
        maximum_gap_distance=20.0,
        direction_window_radius=7,
        maximum_direction_difference_degrees=20.0,
        maximum_connection_angle_degrees=25.0,
        maximum_lateral_offset_pixels=4.0,
        minimum_gap_response_fraction=0.25,
        require_mutual_best_match=True,
        link_thickness=1,
    )
    preview_config = CandidateLinkerPreviewConfig(
        crop_padding=15,
        skeleton_preview_thickness=3,
    )
    writer_config = ResultWriterConfig(
        output_folder_name="candidate_linker_results",
        run_number_width=3,
        max_run_number=999,
        save_comparison=False,
        save_processed=True,
        comparison_suffix="_candidate_linker_comparison.png",
        processed_suffix="_candidate_linker.png",
    )

    processed_count, images_without_json, output_dir = _run_standalone_dataset(
        dataset_dir,
        pipeline,
        linker_config,
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
