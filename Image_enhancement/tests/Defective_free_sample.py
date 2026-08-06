"""Defect-free sample synthesis for silver-plated power-box images.

流程备注:
1. 遍历源目录中每个 JSON，找到同名图片；如果 JSON 内没有 Scratch、Pit、
   Stain 三类缺陷，则跳过该样本。
2. 读取 Silver box 作为可修复的物体范围，并把缺陷标注转成 mask。每个缺陷
   会按类别自适应向外扩张，避免只覆盖缺陷中心导致边缘残留。
3. donor patch 的来源只允许在 Silver box 内，并排除所有扩张后的缺陷区域和
   The_avoided_area；这样不会从缺陷、避让区域或物体外部取像素。
4. 对扩张后连在一起的缺陷合并成 repair region。小的非划痕缺陷优先用
   TELEA 修复；划痕始终使用同图内正常区域 donor patch 修复。
5. 纯 Scratch 的长边达到阈值后，沿真实骨架计算最大宽度。宽度小于 5 像素时
   使用同尺寸正常区域整体修复；否则使用 9~15 像素的骨架小图块修复。
6. 划痕 donor 只能来自 Silver box 向内收缩 30 像素后的区域。法线候选越界时
   裁掉无效像素并逐步扩大周边搜索，直至收集到足量且互不重复的有效像素。
7. 小图块根据亮度、梯度、纹理和重叠区连续性评分，从最优候选中随机选择，
   只写入扩张后的划痕 mask；图块间加权融合，最后对整个 mask 边缘羽化。
8. 其他 donor patch 先在局部搜索，再扩大范围；找到后根据缺陷周边干净边界
   拟合平滑颜色校正，再按 mask 羽化融合。
9. 每次运行会在 Defective_free 目录内创建递增数字文件夹（如 01、02）；输出
   图片和同名 JSON 保存到该数字文件夹，JSON 的 objects 中只保留 Silver box。
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SOURCE_DIR = Path(r"E:\projects\datasets\Power_box\old\results\results")
OUTPUT_DIR_NAME = "Defective_free"
INPAINT_RADIUS = 5
SEARCH_MARGIN = 300
MATCH_BAND = 12
FEATHER_PIXELS = 5
SCRATCH_FEATHER_PIXELS = 3
LONG_SCRATCH_MIN_LENGTH = 96
SCRATCH_PATCH_MIN_SIZE = 9
SCRATCH_PATCH_MAX_SIZE = 15
SCRATCH_PATCH_TOP_K = 3
SCRATCH_PATCH_RANDOM_SEED = 20260806
SCRATCH_DONOR_INSET_PIXELS = 50
THIN_SCRATCH_MAX_WIDTH = 5.0

IMAGE_EXTENSIONS = (
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
)

SILVER_BOX_KEY = "silver_box"
AVOIDED_AREA_KEY = "the_avoided_area"
DEFECT_CATEGORY_KEYS = {"scratch", "pit", "stain"}
CATEGORY_BITS = {"scratch": 1, "pit": 2, "stain": 4}


@dataclass(frozen=True)
class RepairConfig:
    inpaint_radius: int = INPAINT_RADIUS
    search_margin: int = SEARCH_MARGIN
    match_band: int = MATCH_BAND
    feather_pixels: int = FEATHER_PIXELS
    scratch_feather_pixels: int = SCRATCH_FEATHER_PIXELS
    long_scratch_min_length: int = LONG_SCRATCH_MIN_LENGTH
    scratch_patch_min_size: int = SCRATCH_PATCH_MIN_SIZE
    scratch_patch_max_size: int = SCRATCH_PATCH_MAX_SIZE
    scratch_patch_top_k: int = SCRATCH_PATCH_TOP_K
    scratch_patch_random_seed: int = SCRATCH_PATCH_RANDOM_SEED
    scratch_donor_inset_pixels: int = SCRATCH_DONOR_INSET_PIXELS

    def validate(self) -> None:
        if self.inpaint_radius < 1:
            raise ValueError("inpaint_radius must be at least 1")
        if self.search_margin < 1:
            raise ValueError("search_margin must be at least 1")
        if self.match_band < 2:
            raise ValueError("match_band must be at least 2")
        if self.feather_pixels < 0:
            raise ValueError("feather_pixels must not be negative")
        if self.scratch_feather_pixels < 0:
            raise ValueError("scratch_feather_pixels must not be negative")
        if self.long_scratch_min_length < 1:
            raise ValueError("long_scratch_min_length must be at least 1")
        if self.scratch_patch_min_size < 3 or self.scratch_patch_min_size % 2 == 0:
            raise ValueError("scratch_patch_min_size must be an odd integer >= 3")
        if self.scratch_patch_max_size < self.scratch_patch_min_size:
            raise ValueError(
                "scratch_patch_max_size must be >= scratch_patch_min_size"
            )
        if self.scratch_patch_max_size % 2 == 0:
            raise ValueError("scratch_patch_max_size must be odd")
        if self.scratch_patch_top_k < 1:
            raise ValueError("scratch_patch_top_k must be at least 1")
        if self.scratch_donor_inset_pixels < 0:
            raise ValueError("scratch_donor_inset_pixels must not be negative")


@dataclass(frozen=True)
class RepairRegion:
    x: int
    y: int
    mask: np.ndarray
    categories: frozenset[str]
    raw_thickness: float
    raw_mask: np.ndarray | None = None

    @property
    def width(self) -> int:
        return int(self.mask.shape[1])

    @property
    def height(self) -> int:
        return int(self.mask.shape[0])


@dataclass(frozen=True)
class ScratchDonorPatch:
    rgb: np.ndarray
    low_frequency: np.ndarray
    gradient: np.ndarray
    texture_energy: np.ndarray


def normalize_category(value: object) -> str:
    text = str(value).strip().lower().replace(" ", "_")
    return "_".join(part for part in text.split("_") if part)


def read_json(json_path: Path) -> dict:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {json_path}")
    if not isinstance(data.get("objects"), list):
        raise ValueError(f'JSON must contain an "objects" list: {json_path}')
    return data


def find_image_for_json(json_path: Path) -> Path:
    for suffix in IMAGE_EXTENSIONS:
        image_path = json_path.with_suffix(suffix)
        if image_path.exists():
            return image_path

    candidates = {
        path.suffix.lower(): path
        for path in json_path.parent.glob(f"{json_path.stem}.*")
        if path.is_file()
    }
    for suffix in IMAGE_EXTENSIONS:
        if suffix in candidates:
            return candidates[suffix]

    raise FileNotFoundError(f"No same-name image found for JSON file: {json_path}")


def objects_by_category(data: dict, category_key: str) -> list[dict]:
    return [
        item
        for item in data.get("objects", [])
        if isinstance(item, dict)
        and normalize_category(item.get("category")) == category_key
    ]


def defect_objects(data: dict) -> list[dict]:
    return [
        item
        for item in data.get("objects", [])
        if isinstance(item, dict)
        and normalize_category(item.get("category")) in DEFECT_CATEGORY_KEYS
    ]


def segmentation_points(item: dict) -> list[tuple[int, int]]:
    segmentation = item.get("segmentation")
    if not isinstance(segmentation, list):
        return []

    points: list[tuple[int, int]] = []
    for point in segmentation:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            points.append((round(float(point[0])), round(float(point[1]))))
        except (TypeError, ValueError):
            continue
    return points if len(points) >= 3 else []


def build_mask(size: tuple[int, int], objects: list[dict]) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)

    for item in objects:
        points = segmentation_points(item)
        if not points:
            continue

        polygon = np.asarray(points, dtype=np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [polygon], 255)

    return mask


def mask_bounding_rect(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(mask)
    return None if points is None else cv2.boundingRect(points)


def estimate_mask_thickness(mask: np.ndarray) -> float:
    rect = mask_bounding_rect(mask)
    if rect is None:
        return 0.0

    x, y, width, height = rect
    cropped = mask[y : y + height, x : x + width]
    distance = cv2.distanceTransform(cropped, cv2.DIST_L2, 5)
    return float(distance.max() * 2.0)


def defect_expansion_pixels(mask: np.ndarray, category: str) -> int:
    thickness = estimate_mask_thickness(mask)
    if category == "scratch":
        return int(np.clip(round(thickness * 0.35), 5, 10))
    if category == "pit":
        return int(np.clip(round(thickness * 0.18), 2, 7))
    return int(np.clip(round(thickness * 0.10), 3, 10))


def dilate_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.copy()
    kernel_size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.dilate(mask, kernel, iterations=1)


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.copy()
    kernel_size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    return cv2.erode(mask, kernel, iterations=1)


def build_repair_regions(
    expanded_union: np.ndarray,
    raw_union: np.ndarray,
    category_map: np.ndarray,
) -> list[RepairRegion]:
    contours, _ = cv2.findContours(
        expanded_union,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    regions: list[RepairRegion] = []

    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        shifted_contour = contour - np.array([[[x, y]]], dtype=np.int32)
        local_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(local_mask, [shifted_contour], -1, 255, cv2.FILLED)

        raw_local = cv2.bitwise_and(
            raw_union[y : y + height, x : x + width],
            local_mask,
        )
        category_values = category_map[y : y + height, x : x + width][
            local_mask > 0
        ]
        category_bits = (
            int(np.bitwise_or.reduce(category_values))
            if category_values.size
            else 0
        )
        categories = frozenset(
            category
            for category, bit in CATEGORY_BITS.items()
            if category_bits & bit
        )
        regions.append(
            RepairRegion(
                x=x,
                y=y,
                mask=local_mask,
                categories=categories,
                raw_thickness=estimate_mask_thickness(raw_local),
                raw_mask=raw_local,
            )
        )

    return sorted(regions, key=lambda region: (region.y, region.x))


def build_texture_features(
    source: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lightness = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)[:, :, 0]
    low_frequency = cv2.GaussianBlur(lightness, (0, 0), sigmaX=3.0)
    gradient_x = cv2.Sobel(low_frequency, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(low_frequency, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.clip(cv2.magnitude(gradient_x, gradient_y), 0, 255).astype(
        np.uint8
    )
    residual = lightness.astype(np.float32) - cv2.GaussianBlur(
        lightness.astype(np.float32),
        (0, 0),
        sigmaX=1.5,
    )
    texture_energy = np.sqrt(
        cv2.GaussianBlur(residual * residual, (0, 0), sigmaX=2.0)
    )
    texture_energy = np.clip(texture_energy * 8.0, 0, 255).astype(np.uint8)
    return low_frequency, gradient, texture_energy


def masked_mse_map(
    search: np.ndarray,
    template: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    search_float = search.astype(np.float32)
    template_float = template.astype(np.float32)
    mask_float = mask.astype(np.float32)
    sample_count = float(mask_float.sum())
    if sample_count <= 0:
        raise ValueError("The donor matching mask is empty.")

    search_squared_sum = cv2.matchTemplate(
        search_float * search_float,
        mask_float,
        cv2.TM_CCORR,
    )
    cross_sum = cv2.matchTemplate(
        search_float,
        template_float * mask_float,
        cv2.TM_CCORR,
    )
    template_squared_sum = float(
        np.sum(template_float * template_float * mask_float)
    )
    mse = (
        search_squared_sum - 2.0 * cross_sum + template_squared_sum
    ) / sample_count
    return np.maximum(mse, 0.0)


def masked_mean_map(search: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mask_float = mask.astype(np.float32)
    sample_count = float(mask_float.sum())
    if sample_count <= 0:
        raise ValueError("The donor statistic mask is empty.")
    value_sum = cv2.matchTemplate(
        search.astype(np.float32),
        mask_float,
        cv2.TM_CCORR,
    )
    return value_sum / sample_count


def masked_mean_std_maps(
    search: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    search_float = search.astype(np.float32)
    mean = masked_mean_map(search_float, mask)
    squared_mean = masked_mean_map(search_float * search_float, mask)
    variance = np.maximum(squared_mean - mean * mean, 0.0)
    return mean, np.sqrt(variance)


def expanded_rectangle(
    x: int,
    y: int,
    width: int,
    height: int,
    margin: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    left = max(0, x - margin)
    top = max(0, y - margin)
    right = min(image_width, x + width + margin)
    bottom = min(image_height, y + height + margin)
    return left, top, right, bottom


def region_mask_in_rectangle(
    region: RepairRegion,
    rectangle: tuple[int, int, int, int],
) -> np.ndarray:
    left, top, right, bottom = rectangle
    mask = np.zeros((bottom - top, right - left), dtype=np.uint8)
    offset_x = region.x - left
    offset_y = region.y - top
    mask[
        offset_y : offset_y + region.height,
        offset_x : offset_x + region.width,
    ] = region.mask
    return mask


def find_donor_patch(
    source: np.ndarray,
    low_frequency: np.ndarray,
    gradient: np.ndarray,
    texture_energy: np.ndarray,
    invalid_source_mask: np.ndarray,
    region: RepairRegion,
    config: RepairConfig,
    candidate_invalid_source_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    image_height, image_width = source.shape[:2]
    donor_invalid_mask = (
        candidate_invalid_source_mask
        if candidate_invalid_source_mask is not None
        else invalid_source_mask
    )
    template_rect = expanded_rectangle(
        region.x,
        region.y,
        region.width,
        region.height,
        config.match_band,
        image_width,
        image_height,
    )
    left, top, right, bottom = template_rect
    template_mask = region_mask_in_rectangle(region, template_rect)
    ring_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (config.match_band * 2 + 1, config.match_band * 2 + 1),
    )
    match_ring = (cv2.dilate(template_mask, ring_kernel) > 0) & (
        template_mask == 0
    )
    match_ring &= invalid_source_mask[top:bottom, left:right] == 0
    if np.count_nonzero(match_ring) < 20:
        raise ValueError("Not enough clean boundary pixels for donor matching.")

    support_mask = ((template_mask > 0) | match_ring).astype(np.uint8)
    ring_mask = match_ring.astype(np.uint8)
    template_low = low_frequency[top:bottom, left:right]
    template_gradient = gradient[top:bottom, left:right]
    template_texture = texture_energy[top:bottom, left:right]
    template_height, template_width = template_mask.shape
    target_texture_level = float(np.mean(template_texture[match_ring]))
    target_texture_std = float(np.std(template_texture[match_ring]))
    target_gradient_level = float(np.mean(template_gradient[match_ring]))
    target_gradient_std = float(np.std(template_gradient[match_ring]))
    target_low_level = float(np.mean(template_low[match_ring]))
    target_low_std = float(np.std(template_low[match_ring]))

    margins = [config.search_margin, config.search_margin * 2]
    margins.append(max(image_width, image_height))
    checked_margins: set[int] = set()

    for margin in margins:
        if margin in checked_margins:
            continue
        checked_margins.add(margin)
        search_rect = expanded_rectangle(
            left,
            top,
            template_width,
            template_height,
            margin,
            image_width,
            image_height,
        )
        search_left, search_top, search_right, search_bottom = search_rect
        search_height = search_bottom - search_top
        search_width = search_right - search_left
        if search_height < template_height or search_width < template_width:
            continue

        invalid_search = (
            donor_invalid_mask[
                search_top:search_bottom,
                search_left:search_right,
            ]
            > 0
        ).astype(np.uint8)
        candidate_support_mask = (
            np.ones_like(support_mask)
            if candidate_invalid_source_mask is not None
            else support_mask
        )
        invalid_count = cv2.matchTemplate(
            invalid_search,
            candidate_support_mask,
            cv2.TM_CCORR,
        )
        valid_candidates = invalid_count < 0.5
        if not valid_candidates.any():
            continue

        low_score = masked_mse_map(
            low_frequency[search_top:search_bottom, search_left:search_right],
            template_low,
            ring_mask,
        )
        gradient_score = masked_mse_map(
            gradient[search_top:search_bottom, search_left:search_right],
            template_gradient,
            ring_mask,
        )
        if "scratch" in region.categories:
            texture_search = texture_energy[
                search_top:search_bottom,
                search_left:search_right,
            ]
            gradient_search = gradient[
                search_top:search_bottom,
                search_left:search_right,
            ]
            low_search = low_frequency[
                search_top:search_bottom,
                search_left:search_right,
            ]
            center_mask = (template_mask > 0).astype(np.uint8)
            texture_score = masked_mse_map(
                texture_search,
                template_texture,
                ring_mask,
            )
            candidate_texture_level, candidate_texture_std = (
                masked_mean_std_maps(texture_search, center_mask)
            )
            candidate_gradient_level, candidate_gradient_std = (
                masked_mean_std_maps(gradient_search, center_mask)
            )
            candidate_low_level, candidate_low_std = masked_mean_std_maps(
                low_search,
                center_mask,
            )
            score = (
                low_score
                + gradient_score * 0.20
                + texture_score * 0.50
                + (candidate_texture_level - target_texture_level) ** 2 * 0.75
                + (candidate_texture_std - target_texture_std) ** 2 * 0.25
                + (candidate_gradient_level - target_gradient_level) ** 2 * 0.35
                + (candidate_gradient_std - target_gradient_std) ** 2 * 0.20
                + (candidate_low_level - target_low_level) ** 2 * 0.20
                + (candidate_low_std - target_low_std) ** 2 * 0.30
            )
        else:
            score = low_score + gradient_score * 0.35
        score[~valid_candidates] = np.inf
        if not np.isfinite(score).any():
            continue

        candidate_y, candidate_x = np.unravel_index(
            int(np.argmin(score)),
            score.shape,
        )
        donor_left = search_left + int(candidate_x)
        donor_top = search_top + int(candidate_y)
        donor_rect = (
            donor_left,
            donor_top,
            donor_left + template_width,
            donor_top + template_height,
        )
        donor_patch = source[
            donor_top : donor_top + template_height,
            donor_left : donor_left + template_width,
        ].copy()
        return donor_patch, match_ring, template_rect

    raise ValueError("No clean donor patch was found inside Silver box.")


def fit_smooth_color_correction(
    target_patch: np.ndarray,
    donor_patch: np.ndarray,
    match_ring: np.ndarray,
) -> np.ndarray:
    target_smooth = cv2.GaussianBlur(
        target_patch.astype(np.float32),
        (0, 0),
        sigmaX=3.0,
    )
    donor_smooth = cv2.GaussianBlur(
        donor_patch.astype(np.float32),
        (0, 0),
        sigmaX=3.0,
    )
    ys, xs = np.nonzero(match_ring)
    if len(xs) == 0:
        return np.zeros_like(target_smooth)

    max_samples = 5000
    if len(xs) > max_samples:
        sample_indices = np.linspace(0, len(xs) - 1, max_samples).astype(int)
        xs = xs[sample_indices]
        ys = ys[sample_indices]

    height, width = match_ring.shape
    x_scale = max(1.0, (width - 1) * 0.5)
    y_scale = max(1.0, (height - 1) * 0.5)
    x_values = (xs.astype(np.float32) - (width - 1) * 0.5) / x_scale
    y_values = (ys.astype(np.float32) - (height - 1) * 0.5) / y_scale
    design = np.column_stack(
        (x_values, y_values, np.ones_like(x_values))
    ).astype(np.float32)
    differences = target_smooth[ys, xs] - donor_smooth[ys, xs]
    coefficients, _, _, _ = np.linalg.lstsq(design, differences, rcond=None)

    grid_y, grid_x = np.indices((height, width), dtype=np.float32)
    grid_x = (grid_x - (width - 1) * 0.5) / x_scale
    grid_y = (grid_y - (height - 1) * 0.5) / y_scale
    correction = (
        grid_x[:, :, None] * coefficients[0]
        + grid_y[:, :, None] * coefficients[1]
        + coefficients[2]
    )
    return np.clip(correction, -30.0, 30.0)


def blend_donor_patch(
    result: np.ndarray,
    source: np.ndarray,
    donor_patch: np.ndarray,
    match_ring: np.ndarray,
    template_rect: tuple[int, int, int, int],
    region: RepairRegion,
    feather_pixels: int,
) -> None:
    left, top, right, bottom = template_rect
    target_patch = source[top:bottom, left:right]
    target_mask = region_mask_in_rectangle(region, template_rect)
    color_correction = fit_smooth_color_correction(
        target_patch,
        donor_patch,
        match_ring,
    )
    adjusted_donor = np.clip(
        donor_patch.astype(np.float32) + color_correction,
        0,
        255,
    )

    distance = cv2.distanceTransform(target_mask, cv2.DIST_L2, 5)
    if feather_pixels <= 0:
        alpha = (target_mask > 0).astype(np.float32)
    else:
        effective_feather = min(
            float(feather_pixels),
            max(1.0, float(distance.max()) * 0.5),
        )
        alpha = np.clip(distance / effective_feather, 0.0, 1.0)
    alpha = alpha[:, :, None]

    result_patch = result[top:bottom, left:right].astype(np.float32)
    blended = result_patch * (1.0 - alpha) + adjusted_donor * alpha
    result[top:bottom, left:right] = np.clip(blended, 0, 255).astype(np.uint8)


def telea_is_safe(
    region: RepairRegion,
    invalid_source_mask: np.ndarray,
    radius: int,
) -> bool:
    image_height, image_width = invalid_source_mask.shape
    padding = radius + 2
    rectangle = expanded_rectangle(
        region.x,
        region.y,
        region.width,
        region.height,
        padding,
        image_width,
        image_height,
    )
    target_mask = region_mask_in_rectangle(region, rectangle)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (padding * 2 + 1, padding * 2 + 1),
    )
    neighborhood = (cv2.dilate(target_mask, kernel) > 0) & (target_mask == 0)
    left, top, right, bottom = rectangle
    invalid_neighbors = invalid_source_mask[top:bottom, left:right] > 0
    return not np.any(neighborhood & invalid_neighbors)


def repair_with_telea(
    result: np.ndarray,
    region: RepairRegion,
    radius: int,
) -> None:
    image_height, image_width = result.shape[:2]
    padding = radius * 2 + 2
    rectangle = expanded_rectangle(
        region.x,
        region.y,
        region.width,
        region.height,
        padding,
        image_width,
        image_height,
    )
    left, top, right, bottom = rectangle
    target_mask = region_mask_in_rectangle(region, rectangle)
    result_patch = result[top:bottom, left:right]
    repaired_patch = cv2.inpaint(
        result_patch,
        target_mask,
        float(radius),
        cv2.INPAINT_TELEA,
    )
    target = target_mask > 0
    result_patch[target] = repaired_patch[target]


def morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    current = cv2.copyMakeBorder(
        binary,
        1,
        1,
        1,
        1,
        cv2.BORDER_CONSTANT,
        value=0,
    )
    skeleton = np.zeros_like(current)
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while cv2.countNonZero(current) > 0:
        eroded = cv2.erode(current, kernel)
        opened = cv2.dilate(eroded, kernel)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(current, opened))
        current = eroded

    return skeleton[1:-1, 1:-1]


def maximum_scratch_width_along_skeleton(region: RepairRegion) -> float:
    raw_mask = region.raw_mask if region.raw_mask is not None else region.mask
    skeleton = morphological_skeleton(raw_mask)
    if not np.any(skeleton):
        return 0.0
    distance_to_edge = cv2.distanceTransform(raw_mask, cv2.DIST_L2, 5)
    return float(np.max(distance_to_edge[skeleton > 0]) * 2.0)


def adaptive_scratch_patch_size(
    region: RepairRegion,
    config: RepairConfig,
) -> int:
    estimated = int(round(region.raw_thickness * 0.75))
    size = int(
        np.clip(
            estimated,
            config.scratch_patch_min_size,
            config.scratch_patch_max_size,
        )
    )
    if size % 2 == 0:
        size += 1 if size < config.scratch_patch_max_size else -1
    return size


def skeleton_guided_centers(
    mask: np.ndarray,
    skeleton: np.ndarray,
    patch_size: int,
) -> list[tuple[int, int]]:
    radius = patch_size // 2
    spacing = max(2, radius)
    remaining_skeleton = skeleton.copy()
    centers: list[tuple[int, int]] = []

    while np.any(remaining_skeleton):
        y, x = np.argwhere(remaining_skeleton > 0)[0]
        centers.append((int(x), int(y)))
        cv2.circle(
            remaining_skeleton,
            (int(x), int(y)),
            spacing,
            0,
            cv2.FILLED,
        )

    covered = np.zeros(mask.shape, dtype=bool)

    def mark_covered(center_x: int, center_y: int) -> None:
        left = max(0, center_x - radius)
        top = max(0, center_y - radius)
        right = min(mask.shape[1], center_x + radius + 1)
        bottom = min(mask.shape[0], center_y + radius + 1)
        covered[top:bottom, left:right] |= mask[top:bottom, left:right] > 0

    for center_x, center_y in centers:
        mark_covered(center_x, center_y)

    while np.any((mask > 0) & ~covered):
        y, x = np.argwhere((mask > 0) & ~covered)[0]
        centers.append((int(x), int(y)))
        mark_covered(int(x), int(y))

    return centers


def local_skeleton_frame(
    skeleton_points: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    deltas = skeleton_points - np.array([center_y, center_x])
    squared_distances = np.sum(deltas * deltas, axis=1)
    nearest_index = int(np.argmin(squared_distances))
    nearest_y, nearest_x = skeleton_points[nearest_index]
    nearby = skeleton_points[squared_distances <= radius * radius]
    if len(nearby) < 3:
        nearest_indices = np.argsort(squared_distances)[: min(12, len(skeleton_points))]
        nearby = skeleton_points[nearest_indices]

    xy_points = nearby[:, ::-1].astype(np.float32)
    if len(xy_points) >= 2:
        covariance = np.cov(xy_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        tangent = eigenvectors[:, int(np.argmax(eigenvalues))].astype(np.float32)
    else:
        tangent = np.array([1.0, 0.0], dtype=np.float32)
    tangent /= max(float(np.linalg.norm(tangent)), 1e-6)
    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    return tangent, normal, (int(nearest_x), int(nearest_y))


def patch_feature_statistics(
    low_frequency: np.ndarray,
    gradient: np.ndarray,
    texture_energy: np.ndarray,
    rectangle: tuple[int, int, int, int],
    valid_mask: np.ndarray | None = None,
) -> np.ndarray | None:
    left, top, right, bottom = rectangle
    features = (
        low_frequency[top:bottom, left:right],
        gradient[top:bottom, left:right],
        texture_energy[top:bottom, left:right],
    )
    if valid_mask is None:
        valid_mask = np.ones(features[0].shape, dtype=bool)
    if np.count_nonzero(valid_mask) < 5:
        return None

    values: list[float] = []
    for feature in features:
        selected = feature[valid_mask].astype(np.float32)
        values.extend((float(np.mean(selected)), float(np.std(selected))))
    return np.asarray(values, dtype=np.float32)


def collect_scratch_donor_patch(
    source: np.ndarray,
    low_frequency: np.ndarray,
    gradient: np.ndarray,
    texture_energy: np.ndarray,
    invalid_source_mask: np.ndarray,
    center_x: float,
    center_y: float,
    target_width: int,
    target_height: int,
    search_margin: int,
) -> ScratchDonorPatch | None:
    image_height, image_width = invalid_source_mask.shape
    required_pixels = target_width * target_height
    ideal_left = int(round(center_x - (target_width - 1) * 0.5))
    ideal_top = int(round(center_y - (target_height - 1) * 0.5))
    expansion_step = max(2, min(target_width, target_height) // 2)
    valid_global: np.ndarray | None = None

    for margin in range(0, search_margin + expansion_step, expansion_step):
        left = max(0, ideal_left - margin)
        top = max(0, ideal_top - margin)
        right = min(image_width, ideal_left + target_width + margin)
        bottom = min(image_height, ideal_top + target_height + margin)
        if left >= right or top >= bottom:
            continue

        valid_local = np.argwhere(
            invalid_source_mask[top:bottom, left:right] == 0
        )
        if len(valid_local) < required_pixels:
            continue
        valid_global = valid_local + np.array([top, left])
        break

    if valid_global is None:
        return None

    ideal_y, ideal_x = np.indices((target_height, target_width))
    ideal_points = np.column_stack(
        (
            (ideal_y + ideal_top).ravel(),
            (ideal_x + ideal_left).ravel(),
        )
    ).astype(np.float32)

    pool_limit = max(required_pixels * 8, 512)
    if len(valid_global) > pool_limit:
        center = np.array([center_y, center_x], dtype=np.float32)
        center_distances = np.sum(
            (valid_global.astype(np.float32) - center) ** 2,
            axis=1,
        )
        nearest_pool = np.argpartition(center_distances, pool_limit - 1)[
            :pool_limit
        ]
        valid_global = valid_global[nearest_pool]

    squared_distances = np.sum(
        (
            ideal_points[:, None, :]
            - valid_global[None, :, :].astype(np.float32)
        )
        ** 2,
        axis=2,
    )
    assignment_order = np.argsort(np.min(squared_distances, axis=1))
    available = np.ones(len(valid_global), dtype=bool)
    selected_indices = np.empty(required_pixels, dtype=np.int32)
    for target_index in assignment_order:
        distances = squared_distances[target_index].copy()
        distances[~available] = np.inf
        source_index = int(np.argmin(distances))
        selected_indices[target_index] = source_index
        available[source_index] = False

    selected = valid_global[selected_indices]
    selected_y = selected[:, 0]
    selected_x = selected[:, 1]

    def gather(array: np.ndarray) -> np.ndarray:
        values = array[selected_y, selected_x]
        return values.reshape((target_height, target_width) + values.shape[1:])

    return ScratchDonorPatch(
        rgb=gather(source),
        low_frequency=gather(low_frequency),
        gradient=gather(gradient),
        texture_energy=gather(texture_energy),
    )


def candidate_scratch_donor_patches(
    source: np.ndarray,
    low_frequency: np.ndarray,
    gradient: np.ndarray,
    texture_energy: np.ndarray,
    invalid_source_mask: np.ndarray,
    nearest_skeleton: tuple[int, int],
    tangent: np.ndarray,
    normal: np.ndarray,
    half_scratch_width: float,
    target_width: int,
    target_height: int,
    patch_size: int,
    search_margin: int,
) -> list[ScratchDonorPatch]:
    skeleton_x, skeleton_y = nearest_skeleton
    candidates: list[ScratchDonorPatch] = []
    seen: set[tuple[int, int]] = set()
    max_candidates = 12

    def add_candidate(center_x: float, center_y: float) -> None:
        key = (int(round(center_x)), int(round(center_y)))
        if key in seen or len(candidates) >= max_candidates:
            return
        seen.add(key)
        candidate = collect_scratch_donor_patch(
            source,
            low_frequency,
            gradient,
            texture_energy,
            invalid_source_mask,
            center_x,
            center_y,
            target_width,
            target_height,
            search_margin,
        )
        if candidate is not None:
            candidates.append(candidate)

    base_offset = half_scratch_width + max(target_width, target_height) * 0.5 + 2.0
    radial_step = max(2.0, patch_size * 0.5)
    tangent_offsets = (0.0, -patch_size, patch_size, -2.0 * patch_size, 2.0 * patch_size)
    for radial_index in range(8):
        offset = base_offset + radial_index * radial_step
        for side in (-1.0, 1.0):
            for tangent_offset in tangent_offsets:
                center = (
                    np.array([skeleton_x, skeleton_y], dtype=np.float32)
                    + normal * (side * offset)
                    + tangent * tangent_offset
                )
                add_candidate(float(center[0]), float(center[1]))
                if len(candidates) >= max_candidates:
                    return candidates

    if len(candidates) >= max_candidates:
        return candidates

    fallback_margin = min(search_margin, patch_size * 12)
    step = max(3, patch_size // 2)
    offsets = [
        (offset_x, offset_y)
        for offset_y in range(-fallback_margin, fallback_margin + 1, step)
        for offset_x in range(-fallback_margin, fallback_margin + 1, step)
    ]
    offsets.sort(key=lambda value: value[0] * value[0] + value[1] * value[1])
    for offset_x, offset_y in offsets:
        add_candidate(skeleton_x + offset_x, skeleton_y + offset_y)
        if len(candidates) >= max_candidates:
            break
    return candidates


def repair_scratch_with_skeleton_patches(
    result: np.ndarray,
    source: np.ndarray,
    low_frequency: np.ndarray,
    gradient: np.ndarray,
    texture_energy: np.ndarray,
    invalid_source_mask: np.ndarray,
    candidate_invalid_source_mask: np.ndarray,
    region: RepairRegion,
    config: RepairConfig,
) -> bool:
    skeleton = morphological_skeleton(region.mask)
    skeleton_points = np.argwhere(skeleton > 0)
    if len(skeleton_points) == 0:
        return False

    patch_size = adaptive_scratch_patch_size(region, config)
    centers = skeleton_guided_centers(region.mask, skeleton, patch_size)
    distance_to_edge = cv2.distanceTransform(region.mask, cv2.DIST_L2, 5)
    accumulation = np.zeros((region.height, region.width, 3), dtype=np.float32)
    accumulated_weight = np.zeros((region.height, region.width), dtype=np.float32)
    rng_seed = (
        config.scratch_patch_random_seed
        + region.x * 73856093
        + region.y * 19349663
        + region.width * 83492791
    ) & 0xFFFFFFFF
    rng = np.random.default_rng(rng_seed)

    for center_x, center_y in centers:
        radius = patch_size // 2
        local_left = max(0, center_x - radius)
        local_top = max(0, center_y - radius)
        local_right = min(region.width, center_x + radius + 1)
        local_bottom = min(region.height, center_y + radius + 1)
        target_mask = (
            region.mask[local_top:local_bottom, local_left:local_right] > 0
        )
        if not np.any(target_mask):
            continue

        target_rect = (
            region.x + local_left,
            region.y + local_top,
            region.x + local_right,
            region.y + local_bottom,
        )
        target_width = local_right - local_left
        target_height = local_bottom - local_top
        tangent, normal, nearest_local = local_skeleton_frame(
            skeleton_points,
            center_x,
            center_y,
            patch_size * 2,
        )
        nearest_x, nearest_y = nearest_local
        nearest_global = (region.x + nearest_x, region.y + nearest_y)
        half_scratch_width = float(distance_to_edge[nearest_y, nearest_x])

        context_margin = patch_size * 2
        context_rect = expanded_rectangle(
            target_rect[0],
            target_rect[1],
            target_width,
            target_height,
            context_margin,
            source.shape[1],
            source.shape[0],
        )
        context_left, context_top, context_right, context_bottom = context_rect
        context_valid = (
            invalid_source_mask[
                context_top:context_bottom,
                context_left:context_right,
            ]
            == 0
        )
        target_statistics = patch_feature_statistics(
            low_frequency,
            gradient,
            texture_energy,
            context_rect,
            context_valid,
        )
        if target_statistics is None:
            continue
        context_rgb = source[
            context_top:context_bottom,
            context_left:context_right,
        ][context_valid]
        target_rgb_mean = np.mean(context_rgb.astype(np.float32), axis=0)

        candidate_patches = candidate_scratch_donor_patches(
            source,
            low_frequency,
            gradient,
            texture_energy,
            candidate_invalid_source_mask,
            nearest_global,
            tangent,
            normal,
            half_scratch_width,
            target_width,
            target_height,
            patch_size,
            config.search_margin,
        )
        scored_candidates: list[tuple[float, np.ndarray]] = []
        existing_weight = accumulated_weight[
            local_top:local_bottom,
            local_left:local_right,
        ]
        overlap = target_mask & (existing_weight > 0)

        for candidate in candidate_patches:
            candidate_statistics_values: list[float] = []
            for feature in (
                candidate.low_frequency,
                candidate.gradient,
                candidate.texture_energy,
            ):
                feature_float = feature.astype(np.float32)
                candidate_statistics_values.extend(
                    (float(np.mean(feature_float)), float(np.std(feature_float)))
                )
            candidate_statistics = np.asarray(
                candidate_statistics_values,
                dtype=np.float32,
            )
            feature_difference = candidate_statistics - target_statistics
            feature_weights = np.array(
                [1.0, 0.35, 0.20, 0.15, 0.55, 0.30],
                dtype=np.float32,
            )
            score = float(np.sum(feature_difference * feature_difference * feature_weights))

            donor_patch = candidate.rgb.astype(np.float32)
            color_correction = np.clip(
                target_rgb_mean - np.mean(donor_patch, axis=(0, 1)),
                -25.0,
                25.0,
            )
            adjusted_donor = np.clip(donor_patch + color_correction, 0, 255)

            if np.any(overlap):
                current_average = accumulation[
                    local_top:local_bottom,
                    local_left:local_right,
                ] / np.maximum(existing_weight[:, :, None], 1e-6)
                overlap_difference = (
                    adjusted_donor[overlap] - current_average[overlap]
                )
                score += float(np.mean(overlap_difference * overlap_difference)) * 0.35
            scored_candidates.append((score, adjusted_donor))

        if not scored_candidates:
            continue
        scored_candidates.sort(key=lambda item: item[0])
        top_candidates = scored_candidates[: config.scratch_patch_top_k]
        top_scores = np.asarray([item[0] for item in top_candidates], dtype=np.float64)
        score_scale = max(float(np.std(top_scores)), 1.0)
        probabilities = np.exp(-(top_scores - top_scores.min()) / score_scale)
        probabilities /= probabilities.sum()
        chosen_index = int(rng.choice(len(top_candidates), p=probabilities))
        chosen_patch = top_candidates[chosen_index][1]

        grid_y, grid_x = np.indices((target_height, target_width), dtype=np.float32)
        normalized_x = (grid_x - (target_width - 1) * 0.5) / max(target_width * 0.5, 1.0)
        normalized_y = (grid_y - (target_height - 1) * 0.5) / max(target_height * 0.5, 1.0)
        tile_weight = np.exp(-2.0 * (normalized_x * normalized_x + normalized_y * normalized_y))
        tile_weight *= target_mask
        accumulation[
            local_top:local_bottom,
            local_left:local_right,
        ] += chosen_patch * tile_weight[:, :, None]
        accumulated_weight[
            local_top:local_bottom,
            local_left:local_right,
        ] += tile_weight

    target = region.mask > 0
    if np.any(target & (accumulated_weight <= 0)):
        return False

    synthesized = accumulation / np.maximum(accumulated_weight[:, :, None], 1e-6)
    distance = cv2.distanceTransform(region.mask, cv2.DIST_L2, 5)
    if config.scratch_feather_pixels <= 0:
        alpha = target.astype(np.float32)
    else:
        alpha = np.clip(distance / float(config.scratch_feather_pixels), 0.0, 1.0)
    alpha = alpha[:, :, None]

    result_patch = result[
        region.y : region.y + region.height,
        region.x : region.x + region.width,
    ].astype(np.float32)
    blended = result_patch * (1.0 - alpha) + synthesized * alpha
    result[
        region.y : region.y + region.height,
        region.x : region.x + region.width,
    ] = np.clip(blended, 0, 255).astype(np.uint8)
    return True


def should_use_telea(region: RepairRegion) -> bool:
    if "scratch" in region.categories:
        return False
    max_dimension = max(region.width, region.height)
    return max_dimension <= 24 and region.raw_thickness <= 16


def is_long_pure_scratch(
    region: RepairRegion,
    config: RepairConfig,
) -> bool:
    return (
        region.categories == frozenset({"scratch"})
        and max(region.width, region.height) >= config.long_scratch_min_length
    )


def repair_region(
    result: np.ndarray,
    source: np.ndarray,
    low_frequency: np.ndarray,
    gradient: np.ndarray,
    texture_energy: np.ndarray,
    invalid_source_mask: np.ndarray,
    scratch_candidate_invalid_source_mask: np.ndarray,
    region: RepairRegion,
    config: RepairConfig,
) -> str:
    donor_candidate_invalid_mask = (
        scratch_candidate_invalid_source_mask
        if "scratch" in region.categories
        else invalid_source_mask
    )
    safe_for_telea = telea_is_safe(
        region,
        invalid_source_mask,
        config.inpaint_radius,
    )
    if should_use_telea(region) and safe_for_telea:
        repair_with_telea(result, region, config.inpaint_radius)
        return "telea"

    long_pure_scratch = is_long_pure_scratch(region, config)
    maximum_scratch_width = (
        maximum_scratch_width_along_skeleton(region)
        if long_pure_scratch
        else 0.0
    )
    thin_long_scratch = (
        long_pure_scratch and maximum_scratch_width < THIN_SCRATCH_MAX_WIDTH
    )
    skeleton_patches_attempted = False
    if long_pure_scratch and not thin_long_scratch:
        skeleton_patches_attempted = True
        repaired = repair_scratch_with_skeleton_patches(
            result=result,
            source=source,
            low_frequency=low_frequency,
            gradient=gradient,
            texture_energy=texture_energy,
            invalid_source_mask=invalid_source_mask,
            candidate_invalid_source_mask=donor_candidate_invalid_mask,
            region=region,
            config=config,
        )
        if repaired:
            return "skeleton_patches"

    try:
        donor_patch, match_ring, template_rect = find_donor_patch(
            source=source,
            low_frequency=low_frequency,
            gradient=gradient,
            texture_energy=texture_energy,
            invalid_source_mask=invalid_source_mask,
            region=region,
            config=config,
            candidate_invalid_source_mask=donor_candidate_invalid_mask,
        )
    except ValueError:
        if (
            "scratch" in region.categories
            and not skeleton_patches_attempted
            and not thin_long_scratch
        ):
            repaired = repair_scratch_with_skeleton_patches(
                result=result,
                source=source,
                low_frequency=low_frequency,
                gradient=gradient,
                texture_energy=texture_energy,
                invalid_source_mask=invalid_source_mask,
                candidate_invalid_source_mask=donor_candidate_invalid_mask,
                region=region,
                config=config,
            )
            if repaired:
                return "skeleton_patches_fallback"
            raise
        if safe_for_telea:
            repair_with_telea(result, region, config.inpaint_radius)
            return "telea_fallback"
        raise

    blend_donor_patch(
        result=result,
        source=source,
        donor_patch=donor_patch,
        match_ring=match_ring,
        template_rect=template_rect,
        region=region,
        feather_pixels=(
            config.scratch_feather_pixels
            if "scratch" in region.categories
            else config.feather_pixels
        ),
    )
    return "thin_scratch_donor" if thin_long_scratch else "donor"


def write_silver_box_json(data: dict, output_json_path: Path) -> None:
    output_data = copy.deepcopy(data)
    output_data["objects"] = objects_by_category(output_data, SILVER_BOX_KEY)

    with output_json_path.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)
        file.write("\n")


def next_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    numeric_indices = [
        int(path.name)
        for path in output_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    run_dir = output_root / f"{max(numeric_indices, default=0) + 1:02d}"
    run_dir.mkdir(parents=False, exist_ok=False)
    return run_dir


def process_one(
    json_path: Path,
    output_dir: Path,
    config: RepairConfig,
) -> bool:
    data = read_json(json_path)
    defects = defect_objects(data)
    if not defects:
        return False

    image_path = find_image_for_json(json_path)
    with Image.open(image_path) as image_file:
        source = np.asarray(image_file.convert("RGB"))

    height, width = source.shape[:2]
    silver_objects = objects_by_category(data, SILVER_BOX_KEY)
    if not silver_objects:
        raise ValueError(f"No Silver box object found in: {json_path}")

    silver_mask = build_mask((width, height), silver_objects)
    if not silver_mask.any():
        raise ValueError(f"Silver box mask is empty in: {json_path}")

    raw_union = np.zeros((height, width), dtype=np.uint8)
    expanded_union = np.zeros((height, width), dtype=np.uint8)
    category_map = np.zeros((height, width), dtype=np.uint8)
    valid_defect_count = 0

    for item in defects:
        mask = build_mask((width, height), [item])
        if not mask.any():
            continue

        category = normalize_category(item.get("category"))
        expansion_pixels = defect_expansion_pixels(mask, category)
        expanded = dilate_mask(mask, expansion_pixels)
        raw_union = cv2.bitwise_or(raw_union, mask)
        expanded_union = cv2.bitwise_or(expanded_union, expanded)
        category_map[expanded > 0] |= CATEGORY_BITS[category]
        valid_defect_count += 1

    if valid_defect_count == 0:
        raise ValueError(f"All defect segmentations are invalid in: {json_path}")

    expanded_union = cv2.bitwise_and(expanded_union, silver_mask)
    raw_union = cv2.bitwise_and(raw_union, silver_mask)
    if not expanded_union.any():
        raise ValueError(f"All defects are outside Silver box in: {json_path}")

    avoided_objects = objects_by_category(data, AVOIDED_AREA_KEY)
    avoided_mask = build_mask((width, height), avoided_objects)
    invalid_source_mask = np.where(
        (silver_mask == 0) | (expanded_union > 0) | (avoided_mask > 0),
        255,
        0,
    ).astype(np.uint8)
    scratch_donor_mask = erode_mask(
        silver_mask,
        config.scratch_donor_inset_pixels,
    )
    scratch_candidate_invalid_source_mask = np.where(
        (scratch_donor_mask == 0)
        | (expanded_union > 0)
        | (avoided_mask > 0),
        255,
        0,
    ).astype(np.uint8)

    regions = build_repair_regions(expanded_union, raw_union, category_map)
    if not regions:
        raise ValueError(f"No repairable defect region found in: {json_path}")

    low_frequency, gradient, texture_energy = build_texture_features(source)

    result = source.copy()
    for index, region in enumerate(regions, start=1):
        try:
            repair_region(
                result=result,
                source=source,
                low_frequency=low_frequency,
                gradient=gradient,
                texture_energy=texture_energy,
                invalid_source_mask=invalid_source_mask,
                scratch_candidate_invalid_source_mask=(
                    scratch_candidate_invalid_source_mask
                ),
                region=region,
                config=config,
            )
        except ValueError as error:
            categories = ", ".join(sorted(region.categories)) or "unknown"
            raise ValueError(
                f"Failed to repair region {index} ({categories}) in "
                f"{json_path.name}: {error}"
            ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(output_dir / image_path.name)
    write_silver_box_json(data, output_dir / json_path.name)
    return True


def process_folder(
    source_dir: Path = SOURCE_DIR,
    output_dir: Path | None = None,
    config: RepairConfig | None = None,
) -> tuple[int, int, Path]:
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {source_dir}")

    repair_config = config or RepairConfig()
    repair_config.validate()

    output_root = output_dir or source_dir / OUTPUT_DIR_NAME
    resolved_output_dir = next_run_dir(output_root)

    processed = 0
    skipped = 0
    for json_path in sorted(source_dir.glob("*.json")):
        if process_one(json_path, resolved_output_dir, repair_config):
            processed += 1
        else:
            skipped += 1

    return processed, skipped, resolved_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair Scratch, Pit, and Stain regions with hybrid TELEA and "
            "matched clean-patch synthesis inside Silver box."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=SOURCE_DIR,
        help=f"Folder containing same-name images and JSON files. Default: {SOURCE_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output root folder; an incrementing numeric run folder is created "
            f"inside it. Default: {OUTPUT_DIR_NAME} inside source-dir."
        ),
    )
    parser.add_argument(
        "--inpaint-radius",
        type=int,
        default=INPAINT_RADIUS,
        help=f"TELEA radius for thin or small defects. Default: {INPAINT_RADIUS}",
    )
    parser.add_argument(
        "--search-margin",
        type=int,
        default=SEARCH_MARGIN,
        help=f"Initial local donor-search margin. Default: {SEARCH_MARGIN}",
    )
    parser.add_argument(
        "--match-band",
        type=int,
        default=MATCH_BAND,
        help=f"Clean boundary width used for donor matching. Default: {MATCH_BAND}",
    )
    parser.add_argument(
        "--feather-pixels",
        type=int,
        default=FEATHER_PIXELS,
        help=f"Donor-patch boundary blend width. Default: {FEATHER_PIXELS}",
    )
    parser.add_argument(
        "--scratch-feather-pixels",
        type=int,
        default=SCRATCH_FEATHER_PIXELS,
        help=(
            "Scratch donor-patch boundary blend width. "
            f"Default: {SCRATCH_FEATHER_PIXELS}"
        ),
    )
    parser.add_argument(
        "--long-scratch-min-length",
        type=int,
        default=LONG_SCRATCH_MIN_LENGTH,
        help=(
            "Minimum Scratch bounding-box length for skeleton-guided patch "
            f"repair. Default: {LONG_SCRATCH_MIN_LENGTH}"
        ),
    )
    parser.add_argument(
        "--scratch-patch-min-size",
        type=int,
        default=SCRATCH_PATCH_MIN_SIZE,
        help=f"Minimum odd Scratch patch size. Default: {SCRATCH_PATCH_MIN_SIZE}",
    )
    parser.add_argument(
        "--scratch-patch-max-size",
        type=int,
        default=SCRATCH_PATCH_MAX_SIZE,
        help=f"Maximum odd Scratch patch size. Default: {SCRATCH_PATCH_MAX_SIZE}",
    )
    parser.add_argument(
        "--scratch-patch-top-k",
        type=int,
        default=SCRATCH_PATCH_TOP_K,
        help=(
            "Randomly select from this many best Scratch donor candidates. "
            f"Default: {SCRATCH_PATCH_TOP_K}"
        ),
    )
    parser.add_argument(
        "--scratch-patch-random-seed",
        type=int,
        default=SCRATCH_PATCH_RANDOM_SEED,
        help=(
            "Random seed for reproducible Scratch patch selection. "
            f"Default: {SCRATCH_PATCH_RANDOM_SEED}"
        ),
    )
    parser.add_argument(
        "--scratch-donor-inset-pixels",
        type=int,
        default=SCRATCH_DONOR_INSET_PIXELS,
        help=(
            "Require every sampled Scratch donor pixel to stay inside Silver box "
            f"after this inward offset. Default: {SCRATCH_DONOR_INSET_PIXELS}"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RepairConfig(
        inpaint_radius=args.inpaint_radius,
        search_margin=args.search_margin,
        match_band=args.match_band,
        feather_pixels=args.feather_pixels,
        scratch_feather_pixels=args.scratch_feather_pixels,
        long_scratch_min_length=args.long_scratch_min_length,
        scratch_patch_min_size=args.scratch_patch_min_size,
        scratch_patch_max_size=args.scratch_patch_max_size,
        scratch_patch_top_k=args.scratch_patch_top_k,
        scratch_patch_random_seed=args.scratch_patch_random_seed,
        scratch_donor_inset_pixels=args.scratch_donor_inset_pixels,
    )
    processed, skipped, output_dir = process_folder(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        config=config,
    )
    print(
        f"Processed {processed} image(s); skipped {skipped} image(s) without "
        f"defects. Results saved to: {output_dir}"
    )


if __name__ == "__main__":
    main()
