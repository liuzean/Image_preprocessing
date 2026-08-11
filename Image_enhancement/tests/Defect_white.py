from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SOURCE_DIR = Path(r"E:\projects\datasets\Power_box\old\results\results")
OUTPUT_DIR_NAME = "Defect_white"
TARGET_CATEGORY_KEYS = {"scratch", "pit"}
ANNOTATION_EXPAND_PIXELS = 5
ANNOTATION_LINE_THICKNESS = 2
ANNOTATION_COLOR_RGB = (255, 0, 0)
SKELETON_COLOR_RGB = (0, 255, 0)

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

    raise FileNotFoundError(f"No same-name image found for: {json_path}")


def target_objects(data: dict) -> list[dict]:
    return [
        item
        for item in data.get("objects", [])
        if isinstance(item, dict)
        and normalize_category(item.get("category")) in TARGET_CATEGORY_KEYS
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


def object_mask(size: tuple[int, int], item: dict) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)
    points = segmentation_points(item)
    if not points:
        return mask

    polygon = np.asarray(points, dtype=np.int32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    cv2.fillPoly(mask, [polygon], 255)
    return mask


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


def mask_bounding_rect(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(mask)
    return None if points is None else cv2.boundingRect(points)


def object_whitening_alpha(
    mask: np.ndarray,
    base_whiten_strength: float,
    skeleton_gradient_strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.zeros(mask.shape, dtype=np.float32)
    skeleton_mask = np.zeros(mask.shape, dtype=np.uint8)
    rect = mask_bounding_rect(mask)
    if rect is None:
        return alpha, skeleton_mask

    x, y, width, height = rect
    cropped_mask = mask[y : y + height, x : x + width]
    skeleton = morphological_skeleton(cropped_mask)

    # distanceTransform measures each defect pixel's distance to the skeleton.
    distance_input = np.full(cropped_mask.shape, 255, dtype=np.uint8)
    distance_input[skeleton > 0] = 0
    distance_to_skeleton = cv2.distanceTransform(
        distance_input,
        cv2.DIST_L2,
        5,
    )

    inside = cropped_mask > 0
    max_distance = float(distance_to_skeleton[inside].max())
    if max_distance > 0:
        skeleton_weight = 1.0 - distance_to_skeleton / max_distance
        skeleton_weight = np.clip(skeleton_weight, 0.0, 1.0)
    else:
        skeleton_weight = np.ones(cropped_mask.shape, dtype=np.float32)

    cropped_alpha = np.zeros(cropped_mask.shape, dtype=np.float32)
    cropped_alpha[inside] = np.clip(
        base_whiten_strength
        + skeleton_gradient_strength * skeleton_weight[inside],
        0.0,
        1.0,
    )
    alpha[y : y + height, x : x + width] = cropped_alpha
    skeleton_mask[y : y + height, x : x + width] = skeleton
    return alpha, skeleton_mask


def whiten_defects(
    image: np.ndarray,
    objects: list[dict],
    base_whiten_strengths: dict[str, float],
    skeleton_gradient_strengths: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    combined_alpha = np.zeros((height, width), dtype=np.float32)
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    combined_skeleton = np.zeros((height, width), dtype=np.uint8)

    for item in objects:
        category_key = normalize_category(item.get("category"))
        if category_key not in base_whiten_strengths:
            continue

        mask = object_mask((width, height), item)
        if not np.any(mask):
            continue
        alpha, skeleton_mask = object_whitening_alpha(
            mask,
            base_whiten_strengths[category_key],
            skeleton_gradient_strengths[category_key],
        )
        combined_alpha = np.maximum(combined_alpha, alpha)
        combined_mask = cv2.bitwise_or(combined_mask, mask)
        combined_skeleton = cv2.bitwise_or(combined_skeleton, skeleton_mask)

    alpha_3d = combined_alpha[..., None]
    whitened = image.astype(np.float32) * (1.0 - alpha_3d) + 255.0 * alpha_3d
    return (
        np.clip(np.rint(whitened), 0, 255).astype(np.uint8),
        combined_mask,
        combined_skeleton,
    )


def draw_expanded_annotation(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if not np.any(mask):
        return image

    kernel_size = ANNOTATION_EXPAND_PIXELS * 2 + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(
        expanded_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    annotated = image.copy()
    cv2.drawContours(
        annotated,
        contours,
        -1,
        ANNOTATION_COLOR_RGB,
        ANNOTATION_LINE_THICKNESS,
    )
    return annotated


def draw_skeleton(
    image: np.ndarray,
    skeleton_mask: np.ndarray,
    skeleton_width: int,
) -> np.ndarray:
    if not np.any(skeleton_mask):
        return image

    if skeleton_width == 1:
        visible_skeleton = skeleton_mask
    else:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (skeleton_width, skeleton_width),
        )
        visible_skeleton = cv2.dilate(skeleton_mask, kernel, iterations=1)

    visualized = image.copy()
    visualized[visible_skeleton > 0] = SKELETON_COLOR_RGB
    return visualized


def next_run_dir(source_dir: Path) -> Path:
    numeric_indices = [
        int(path.name)
        for path in source_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    next_index = max(numeric_indices, default=0) + 1
    return source_dir / f"{next_index:02d}"


def validate_strength(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0, got {value}")


def validate_category_strengths(name: str, strengths: dict[str, float]) -> None:
    missing_categories = TARGET_CATEGORY_KEYS - set(strengths)
    if missing_categories:
        joined = ", ".join(sorted(missing_categories))
        raise ValueError(f"{name} is missing category strength(s): {joined}")

    for category_key, value in strengths.items():
        if category_key not in TARGET_CATEGORY_KEYS:
            raise ValueError(f"{name} contains unsupported category: {category_key}")
        validate_strength(f"{name}[{category_key}]", value)


def save_rgb(image: np.ndarray, output_path: Path) -> None:
    pil_image = Image.fromarray(image, mode="RGB")
    if output_path.suffix.lower() in {".jpg", ".jpeg"}:
        pil_image.save(output_path, quality=95, subsampling=0)
    else:
        pil_image.save(output_path)


def process_dataset(
    source_dir: Path,
    base_whiten_strengths: dict[str, float],
    skeleton_gradient_strengths: dict[str, float],
    enable_red_outline: bool,
    enable_green_skeleton: bool,
    skeleton_visualization_width: int,
) -> tuple[int, int, Path]:
    validate_category_strengths("base_whiten_strengths", base_whiten_strengths)
    validate_category_strengths(
        "skeleton_gradient_strengths",
        skeleton_gradient_strengths,
    )
    if skeleton_visualization_width < 1:
        raise ValueError("skeleton_visualization_width must be at least 1")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")

    json_paths = sorted(source_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found in: {source_dir}")

    output_root = source_dir / OUTPUT_DIR_NAME
    output_root.mkdir(parents=False, exist_ok=True)
    output_dir = next_run_dir(output_root)
    output_dir.mkdir(parents=False, exist_ok=False)
    processed_count = 0
    skipped_count = 0

    for json_path in json_paths:
        data = read_json(json_path)
        objects = target_objects(data)
        if not objects:
            skipped_count += 1
            continue

        image_path = find_image_for_json(json_path)
        with Image.open(image_path) as image_file:
            image = np.asarray(image_file.convert("RGB"))

        output_image, defect_mask, skeleton_mask = whiten_defects(
            image,
            objects,
            base_whiten_strengths,
            skeleton_gradient_strengths,
        )
        if enable_red_outline:
            output_image = draw_expanded_annotation(output_image, defect_mask)
        if enable_green_skeleton:
            output_image = draw_skeleton(
                output_image,
                skeleton_mask,
                skeleton_visualization_width,
            )

        save_rgb(output_image, output_dir / image_path.name)
        shutil.copy2(json_path, output_dir / json_path.name)
        processed_count += 1

    return processed_count, skipped_count, output_dir


if __name__ == "__main__":
    # Base whitening strength for the whole defect mask, range 0.0~1.0.基础增白强度，范围0.0~1.0
    SCRATCH_BASE_WHITEN_STRENGTH = 0.05
    PIT_BASE_WHITEN_STRENGTH = 0.05

    # Extra whitening strength at the skeleton. The effect decays toward mask edges.
    SCRATCH_SKELETON_GRADIENT_STRENGTH = 0.4
    PIT_SKELETON_GRADIENT_STRENGTH = 0.3

    BASE_WHITEN_STRENGTHS = {
        "scratch": SCRATCH_BASE_WHITEN_STRENGTH,
        "pit": PIT_BASE_WHITEN_STRENGTH,
    }
    SKELETON_GRADIENT_STRENGTHS = {
        "scratch": SCRATCH_SKELETON_GRADIENT_STRENGTH,
        "pit": PIT_SKELETON_GRADIENT_STRENGTH,
    }

    # True draws a red outline expanded by 5 pixels around each defect.
    ENABLE_RED_OUTLINE = False
    # True overlays the detected skeleton in green on the output image.
    ENABLE_GREEN_SKELETON = False
    # Green skeleton visualization width in pixels. This does not affect whitening.
    SKELETON_VISUALIZATION_WIDTH = 1

    processed, skipped, result_dir = process_dataset(
        source_dir=SOURCE_DIR,
        base_whiten_strengths=BASE_WHITEN_STRENGTHS,
        skeleton_gradient_strengths=SKELETON_GRADIENT_STRENGTHS,
        enable_red_outline=ENABLE_RED_OUTLINE,
        enable_green_skeleton=ENABLE_GREEN_SKELETON,
        skeleton_visualization_width=SKELETON_VISUALIZATION_WIDTH,
    )
    print(f"Processed {processed} image(s), skipped {skipped} without Scratch/Pit.")
    print(f"Results saved to: {result_dir}")
