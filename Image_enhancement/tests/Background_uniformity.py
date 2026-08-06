from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


SOURCE_DIR = Path(r"E:\projects\datasets\Power_box\old\results\results")
OUTPUT_DIR_NAME = "Background_uniformity_results"
STANDARD_STEM = "001"
SHRINK_PIXELS = 30

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
CLEAN_AREA_KEY = "clean_area"
AVOIDED_AREA_KEY = "the_avoided_area"

Rect = tuple[int, int, int, int]


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
    raise FileNotFoundError(f"No same-name image found for JSON file: {json_path}")


def objects_by_category(data: dict, category_key: str) -> list[dict]:
    return [
        item
        for item in data.get("objects", [])
        if isinstance(item, dict)
        and normalize_category(item.get("category")) == category_key
    ]


def segmentation_points(item: dict) -> list[tuple[int, int]]:
    segmentation = item.get("segmentation")
    if not isinstance(segmentation, list):
        return []

    points: list[tuple[int, int]] = []
    for point in segmentation:
        if not isinstance(point, list) or len(point) < 2:
            continue
        points.append((round(float(point[0])), round(float(point[1]))))
    return points if len(points) >= 3 else []


def build_mask(
    size: tuple[int, int],
    objects: list[dict],
) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)

    for item in objects:
        points = segmentation_points(item)
        if not points:
            continue
        polygon = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [polygon], 255)

    return mask


def largest_inner_rectangle(mask: np.ndarray) -> Rect:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        raise ValueError("Mask is empty; cannot calculate largest inner rectangle.")

    x_offset = int(xs.min())
    y_offset = int(ys.min())
    cropped = mask[y_offset : int(ys.max()) + 1, x_offset : int(xs.max()) + 1] > 0
    height, width = cropped.shape

    best_area = 0
    best_rect: Rect | None = None
    heights = [0] * width

    for y in range(height):
        row = cropped[y]
        for x, inside in enumerate(row):
            heights[x] = heights[x] + 1 if inside else 0

        stack: list[tuple[int, int]] = []
        for x in range(width + 1):
            current_height = heights[x] if x < width else 0
            start = x

            while stack and stack[-1][1] > current_height:
                left, rect_height = stack.pop()
                area = rect_height * (x - left)
                if area > best_area:
                    best_area = area
                    best_rect = (
                        x_offset + left,
                        y_offset + y + 1 - rect_height,
                        x_offset + x,
                        y_offset + y + 1,
                    )
                start = left

            if not stack or stack[-1][1] < current_height:
                stack.append((start, current_height))

    if best_rect is None:
        raise ValueError("No valid inner rectangle found in mask.")
    return best_rect


def load_standard_patch(source_dir: Path) -> np.ndarray:
    standard_json_path = source_dir / f"{STANDARD_STEM}.json"
    if not standard_json_path.exists():
        raise FileNotFoundError(f"Standard JSON does not exist: {standard_json_path}")

    standard_data = read_json(standard_json_path)
    clean_objects = objects_by_category(standard_data, CLEAN_AREA_KEY)
    if not clean_objects:
        raise ValueError(f"No Clean_area object found in: {standard_json_path}")

    standard_image_path = find_image_for_json(standard_json_path)
    with Image.open(standard_image_path) as image:
        standard_image = np.array(image.convert("RGB"))

    height, width = standard_image.shape[:2]
    clean_mask = build_mask((width, height), clean_objects)
    left, top, right, bottom = largest_inner_rectangle(clean_mask)
    patch = standard_image[top:bottom, left:right].copy()

    if patch.size == 0:
        raise ValueError(
            f"Clean_area largest inner rectangle is empty in: {standard_json_path}"
        )
    return patch


def shrink_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    if pixels <= 0:
        return mask.copy()

    kernel_size = pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask, kernel, iterations=1)


def fill_with_standard_patch(
    image: np.ndarray,
    target_mask: np.ndarray,
    patch: np.ndarray,
) -> np.ndarray:
    result = image.copy()
    ys, xs = np.nonzero(target_mask)
    if len(xs) == 0:
        return result

    patch_height, patch_width = patch.shape[:2]
    if patch_height == 0 or patch_width == 0:
        raise ValueError("Standard patch must not be empty.")

    min_x = int(xs.min())
    max_x = int(xs.max()) + 1
    min_y = int(ys.min())
    max_y = int(ys.max()) + 1

    for y in range(min_y, max_y, patch_height):
        tile_bottom = min(y + patch_height, max_y)
        tile_height = tile_bottom - y

        for x in range(min_x, max_x, patch_width):
            tile_right = min(x + patch_width, max_x)
            tile_width = tile_right - x
            mask_window = target_mask[y:tile_bottom, x:tile_right] > 0
            if not mask_window.any():
                continue

            result_window = result[y:tile_bottom, x:tile_right]
            patch_window = patch[:tile_height, :tile_width]
            result_window[mask_window] = patch_window[mask_window]

    return result


def restore_avoided_area(
    result: np.ndarray,
    original: np.ndarray,
    avoided_mask: np.ndarray,
    silver_mask: np.ndarray,
) -> np.ndarray:
    restore_mask = (avoided_mask > 0) & (silver_mask > 0)
    if restore_mask.any():
        result[restore_mask] = original[restore_mask]
    return result


def write_silver_box_json(data: dict, output_json_path: Path) -> None:
    output_data = copy.deepcopy(data)
    output_data["objects"] = objects_by_category(output_data, SILVER_BOX_KEY)

    with output_json_path.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, ensure_ascii=False, indent=4)
        file.write("\n")


def process_one(
    json_path: Path,
    output_dir: Path,
    standard_patch: np.ndarray,
    shrink_pixels: int,
) -> None:
    data = read_json(json_path)
    image_path = find_image_for_json(json_path)

    with Image.open(image_path) as source:
        image = np.array(source.convert("RGB"))

    height, width = image.shape[:2]
    silver_objects = objects_by_category(data, SILVER_BOX_KEY)
    if not silver_objects:
        raise ValueError(f"No Silver box object found in: {json_path}")

    silver_mask = build_mask((width, height), silver_objects)
    shrinked_silver_mask = shrink_mask(silver_mask, shrink_pixels)

    result = fill_with_standard_patch(image, shrinked_silver_mask, standard_patch)

    avoided_objects = objects_by_category(data, AVOIDED_AREA_KEY)
    if avoided_objects:
        avoided_mask = build_mask((width, height), avoided_objects)
        result = restore_avoided_area(result, image, avoided_mask, silver_mask)

    output_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result).save(output_dir / image_path.name)
    write_silver_box_json(data, output_dir / json_path.name)


def process_folder(
    source_dir: Path = SOURCE_DIR,
    output_dir: Path | None = None,
    shrink_pixels: int = SHRINK_PIXELS,
) -> tuple[int, Path]:
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {source_dir}")

    resolved_output_dir = output_dir or source_dir / OUTPUT_DIR_NAME
    standard_patch = load_standard_patch(source_dir)

    count = 0
    for json_path in sorted(source_dir.glob("*.json")):
        process_one(json_path, resolved_output_dir, standard_patch, shrink_pixels)
        count += 1

    return count, resolved_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tile the largest inner rectangle from 001 Clean_area into each "
            "image's 50-pixel-shrunk Silver box area."
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
            "Output folder. Default: Background_uniformity_results inside "
            "source-dir."
        ),
    )
    parser.add_argument(
        "--shrink-pixels",
        type=int,
        default=SHRINK_PIXELS,
        help=f"Pixels to erode inward from Silver box. Default: {SHRINK_PIXELS}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count, output_dir = process_folder(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        shrink_pixels=args.shrink_pixels,
    )
    print(f"Processed {count} image(s). Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
