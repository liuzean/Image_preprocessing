from __future__ import annotations

import argparse
import copy
from pathlib import Path
import json

import cv2
import numpy as np
from PIL import Image


BASE_DIR = Path(r"E:\projects\datasets\Power_box\old\results\results\Defective_free\11")
TARGET_FOLDERS = (
    "Power_box_1short",
    "Power_box_2long",
    "Power_box_3long",
    "Power_box_4long",
    "Power_box_5long",
    "Power_box_6short",
)
AVOIDED_AREA_KEY = "the_avoided_area"
SILVER_BOX_KEY = "silver_box"
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
        raise TypeError(f"JSON root must be an object: {json_path}")
    if not isinstance(data.get("objects"), list):
        raise TypeError(f'JSON must contain an "objects" list: {json_path}')
    return data


def write_json(json_path: Path, data: dict) -> None:
    with json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


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


def segmentation_points(item: dict) -> list[tuple[float, float]]:
    segmentation = item.get("segmentation")
    if not isinstance(segmentation, list):
        return []

    points: list[tuple[float, float]] = []
    for point in segmentation:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            points.append((float(point[0]), float(point[1])))
        except (TypeError, ValueError):
            continue
    return points if len(points) >= 3 else []


def silver_box_objects(data: dict) -> list[dict]:
    return [
        item
        for item in data.get("objects", [])
        if isinstance(item, dict)
        and normalize_category(item.get("category")) == SILVER_BOX_KEY
    ]


def avoided_area_objects(data: dict) -> list[dict]:
    return [
        item
        for item in data.get("objects", [])
        if isinstance(item, dict)
        and normalize_category(item.get("category")) == AVOIDED_AREA_KEY
    ]


def minimum_rectangle_center(objects: list[dict]) -> tuple[float, float]:
    point_groups: list[np.ndarray] = []
    for item in objects:
        points = segmentation_points(item)
        if points:
            point_groups.append(np.asarray(points, dtype=np.float32))
    if not point_groups:
        raise ValueError("Silver box has no valid segmentation points.")
    all_points = np.concatenate(point_groups, axis=0)
    center, _, _ = cv2.minAreaRect(all_points)
    return float(center[0]), float(center[1])


def calculate_translation(
    width: int,
    height: int,
    object_center: tuple[float, float],
) -> tuple[int, int]:
    image_center_x = (width - 1) * 0.5
    image_center_y = (height - 1) * 0.5
    return (
        round(image_center_x - object_center[0]),
        round(image_center_y - object_center[1]),
    )


def translate_segmentation(
    segmentation: object,
    offset_x: int,
    offset_y: int,
) -> list[list[float]] | None:
    if not isinstance(segmentation, list):
        return None

    translated: list[list[float]] = []
    for point in segmentation:
        if not isinstance(point, list) or len(point) < 2:
            continue
        try:
            x_value = point[0]
            y_value = point[1]
            if not isinstance(x_value, (int, float, str)):
                continue
            if not isinstance(y_value, (int, float, str)):
                continue
            translated.append([
                float(x_value) + offset_x,
                float(y_value) + offset_y,
            ])
        except (TypeError, ValueError):
            continue
    return translated if len(translated) >= 3 else None


def translated_avoided_objects(
    avoided_objects: list[dict],
    offset_x: int,
    offset_y: int,
) -> list[dict]:
    translated_objects: list[dict] = []
    for item in avoided_objects:
        translated_item = copy.deepcopy(item)
        translated_segmentation = translate_segmentation(
            translated_item.get("segmentation"),
            offset_x,
            offset_y,
        )
        if translated_segmentation is None:
            raise ValueError("The_avoided_area has invalid segmentation points.")
        translated_item["segmentation"] = translated_segmentation
        translated_objects.append(translated_item)
    return translated_objects


def remove_avoided_objects(data: dict) -> None:
    data["objects"] = [
        item
        for item in data.get("objects", [])
        if not (
            isinstance(item, dict)
            and normalize_category(item.get("category")) == AVOIDED_AREA_KEY
        )
    ]


def copy_avoided_area_objects(base_dir: Path, dry_run: bool) -> None:
    if not base_dir.is_dir():
        raise NotADirectoryError(f"Base directory does not exist: {base_dir}")

    source_json_files = sorted(base_dir.glob("*.json"))
    if not source_json_files:
        print(f"No JSON files found in base directory: {base_dir}")
        return

    updated = 0
    skipped = 0

    for source_json_path in source_json_files:
        try:
            source_data = read_json(source_json_path)
            source_avoided_objects = avoided_area_objects(source_data)
            if not source_avoided_objects:
                print(f"ℹ No {AVOIDED_AREA_KEY} found in {source_json_path.name}")
                skipped += 1
                continue

            source_image_path = find_image_for_json(source_json_path)
            with Image.open(source_image_path) as image_file:
                width, height = image_file.size

            silver_objects = silver_box_objects(source_data)
            if not silver_objects:
                raise ValueError(f"No Silver box object found in: {source_json_path}")

            object_center = minimum_rectangle_center(silver_objects)
            offset_x, offset_y = calculate_translation(width, height, object_center)
            moved_avoided_objects = translated_avoided_objects(
                source_avoided_objects,
                offset_x,
                offset_y,
            )

            for folder_name in TARGET_FOLDERS:
                target_json_path = base_dir / folder_name / source_json_path.name
                if not target_json_path.exists():
                    print(f"⚠ Target JSON not found: {target_json_path}")
                    skipped += 1
                    continue

                target_image_path = find_image_for_json(target_json_path)
                with Image.open(target_image_path) as image_file:
                    target_width, target_height = image_file.size
                if (target_width, target_height) != (width, height):
                    raise ValueError(
                        f"Image size mismatch for {target_json_path.name}: "
                        f"source={(width, height)}, target={(target_width, target_height)}"
                    )

                target_data = read_json(target_json_path)
                remove_avoided_objects(target_data)
                target_data["objects"].extend(copy.deepcopy(item) for item in moved_avoided_objects)

                if dry_run:
                    print(
                        f"[dry-run] {source_json_path.name} -> {target_json_path}: "
                        f"copied {len(moved_avoided_objects)} {AVOIDED_AREA_KEY} object(s) "
                        f"with offset x={offset_x}, y={offset_y}"
                    )
                else:
                    write_json(target_json_path, target_data)
                    print(
                        f"✓ {source_json_path.name} -> {target_json_path.name}: "
                        f"copied {len(moved_avoided_objects)} {AVOIDED_AREA_KEY} object(s) "
                        f"with offset x={offset_x}, y={offset_y}"
                    )
                    updated += 1
        except (ValueError, TypeError, json.JSONDecodeError, FileNotFoundError) as exc:
            print(f"✗ Error processing {source_json_path.name}: {exc}")
            skipped += 1

    print(
        f"\n总计：{'将更新' if dry_run else '更新了'} {updated} 个 JSON 文件，"
        f"跳过 {skipped} 个源/目标项"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Copy {AVOIDED_AREA_KEY} objects from root JSON files into same-name JSON "
            "files under the six target folders, using the same translation as "
            "Center_placement.py."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show planned changes without modifying files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print("Dry run mode - no files will be modified.\n")

    copy_avoided_area_objects(BASE_DIR, args.dry_run)


if __name__ == "__main__":
    main()