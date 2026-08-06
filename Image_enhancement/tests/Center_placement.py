# 图片中的物体摆放到图片的中心位置，只适合纯背景的情况。

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT_DIR = Path(
    r"E:\projects\datasets\Power_box\old\results\results\Defective_free\11"
)
TARGET_FOLDERS = (
    "Power_box_1short",
    "Power_box_2long",
    "Power_box_3long",
    "Power_box_4long",
    "Power_box_5long",
    "Power_box_6short",
)
SILVER_BOX_KEY = "silver_box"
IMAGE_EXTENSIONS = (
    ".bmp",
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


def build_mask(width: int, height: int, objects: list[dict]) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in objects:
        points = segmentation_points(item)
        if not points:
            continue
        polygon = np.rint(np.asarray(points, dtype=np.float32)).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [polygon], 255)
    return mask


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
        int(round(image_center_x - object_center[0])),
        int(round(image_center_y - object_center[1])),
    )


def translated_slices(
    width: int,
    height: int,
    offset_x: int,
    offset_y: int,
) -> tuple[slice, slice, slice, slice]:
    source_left = max(0, -offset_x)
    source_top = max(0, -offset_y)
    source_right = min(width, width - offset_x)
    source_bottom = min(height, height - offset_y)
    if source_left >= source_right or source_top >= source_bottom:
        raise ValueError("Translation moves the entire object outside the image.")
    target_left = source_left + offset_x
    target_top = source_top + offset_y
    target_right = source_right + offset_x
    target_bottom = source_bottom + offset_y
    return (
        slice(source_top, source_bottom),
        slice(source_left, source_right),
        slice(target_top, target_bottom),
        slice(target_left, target_right),
    )


def center_object_on_black(
    image: np.ndarray,
    silver_mask: np.ndarray,
    offset_x: int,
    offset_y: int,
) -> np.ndarray:
    height, width = silver_mask.shape
    masked_object = np.zeros_like(image)
    masked_object[silver_mask > 0] = image[silver_mask > 0]
    centered = np.zeros_like(image)
    source_y, source_x, target_y, target_x = translated_slices(
        width,
        height,
        offset_x,
        offset_y,
    )
    centered[target_y, target_x] = masked_object[source_y, source_x]
    translated_mask = np.zeros_like(silver_mask)
    translated_mask[target_y, target_x] = silver_mask[source_y, source_x]
    if np.count_nonzero(translated_mask) != np.count_nonzero(silver_mask):
        raise ValueError("Centering would clip part of the Silver box object.")
    return centered


def translate_annotations(data: dict, offset_x: int, offset_y: int) -> dict:
    translated = json.loads(json.dumps(data))
    for item in translated.get("objects", []):
        if not isinstance(item, dict):
            continue
        segmentation = item.get("segmentation")
        if not isinstance(segmentation, list):
            continue
        for point in segmentation:
            if not isinstance(point, list) or len(point) < 2:
                continue
            try:
                point[0] = float(point[0]) + offset_x
                point[1] = float(point[1]) + offset_y
            except (TypeError, ValueError):
                continue
    return translated


def image_save_options(image_format: str | None) -> dict:
    if image_format in {"JPEG", "JPG"}:
        return {"quality": 95, "subsampling": 0}
    return {}


def overwrite_pair(
    image_path: Path,
    json_path: Path,
    image: np.ndarray,
    data: dict,
    image_format: str | None,
) -> None:
    temporary_image = image_path.with_name(
        f".{image_path.stem}.center_tmp{image_path.suffix}"
    )
    temporary_json = json_path.with_name(f".{json_path.stem}.center_tmp.json")
    try:
        Image.fromarray(image).save(
            temporary_image,
            format=image_format,
            **image_save_options(image_format),
        )
        with temporary_json.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.write("\n")
        temporary_image.replace(image_path)
        temporary_json.replace(json_path)
    finally:
        temporary_image.unlink(missing_ok=True)
        temporary_json.unlink(missing_ok=True)


def process_pair(json_path: Path, overwrite: bool) -> tuple[Path, int, int]:
    data = read_json(json_path)
    silver_objects = silver_box_objects(data)
    if not silver_objects:
        raise ValueError(f"No Silver box object found in: {json_path}")
    image_path = find_image_for_json(json_path)
    with Image.open(image_path) as image_file:
        image_format = image_file.format
        image = np.asarray(image_file.convert("RGB"))
    height, width = image.shape[:2]
    silver_mask = build_mask(width, height, silver_objects)
    if not np.any(silver_mask):
        raise ValueError(f"Silver box mask is empty in: {json_path}")

    object_center = minimum_rectangle_center(silver_objects)
    offset_x, offset_y = calculate_translation(width, height, object_center)
    centered_image = center_object_on_black(
        image,
        silver_mask,
        offset_x,
        offset_y,
    )
    translated_data = translate_annotations(data, offset_x, offset_y)

    translated_center = minimum_rectangle_center(
        silver_box_objects(translated_data)
    )
    expected_center = ((width - 1) * 0.5, (height - 1) * 0.5)
    center_error = max(
        abs(translated_center[0] - expected_center[0]),
        abs(translated_center[1] - expected_center[1]),
    )
    if center_error > 1.0:
        raise ValueError(f"Translated Silver box is not centered: {json_path}")

    if overwrite:
        overwrite_pair(
            image_path,
            json_path,
            centered_image,
            translated_data,
            image_format,
        )
    return image_path, offset_x, offset_y


def process_folders(root_dir: Path, overwrite: bool) -> int:
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root directory does not exist: {root_dir}")
    folder_paths = [root_dir / name for name in TARGET_FOLDERS]
    missing_folders = [path for path in folder_paths if not path.is_dir()]
    if missing_folders:
        missing = ", ".join(str(path) for path in missing_folders)
        raise FileNotFoundError(f"Missing target folder(s): {missing}")

    json_paths = [
        json_path
        for folder_path in folder_paths
        for json_path in sorted(folder_path.glob("*.json"))
    ]
    if not json_paths:
        raise FileNotFoundError(f"No JSON files found below: {root_dir}")
    for json_path in json_paths:
        find_image_for_json(json_path)
    for json_path in json_paths:
        image_path, offset_x, offset_y = process_pair(json_path, overwrite)
        mode = "overwritten" if overwrite else "checked"
        print(
            f"{mode}: {image_path} "
            f"(translation x={offset_x}, y={offset_y})"
        )
    return len(json_paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Center Silver box objects on a black background and overwrite "
            "same-name images and JSON annotations."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=ROOT_DIR,
        help=f"Folder containing the six Power_box folders. Default: {ROOT_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all pairs and translations without overwriting files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = process_folders(args.root_dir, overwrite=not args.dry_run)
    action = "Validated" if args.dry_run else "Centered and overwritten"
    print(f"{action} {count} image/JSON pair(s).")


if __name__ == "__main__":
    main()
