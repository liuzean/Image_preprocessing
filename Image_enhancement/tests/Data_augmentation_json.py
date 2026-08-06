from __future__ import annotations

import argparse
import copy
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

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
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
AUGMENTED_NAME_PATTERN = re.compile(
    r"^(?P<source>\d{3})_aug_r(?P<angle>\d{3})"
    r"(?:_(?P<flip>vflip|hflip))?$"
)
SILVER_BOX_KEY = "silver_box"


@dataclass(frozen=True)
class Augmentation:
    source_stem: str
    angle: int
    flip: str | None


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


def parse_augmentation(image_path: Path) -> Augmentation | None:
    match = AUGMENTED_NAME_PATTERN.fullmatch(image_path.stem)
    if match is None:
        return None
    angle = int(match.group("angle"))
    if angle < 0 or angle > 360 or angle % 20 != 0:
        raise ValueError(f"Unsupported augmentation angle: {image_path.name}")
    return Augmentation(
        source_stem=match.group("source"),
        angle=angle,
        flip=match.group("flip"),
    )


def rotate_point(
    x: float,
    y: float,
    angle: int,
    width: int,
    height: int,
) -> tuple[float, float]:
    normalized_angle = angle % 360
    if normalized_angle == 0:
        return x, y
    if normalized_angle == 180:
        # Pillow uses its exact ROTATE_180 transpose fast path.
        return (width - 1) - x, (height - 1) - y

    center_x = width * 0.5
    center_y = height * 0.5
    radians = math.radians(normalized_angle)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    relative_x = x - center_x
    relative_y = y - center_y
    return (
        cosine * relative_x + sine * relative_y + center_x,
        -sine * relative_x + cosine * relative_y + center_y,
    )


def transform_point(
    x: float,
    y: float,
    augmentation: Augmentation,
    width: int,
    height: int,
) -> tuple[float, float]:
    transformed_x, transformed_y = rotate_point(
        x,
        y,
        augmentation.angle,
        width,
        height,
    )
    if augmentation.flip == "vflip":
        transformed_y = (height - 1) - transformed_y
    elif augmentation.flip == "hflip":
        transformed_x = (width - 1) - transformed_x
    return transformed_x, transformed_y


def clip_polygon_to_image(
    points: list[tuple[float, float]],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    boundaries = (
        (lambda point: point[0] >= 0.0, lambda a, b: _vertical_intersection(a, b, 0.0)),
        (
            lambda point: point[0] <= width - 1.0,
            lambda a, b: _vertical_intersection(a, b, width - 1.0),
        ),
        (lambda point: point[1] >= 0.0, lambda a, b: _horizontal_intersection(a, b, 0.0)),
        (
            lambda point: point[1] <= height - 1.0,
            lambda a, b: _horizontal_intersection(a, b, height - 1.0),
        ),
    )
    clipped = points
    for inside, intersection in boundaries:
        if not clipped:
            break
        output: list[tuple[float, float]] = []
        previous = clipped[-1]
        previous_inside = inside(previous)
        for current in clipped:
            current_inside = inside(current)
            if current_inside:
                if not previous_inside:
                    output.append(intersection(previous, current))
                output.append(current)
            elif previous_inside:
                output.append(intersection(previous, current))
            previous = current
            previous_inside = current_inside
        clipped = output
    return clipped


def _vertical_intersection(
    start: tuple[float, float],
    end: tuple[float, float],
    x_value: float,
) -> tuple[float, float]:
    delta_x = end[0] - start[0]
    ratio = 0.0 if abs(delta_x) < 1e-12 else (x_value - start[0]) / delta_x
    return x_value, start[1] + ratio * (end[1] - start[1])


def _horizontal_intersection(
    start: tuple[float, float],
    end: tuple[float, float],
    y_value: float,
) -> tuple[float, float]:
    delta_y = end[1] - start[1]
    ratio = 0.0 if abs(delta_y) < 1e-12 else (y_value - start[1]) / delta_y
    return start[0] + ratio * (end[0] - start[0]), y_value


def transform_segmentation(
    segmentation: object,
    augmentation: Augmentation,
    width: int,
    height: int,
) -> list[list[float]] | None:
    if not isinstance(segmentation, list):
        return None
    transformed: list[tuple[float, float]] = []
    for point in segmentation:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        try:
            transformed.append(
                transform_point(
                    float(point[0]),
                    float(point[1]),
                    augmentation,
                    width,
                    height,
                )
            )
        except (TypeError, ValueError):
            continue
    if len(transformed) < 3:
        return None
    clipped = clip_polygon_to_image(transformed, width, height)
    if len(clipped) < 3:
        return None
    return [[float(x), float(y)] for x, y in clipped]


def build_augmented_json(
    source_data: dict,
    image_path: Path,
    augmentation: Augmentation,
    width: int,
    height: int,
) -> dict:
    output_data = copy.deepcopy(source_data)
    silver_box_count = 0
    for item in output_data.get("objects", []):
        if not isinstance(item, dict):
            continue
        transformed = transform_segmentation(
            item.get("segmentation"),
            augmentation,
            width,
            height,
        )
        if transformed is not None:
            item["segmentation"] = transformed
        if normalize_category(item.get("category")) == SILVER_BOX_KEY:
            if transformed is None:
                raise ValueError(
                    f"Silver box becomes invalid after transforming: {image_path}"
                )
            silver_box_count += 1
    if silver_box_count == 0:
        raise ValueError(f"Source JSON has no Silver box object: {image_path}")

    info = output_data.get("info")
    if not isinstance(info, dict):
        info = {}
        output_data["info"] = info
    info["folder"] = image_path.parent.as_posix()
    info["name"] = image_path.name
    info["width"] = width
    info["height"] = height
    return output_data


def write_json_atomic(data: dict, output_path: Path) -> None:
    temporary_path = output_path.with_name(
        f".{output_path.stem}.augmentation_tmp.json"
    )
    try:
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.write("\n")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def augmented_images(folder_path: Path) -> list[tuple[Path, Augmentation]]:
    results: list[tuple[Path, Augmentation]] = []
    for image_path in sorted(folder_path.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        augmentation = parse_augmentation(image_path)
        if augmentation is not None:
            results.append((image_path, augmentation))
    return results


def process_folder(folder_path: Path, write_files: bool) -> int:
    images = augmented_images(folder_path)
    if not images:
        raise FileNotFoundError(f"No augmented images found in: {folder_path}")

    source_cache: dict[str, tuple[dict, int, int]] = {}
    generated_count = 0
    for image_path, augmentation in images:
        if augmentation.source_stem not in source_cache:
            source_json_path = folder_path / f"{augmentation.source_stem}.json"
            source_image_path = next(
                (
                    folder_path / f"{augmentation.source_stem}{suffix}"
                    for suffix in sorted(IMAGE_EXTENSIONS)
                    if (folder_path / f"{augmentation.source_stem}{suffix}").exists()
                ),
                None,
            )
            if not source_json_path.exists():
                raise FileNotFoundError(f"Source JSON does not exist: {source_json_path}")
            if source_image_path is None:
                raise FileNotFoundError(
                    f"Source image does not exist for: {source_json_path}"
                )
            source_data = read_json(source_json_path)
            with Image.open(source_image_path) as source_image:
                width, height = source_image.size
            source_cache[augmentation.source_stem] = (
                source_data,
                width,
                height,
            )

        source_data, width, height = source_cache[augmentation.source_stem]
        with Image.open(image_path) as augmented_image:
            if augmented_image.size != (width, height):
                raise ValueError(
                    f"Augmented image size differs from source: {image_path}"
                )
        output_data = build_augmented_json(
            source_data,
            image_path,
            augmentation,
            width,
            height,
        )
        if write_files:
            write_json_atomic(output_data, image_path.with_suffix(".json"))
        generated_count += 1
    return generated_count


def process_folders(root_dir: Path, write_files: bool) -> int:
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root directory does not exist: {root_dir}")
    folder_paths = [root_dir / name for name in TARGET_FOLDERS]
    missing_folders = [path for path in folder_paths if not path.is_dir()]
    if missing_folders:
        missing = ", ".join(str(path) for path in missing_folders)
        raise FileNotFoundError(f"Missing target folder(s): {missing}")

    total = 0
    for folder_path in folder_paths:
        count = process_folder(folder_path, write_files)
        total += count
        action = "generated" if write_files else "validated"
        print(f"{folder_path.name}: {count} augmented JSON file(s) {action}.")
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate transformed JSON annotations for images created by "
            "Data_augmentation.py."
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
        help="Validate all transformations without writing JSON files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = process_folders(args.root_dir, write_files=not args.dry_run)
    action = "Validated" if args.dry_run else "Generated"
    print(f"{action} {count} augmented JSON file(s).")


if __name__ == "__main__":
    main()
