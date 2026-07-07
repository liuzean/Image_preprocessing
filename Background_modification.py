from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = [
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
]
BACKGROUND_RGB = (0, 0, 0)


def find_image_for_json(json_path: Path) -> Path:
    for suffix in IMAGE_EXTENSIONS:
        image_path = json_path.with_suffix(suffix)
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"No same-name image found for JSON file: {json_path}")


def load_segmentations(json_path: Path) -> list[list[tuple[float, float]]]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    objects = data.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f'JSON must contain an "objects" list: {json_path}')

    segmentations: list[list[tuple[float, float]]] = []
    for item in objects:
        if not isinstance(item, dict):
            continue

        segmentation = item.get("segmentation")
        if not isinstance(segmentation, list) or len(segmentation) < 3:
            continue

        points: list[tuple[float, float]] = []
        for point in segmentation:
            if not isinstance(point, list) or len(point) < 2:
                continue
            points.append((float(point[0]), float(point[1])))

        if len(points) >= 3:
            segmentations.append(points)

    return segmentations


def build_mask(size: tuple[int, int], segmentations: list[list[tuple[float, float]]]) -> Image.Image:
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)

    for points in segmentations:
        draw.polygon(points, fill=255)

    return mask


def blacken_background(image_path: Path, json_path: Path, output_dir: Path) -> None:
    segmentations = load_segmentations(json_path)
    if not segmentations:
        raise ValueError(f"No valid mask segmentation found in: {json_path}")

    with Image.open(image_path) as image:
        image_rgb = image.convert("RGB")
        mask = build_mask(image_rgb.size, segmentations)
        result = Image.new("RGB", image_rgb.size, BACKGROUND_RGB)
        result.paste(image_rgb, mask=mask)

    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(output_dir / image_path.name)


def process_folder(target_dir: Path) -> int:
    if not target_dir.exists() or not target_dir.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {target_dir}")

    output_dir = target_dir / "results"
    count = 0

    for json_path in sorted(target_dir.glob("*.json")):
        image_path = find_image_for_json(json_path)
        blacken_background(image_path, json_path, output_dir)
        count += 1

    return count


def main(target_dir: Path) -> None:
    count = process_folder(target_dir)
    print(f"Processed {count} images. Results saved to: {target_dir / 'results'}")


if __name__ == "__main__":
    TARGET_DIR = Path(r"E:\projects\datasets\Power_box\Power_box_2long")
    main(TARGET_DIR)
