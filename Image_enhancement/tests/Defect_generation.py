from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


NORMAL_DIR = Path(
    r"E:\projects\datasets\Power_box\old\results\results\Defective_free"
)
DEFECTIVE_DIR = Path(
    r"E:\projects\datasets\Power_box\old\results\results"
)
DEFECT_CATEGORIES = ("Scratch", "Pit", "Stain")
SILVER_BOX_KEY = "silver_box"
AVOIDED_AREA_KEY = "the_avoided_area"
DEFECTS_PER_IMAGE = 3
MAX_PLACEMENT_ATTEMPTS = 800
EDGE_FEATHER_PIXELS = 3
ANNOTATION_EXPAND_PIXELS = 5
ANNOTATION_LINE_THICKNESS = 2
ANNOTATION_COLOR_RGB = (255, 0, 0)

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


@dataclass(frozen=True)
class DefectPatch:
    category: str
    source_stem: str
    index: int
    rgb_patch: np.ndarray
    delta_patch: np.ndarray | None
    mask: np.ndarray

    @property
    def sort_key(self) -> tuple[str, int, int, int]:
        return (
            self.source_stem,
            self.index,
            int(self.mask.shape[0]),
            int(self.mask.shape[1]),
        )


@dataclass
class DefectBag:
    patches: list[DefectPatch]
    remaining: list[DefectPatch]

    @classmethod
    def from_patches(cls, patches: list[DefectPatch]) -> "DefectBag":
        return cls(patches=sorted(patches, key=lambda item: item.sort_key), remaining=[])

    def draw(self, count: int, rng: random.Random) -> list[DefectPatch]:
        if len(self.patches) < count:
            raise ValueError(
                f"Need at least {count} {self.patches[0].category if self.patches else ''} "
                "defect patches."
            )
        if len(self.remaining) < count:
            self.remaining = self.patches.copy()
        selected = rng.sample(self.remaining, count)
        selected_ids = {id(item) for item in selected}
        self.remaining = [
            item for item in self.remaining if id(item) not in selected_ids
        ]
        return selected


@dataclass(frozen=True)
class Placement:
    patch: DefectPatch
    x: int
    y: int
    rgb_patch: np.ndarray
    delta_patch: np.ndarray | None
    rotated_mask: np.ndarray
    final_mask: np.ndarray


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


def same_name_image(folder: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        image_path = folder / f"{stem}{suffix}"
        if image_path.exists():
            return image_path
    return None


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
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        points.append((int(round(float(point[0]))), int(round(float(point[1])))))
    return points


def build_mask(size: tuple[int, int], objects: list[dict]) -> np.ndarray:
    width, height = size
    mask = np.zeros((height, width), dtype=np.uint8)

    for item in objects:
        points = segmentation_points(item)
        if len(points) < 3:
            continue
        polygon = np.array(points, dtype=np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
        polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [polygon], 255)

    return mask


def mask_bounding_rect(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    return cv2.boundingRect(points)


def read_rgb(image_path: Path) -> np.ndarray:
    with Image.open(image_path) as image_file:
        return np.array(image_file.convert("RGB"))


def crop_patch(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    rect = mask_bounding_rect(mask)
    if rect is None:
        return None

    x, y, width, height = rect
    image_height, image_width = mask.shape
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image_width, x + width + padding)
    bottom = min(image_height, y + height + padding)
    return image[top:bottom, left:right].copy(), mask[top:bottom, left:right].copy()


def extract_defect_patches(
    defective_dir: Path,
    normal_dir: Path,
    padding: int,
) -> dict[str, list[DefectPatch]]:
    collected = {category: [] for category in DEFECT_CATEGORIES}

    for json_path in sorted(defective_dir.glob("*.json")):
        data = read_json(json_path)
        defective_image_path = find_image_for_json(json_path)
        defective_image = read_rgb(defective_image_path)
        height, width = defective_image.shape[:2]

        paired_normal_path = same_name_image(normal_dir, json_path.stem)
        paired_normal = (
            read_rgb(paired_normal_path)
            if paired_normal_path is not None
            else None
        )
        if paired_normal is not None and paired_normal.shape != defective_image.shape:
            paired_normal = None

        category_counts = {category: 0 for category in DEFECT_CATEGORIES}
        for item in data.get("objects", []):
            category = str(item.get("category", "")).strip()
            normalized = normalize_category(category)
            canonical = next(
                (
                    defect_category
                    for defect_category in DEFECT_CATEGORIES
                    if normalize_category(defect_category) == normalized
                ),
                None,
            )
            if canonical is None:
                continue

            mask = build_mask((width, height), [item])
            cropped = crop_patch(defective_image, mask, padding)
            if cropped is None:
                continue
            rgb_patch, patch_mask = cropped
            if not np.any(patch_mask):
                continue

            delta_patch = None
            if paired_normal is not None:
                normal_crop = crop_patch(paired_normal, mask, padding)
                if normal_crop is not None and normal_crop[0].shape == rgb_patch.shape:
                    delta_patch = (
                        rgb_patch.astype(np.int16) - normal_crop[0].astype(np.int16)
                    )

            category_counts[canonical] += 1
            collected[canonical].append(
                DefectPatch(
                    category=canonical,
                    source_stem=json_path.stem,
                    index=category_counts[canonical],
                    rgb_patch=rgb_patch,
                    delta_patch=delta_patch,
                    mask=patch_mask,
                )
            )

    for category, patches in collected.items():
        if len(patches) < DEFECTS_PER_IMAGE:
            raise ValueError(
                f"{category} only has {len(patches)} usable patches; "
                f"need at least {DEFECTS_PER_IMAGE}."
            )
        patches.sort(key=lambda item: item.sort_key)

    return collected


def next_run_dir(category_dir: Path) -> Path:
    category_dir.mkdir(parents=True, exist_ok=True)
    max_index = 0
    for child in category_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            max_index = max(max_index, int(child.name))
    return category_dir / f"{max_index + 1:02d}"


def rotate_patch(
    patch: DefectPatch,
    angle_degrees: float,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    height, width = patch.mask.shape
    center = ((width - 1) * 0.5, (height - 1) * 0.5)
    matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    cos_value = abs(matrix[0, 0])
    sin_value = abs(matrix[0, 1])
    new_width = int(np.ceil(height * sin_value + width * cos_value))
    new_height = int(np.ceil(height * cos_value + width * sin_value))
    matrix[0, 2] += new_width * 0.5 - center[0]
    matrix[1, 2] += new_height * 0.5 - center[1]

    rotated_rgb = cv2.warpAffine(
        patch.rgb_patch,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    rotated_mask = cv2.warpAffine(
        patch.mask,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    rotated_delta = None
    if patch.delta_patch is not None:
        rotated_delta = cv2.warpAffine(
            patch.delta_patch,
            matrix,
            (new_width, new_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    rect = mask_bounding_rect(rotated_mask)
    if rect is None:
        return rotated_rgb[:0, :0], rotated_delta, rotated_mask[:0, :0]

    x, y, width, height = rect
    cropped_rgb = rotated_rgb[y : y + height, x : x + width].copy()
    cropped_mask = rotated_mask[y : y + height, x : x + width].copy()
    cropped_delta = (
        None
        if rotated_delta is None
        else rotated_delta[y : y + height, x : x + width].copy()
    )
    return cropped_rgb, cropped_delta, cropped_mask


def random_top_left(
    image_width: int,
    image_height: int,
    patch_width: int,
    patch_height: int,
    category: str,
    rng: random.Random,
) -> tuple[int, int]:
    if category == "Pit":
        max_x = max(0, image_width - patch_width)
        max_y = max(0, image_height - patch_height)
        return rng.randint(0, max_x), rng.randint(0, max_y)

    min_x = min(0, image_width - patch_width)
    min_y = min(0, image_height - patch_height)
    return (
        rng.randint(min_x, image_width - 1),
        rng.randint(min_y, image_height - 1),
    )


def paste_mask(
    canvas_shape: tuple[int, int],
    patch_mask: np.ndarray,
    x: int,
    y: int,
) -> tuple[np.ndarray, tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    image_height, image_width = canvas_shape
    patch_height, patch_width = patch_mask.shape
    left = max(0, x)
    top = max(0, y)
    right = min(image_width, x + patch_width)
    bottom = min(image_height, y + patch_height)
    if left >= right or top >= bottom:
        return None

    patch_left = left - x
    patch_top = top - y
    patch_right = patch_left + (right - left)
    patch_bottom = patch_top + (bottom - top)
    canvas = np.zeros((image_height, image_width), dtype=np.uint8)
    canvas[top:bottom, left:right] = patch_mask[
        patch_top:patch_bottom,
        patch_left:patch_right,
    ]
    return (
        canvas,
        (left, top, right, bottom),
        (patch_left, patch_top, patch_right, patch_bottom),
    )


def find_placement(
    patch: DefectPatch,
    silver_mask: np.ndarray,
    avoided_mask: np.ndarray,
    used_mask: np.ndarray,
    rng: random.Random,
    max_attempts: int,
) -> Placement:
    image_height, image_width = silver_mask.shape

    for _ in range(max_attempts):
        angle = rng.uniform(0.0, 360.0)
        rotated_rgb, rotated_delta, rotated_mask = rotate_patch(patch, angle)
        if rotated_mask.size == 0:
            continue

        patch_height, patch_width = rotated_mask.shape
        x, y = random_top_left(
            image_width,
            image_height,
            patch_width,
            patch_height,
            patch.category,
            rng,
        )
        pasted = paste_mask((image_height, image_width), rotated_mask, x, y)
        if pasted is None:
            continue

        placed_mask, _, _ = pasted
        original_area = int(np.count_nonzero(rotated_mask))
        if original_area == 0:
            continue

        if patch.category == "Pit":
            if np.any((placed_mask > 0) & (silver_mask == 0)):
                continue
            if np.any((placed_mask > 0) & (avoided_mask > 0)):
                continue
            final_mask = placed_mask
        else:
            final_mask = np.where(
                (placed_mask > 0) & (silver_mask > 0) & (avoided_mask == 0),
                255,
                0,
            ).astype(np.uint8)
            if np.count_nonzero(final_mask) < original_area / 3.0:
                continue

        if np.any((final_mask > 0) & (used_mask > 0)):
            continue

        return Placement(
            patch=patch,
            x=x,
            y=y,
            rgb_patch=rotated_rgb,
            delta_patch=rotated_delta,
            rotated_mask=rotated_mask,
            final_mask=final_mask,
        )

    raise ValueError(
        f"Could not place {patch.category} patch from {patch.source_stem} "
        f"after {max_attempts} attempts."
    )


def local_mean_color(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    pixels = image[mask > 0]
    if pixels.size == 0:
        return np.zeros(3, dtype=np.float32)
    return pixels.astype(np.float32).mean(axis=0)


def apply_placement(
    image: np.ndarray,
    placement: Placement,
    output_mask: np.ndarray,
    feather_pixels: int,
) -> None:
    patch_height, patch_width = placement.rotated_mask.shape
    image_height, image_width = output_mask.shape
    left = max(0, placement.x)
    top = max(0, placement.y)
    right = min(image_width, placement.x + patch_width)
    bottom = min(image_height, placement.y + patch_height)
    if left >= right or top >= bottom:
        return

    patch_left = left - placement.x
    patch_top = top - placement.y
    patch_right = patch_left + (right - left)
    patch_bottom = patch_top + (bottom - top)

    local_final_mask = placement.final_mask[top:bottom, left:right]
    if not np.any(local_final_mask):
        return

    target_patch = image[top:bottom, left:right].astype(np.float32)
    if placement.delta_patch is not None:
        delta = placement.delta_patch[
            patch_top:patch_bottom,
            patch_left:patch_right,
        ].astype(np.float32)
        synthetic_patch = np.clip(target_patch + delta, 0, 255)
    else:
        source_patch = placement.rgb_patch[
            patch_top:patch_bottom,
            patch_left:patch_right,
        ].astype(np.float32)
        source_mask = placement.rotated_mask[
            patch_top:patch_bottom,
            patch_left:patch_right,
        ]
        color_shift = local_mean_color(target_patch.astype(np.uint8), local_final_mask)
        color_shift -= local_mean_color(source_patch.astype(np.uint8), source_mask)
        synthetic_patch = np.clip(source_patch + color_shift, 0, 255)

    if feather_pixels <= 0:
        alpha = (local_final_mask > 0).astype(np.float32)
    else:
        distance = cv2.distanceTransform(local_final_mask, cv2.DIST_L2, 5)
        alpha = np.clip(distance / float(feather_pixels), 0.0, 1.0)
    alpha = alpha[:, :, None]

    blended = target_patch * (1.0 - alpha) + synthetic_patch * alpha
    image[top:bottom, left:right] = np.clip(blended, 0, 255).astype(np.uint8)
    output_mask_patch = output_mask[top:bottom, left:right]
    output_mask_patch[local_final_mask > 0] = 255


def draw_expanded_defect_annotation(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
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


def normal_json_files(normal_dir: Path) -> list[Path]:
    return sorted(normal_dir.glob("*.json"))


def find_source_annotation_json(defective_dir: Path, stem: str) -> Path:
    candidates = (
        defective_dir / f"{stem}.json",
        defective_dir / "Defective_old" / f"{stem}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"No original same-name annotation JSON found for {stem}. "
        f"The_avoided_area must be read from the original annotations. Searched: {searched}"
    )


def process_image_for_category(
    json_path: Path,
    source_annotation_path: Path,
    image_path: Path,
    category: str,
    bag: DefectBag,
    output_dir: Path,
    rng: random.Random,
    max_attempts: int,
    feather_pixels: int,
) -> None:
    normal_data = read_json(json_path)
    source_data = read_json(source_annotation_path)
    image = read_rgb(image_path)
    height, width = image.shape[:2]

    silver_objects = objects_by_category(normal_data, SILVER_BOX_KEY)
    if not silver_objects:
        silver_objects = objects_by_category(source_data, SILVER_BOX_KEY)
    silver_mask = build_mask((width, height), silver_objects)
    if not np.any(silver_mask):
        raise ValueError(
            f"No valid Silver box mask found in {json_path} or "
            f"{source_annotation_path}"
        )

    avoided_mask = build_mask(
        (width, height),
        objects_by_category(source_data, AVOIDED_AREA_KEY),
    )
    output_mask = np.zeros((height, width), dtype=np.uint8)

    selected_patches = bag.draw(DEFECTS_PER_IMAGE, rng)
    for patch in selected_patches:
        placement = find_placement(
            patch=patch,
            silver_mask=silver_mask,
            avoided_mask=avoided_mask,
            used_mask=output_mask,
            rng=rng,
            max_attempts=max_attempts,
        )
        apply_placement(
            image=image,
            placement=placement,
            output_mask=output_mask,
            feather_pixels=feather_pixels,
        )

    avoided_overlap = int(
        np.count_nonzero((output_mask > 0) & (avoided_mask > 0))
    )
    if avoided_overlap:
        raise RuntimeError(
            f"Generated {category} mask for {image_path.name} overlaps "
            f"The_avoided_area by {avoided_overlap} pixel(s)."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    annotated_image = draw_expanded_defect_annotation(image, output_mask)
    Image.fromarray(annotated_image).save(output_dir / image_path.name)
    Image.fromarray(output_mask).save(mask_dir / image_path.name)


def process_category(
    category: str,
    json_files: list[Path],
    bag: DefectBag,
    defective_dir: Path,
    rng: random.Random,
    max_attempts: int,
    feather_pixels: int,
) -> tuple[int, Path]:
    run_dir = next_run_dir(defective_dir / category)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "mask").mkdir(parents=True, exist_ok=True)

    processed = 0
    for json_path in json_files:
        image_path = find_image_for_json(json_path)
        source_annotation_path = find_source_annotation_json(
            defective_dir,
            json_path.stem,
        )
        process_image_for_category(
            json_path=json_path,
            source_annotation_path=source_annotation_path,
            image_path=image_path,
            category=category,
            bag=bag,
            output_dir=run_dir,
            rng=rng,
            max_attempts=max_attempts,
            feather_pixels=feather_pixels,
        )
        processed += 1

    return processed, run_dir


def process_all(
    normal_dir: Path,
    defective_dir: Path,
    seed: int | None,
    patch_padding: int,
    max_attempts: int,
    feather_pixels: int,
) -> dict[str, tuple[int, Path]]:
    if not normal_dir.is_dir():
        raise NotADirectoryError(f"Normal folder does not exist: {normal_dir}")
    if not defective_dir.is_dir():
        raise NotADirectoryError(f"Defective folder does not exist: {defective_dir}")

    rng = random.Random(seed)
    json_files = normal_json_files(normal_dir)
    if not json_files:
        raise ValueError(f"No normal JSON files found in: {normal_dir}")

    patches_by_category = extract_defect_patches(
        defective_dir=defective_dir,
        normal_dir=normal_dir,
        padding=patch_padding,
    )

    results: dict[str, tuple[int, Path]] = {}
    for category in DEFECT_CATEGORIES:
        bag = DefectBag.from_patches(patches_by_category[category])
        results[category] = process_category(
            category=category,
            json_files=json_files,
            bag=bag,
            defective_dir=defective_dir,
            rng=rng,
            max_attempts=max_attempts,
            feather_pixels=feather_pixels,
        )
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate augmented Scratch, Pit, and Stain samples from normal "
            "images and annotated initial defect samples."
        )
    )
    parser.add_argument(
        "--normal-dir",
        type=Path,
        default=NORMAL_DIR,
        help=f"Folder containing normal same-name images and JSON files. Default: {NORMAL_DIR}",
    )
    parser.add_argument(
        "--defective-dir",
        type=Path,
        default=DEFECTIVE_DIR,
        help=(
            "Folder containing initial defective images and JSON files, and the "
            f"category output folders. Default: {DEFECTIVE_DIR}"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Omit for non-deterministic placement and sampling.",
    )
    parser.add_argument(
        "--patch-padding",
        type=int,
        default=8,
        help="Extra pixels kept around extracted defect masks. Default: 8",
    )
    parser.add_argument(
        "--max-placement-attempts",
        type=int,
        default=MAX_PLACEMENT_ATTEMPTS,
        help=(
            "Maximum random placement attempts for each defect. "
            f"Default: {MAX_PLACEMENT_ATTEMPTS}"
        ),
    )
    parser.add_argument(
        "--feather-pixels",
        type=int,
        default=EDGE_FEATHER_PIXELS,
        help=f"Boundary feather width when applying defects. Default: {EDGE_FEATHER_PIXELS}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = process_all(
        normal_dir=args.normal_dir,
        defective_dir=args.defective_dir,
        seed=args.seed,
        patch_padding=args.patch_padding,
        max_attempts=args.max_placement_attempts,
        feather_pixels=args.feather_pixels,
    )
    for category, (count, output_dir) in results.items():
        print(f"{category}: generated {count} image(s) in {output_dir}")


if __name__ == "__main__":
    main()
