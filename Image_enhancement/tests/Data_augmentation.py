from __future__ import annotations

import argparse
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
AUGMENTATION_MARKER = "_aug_"
ROTATION_STEP_DEGREES = 20


def is_source_image(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and AUGMENTATION_MARKER not in path.stem
    )


def rotation_angles() -> tuple[int, ...]:
    return tuple(range(ROTATION_STEP_DEGREES, 361, ROTATION_STEP_DEGREES))


def augmentations_per_source() -> int:
    return 2 + len(rotation_angles()) * 3


def output_path(
    source_path: Path,
    angle: int,
    operation: str | None = None,
) -> Path:
    suffix = f"_aug_r{angle:03d}"
    if operation is not None:
        suffix += f"_{operation}"
    return source_path.with_name(f"{source_path.stem}{suffix}{source_path.suffix}")


def save_options(image_format: str | None) -> dict:
    if image_format in {"JPEG", "JPG"}:
        return {"quality": 95, "subsampling": 0}
    return {}


def save_image_atomic(
    image: Image.Image,
    destination: Path,
    image_format: str | None,
) -> None:
    temporary_path = destination.with_name(
        f".{destination.stem}.augmentation_tmp{destination.suffix}"
    )
    try:
        image.save(
            temporary_path,
            format=image_format,
            **save_options(image_format),
        )
        temporary_path.replace(destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def rotate_on_black(image: Image.Image, angle: int) -> Image.Image:
    if angle % 360 == 0:
        return image.copy()
    return image.rotate(
        angle,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(0, 0, 0),
    )


def augment_image(source_path: Path, write_files: bool) -> int:
    with Image.open(source_path) as image_file:
        image_format = image_file.format
        source = image_file.convert("RGB")

    generated_count = 0

    initial_variants = (
        (
            source.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
            output_path(source_path, 0, "vflip"),
        ),
        (
            source.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
            output_path(source_path, 0, "hflip"),
        ),
    )
    for variant, destination in initial_variants:
        if write_files:
            save_image_atomic(variant, destination, image_format)
        generated_count += 1

    for angle in rotation_angles():
        rotated = rotate_on_black(source, angle)
        variants = (
            (rotated, output_path(source_path, angle)),
            (
                rotated.transpose(Image.Transpose.FLIP_TOP_BOTTOM),
                output_path(source_path, angle, "vflip"),
            ),
            (
                rotated.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
                output_path(source_path, angle, "hflip"),
            ),
        )
        for variant, destination in variants:
            if write_files:
                save_image_atomic(variant, destination, image_format)
            generated_count += 1

    return generated_count


def process_folders(root_dir: Path, write_files: bool) -> tuple[int, int]:
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root directory does not exist: {root_dir}")

    folder_paths = [root_dir / name for name in TARGET_FOLDERS]
    missing_folders = [path for path in folder_paths if not path.is_dir()]
    if missing_folders:
        missing = ", ".join(str(path) for path in missing_folders)
        raise FileNotFoundError(f"Missing target folder(s): {missing}")

    source_count = 0
    generated_count = 0
    for folder_path in folder_paths:
        source_images = sorted(
            path for path in folder_path.iterdir() if is_source_image(path)
        )
        if not source_images:
            raise FileNotFoundError(f"No source images found in: {folder_path}")

        folder_generated = 0
        if write_files:
            for source_path in source_images:
                folder_generated += augment_image(source_path, write_files=True)
        else:
            folder_generated = len(source_images) * augmentations_per_source()
        source_count += len(source_images)
        generated_count += folder_generated
        action = "generated" if write_files else "planned"
        print(
            f"{folder_path.name}: {len(source_images)} source image(s), "
            f"{folder_generated} augmentation(s) {action}."
        )

    return source_count, generated_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate 20-degree rotations and vertical/horizontal flips in "
            "the six Power_box image folders."
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
        help="Count planned augmentations without writing image files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_count, generated_count = process_folders(
        args.root_dir,
        write_files=not args.dry_run,
    )
    action = "Would generate" if args.dry_run else "Generated"
    print(
        f"{action} {generated_count} augmentation(s) from "
        f"{source_count} source image(s)."
    )


if __name__ == "__main__":
    main()
