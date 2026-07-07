from __future__ import annotations

import argparse
import uuid
from pathlib import Path


IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args(default_dir: Path, default_start: int) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rename images by modified time from oldest to newest, and rename "
            "the same-stem JSON files to match."
        )
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=default_dir,
        help=f"Target folder. Default: {default_dir}",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional filename prefix, for example power_box_.",
    )
    parser.add_argument(
        "--start",   #图片开始的编号
        type=int,
        default=default_start,
        help="Starting number for renamed files.",
    )
    parser.add_argument(
        "--digits",   #限制图片编号的位数，最大为3位
        type=int,
        default=3,
        help="Zero-padding width for renamed files. Maximum: 3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the planned changes without renaming files.",
    )
    return parser.parse_args()


def find_images(folder: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda path: (path.stat().st_mtime, path.name.lower()),
    )


def build_rename_plan(
    images: list[Path],
    prefix: str,
    start: int,
    digits: int,
) -> list[tuple[Path, Path, Path, Path]]:
    if digits < 1 or digits > 3:
        raise ValueError("--digits must be between 1 and 3.")
    if start < 1:
        raise ValueError("--start must be greater than or equal to 1.")
    if start + len(images) - 1 > 999:
        raise ValueError("File numbering exceeds 999. File names are limited to 3 digits.")

    plan: list[tuple[Path, Path, Path, Path]] = []

    for index, image_path in enumerate(images, start=start):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            raise FileNotFoundError(f"Missing JSON file for image: {image_path}")

        new_stem = f"{prefix}{index:0{digits}d}"
        new_image_path = image_path.with_name(f"{new_stem}{image_path.suffix.lower()}")
        new_json_path = image_path.with_name(f"{new_stem}.json")
        plan.append((image_path, json_path, new_image_path, new_json_path))

    return plan


def ensure_no_external_conflicts(plan: list[tuple[Path, Path, Path, Path]]) -> None:
    sources = {path.resolve() for item in plan for path in item[:2]}

    for _, _, target_image, target_json in plan:
        for target in (target_image, target_json):
            if target.exists() and target.resolve() not in sources:
                raise FileExistsError(f"Target already exists and is not part of rename plan: {target}")


def print_plan(plan: list[tuple[Path, Path, Path, Path]]) -> None:
    for image_path, json_path, new_image_path, new_json_path in plan:
        print(f"{image_path.name} -> {new_image_path.name}")
        print(f"{json_path.name} -> {new_json_path.name}")


def rename_files(plan: list[tuple[Path, Path, Path, Path]]) -> None:
    temp_plan: list[tuple[Path, Path]] = []
    final_plan: list[tuple[Path, Path]] = []

    for image_path, json_path, new_image_path, new_json_path in plan:
        token = uuid.uuid4().hex
        temp_image_path = image_path.with_name(f".rename_tmp_{token}{image_path.suffix}")
        temp_json_path = json_path.with_name(f".rename_tmp_{token}.json")

        temp_plan.extend([(image_path, temp_image_path), (json_path, temp_json_path)])
        final_plan.extend([(temp_image_path, new_image_path), (temp_json_path, new_json_path)])

    completed_temp_renames: list[tuple[Path, Path]] = []
    try:
        for source, target in temp_plan:
            source.rename(target)
            completed_temp_renames.append((source, target))

        for source, target in final_plan:
            source.rename(target)
    except Exception:
        for original, temporary in reversed(completed_temp_renames):
            if temporary.exists() and not original.exists():
                temporary.rename(original)
        raise


def main(default_dir: Path, default_start: int) -> None:
    args = parse_args(default_dir, default_start)
    folder = args.dir

    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")

    images = find_images(folder)
    if not images:
        print(f"No image files found in: {folder}")
        return

    plan = build_rename_plan(images, args.prefix, args.start, args.digits)
    ensure_no_external_conflicts(plan)
    print_plan(plan)

    if args.dry_run:
        print("Dry run only. No files were renamed.")
        return

    rename_files(plan)
    print(f"Renamed {len(plan)} image files and {len(plan)} JSON files.")


if __name__ == "__main__":
    DEFAULT_DIR = Path(r"E:\projects\datasets\Power_box\Power_box_1")
    DEFAULT_START = 1  #文件编号从1开始
    main(DEFAULT_DIR, DEFAULT_START)
