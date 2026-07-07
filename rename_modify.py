from __future__ import annotations

import argparse
import json
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
            "Rename images and same-stem JSON files by image modified time, "
            "then update info.folder and info.name in each JSON file."
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
        "--start",
        type=int,
        default=default_start,
        help="Starting number for renamed files.",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=3,
        help="Zero-padding width for renamed files. Maximum: 3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned changes without renaming files or modifying JSON content.",
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
) -> list[tuple[Path, Path | None, Path, Path | None]]:
    if digits < 1 or digits > 3:
        raise ValueError("--digits must be between 1 and 3.")
    if start < 1:
        raise ValueError("--start must be greater than or equal to 1.")
    if start + len(images) - 1 > 999:
        raise ValueError("File numbering exceeds 999. File names are limited to 3 digits.")

    plan: list[tuple[Path, Path | None, Path, Path | None]] = []
    for index, image_path in enumerate(images, start=start):
        json_path = image_path.with_suffix(".json")
        if not json_path.exists():
            json_path = None

        new_stem = f"{prefix}{index:0{digits}d}"
        new_image_path = image_path.with_name(f"{new_stem}{image_path.suffix.lower()}")
        new_json_path = image_path.with_name(f"{new_stem}.json") if json_path else None
        plan.append((image_path, json_path, new_image_path, new_json_path))

    return plan


def ensure_no_external_conflicts(plan: list[tuple[Path, Path | None, Path, Path | None]]) -> None:
    sources = {path.resolve() for item in plan for path in item[:2] if path is not None}

    for _, _, target_image, target_json in plan:
        for target in (target_image, target_json):
            if target is None:
                continue
            if target.exists() and target.resolve() not in sources:
                raise FileExistsError(f"Target already exists and is not part of rename plan: {target}")


def print_rename_plan(plan: list[tuple[Path, Path | None, Path, Path | None]]) -> None:
    for image_path, json_path, new_image_path, new_json_path in plan:
        print(f"{image_path.name} -> {new_image_path.name}")
        if json_path and new_json_path:
            print(f"{json_path.name} -> {new_json_path.name}")
        else:
            print(f"{image_path.stem}.json not found. JSON rename/update skipped.")


def rename_files(plan: list[tuple[Path, Path | None, Path, Path | None]]) -> None:
    temp_plan: list[tuple[Path, Path]] = []
    final_plan: list[tuple[Path, Path]] = []

    for image_path, json_path, new_image_path, new_json_path in plan:
        token = uuid.uuid4().hex
        temp_image_path = image_path.with_name(f".rename_tmp_{token}{image_path.suffix}")

        temp_plan.append((image_path, temp_image_path))
        final_plan.append((temp_image_path, new_image_path))

        if json_path and new_json_path:
            temp_json_path = json_path.with_name(f".rename_tmp_{token}.json")
            temp_plan.append((json_path, temp_json_path))
            final_plan.append((temp_json_path, new_json_path))

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


def build_new_name(json_path: Path, current_name: object) -> str:
    suffix = ""
    if isinstance(current_name, str):
        suffix = Path(current_name).suffix
    return f"{json_path.stem}{suffix}"


def update_json_file(json_path: Path, target_dir: Path) -> None:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {json_path}")
    if not isinstance(data.get("info"), dict):
        raise ValueError(f'JSON must contain an "info" object: {json_path}')

    info = data["info"]
    info["folder"] = str(target_dir)
    info["name"] = build_new_name(json_path, info.get("name"))

    data.pop("folder", None)
    data.pop("name", None)

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)
        file.write("\n")


def update_json_files(json_paths: list[Path], target_dir: Path) -> int:
    count = 0
    for json_path in sorted(json_paths):
        update_json_file(json_path, target_dir)
        count += 1
    return count


def main(default_dir: Path, default_start: int) -> None:
    args = parse_args(default_dir, default_start)
    folder = args.dir

    if not folder.exists() or not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")

    images = find_images(folder)
    if not images:
        print(f"No image files found in: {folder}")
        return

    rename_plan = build_rename_plan(images, args.prefix, args.start, args.digits)
    ensure_no_external_conflicts(rename_plan)
    print_rename_plan(rename_plan)

    if args.dry_run:
        print("Dry run only. No files were renamed and no JSON content was modified.")
        return

    rename_files(rename_plan)
    renamed_json_paths = [new_json for _, _, _, new_json in rename_plan if new_json is not None]
    print(f"Renamed {len(rename_plan)} image files and {len(renamed_json_paths)} JSON files.")

    updated_count = update_json_files(renamed_json_paths, folder)
    print(f"Updated {updated_count} JSON files in: {folder}")


if __name__ == "__main__":
    DEFAULT_DIR = Path(r"E:\projects\datasets\Power_box\原图")
    DEFAULT_START = 1
    main(DEFAULT_DIR, DEFAULT_START)
