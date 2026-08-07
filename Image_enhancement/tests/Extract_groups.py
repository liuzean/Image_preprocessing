from __future__ import annotations

from dataclasses import dataclass
import argparse
import random
import shutil
from pathlib import Path


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
DEFECT_GROUPS = ("Pit", "Scratch", "Stain")
PAIRS_PER_GROUP = 20
DEFAULT_RANDOM_SEED = 20260806
IMAGE_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


@dataclass(frozen=True)
class ImageJsonPair:
    image_path: Path
    json_path: Path


def collect_pairs(folder_path: Path) -> list[ImageJsonPair]:
    image_paths = sorted(
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    missing_json = [
        image_path
        for image_path in image_paths
        if not image_path.with_suffix(".json").exists()
    ]
    if missing_json:
        examples = ", ".join(path.name for path in missing_json[:5])
        raise FileNotFoundError(
            f"{len(missing_json)} image(s) have no same-name JSON in "
            f"{folder_path}: {examples}"
        )
    return [
        ImageJsonPair(
            image_path=image_path,
            json_path=image_path.with_suffix(".json"),
        )
        for image_path in image_paths
    ]


def select_groups(
    pairs: list[ImageJsonPair],
    rng: random.Random,
) -> dict[str, list[ImageJsonPair]]:
    required_count = len(DEFECT_GROUPS) * PAIRS_PER_GROUP
    if len(pairs) < required_count:
        raise ValueError(
            f"Need at least {required_count} image/JSON pairs, got {len(pairs)}"
        )
    selected = rng.sample(pairs, required_count)
    return {
        group_name: selected[
            index * PAIRS_PER_GROUP : (index + 1) * PAIRS_PER_GROUP
        ]
        for index, group_name in enumerate(DEFECT_GROUPS)
    }


def reset_group_folder(group_path: Path) -> None:
    group_path.mkdir(parents=False, exist_ok=True)
    for path in group_path.iterdir():
        if path.is_file() or path.is_symlink():
            path.unlink()


def write_groups(
    folder_path: Path,
    groups: dict[str, list[ImageJsonPair]],
) -> None:
    for group_name in DEFECT_GROUPS:
        reset_group_folder(folder_path / group_name)

    for group_name, pairs in groups.items():
        group_path = folder_path / group_name
        for pair in pairs:
            shutil.copy2(pair.image_path, group_path / pair.image_path.name)
            shutil.copy2(pair.json_path, group_path / pair.json_path.name)

        copied_images = [
            path
            for path in group_path.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        copied_json = list(group_path.glob("*.json"))
        if (
            len(copied_images) != PAIRS_PER_GROUP
            or len(copied_json) != PAIRS_PER_GROUP
        ):
            raise RuntimeError(
                f"Incorrect copied pair count in {group_path}: "
                f"images={len(copied_images)}, json={len(copied_json)}"
            )


def process_folders(
    root_dir: Path,
    random_seed: int,
    write_files: bool,
) -> tuple[int, int]:
    if not root_dir.is_dir():
        raise NotADirectoryError(f"Root directory does not exist: {root_dir}")

    folder_paths = [root_dir / name for name in TARGET_FOLDERS]
    missing_folders = [path for path in folder_paths if not path.is_dir()]
    if missing_folders:
        missing = ", ".join(str(path) for path in missing_folders)
        raise FileNotFoundError(f"Missing target folder(s): {missing}")

    rng = random.Random(random_seed)
    plans: dict[Path, dict[str, list[ImageJsonPair]]] = {}
    for folder_path in folder_paths:
        pairs = collect_pairs(folder_path)
        plans[folder_path] = select_groups(pairs, rng)

    total_selected = 0
    for folder_path, groups in plans.items():
        if write_files:
            write_groups(folder_path, groups)
        selected_count = sum(len(pairs) for pairs in groups.values())
        total_selected += selected_count
        action = "copied" if write_files else "planned"
        print(
            f"{folder_path.name}: {selected_count} pair(s) {action}; "
            + ", ".join(
                f"{group_name}={len(groups[group_name])}"
                for group_name in DEFECT_GROUPS
            )
        )
    return len(folder_paths), total_selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Randomly copy 60 image/JSON pairs from each Power_box folder "
            "into Pit, Scratch, and Stain groups of 20 pairs each."
        )
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=ROOT_DIR,
        help=f"Folder containing the six Power_box folders. Default: {ROOT_DIR}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help=f"Reproducible random seed. Default: {DEFAULT_RANDOM_SEED}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and plan selections without creating or clearing folders.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    folder_count, selected_count = process_folders(
        args.root_dir,
        random_seed=args.seed,
        write_files=not args.dry_run,
    )
    action = "Planned" if args.dry_run else "Copied"
    print(
        f"{action} {selected_count} image/JSON pair selections across "
        f"{folder_count} folder(s)."
    )


if __name__ == "__main__":
    main()
