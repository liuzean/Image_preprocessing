from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps


ROOT_DIR = Path(r"E:\projects\datasets\Power_box\old\frangi_results")
LEFT_DIR_NAME = "013"
RIGHT_DIR_NAME = "018"
OUTPUT_DIR_NAME = "Assemble_frangi"
IMAGE_CATEGORIES = (
    "frangi_bright",
    "frangi_dark",
    "frangi_response",
)
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


def get_image_category(file_name: str) -> str | None:
    """Return the Frangi category found in a file name."""
    lower_name = file_name.lower()
    for category in IMAGE_CATEGORIES:
        if category in lower_name:
            return category
    return None


def find_target_images(folder: Path) -> dict[str, tuple[Path, str]]:
    """Return target images indexed by their exact file name."""
    if not folder.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {folder}")

    images: dict[str, tuple[Path, str]] = {}
    for path in folder.iterdir():
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        category = get_image_category(path.name)
        if category is not None:
            images[path.name] = (path, category)

    return images


def assemble_pair(left_path: Path, right_path: Path, output_path: Path) -> None:
    """Join two images horizontally without resizing them."""
    with Image.open(left_path) as left_source, Image.open(right_path) as right_source:
        left = ImageOps.exif_transpose(left_source).convert("RGB")
        right = ImageOps.exif_transpose(right_source).convert("RGB")

        result = Image.new(
            "RGB",
            (left.width + right.width, max(left.height, right.height)),
            color=(0, 0, 0),
        )
        result.paste(left, (0, 0))
        result.paste(right, (left.width, 0))
        result.save(output_path)


def create_next_output_dir(root_dir: Path) -> Path:
    """Create the next numbered run directory and all required subdirectories."""
    output_root = root_dir / OUTPUT_DIR_NAME
    output_root.mkdir(parents=True, exist_ok=True)

    numeric_indices = [
        int(path.name)
        for path in output_root.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    next_index = max(numeric_indices, default=0) + 1
    if next_index > 999:
        raise RuntimeError(
            f"No output index is available: {output_root} has reached 999"
        )

    output_dir = output_root / f"{next_index:03d}"
    output_dir.mkdir(parents=False, exist_ok=False)

    for category in IMAGE_CATEGORIES:
        (output_dir / category).mkdir(parents=False, exist_ok=False)

    source_pair_dir = output_dir / f"{LEFT_DIR_NAME}_{RIGHT_DIR_NAME}"
    source_pair_dir.mkdir(parents=False, exist_ok=False)
    return output_dir


def assemble_matching_images(
    root_dir: Path = ROOT_DIR,
) -> tuple[dict[str, int], list[str], list[str], Path]:
    """Match, assemble, and route Frangi images into category directories."""
    left_images = find_target_images(root_dir / LEFT_DIR_NAME)
    right_images = find_target_images(root_dir / RIGHT_DIR_NAME)
    output_dir = create_next_output_dir(root_dir)

    category_counts = {category: 0 for category in IMAGE_CATEGORIES}
    matching_names = sorted(left_images.keys() & right_images.keys())
    for name in matching_names:
        left_path, category = left_images[name]
        right_path, right_category = right_images[name]
        if category != right_category:
            raise ValueError(
                f"Category mismatch for same-name image {name}: "
                f"{category} != {right_category}"
            )

        assemble_pair(left_path, right_path, output_dir / category / name)
        category_counts[category] += 1

    only_in_left = sorted(left_images.keys() - right_images.keys())
    only_in_right = sorted(right_images.keys() - left_images.keys())
    return category_counts, only_in_left, only_in_right, output_dir


def main() -> None:
    category_counts, only_in_left, only_in_right, output_dir = (
        assemble_matching_images()
    )
    total_count = sum(category_counts.values())
    print(f"Assembled {total_count} image pair(s). Results saved to: {output_dir}")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")

    if only_in_left:
        print(f"Skipped {len(only_in_left)} image(s) found only in {LEFT_DIR_NAME}:")
        for name in only_in_left:
            print(f"  {name}")

    if only_in_right:
        print(f"Skipped {len(only_in_right)} image(s) found only in {RIGHT_DIR_NAME}:")
        for name in only_in_right:
            print(f"  {name}")


if __name__ == "__main__":
    main()
