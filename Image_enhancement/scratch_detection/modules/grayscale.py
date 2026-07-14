from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

#统一后续算法的数据格式并减少计算量。

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Image_enhancement.scratch_detection.io.dataset import (  # noqa: E402
    DEFAULT_DATASET_DIR,
    read_image,
    scan_image_json_pairs,
)
from Image_enhancement.scratch_detection.io.result_writer import (  # noqa: E402
    ResultWriter,
    ResultWriterConfig,
)


@dataclass(frozen=True)
class GrayscaleResult:
    image: np.ndarray


def convert_to_grayscale_float32(image: np.ndarray) -> GrayscaleResult:
    """Convert an image to one-channel float32 without changing its scale."""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")

    if image.ndim == 2:
        grayscale = image
    elif image.ndim == 3 and image.shape[2] == 1:
        grayscale = image[:, :, 0]
    elif image.ndim == 3 and image.shape[2] == 3:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(
            "image must be grayscale, BGR, or BGRA with two or three dimensions"
        )

    return GrayscaleResult(image=grayscale.astype(np.float32, copy=True))


def create_preview_image(grayscale_float32: np.ndarray) -> np.ndarray:
    """Create a BGR uint8 image only for standalone result visualization."""
    if grayscale_float32.ndim != 2:
        raise ValueError("grayscale_float32 must be a one-channel image")

    grayscale_uint8 = np.clip(grayscale_float32, 0, 255).astype(np.uint8)
    return cv2.cvtColor(grayscale_uint8, cv2.COLOR_GRAY2BGR)


def process_dataset(
    dataset_dir: Path,
    writer_config: ResultWriterConfig,
) -> tuple[int, list[Path], Path]:
    scan_result = scan_image_json_pairs(dataset_dir)
    writer = ResultWriter(dataset_dir, writer_config)

    processed_count = 0
    for pair in scan_result.pairs:
        original_image = read_image(pair.image_path)
        grayscale_result = convert_to_grayscale_float32(original_image)
        preview_image = create_preview_image(grayscale_result.image)
        writer.save_result(pair.image_path.stem, original_image, preview_image)
        processed_count += 1

    return processed_count, scan_result.images_without_json, writer.output_dir


def main() -> None:
    dataset_dir = DEFAULT_DATASET_DIR
    writer_config = ResultWriterConfig(
        output_folder_name="grayscale_results",
        run_number_width=3,
        max_run_number=999,
        save_comparison=True,
        save_processed=False,
        comparison_suffix="_grayscale_comparison.png",
    )

    processed_count, images_without_json, output_dir = process_dataset(
        dataset_dir,
        writer_config,
    )
    print(f"Processed {processed_count} images. Results saved to: {output_dir}")

    if images_without_json:
        print("Skipped images without a same-name JSON file:")
        for image_path in images_without_json:
            print(f"  {image_path.name}")


if __name__ == "__main__":
    main()
