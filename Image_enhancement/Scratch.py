from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


IMAGE_EXTENSIONS = (
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
)


def build_output_stem(image_stem: str, revised_content: str = "") -> str:
    if not revised_content:
        return image_stem
    return f"{image_stem}_{revised_content}"


def create_next_result_dir(target_dir: Path) -> Path:
    results_dir = target_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    numeric_names = [
        int(path.name)
        for path in results_dir.iterdir()
        if path.is_dir() and path.name.isdigit()
    ]
    next_index = max(numeric_names, default=0) + 1
    output_dir = results_dir / f"{next_index:02d}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def read_image(image_path: Path) -> np.ndarray:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")
    return image


def write_image(image_path: Path, image: np.ndarray) -> None:
    image_path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(image_path.suffix, image)
    if not success:
        raise ValueError(f"Failed to encode image: {image_path}")
    encoded.tofile(str(image_path))


def find_image_for_json(json_path: Path) -> Path:
    for suffix in IMAGE_EXTENSIONS:
        image_path = json_path.with_suffix(suffix)
        if image_path.exists():
            return image_path
    raise FileNotFoundError(f"No same-name image found for JSON file: {json_path}")


def load_segmentations(json_path: Path) -> list[np.ndarray]:
    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    objects = data.get("objects")
    if not isinstance(objects, list):
        raise ValueError(f'JSON must contain an "objects" list: {json_path}')

    segmentations: list[np.ndarray] = []
    for item in objects:
        if not isinstance(item, dict):
            continue

        segmentation = item.get("segmentation")
        if not isinstance(segmentation, list) or len(segmentation) < 3:
            continue

        points: list[tuple[float, float]] = []
        for point in segmentation:
            if isinstance(point, list) and len(point) >= 2:
                points.append((float(point[0]), float(point[1])))

        if len(points) >= 3:
            segmentations.append(np.rint(points).astype(np.int32))

    if not segmentations:
        raise ValueError(f"No valid segmentation found in: {json_path}")

    return segmentations


def build_mask(image_shape: tuple[int, ...], segmentations: list[np.ndarray]) -> np.ndarray:
    height, width = image_shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    clipped_segmentations = []
    for points in segmentations:
        clipped = points.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
        clipped_segmentations.append(clipped)

    cv2.fillPoly(mask, clipped_segmentations, 255)
    return mask


def erode_mask(mask: np.ndarray, kernel_size: int = 31, iterations: int = 1) -> np.ndarray:
    kernel_size = ensure_odd_kernel_size(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.erode(mask, kernel, iterations=iterations)


def ensure_odd_kernel_size(kernel_size: int) -> int:
    if kernel_size < 3:
        return 3
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


class EnhancementModule(Protocol):
    name: str

    def apply(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        ...


@dataclass
class LargeScaleGaussianBackgroundSubtraction:
    """Enhance local defects by subtracting a large-scale Gaussian background."""

    gaussian_kernel_size: int = 151
    sigma: float = 0.0
    offset: float = 128.0
    name: str = "large_gaussian_background_subtraction"

    def apply(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        kernel_size = ensure_odd_kernel_size(self.gaussian_kernel_size)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        background = cv2.GaussianBlur(gray, (kernel_size, kernel_size), self.sigma)
        enhanced = np.clip(gray - background + self.offset, 0, 255).astype(np.uint8)

        result = image_bgr.copy()
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        result[mask > 0] = enhanced_bgr[mask > 0]
        return result


@dataclass
class MultiDirectionGaborLineEnhancement:
    """Enhance line-like scratches by taking the maximum response across Gabor directions."""

    angles_degrees: list[float] = field(
        default_factory=lambda: [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]
    )
    kernel_size: int = 31
    sigma: float = 4.0
    wavelength: float = 10.0
    gamma: float = 0.5
    psi: float = 0.0
    name: str = "multi_direction_gabor_line_enhancement"

    def apply(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        kernel_size = ensure_odd_kernel_size(self.kernel_size)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        max_response = np.zeros_like(gray, dtype=np.float32)

        for angle in self.angles_degrees:
            theta = np.deg2rad(angle)
            kernel = cv2.getGaborKernel(
                (kernel_size, kernel_size),
                self.sigma,
                theta,
                self.wavelength,
                self.gamma,
                self.psi,
                ktype=cv2.CV_32F,
            )
            kernel -= kernel.mean()
            response = cv2.filter2D(gray, cv2.CV_32F, kernel)
            max_response = np.maximum(max_response, np.abs(response))

        enhanced = np.zeros_like(gray, dtype=np.uint8)
        mask_pixels = mask > 0
        if np.any(mask_pixels):
            masked_response = max_response[mask_pixels]
            min_value = float(masked_response.min())
            max_value = float(masked_response.max())
            if max_value > min_value:
                normalized = (max_response - min_value) * 255.0 / (max_value - min_value)
                enhanced = np.clip(normalized, 0, 255).astype(np.uint8)

        result = image_bgr.copy()
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        result[mask_pixels] = enhanced_bgr[mask_pixels]
        return result


@dataclass
class ScratchEnhancementPipeline:
    modules: list[EnhancementModule]
    mask_erode_kernel_size: int = 31
    mask_erode_iterations: int = 1

    def process(self, image_bgr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        processing_mask = erode_mask(
            mask,
            kernel_size=self.mask_erode_kernel_size,
            iterations=self.mask_erode_iterations,
        )

        result = image_bgr.copy()
        for module in self.modules:
            result = module.apply(result, processing_mask)

        return result, processing_mask


def process_image(
    json_path: Path,
    output_dir: Path,
    pipeline: ScratchEnhancementPipeline,
    revised_content: str = "",
) -> None:
    image_path = find_image_for_json(json_path)
    image = read_image(image_path)
    segmentations = load_segmentations(json_path)
    mask = build_mask(image.shape, segmentations)
    result, _ = pipeline.process(image, mask)
    comparison = np.hstack((image, result))

    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = build_output_stem(image_path.stem, revised_content)
    write_image(output_dir / f"{output_stem}_comparison.png", comparison)
    write_image(output_dir / f"{output_stem}_enhanced.png", result)


def process_folder(
    target_dir: Path,
    pipeline: ScratchEnhancementPipeline,
    revised_content: str = "",
) -> tuple[int, Path]:
    if not target_dir.exists() or not target_dir.is_dir():
        raise NotADirectoryError(f"Folder does not exist: {target_dir}")

    output_dir = create_next_result_dir(target_dir)
    count = 0
    for json_path in sorted(target_dir.glob("*.json")):
        process_image(json_path, output_dir, pipeline, revised_content)
        count += 1

    return count, output_dir


def main(target_dir: Path, revised_content: str = "") -> None:
    pipeline = ScratchEnhancementPipeline(
        modules=[
            LargeScaleGaussianBackgroundSubtraction(
                gaussian_kernel_size=151,
                sigma=0,
                offset=128,
            ),
            MultiDirectionGaborLineEnhancement(
                angles_degrees=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165],
                kernel_size=21,
                sigma=3.0,
                wavelength=6.0,
                gamma=0.5,
                psi=0.0,
            )
        ],
        mask_erode_kernel_size=31,
        mask_erode_iterations=1,
    )
    count, output_dir = process_folder(target_dir, pipeline, revised_content)
    print(f"Processed {count} images. Results saved to: {output_dir}")


if __name__ == "__main__":
    target_dir = Path(r"E:\projects\datasets\Power_box\Power_box_3long")
    Revised_content = "background_gabor"
    main(target_dir, Revised_content)
