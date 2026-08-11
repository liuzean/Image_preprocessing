from __future__ import annotations
from fontTools.ufoLib.utils import F

import random
import time
from pathlib import Path

import numpy as np
from PIL import Image

if __package__:
    from . import Defect_generation as generation
    from . import Defect_white as whitening
else:
    import Defect_generation as generation
    import Defect_white as whitening


WHITENED_CATEGORY_KEYS = {"scratch", "pit"}


def whiten_generated_defects(
    image: np.ndarray,
    defect_masks: list[np.ndarray],
    category: str,
    base_whiten_strengths: dict[str, float],
    skeleton_gradient_strengths: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    category_key = generation.normalize_category(category)
    combined_skeleton = np.zeros(image.shape[:2], dtype=np.uint8)
    if category_key not in WHITENED_CATEGORY_KEYS:
        return image, combined_skeleton

    combined_alpha = np.zeros(image.shape[:2], dtype=np.float32)
    for defect_mask in defect_masks:
        alpha, skeleton_mask = whitening.object_whitening_alpha(
            defect_mask,
            base_whiten_strengths[category_key],
            skeleton_gradient_strengths[category_key],
        )
        combined_alpha = np.maximum(combined_alpha, alpha)
        combined_skeleton = np.maximum(combined_skeleton, skeleton_mask)

    alpha_3d = combined_alpha[..., None]
    whitened = image.astype(np.float32) * (1.0 - alpha_3d) + 255.0 * alpha_3d
    return np.clip(np.rint(whitened), 0, 255).astype(np.uint8), combined_skeleton


def process_image_for_category(
    json_path: Path,
    image_path: Path,
    category: str,
    bag: generation.DefectBag,
    output_dir: Path,
    rng: random.Random,
    max_attempts: int,
    silver_box_inset_pixels: int,
    base_whiten_strengths: dict[str, float],
    skeleton_gradient_strengths: dict[str, float],
    enable_pit_red_contour: bool,
    enable_scratch_red_contour: bool,
    enable_stain_red_contour: bool,
    enable_green_skeleton: bool,
    skeleton_visualization_width: int,
) -> list[generation.DefectPatch]:
    normal_data = generation.read_json(json_path)
    image = generation.read_rgb(image_path)
    height, width = image.shape[:2]

    silver_objects = generation.objects_by_category(
        normal_data,
        generation.SILVER_BOX_KEY,
    )
    silver_mask = generation.build_mask((width, height), silver_objects)
    if not np.any(silver_mask):
        raise ValueError(f"No valid Silver box mask found in {json_path}")
    inset_silver_mask = generation.inset_mask(
        silver_mask,
        silver_box_inset_pixels,
    )
    if not np.any(inset_silver_mask):
        raise ValueError(
            f"Silver box becomes empty after a {silver_box_inset_pixels}-pixel "
            f"inset in {json_path}"
        )

    avoided_mask = generation.build_mask(
        (width, height),
        generation.objects_by_category(normal_data, generation.AVOIDED_AREA_KEY),
    )
    output_mask = np.zeros((height, width), dtype=np.uint8)
    placed_masks: list[np.ndarray] = []

    selected_patches = bag.draw(generation.DEFECTS_PER_IMAGE, rng)
    for patch in selected_patches:
        placement = generation.find_placement(
            patch=patch,
            silver_mask=silver_mask,
            inset_silver_mask=inset_silver_mask,
            avoided_mask=avoided_mask,
            used_mask=output_mask,
            rng=rng,
            max_attempts=max_attempts,
        )
        generation.apply_placement(
            image=image,
            placement=placement,
            output_mask=output_mask,
        )
        placed_masks.append(placement.final_mask)

    avoided_overlap = int(np.count_nonzero((output_mask > 0) & (avoided_mask > 0)))
    if avoided_overlap:
        raise RuntimeError(
            f"Generated {category} mask for {image_path.name} overlaps "
            f"The_avoided_area by {avoided_overlap} pixel(s)."
        )

    output_image, skeleton_mask = whiten_generated_defects(
        image=image,
        defect_masks=placed_masks,
        category=category,
        base_whiten_strengths=base_whiten_strengths,
        skeleton_gradient_strengths=skeleton_gradient_strengths,
    )
    if enable_green_skeleton and generation.normalize_category(category) in WHITENED_CATEGORY_KEYS:
        output_image = whitening.draw_skeleton(
            output_image,
            skeleton_mask,
            skeleton_visualization_width,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(output_image).save(output_dir / image_path.name)
    Image.fromarray(output_mask).save(mask_dir / image_path.name)

    if generation.red_contour_enabled_for_category(
        category,
        enable_pit_red_contour,
        enable_scratch_red_contour,
        enable_stain_red_contour,
    ):
        red_contour_dir = output_dir / f"{category}_red_contour"
        red_contour_dir.mkdir(parents=True, exist_ok=True)
        annotated_image = generation.draw_expanded_defect_annotation(
            output_image,
            output_mask,
        )
        Image.fromarray(annotated_image).save(red_contour_dir / image_path.name)
    return selected_patches


def process_category(
    category: str,
    normal_dir: Path,
    bag: generation.DefectBag,
    rng: random.Random,
    max_attempts: int,
    silver_box_inset_pixels: int,
    base_whiten_strengths: dict[str, float],
    skeleton_gradient_strengths: dict[str, float],
    enable_pit_red_contour: bool,
    enable_scratch_red_contour: bool,
    enable_stain_red_contour: bool,
    enable_green_skeleton: bool,
    skeleton_visualization_width: int,
    enable_txt_log: bool,
) -> tuple[int, Path]:
    json_files = generation.normal_json_files(normal_dir)
    if not json_files:
        raise ValueError(f"No normal JSON files found in: {normal_dir}")

    run_dir = generation.next_run_dir(normal_dir / "results")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "mask").mkdir(parents=True, exist_ok=True)

    log_file = None
    if enable_txt_log:
        log_path = run_dir / f"{normal_dir.name}.txt"
        log_file = log_path.open("w", encoding="utf-8", newline="")
        generation.write_defect_log_header(log_file)

    processed = 0
    try:
        for json_path in json_files:
            image_path = generation.find_image_for_json(json_path)
            selected_patches = process_image_for_category(
                json_path=json_path,
                image_path=image_path,
                category=category,
                bag=bag,
                output_dir=run_dir,
                rng=rng,
                max_attempts=max_attempts,
                silver_box_inset_pixels=silver_box_inset_pixels,
                base_whiten_strengths=base_whiten_strengths,
                skeleton_gradient_strengths=skeleton_gradient_strengths,
                enable_pit_red_contour=enable_pit_red_contour,
                enable_scratch_red_contour=enable_scratch_red_contour,
                enable_stain_red_contour=enable_stain_red_contour,
                enable_green_skeleton=enable_green_skeleton,
                skeleton_visualization_width=skeleton_visualization_width,
            )
            if log_file is not None:
                generation.write_defect_log_rows(
                    log_file,
                    image_path.name,
                    selected_patches,
                )
                log_file.flush()
            processed += 1
    finally:
        if log_file is not None:
            log_file.close()

    return processed, run_dir


def process_all(
    normal_dir: Path,
    defective_dir: Path,
    seed: int | None,
    patch_padding: int,
    max_attempts: int,
    silver_box_inset_pixels: int,
    base_whiten_strengths: dict[str, float],
    skeleton_gradient_strengths: dict[str, float],
    enable_pit_red_contour: bool,
    enable_scratch_red_contour: bool,
    enable_stain_red_contour: bool,
    enable_green_skeleton: bool,
    skeleton_visualization_width: int,
    enable_txt_log: bool,
) -> list[tuple[str, str, int, Path]]:
    if not normal_dir.is_dir():
        raise NotADirectoryError(f"Normal folder does not exist: {normal_dir}")
    if not defective_dir.is_dir():
        raise NotADirectoryError(f"Defective folder does not exist: {defective_dir}")
    whitening.validate_category_strengths(
        "base_whiten_strengths",
        base_whiten_strengths,
    )
    whitening.validate_category_strengths(
        "skeleton_gradient_strengths",
        skeleton_gradient_strengths,
    )
    if skeleton_visualization_width < 1:
        raise ValueError("skeleton_visualization_width must be at least 1")

    rng = random.Random(seed)
    normal_folders = generation.normal_category_folders(normal_dir)
    patches_by_category = generation.extract_defect_patches(
        defective_dir=defective_dir,
        padding=patch_padding,
    )
    bags = {
        category: generation.DefectBag.from_patches(patches_by_category[category])
        for category in generation.DEFECT_CATEGORIES
    }

    results: list[tuple[str, str, int, Path]] = []
    for normal_folder in normal_folders:
        processed, output_dir = process_category(
            category=normal_folder.category,
            normal_dir=normal_folder.path,
            bag=bags[normal_folder.category],
            rng=rng,
            max_attempts=max_attempts,
            silver_box_inset_pixels=silver_box_inset_pixels,
            base_whiten_strengths=base_whiten_strengths,
            skeleton_gradient_strengths=skeleton_gradient_strengths,
            enable_pit_red_contour=enable_pit_red_contour,
            enable_scratch_red_contour=enable_scratch_red_contour,
            enable_stain_red_contour=enable_stain_red_contour,
            enable_green_skeleton=enable_green_skeleton,
            skeleton_visualization_width=skeleton_visualization_width,
            enable_txt_log=enable_txt_log,
        )
        results.append(
            (
                normal_folder.group_name,
                normal_folder.category,
                processed,
                output_dir,
            )
        )
    return results


def main(
    normal_dir: Path,
    defective_dir: Path,
    silver_box_inset_pixels: int,
    base_whiten_strengths: dict[str, float],
    skeleton_gradient_strengths: dict[str, float],
    enable_pit_red_contour: bool,
    enable_scratch_red_contour: bool,
    enable_stain_red_contour: bool,
    enable_green_skeleton: bool,
    skeleton_visualization_width: int,
    enable_txt_log: bool,
) -> None:
    args = generation.parse_args(normal_dir, defective_dir)
    start_time = time.perf_counter()
    results = process_all(
        normal_dir=args.normal_dir,
        defective_dir=args.defective_dir,
        seed=args.seed,
        patch_padding=args.patch_padding,
        max_attempts=args.max_placement_attempts,
        silver_box_inset_pixels=silver_box_inset_pixels,
        base_whiten_strengths=base_whiten_strengths,
        skeleton_gradient_strengths=skeleton_gradient_strengths,
        enable_pit_red_contour=enable_pit_red_contour,
        enable_scratch_red_contour=enable_scratch_red_contour,
        enable_stain_red_contour=enable_stain_red_contour,
        enable_green_skeleton=enable_green_skeleton,
        skeleton_visualization_width=skeleton_visualization_width,
        enable_txt_log=args.enable_txt_log and enable_txt_log,
    )
    elapsed_seconds = time.perf_counter() - start_time
    for group_name, category, count, output_dir in results:
        print(f"{group_name}/{category}: generated {count} image(s) in {output_dir}")
    print(f"Total runtime: {elapsed_seconds:.2f} s")


if __name__ == "__main__":
    NORMAL_DIR = Path(
        r"E:\projects\datasets\Power_box\old\results\results\Defective_free\11"
    )
    DEFECTIVE_DIR = Path(
        r"E:\projects\datasets\Power_box\old\results\results"
    )
    SILVER_BOX_INSET_PIXELS = 10

    SCRATCH_BASE_WHITEN_STRENGTH = 0.05
    PIT_BASE_WHITEN_STRENGTH = 0.05
    SCRATCH_SKELETON_GRADIENT_STRENGTH = 0.4
    PIT_SKELETON_GRADIENT_STRENGTH = 0.3
    BASE_WHITEN_STRENGTHS = {
        "scratch": SCRATCH_BASE_WHITEN_STRENGTH,
        "pit": PIT_BASE_WHITEN_STRENGTH,
    }
    SKELETON_GRADIENT_STRENGTHS = {
        "scratch": SCRATCH_SKELETON_GRADIENT_STRENGTH,
        "pit": PIT_SKELETON_GRADIENT_STRENGTH,
    }

    ENABLE_PIT_RED_CONTOUR = True
    ENABLE_SCRATCH_RED_CONTOUR = True
    ENABLE_STAIN_RED_CONTOUR = True
    ENABLE_GREEN_SKELETON = False
    SKELETON_VISUALIZATION_WIDTH = 1
    ENABLE_TXT_LOG = True

    main(
        normal_dir=NORMAL_DIR,
        defective_dir=DEFECTIVE_DIR,
        silver_box_inset_pixels=SILVER_BOX_INSET_PIXELS,
        base_whiten_strengths=BASE_WHITEN_STRENGTHS,
        skeleton_gradient_strengths=SKELETON_GRADIENT_STRENGTHS,
        enable_pit_red_contour=ENABLE_PIT_RED_CONTOUR,
        enable_scratch_red_contour=ENABLE_SCRATCH_RED_CONTOUR,
        enable_stain_red_contour=ENABLE_STAIN_RED_CONTOUR,
        enable_green_skeleton=ENABLE_GREEN_SKELETON,
        skeleton_visualization_width=SKELETON_VISUALIZATION_WIDTH,
        enable_txt_log=ENABLE_TXT_LOG,
    )
