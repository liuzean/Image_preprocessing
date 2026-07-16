from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Image_enhancement.scratch_detection.modules.background import (
    BackgroundCorrectionConfig,
    subtract_mask_aware_gaussian_background,
)
from Image_enhancement.scratch_detection.modules.erode_mask import (
    ErodeMaskConfig,
    process_image as process_mask,
)
from Image_enhancement.scratch_detection.modules.grayscale import (
    convert_to_grayscale_float32,
)
from Image_enhancement.scratch_detection.modules.gabor import (
    MultiDirectionGaborConfig,
    enhance_multi_direction_gabor,
)
from Image_enhancement.scratch_detection.modules.frangi import (
    FrangiConfig,
    enhance_frangi,
)


LINE_ENHANCEMENT_METHODS = {"frangi", "gabor"}
FRANGI_RESPONSE_MODES = {"bright", "dark", "combined"}


@dataclass(frozen=True)
class ScratchPipelineResult:
    working_image: np.ndarray
    line_enhancement_method: str
    grayscale_image: np.ndarray
    background_image: np.ndarray
    original_mask: np.ndarray
    processing_mask: np.ndarray
    gabor_response_image: np.ndarray | None
    gabor_signed_response_image: np.ndarray | None
    gabor_direction_map_degrees: np.ndarray | None
    frangi_response_image: np.ndarray | None
    frangi_bright_response_image: np.ndarray | None
    frangi_dark_response_image: np.ndarray | None


class ScratchDetectionPipeline:
    """Run the configured scratch-detection stages for one image."""

    def __init__(
        self,
        erode_mask_config: ErodeMaskConfig,
        background_config: BackgroundCorrectionConfig,
        gabor_config: MultiDirectionGaborConfig,
        frangi_config: FrangiConfig,
        line_enhancement_method: str = "frangi",
        frangi_response_mode: str = "bright",
    ) -> None:
        if line_enhancement_method not in LINE_ENHANCEMENT_METHODS:
            supported = ", ".join(sorted(LINE_ENHANCEMENT_METHODS))
            raise ValueError(
                f"line_enhancement_method must be one of: {supported}"
            )
        if frangi_response_mode not in FRANGI_RESPONSE_MODES:
            supported = ", ".join(sorted(FRANGI_RESPONSE_MODES))
            raise ValueError(f"frangi_response_mode must be one of: {supported}")

        self.erode_mask_config = erode_mask_config
        self.background_config = background_config
        self.gabor_config = gabor_config
        self.frangi_config = frangi_config
        self.line_enhancement_method = line_enhancement_method
        self.frangi_response_mode = frangi_response_mode

    def run(
        self,
        image: np.ndarray,
        annotation: dict,
    ) -> ScratchPipelineResult:
        if (
            not isinstance(image, np.ndarray)
            or image.ndim != 3
            or image.shape[2] != 3
        ):
            raise ValueError("image must be a BGR array with shape (height, width, 3)")

        mask_result = process_mask(
            image,
            annotation,
            self.erode_mask_config,
        )
        grayscale_result = convert_to_grayscale_float32(image)
        background_result = subtract_mask_aware_gaussian_background(
            grayscale_result.image,
            mask_result.original_mask,
            self.background_config,
        )
        gabor_result = None
        frangi_result = None
        if self.line_enhancement_method == "gabor":
            gabor_result = enhance_multi_direction_gabor(
                background_result.corrected_image,
                mask_result.eroded_mask,
                self.gabor_config,
            )
            working_image = gabor_result.response_image
        else:
            frangi_result = enhance_frangi(
                background_result.corrected_image,
                mask_result.eroded_mask,
                self.frangi_config,
            )
            if not self.frangi_config.enabled:
                working_image = frangi_result.response_image
            elif self.frangi_response_mode == "bright":
                working_image = frangi_result.bright_response_image
            elif self.frangi_response_mode == "dark":
                working_image = frangi_result.dark_response_image
            else:
                working_image = frangi_result.response_image

        return ScratchPipelineResult(
            working_image=working_image,
            line_enhancement_method=self.line_enhancement_method,
            grayscale_image=grayscale_result.image,
            background_image=background_result.background_image,
            original_mask=mask_result.original_mask,
            processing_mask=mask_result.eroded_mask,
            gabor_response_image=(
                None if gabor_result is None else gabor_result.response_image
            ),
            gabor_signed_response_image=(
                None if gabor_result is None else gabor_result.signed_response_image
            ),
            gabor_direction_map_degrees=(
                None if gabor_result is None else gabor_result.direction_map_degrees
            ),
            frangi_response_image=(
                None if frangi_result is None else frangi_result.response_image
            ),
            frangi_bright_response_image=(
                None if frangi_result is None else frangi_result.bright_response_image
            ),
            frangi_dark_response_image=(
                None if frangi_result is None else frangi_result.dark_response_image
            ),
        )
