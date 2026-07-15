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


@dataclass(frozen=True)
class ScratchPipelineResult:
    working_image: np.ndarray
    grayscale_image: np.ndarray
    background_image: np.ndarray
    gabor_response_image: np.ndarray
    gabor_signed_response_image: np.ndarray
    gabor_direction_map_degrees: np.ndarray
    original_mask: np.ndarray
    processing_mask: np.ndarray


class ScratchDetectionPipeline:
    """Run the configured scratch-detection stages for one image."""

    def __init__(
        self,
        erode_mask_config: ErodeMaskConfig,
        background_config: BackgroundCorrectionConfig,
        gabor_config: MultiDirectionGaborConfig,
    ) -> None:
        self.erode_mask_config = erode_mask_config
        self.background_config = background_config
        self.gabor_config = gabor_config

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
        gabor_result = enhance_multi_direction_gabor(
            background_result.corrected_image,
            mask_result.eroded_mask,
            self.gabor_config,
        )
        return ScratchPipelineResult(
            working_image=gabor_result.response_image,
            grayscale_image=grayscale_result.image,
            background_image=background_result.background_image,
            gabor_response_image=gabor_result.response_image,
            gabor_signed_response_image=gabor_result.signed_response_image,
            gabor_direction_map_degrees=gabor_result.direction_map_degrees,
            original_mask=mask_result.original_mask,
            processing_mask=mask_result.eroded_mask,
        )
