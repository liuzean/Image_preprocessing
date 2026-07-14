from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Image_enhancement.scratch_detection.modules.erode_mask import (
    ErodeMaskConfig,
    process_image,
)


@dataclass(frozen=True)
class ScratchPipelineResult:
    working_image: np.ndarray
    original_mask: np.ndarray
    processing_mask: np.ndarray


class ScratchDetectionPipeline:
    """Run the configured scratch-detection stages for one image."""

    def __init__(self, erode_mask_config: ErodeMaskConfig) -> None:
        self.erode_mask_config = erode_mask_config

    def run(
        self,
        image: np.ndarray,
        annotation: dict,
    ) -> ScratchPipelineResult:
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a BGR array with shape (height, width, 3)")

        mask_result = process_image(
            image,
            annotation,
            self.erode_mask_config,
        )
        return ScratchPipelineResult(
            working_image=image.copy(),
            original_mask=mask_result.original_mask,
            processing_mask=mask_result.eroded_mask,
        )
