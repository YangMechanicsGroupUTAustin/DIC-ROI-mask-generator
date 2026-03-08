"""SAM2 video predictor lifecycle wrapper.

Provides a clean API over SAM2's video prediction workflow.
Manages model loading, inference state, and GPU memory.
Architecture: single-object now, obj_id parameter for future multi-object.
"""

import os
import logging
from typing import Optional, Callable

import numpy as np
import torch

logger = logging.getLogger("sam2studio.mask_generator")


class MaskGenerator:
    """Manages SAM2 video predictor lifecycle and inference."""

    def __init__(self, base_dir: str = ""):
        self._predictor = None
        self._inference_state = None
        self._config_key: Optional[tuple] = None
        self._base_dir = base_dir or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def initialize(
        self,
        model_cfg: str,
        checkpoint: str,
        device: str = "cuda",
        progress_callback: Optional[Callable[[str], None]] = None,
    ) -> None:
        """Load SAM2 model. Reuses existing if config matches."""
        checkpoint_path = os.path.join(self._base_dir, "checkpoints", checkpoint)
        config_path = os.path.join("sam2", "configs", "sam2.1", model_cfg)
        new_key = (checkpoint_path, config_path, device)

        if self._predictor is not None and self._config_key == new_key:
            logger.info("Reusing existing predictor (config unchanged)")
            return

        # Cleanup old predictor
        self.cleanup()

        if progress_callback:
            progress_callback("Loading SAM2 model...")

        logger.info(f"Initializing SAM2: model={model_cfg}, device={device}")

        from sam2.build_sam import build_sam2_video_predictor
        self._predictor = build_sam2_video_predictor(
            config_path, checkpoint_path, device=device
        )
        self._config_key = new_key
        logger.info("SAM2 model loaded successfully")

    def set_video(self, image_dir: str) -> None:
        """Initialize inference state from image directory."""
        if self._predictor is None:
            raise RuntimeError("Predictor not initialized. Call initialize() first.")

        logger.info(f"Setting video from: {image_dir}")
        self._inference_state = self._predictor.init_state(video_path=image_dir)

    def add_points(
        self,
        frame_idx: int,
        points: np.ndarray,
        labels: np.ndarray,
        obj_id: int = 1,
    ) -> Optional[np.ndarray]:
        """Add annotation points at a specific frame.

        Args:
            frame_idx: 0-based frame index.
            points: Array of shape (N, 2) with (x, y) coordinates.
            labels: Array of shape (N,) with 1=foreground, 0=background.
            obj_id: Object ID (default 1, for future multi-object).

        Returns:
            Preview mask logits for the frame, or None on error.
        """
        if self._predictor is None or self._inference_state is None:
            raise RuntimeError("Predictor or inference state not initialized.")

        _, out_obj_ids, out_mask_logits = self._predictor.add_new_points_or_box(
            inference_state=self._inference_state,
            frame_idx=frame_idx,
            obj_id=obj_id,
            points=points,
            labels=labels,
        )
        return out_mask_logits

    def propagate(
        self,
        threshold: float = 0.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[int, np.ndarray]:
        """Run mask propagation through all frames.

        Args:
            threshold: Mask logit threshold.
            progress_callback: Called with (frame_idx, total_frames).

        Returns:
            Dict mapping frame_idx to binary mask (uint8, 0 or 255).
        """
        if self._predictor is None or self._inference_state is None:
            raise RuntimeError("Predictor or inference state not initialized.")

        video_segments: dict[int, np.ndarray] = {}
        frame_count = 0

        for out_frame_idx, out_obj_ids, out_mask_logits in self._predictor.propagate_in_video(
            self._inference_state
        ):
            mask = (out_mask_logits[0] > threshold).cpu().numpy().squeeze()
            binary_mask = mask.astype(np.uint8) * 255
            video_segments[out_frame_idx] = binary_mask
            frame_count += 1

            if progress_callback:
                progress_callback(frame_count, -1)  # total unknown during propagation

        logger.info(f"Propagation complete: {frame_count} frames")
        return video_segments

    def add_correction(
        self,
        frame_idx: int,
        points: np.ndarray,
        labels: np.ndarray,
        obj_id: int = 1,
    ) -> None:
        """Add correction points at a specific frame for re-propagation."""
        self.add_points(frame_idx, points, labels, obj_id)

    def propagate_from(
        self,
        start_frame_idx: int,
        threshold: float = 0.0,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> dict[int, np.ndarray]:
        """Re-propagate from a specific frame onward.

        Uses the same propagate_in_video but results are filtered
        to only include frames >= start_frame_idx.
        """
        all_segments = self.propagate(threshold, progress_callback)
        return {k: v for k, v in all_segments.items() if k >= start_frame_idx}

    def cleanup(self) -> None:
        """Release model and clear GPU cache."""
        if self._predictor is not None and self._inference_state is not None:
            try:
                self._predictor.reset_state(self._inference_state)
            except Exception as e:
                logger.warning(f"Error resetting state: {e}")

        self._inference_state = None
        self._predictor = None
        self._config_key = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

    @property
    def is_initialized(self) -> bool:
        return self._predictor is not None

    @property
    def has_inference_state(self) -> bool:
        return self._inference_state is not None
