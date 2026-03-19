"""Preprocessing preview controller.

Caches the current frame at full resolution, applies the preprocessing
pipeline, and pushes the result into AppState for display.  This keeps
MainWindow free of preview bookkeeping logic.
"""

import logging
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject

from core.image_processing import load_image_as_rgb
from core.preprocessing import PreprocessingConfig, apply_pipeline

logger = logging.getLogger("sam2studio.preview_controller")


class PreviewController(QObject):
    """Manages preprocessing preview state and rendering.

    Owns the cached full-resolution frame and coordinates with AppState
    to push display images.
    """

    def __init__(self, state, shape_controller, parent: QObject | None = None):
        super().__init__(parent)
        self._state = state
        self._shape_ctrl = shape_controller

        # Full-resolution BGR cache of the current frame
        self._cached_frame: Optional[np.ndarray] = None

    # --- Public API ---

    def cache_frame(self, img_rgb: Optional[np.ndarray] = None) -> None:
        """Cache the current frame at full resolution (BGR).

        If *img_rgb* is given it is used directly (avoids a redundant
        disk read).  Otherwise loads the current frame from disk.
        """
        self._cached_frame = None
        if self._state is None or not self._state.image_files:
            return

        try:
            if img_rgb is None:
                frame = self._state.current_frame
                files = self._state.image_files
                if frame < 1 or frame > len(files):
                    return
                img_rgb = load_image_as_rgb(files[frame - 1])
                if img_rgb is None:
                    return

            self._cached_frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.warning(f"Failed to cache preview frame: {e}")

    def apply_preview(self, config: PreprocessingConfig) -> None:
        """Apply preprocessing to the cached frame and push to AppState.

        Merges shape overlays from ShapeController before applying.
        If the config is identity (no-op), signals the caller to reload
        the original frame instead.
        """
        config = self._shape_ctrl.inject_shapes(config)

        if config.is_identity():
            # Signal: caller should reload original frame
            self._state.set_display_images(
                original=self._state.current_original,
            )
            return

        # Lazy init cache
        if self._cached_frame is None:
            self.cache_frame()
        if self._cached_frame is None:
            return

        processed = apply_pipeline(self._cached_frame, config)
        self._state.set_display_images(
            original=cv2.cvtColor(processed, cv2.COLOR_BGR2RGB),
        )

    def refresh_with_shapes(self, sidebar_config: PreprocessingConfig) -> None:
        """Re-apply preview with current shapes (called after shape changes)."""
        self.apply_preview(sidebar_config)

    def get_config_with_shapes(
        self, config: Optional[PreprocessingConfig] = None,
    ) -> PreprocessingConfig:
        """Build config with shape overlays. If None, uses a default identity."""
        if config is None:
            config = PreprocessingConfig()
        return self._shape_ctrl.inject_shapes(config)
