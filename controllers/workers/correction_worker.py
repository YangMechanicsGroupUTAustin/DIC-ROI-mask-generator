"""Background worker for mid-sequence correction and re-propagation.

Mirrors ProcessingWorker's approach:
- Scales points from original to downsampled coordinates
- Uses streaming frame_callback (no memory accumulation)
- Upscales masks to original resolution before saving
- Passes stop_check for cancellation support
- Uses start_frame_idx to only compute from correction frame forward
"""

import logging
import os
import threading

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.image_processing import get_image_dimensions, imwrite_safe
from core.mask_generator import MaskGenerator

logger = logging.getLogger("sam2studio.correction_worker")

INFERENCE_MAX_SIZE = 1024


class CorrectionWorker(QThread):
    """Background worker for mid-sequence correction and re-propagation."""

    progress = pyqtSignal(int, int, str)
    frame_processed = pyqtSignal(int, object)  # frame_idx, mask(ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        mask_generator: MaskGenerator,
        frame_idx: int,                # 0-based
        points: list[list[float]],
        labels: list[int],
        threshold: float,
        image_files: list[str],
        output_dir: str,
        intermediate_format: str,      # "JPEG (fast)" or "PNG (lossless)"
        mask_output_format: str = "TIFF (default)",
        parent=None,
    ):
        super().__init__(parent)
        self._mask_generator = mask_generator
        self._frame_idx = frame_idx
        self._points = [list(p) for p in points]
        self._labels = list(labels)
        self._threshold = threshold
        self._image_files = list(image_files)
        self._output_dir = output_dir
        self._intermediate_format = intermediate_format
        self._mask_ext = ".png" if "PNG" in mask_output_format else ".tiff"
        self._stop_event = threading.Event()

    def stop(self):
        """Request graceful cancellation (thread-safe)."""
        self._stop_event.set()

    def run(self):
        try:
            self._run_correction()
        except Exception as e:
            logger.exception("Correction failed")
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def _run_correction(self):
        if not self._mask_generator.is_initialized:
            self.error.emit("Model not initialized. Run full processing first.")
            return

        if not self._image_files:
            self.error.emit("No image files available.")
            return

        # Determine scale factors between original and downsampled resolution
        orig_dims = get_image_dimensions(self._image_files[0])
        if orig_dims is None:
            self.error.emit("Cannot read image dimensions.")
            return
        orig_h, orig_w = orig_dims

        fmt = "jpeg" if "jpeg" in self._intermediate_format.lower() else "png"
        ext = ".jpg" if fmt == "jpeg" else ".png"
        converted_dir = os.path.join(self._output_dir, f"converted_{fmt}")
        first_converted = os.path.join(converted_dir, f"000000{ext}")

        conv_dims = get_image_dimensions(first_converted)
        conv_h, conv_w = conv_dims if conv_dims else (orig_h, orig_w)

        scale_x = conv_w / orig_w
        scale_y = conv_h / orig_h
        needs_upscale = (conv_h != orig_h or conv_w != orig_w)

        # Scale correction points from original to downsampled coordinates
        self.progress.emit(0, 0, "Adding correction points...")
        points_arr = np.array(self._points, dtype=np.float32)
        points_arr[:, 0] *= scale_x
        points_arr[:, 1] *= scale_y
        labels_arr = np.array(self._labels, dtype=np.int32)
        self._mask_generator.add_correction(self._frame_idx, points_arr, labels_arr)

        if self._stop_event.is_set():
            return

        # Stream propagation from correction frame with frame_callback
        mask_dir = os.path.join(self._output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        total_remaining = len(self._image_files) - self._frame_idx
        processed_count = 0

        self.progress.emit(0, total_remaining,
                           f"Re-propagating from frame {self._frame_idx}...")

        def on_frame(frame_idx: int, mask: np.ndarray) -> None:
            nonlocal processed_count
            if self._stop_event.is_set():
                return

            # Upscale mask to original resolution if downsampled
            if needs_upscale:
                mask = cv2.resize(
                    mask, (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                )

            mask_path = os.path.join(
                mask_dir, f"mask_{frame_idx:06d}{self._mask_ext}",
            )
            imwrite_safe(mask_path, mask)

            self.frame_processed.emit(frame_idx, mask)

            processed_count += 1
            self.progress.emit(
                processed_count, total_remaining,
                f"Re-propagating ({processed_count}/{total_remaining})",
            )

        self._mask_generator.propagate_from(
            start_frame_idx=self._frame_idx,
            threshold=self._threshold,
            frame_callback=on_frame,
            stop_check=self._stop_event.is_set,
        )

        logger.info(f"Correction complete: {processed_count} frames updated from frame {self._frame_idx}")
