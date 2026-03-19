"""Background worker for standalone preprocessing + save.

Reads original images, applies the preprocessing pipeline, and saves
results to output_dir/preprocessed/. Supports custom frame ranges.
"""

import logging
import os
import threading

from PyQt6.QtCore import QThread, pyqtSignal

from core.image_processing import imread_safe, imwrite_safe
from core.preprocessing import (
    PreprocessingConfig, apply_pipeline, parse_custom_frames,
)

logger = logging.getLogger("sam2studio.preprocessing_save_worker")


class PreprocessingSaveWorker(QThread):
    """Background worker for standalone preprocessing + save."""

    progress = pyqtSignal(int, int, str)    # current, total, message
    finished = pyqtSignal(str)               # output directory path
    error = pyqtSignal(str)

    def __init__(
        self,
        image_files: list[str],
        output_dir: str,
        preprocessing_config: PreprocessingConfig,
        parent=None,
    ):
        super().__init__(parent)
        self._image_files = list(image_files)
        self._output_dir = output_dir
        self._config = preprocessing_config
        self._stop_event = threading.Event()

    def stop(self):
        """Request graceful cancellation (thread-safe)."""
        self._stop_event.set()

    def run(self):
        try:
            self._run_save()
        except Exception as e:
            logger.exception("Preprocessing save failed")
            self.error.emit(str(e))

    def _run_save(self):
        total_files = len(self._image_files)
        if total_files == 0:
            self.error.emit("No image files found.")
            return

        # Determine frame indices to process
        frame_indices = parse_custom_frames(
            self._config.custom_frames, total_files,
        )
        if not frame_indices:
            self.error.emit("No frames selected for preprocessing.")
            return

        dst_dir = os.path.join(self._output_dir, "preprocessed")
        os.makedirs(dst_dir, exist_ok=True)

        total = len(frame_indices)
        written = 0

        for count, idx in enumerate(frame_indices):
            if self._stop_event.is_set():
                break

            src_path = self._image_files[idx]
            img = imread_safe(src_path)
            if img is None:
                logger.warning(f"Preprocessing save: imread failed: {src_path}")
                continue

            processed = apply_pipeline(img, self._config)

            # Use original filename for easy reference
            filename = os.path.basename(src_path)
            # Save as TIFF for consistency with mask output
            name_no_ext = os.path.splitext(filename)[0]
            dst_path = os.path.join(dst_dir, f"{name_no_ext}.tiff")

            success = imwrite_safe(dst_path, processed)
            if not success:
                logger.warning(f"Preprocessing save: imwrite failed: {dst_path}")
                continue

            written += 1
            self.progress.emit(
                count + 1, total,
                f"Saving preprocessed ({count + 1}/{total})",
            )

        logger.info(
            f"Preprocessing save: {written}/{total} files written to {dst_dir}"
        )
        self.finished.emit(dst_dir)
