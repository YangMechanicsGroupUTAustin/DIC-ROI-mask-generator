"""Processing controller for mask generation pipeline.

Orchestrates the full mask generation workflow via QThread workers.
Worker receives immutable data snapshot at construction for thread safety.
"""
import os
import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.mask_generator import MaskGenerator
from core.image_processing import (
    convert_image, create_overlay, get_image_files,
    load_image_as_rgb, create_placeholder_image,
)

logger = logging.getLogger("sam2studio.processing_controller")


class ProcessingWorker(QThread):
    """Background worker for mask generation pipeline.

    Stages:
    1. Convert images to intermediate format (JPEG/PNG)
    2. Initialize SAM2 model
    3. Set video + add annotation points
    4. Propagate masks through all frames
    5. Save binary masks to output directory

    Emits per-frame signals so GUI can update progressively.
    """
    progress = pyqtSignal(int, int, str)              # current, total, message
    frame_processed = pyqtSignal(int, object, object)  # frame_idx, mask(ndarray), overlay(ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
        self,
        mask_generator: MaskGenerator,
        image_files: list[str],
        output_dir: str,
        model_cfg: str,
        checkpoint: str,
        device: str,
        points: list[list[float]],
        labels: list[int],
        threshold: float,
        start_frame: int,             # 1-based
        end_frame: int,               # 1-based
        intermediate_format: str,     # "JPEG (fast)" or "PNG (lossless)"
        force_reprocess: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        # Thread-safe: copy all data at construction
        self._mask_generator = mask_generator
        self._image_files = list(image_files)
        self._output_dir = output_dir
        self._model_cfg = model_cfg
        self._checkpoint = checkpoint
        self._device = device
        self._points = [list(p) for p in points]
        self._labels = list(labels)
        self._threshold = threshold
        self._start_frame = start_frame
        self._end_frame = end_frame
        self._intermediate_format = intermediate_format
        self._force_reprocess = force_reprocess
        self._stop_flag = False

    def stop(self):
        """Request graceful cancellation."""
        self._stop_flag = True

    def run(self):
        try:
            self._run_pipeline()
        except Exception as e:
            logger.exception("Processing failed")
            self.error.emit(str(e))
        finally:
            self.finished.emit()

    def _run_pipeline(self):
        total_files = len(self._image_files)
        if total_files == 0:
            self.error.emit("No image files found.")
            return

        # Determine frame range (0-based internally)
        start_idx = self._start_frame - 1
        end_idx = self._end_frame  # exclusive upper bound for range

        # --- Stage 1: Convert images ---
        fmt = "jpeg" if "jpeg" in self._intermediate_format.lower() else "png"
        ext = ".jpg" if fmt == "jpeg" else ".png"
        converted_dir = os.path.join(self._output_dir, f"converted_{fmt}")
        os.makedirs(converted_dir, exist_ok=True)

        for i, img_path in enumerate(self._image_files):
            if self._stop_flag:
                logger.info("Processing cancelled by user (stage 1)")
                return

            out_name = f"frame_{i:06d}{ext}"
            out_path = os.path.join(converted_dir, out_name)

            if not self._force_reprocess and os.path.exists(out_path):
                continue

            success = convert_image(img_path, out_path, format=fmt)
            if not success:
                self.error.emit(f"Failed to convert: {os.path.basename(img_path)}")
                return

            self.progress.emit(
                i + 1, total_files,
                f"Converting images ({i + 1}/{total_files})",
            )

        # --- Stage 2: Initialize model ---
        if self._stop_flag:
            return

        self.progress.emit(0, 0, "Loading SAM2 model...")
        self._mask_generator.initialize(
            model_cfg=self._model_cfg,
            checkpoint=self._checkpoint,
            device=self._device,
            progress_callback=lambda msg: self.progress.emit(0, 0, msg),
        )

        # --- Stage 3: Set video + add points ---
        if self._stop_flag:
            return

        self.progress.emit(0, 0, "Initializing video predictor...")
        self._mask_generator.set_video(converted_dir)

        if self._points and self._labels:
            points_arr = np.array(self._points, dtype=np.float32)
            labels_arr = np.array(self._labels, dtype=np.int32)
            # Add points at the first frame of the range
            self._mask_generator.add_points(start_idx, points_arr, labels_arr)

        # --- Stage 4: Propagate ---
        if self._stop_flag:
            return

        self.progress.emit(0, total_files, "Propagating masks...")

        def on_propagation_progress(current, total):
            if self._stop_flag:
                return
            self.progress.emit(
                current,
                total if total > 0 else total_files,
                f"Propagating ({current}/{total if total > 0 else '?'})",
            )

        video_segments = self._mask_generator.propagate(
            threshold=self._threshold,
            progress_callback=on_propagation_progress,
        )

        # --- Stage 5: Save masks + emit per-frame ---
        if self._stop_flag:
            return

        mask_dir = os.path.join(self._output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        processed_count = 0
        for frame_idx in sorted(video_segments.keys()):
            if self._stop_flag:
                logger.info("Processing cancelled by user (stage 5)")
                return

            mask = video_segments[frame_idx]

            # Save mask
            mask_filename = f"mask_{frame_idx:06d}.tiff"
            mask_path = os.path.join(mask_dir, mask_filename)
            import cv2
            cv2.imwrite(mask_path, mask)

            # Create overlay for display
            if frame_idx < len(self._image_files):
                original = load_image_as_rgb(self._image_files[frame_idx])
                if original is not None:
                    overlay = create_overlay(original, mask)
                    self.frame_processed.emit(frame_idx, mask, overlay)

            processed_count += 1
            self.progress.emit(
                processed_count, len(video_segments),
                f"Saving masks ({processed_count}/{len(video_segments)})",
            )

        logger.info(f"Processing complete: {processed_count} masks saved to {mask_dir}")


class CorrectionWorker(QThread):
    """Background worker for mid-sequence correction and re-propagation."""
    progress = pyqtSignal(int, int, str)
    frame_processed = pyqtSignal(int, object, object)
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
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

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

        self.progress.emit(0, 0, "Adding correction points...")
        points_arr = np.array(self._points, dtype=np.float32)
        labels_arr = np.array(self._labels, dtype=np.int32)
        self._mask_generator.add_correction(self._frame_idx, points_arr, labels_arr)

        self.progress.emit(0, 0, "Re-propagating from correction frame...")
        segments = self._mask_generator.propagate_from(
            start_frame_idx=self._frame_idx,
            threshold=self._threshold,
        )

        mask_dir = os.path.join(self._output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        count = 0
        for idx in sorted(segments.keys()):
            if self._stop_flag:
                return

            mask = segments[idx]
            mask_path = os.path.join(mask_dir, f"mask_{idx:06d}.tiff")
            import cv2
            cv2.imwrite(mask_path, mask)

            if idx < len(self._image_files):
                original = load_image_as_rgb(self._image_files[idx])
                if original is not None:
                    overlay = create_overlay(original, mask)
                    self.frame_processed.emit(idx, mask, overlay)

            count += 1
            self.progress.emit(
                count, len(segments),
                f"Re-propagating ({count}/{len(segments)})",
            )

        logger.info(f"Correction complete: {count} frames updated")


class ProcessingController(QObject):
    """Coordinates between GUI and processing workers.

    Manages the lifecycle of ProcessingWorker and CorrectionWorker.
    Ensures only one worker runs at a time.
    """
    # Forward signals from workers for GUI consumption
    progress = pyqtSignal(int, int, str)
    frame_processed = pyqtSignal(int, object, object)
    processing_finished = pyqtSignal()
    processing_error = pyqtSignal(str)

    def __init__(self, state, mask_generator: MaskGenerator, parent=None):
        super().__init__(parent)
        self._state = state
        self._mask_generator = mask_generator
        self._worker: Optional[QThread] = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def start_processing(self) -> None:
        """Start the full mask generation pipeline."""
        if self.is_running:
            logger.warning("Processing already in progress")
            return

        model_cfg, checkpoint = self._state.get_model_config()

        worker = ProcessingWorker(
            mask_generator=self._mask_generator,
            image_files=self._state.image_files,
            output_dir=self._state.output_dir,
            model_cfg=model_cfg,
            checkpoint=checkpoint,
            device=self._state.device,
            points=self._state.points,
            labels=self._state.labels,
            threshold=self._state.threshold,
            start_frame=self._state.start_frame,
            end_frame=self._state.end_frame,
            intermediate_format=self._state.intermediate_format,
            force_reprocess=self._state.force_reprocess,
        )
        self._connect_worker(worker)
        self._worker = worker
        worker.start()
        logger.info("Processing started")

    def start_correction(self, frame_idx: int, points: list, labels: list) -> None:
        """Start correction re-propagation from a specific frame."""
        if self.is_running:
            logger.warning("Processing already in progress")
            return

        if not self._mask_generator.is_initialized:
            self.processing_error.emit(
                "Model not initialized. Run full processing first."
            )
            return

        worker = CorrectionWorker(
            mask_generator=self._mask_generator,
            frame_idx=frame_idx,
            points=points,
            labels=labels,
            threshold=self._state.threshold,
            image_files=self._state.image_files,
            output_dir=self._state.output_dir,
        )
        self._connect_worker(worker)
        self._worker = worker
        worker.start()
        logger.info(f"Correction started from frame {frame_idx}")

    def stop_processing(self) -> None:
        """Request graceful cancellation of current worker."""
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            logger.info("Stop requested")

    def _connect_worker(self, worker: QThread) -> None:
        """Connect worker signals to controller signals."""
        worker.progress.connect(self.progress)
        worker.frame_processed.connect(self.frame_processed)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self.processing_error)

    def _on_finished(self) -> None:
        """Handle worker completion."""
        self._worker = None
        self.processing_finished.emit()
        logger.info("Processing finished")
