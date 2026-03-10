"""Processing controller for mask generation pipeline.

Orchestrates the full mask generation workflow via QThread workers.
Worker receives immutable data snapshot at construction for thread safety.
"""
import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.mask_generator import MaskGenerator
from core.image_processing import (
    convert_image, get_image_dimensions, imread_safe, imwrite_safe,
)
from core.preprocessing import PreprocessingConfig, apply_pipeline

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
    frame_processed = pyqtSignal(int, object)          # frame_idx, mask(ndarray)
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
        preprocessing_config: PreprocessingConfig | None = None,
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
        self._preprocessing_config = preprocessing_config or PreprocessingConfig()
        self._stop_event = threading.Event()

    def stop(self):
        """Request graceful cancellation (thread-safe)."""
        self._stop_event.set()

    def run(self):
        try:
            self._run_pipeline()
        except Exception as e:
            logger.exception("Processing failed")
            self.error.emit(str(e))
            try:
                self._mask_generator.cleanup()
            except Exception:
                logger.warning("Failed to cleanup mask generator after error")
        finally:
            self.finished.emit()

    # SAM2 internally resizes to 1024; no benefit in feeding larger images
    INFERENCE_MAX_SIZE = 1024

    def _run_pipeline(self):
        total_files = len(self._image_files)
        if total_files == 0:
            self.error.emit("No image files found.")
            return

        start_idx = self._start_frame - 1

        # Get original image dimensions (for point scaling + mask upscaling)
        orig_dims = get_image_dimensions(self._image_files[0])
        if orig_dims is None:
            self.error.emit("Cannot read first image dimensions.")
            return
        orig_h, orig_w = orig_dims

        # --- Stage 1: Parallel convert + downsample ---
        fmt = "jpeg" if "jpeg" in self._intermediate_format.lower() else "png"
        ext = ".jpg" if fmt == "jpeg" else ".png"
        converted_dir = os.path.join(self._output_dir, f"converted_{fmt}")
        os.makedirs(converted_dir, exist_ok=True)

        conversion_error = self._convert_images_parallel(
            fmt, ext, converted_dir, total_files,
        )
        if conversion_error:
            self.error.emit(conversion_error)
            return

        # --- Stage 1.5: Apply preprocessing to separate directory ---
        # converted_dir stays pristine (reusable across runs).
        # Preprocessed images go to a separate directory for SAM2 input.
        if not self._preprocessing_config.is_identity():
            preprocessed_dir = os.path.join(
                self._output_dir, f"preprocessed_{fmt}",
            )
            os.makedirs(preprocessed_dir, exist_ok=True)
            written = self._apply_preprocessing(
                converted_dir, preprocessed_dir, ext, total_files,
            )
            if written > 0:
                sam2_input_dir = preprocessed_dir
            else:
                logger.warning(
                    "Preprocessing produced no output files, "
                    "falling back to converted images"
                )
                sam2_input_dir = converted_dir
        else:
            sam2_input_dir = converted_dir

        # Determine actual downsampled dimensions from converted file
        first_converted = os.path.join(converted_dir, f"000000{ext}")
        conv_dims = get_image_dimensions(first_converted)
        conv_h, conv_w = conv_dims if conv_dims else (orig_h, orig_w)

        scale_x = conv_w / orig_w
        scale_y = conv_h / orig_h
        needs_upscale = (conv_h != orig_h or conv_w != orig_w)

        if needs_upscale:
            logger.info(
                f"Downsampled {orig_w}x{orig_h} -> {conv_w}x{conv_h} "
                f"(scale: {scale_x:.3f}x{scale_y:.3f})"
            )

        # --- Stage 2: Initialize model ---
        if self._stop_event.is_set():
            return

        if self._mask_generator.is_initialized:
            self.progress.emit(0, 0, "Reusing loaded SAM2 model...")
        else:
            self.progress.emit(0, 0, "Loading SAM2 model...")
        self._mask_generator.initialize(
            model_cfg=self._model_cfg,
            checkpoint=self._checkpoint,
            device=self._device,
            progress_callback=lambda msg: self.progress.emit(0, 0, msg),
        )

        # --- Stage 3: Set video + add points (scaled to downsampled coords) ---
        if self._stop_event.is_set():
            return

        self.progress.emit(0, 0, "Initializing video predictor...")
        self._mask_generator.set_video(sam2_input_dir)

        if self._points and self._labels:
            points_arr = np.array(self._points, dtype=np.float32)
            # Scale point coordinates from original to downsampled space
            points_arr[:, 0] *= scale_x
            points_arr[:, 1] *= scale_y
            labels_arr = np.array(self._labels, dtype=np.int32)
            self._mask_generator.add_points(start_idx, points_arr, labels_arr)

        # --- Stage 4+5: Propagate, upscale, save, preview per frame ---
        if self._stop_event.is_set():
            return

        mask_dir = os.path.join(self._output_dir, "masks")
        os.makedirs(mask_dir, exist_ok=True)

        self.progress.emit(0, total_files, "Propagating masks...")
        processed_count = 0

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

            mask_path = os.path.join(mask_dir, f"mask_{frame_idx:06d}.tiff")
            imwrite_safe(mask_path, mask)

            self.frame_processed.emit(frame_idx, mask)

            processed_count += 1
            self.progress.emit(
                processed_count, total_files,
                f"Propagating & saving ({processed_count}/{total_files})",
            )

        self._mask_generator.propagate(
            threshold=self._threshold,
            frame_callback=on_frame,
            stop_check=self._stop_event.is_set,
        )

        logger.info(f"Processing complete: {processed_count} masks saved to {mask_dir}")

    def _convert_images_parallel(
        self, fmt: str, ext: str, converted_dir: str, total_files: int,
    ) -> str | None:
        """Convert images in parallel using thread pool. Returns error msg or None."""
        # Build list of (index, src_path, dst_path) for files needing conversion
        tasks = []
        for i, img_path in enumerate(self._image_files):
            out_path = os.path.join(converted_dir, f"{i:06d}{ext}")
            if (
                not self._force_reprocess
                and os.path.exists(out_path)
                and os.path.getsize(out_path) > 0
            ):
                continue
            tasks.append((i, img_path, out_path))

        if not tasks:
            self.progress.emit(total_files, total_files, "Images already converted")
            return None

        completed = 0
        # I/O bound task — use more threads than CPU cores
        max_workers = min(8, len(tasks))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    convert_image, src, dst,
                    format=fmt, max_size=self.INFERENCE_MAX_SIZE,
                ): (idx, src)
                for idx, src, dst in tasks
            }
            for future in as_completed(futures):
                if self._stop_event.is_set():
                    pool.shutdown(wait=False, cancel_futures=True)
                    return "cancelled"

                idx, src = futures[future]
                try:
                    if not future.result():
                        return f"Failed to convert: {os.path.basename(src)}"
                except Exception as exc:
                    if self._stop_event.is_set():
                        return "cancelled"
                    return f"Failed to convert: {os.path.basename(src)} ({exc})"

                completed += 1
                self.progress.emit(
                    completed, len(tasks),
                    f"Converting images ({completed}/{len(tasks)})",
                )

        return None

    def _apply_preprocessing(
        self, src_dir: str, dst_dir: str, ext: str, total_files: int,
    ) -> int:
        """Apply preprocessing: read from src_dir, write to dst_dir.

        Source directory (converted images) is never modified, keeping it
        pristine and reusable across runs with different preprocessing.

        Returns:
            Number of files successfully written.
        """
        self.progress.emit(0, total_files, "Applying preprocessing...")
        written = 0

        for i in range(total_files):
            if self._stop_event.is_set():
                return written

            src_path = os.path.join(src_dir, f"{i:06d}{ext}")
            if not os.path.exists(src_path):
                logger.warning(f"Preprocessing: source not found: {src_path}")
                continue

            img = imread_safe(src_path)
            if img is None:
                logger.warning(f"Preprocessing: imread failed: {src_path}")
                continue

            processed = apply_pipeline(img, self._preprocessing_config)
            dst_path = os.path.join(dst_dir, f"{i:06d}{ext}")
            success = imwrite_safe(dst_path, processed)
            if not success:
                logger.warning(f"Preprocessing: imwrite failed: {dst_path}")
                continue

            written += 1
            self.progress.emit(
                i + 1, total_files,
                f"Preprocessing ({i + 1}/{total_files})",
            )

        logger.info(
            f"Preprocessing: {written}/{total_files} files written "
            f"({src_dir} -> {dst_dir})"
        )
        return written


class CorrectionWorker(QThread):
    """Background worker for mid-sequence correction and re-propagation.

    Mirrors ProcessingWorker's approach:
    - Scales points from original to downsampled coordinates
    - Uses streaming frame_callback (no memory accumulation)
    - Upscales masks to original resolution before saving
    - Passes stop_check for cancellation support
    - Uses start_frame_idx to only compute from correction frame forward
    """
    progress = pyqtSignal(int, int, str)
    frame_processed = pyqtSignal(int, object)          # frame_idx, mask(ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    INFERENCE_MAX_SIZE = 1024

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

            mask_path = os.path.join(mask_dir, f"mask_{frame_idx:06d}.tiff")
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


class ProcessingController(QObject):
    """Coordinates between GUI and processing workers.

    Manages the lifecycle of ProcessingWorker and CorrectionWorker.
    Ensures only one worker runs at a time.
    """
    # Forward signals from workers for GUI consumption
    progress = pyqtSignal(int, int, str)
    frame_processed = pyqtSignal(int, object)           # frame_idx, mask
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
            preprocessing_config=self._state.preprocessing_config,
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
            intermediate_format=self._state.intermediate_format,
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
        """Connect worker signals to controller signals.

        Disconnects any previous worker first to prevent duplicate connections.
        """
        if self._worker is not None:
            try:
                self._worker.progress.disconnect(self.progress)
                self._worker.frame_processed.disconnect(self.frame_processed)
                self._worker.finished.disconnect(self._on_finished)
                self._worker.error.disconnect(self.processing_error)
            except (TypeError, RuntimeError):
                pass  # Already disconnected or destroyed

        worker.progress.connect(self.progress)
        worker.frame_processed.connect(self.frame_processed)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self.processing_error)

    def _on_finished(self) -> None:
        """Handle worker completion."""
        self._worker = None
        self.processing_finished.emit()
        logger.info("Processing finished")
