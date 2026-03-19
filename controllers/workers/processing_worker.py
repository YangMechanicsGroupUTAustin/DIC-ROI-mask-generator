"""Background worker for full mask generation pipeline.

Stages:
1. Convert images to intermediate format (JPEG/PNG)
2. Initialize SAM2 model
3. Set video + add annotation points
4. Propagate masks through all frames
5. Save binary masks to output directory

Emits per-frame signals so GUI can update progressively.
"""

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from core.image_processing import (
    convert_image, get_image_dimensions, imwrite_safe,
)
from core.mask_generator import MaskGenerator
from core.preprocessing import PreprocessingConfig

logger = logging.getLogger("sam2studio.processing_worker")

# SAM2 internally resizes to 1024; no benefit in feeding larger images
INFERENCE_MAX_SIZE = 1024


class ProcessingWorker(QThread):
    """Background worker for mask generation pipeline."""

    progress = pyqtSignal(int, int, str)       # current, total, message
    frame_processed = pyqtSignal(int, object)  # frame_idx, mask(ndarray)
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
        skip_existing: bool = False,
        mask_output_format: str = "TIFF (default)",
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
        self._skip_existing = skip_existing
        self._mask_ext = ".png" if "PNG" in mask_output_format else ".tiff"
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

        # --- Stage 1.5: Use preprocessed images if available ---
        preprocessed_dir = os.path.join(
            self._output_dir, f"preprocessed_{fmt}",
        )
        if (
            os.path.isdir(preprocessed_dir)
            and any(
                f.endswith(ext) for f in os.listdir(preprocessed_dir)
            )
        ):
            sam2_input_dir = preprocessed_dir
            logger.info(
                f"Using preprocessed images from {preprocessed_dir}"
            )
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

            mask_path = os.path.join(
                mask_dir, f"mask_{frame_idx:06d}{self._mask_ext}",
            )
            # In resume mode, skip frames that already have a mask file
            if not (self._skip_existing and os.path.exists(mask_path)):
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
                    format=fmt, max_size=INFERENCE_MAX_SIZE,
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
