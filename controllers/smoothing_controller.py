"""Smoothing controller for post-processing mask sequences.

Provides background workers for spatial (Perona-Malik) and temporal
(3D Gaussian) smoothing with progress reporting and cancellation.
"""
import os
import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.spatial_smoothing import perona_malik_smooth
from core.temporal_smoothing import temporal_smooth_sequence
from core.image_processing import get_image_files

logger = logging.getLogger("sam2studio.smoothing_controller")


class SpatialSmoothWorker(QThread):
    """Background worker for spatial smoothing of mask files."""
    progress = pyqtSignal(int, int)       # current, total
    finished = pyqtSignal(str)            # output directory
    error = pyqtSignal(str)

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        num_iterations: int = 50,
        dt: float = 0.1,
        kappa: float = 30.0,
        option: int = 1,
        post_gaussian_sigma: float = 0.0,
        parent=None,
    ):
        super().__init__(parent)
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._num_iterations = num_iterations
        self._dt = dt
        self._kappa = kappa
        self._option = option
        self._post_gaussian_sigma = post_gaussian_sigma
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            self._run_smoothing()
        except Exception as e:
            logger.exception("Spatial smoothing failed")
            self.error.emit(str(e))

    def _run_smoothing(self):
        import cv2

        mask_files = get_image_files(self._input_dir)
        if not mask_files:
            self.error.emit(f"No mask files found in {self._input_dir}")
            return

        os.makedirs(self._output_dir, exist_ok=True)
        total = len(mask_files)

        for i, mask_path in enumerate(mask_files):
            if self._stop_flag:
                logger.info("Spatial smoothing cancelled")
                return

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Could not read mask: {mask_path}")
                continue

            smoothed = perona_malik_smooth(
                mask,
                num_iterations=self._num_iterations,
                dt=self._dt,
                kappa=self._kappa,
                option=self._option,
                post_gaussian_sigma=self._post_gaussian_sigma,
            )

            out_name = os.path.basename(mask_path)
            out_path = os.path.join(self._output_dir, out_name)
            cv2.imwrite(out_path, smoothed)

            self.progress.emit(i + 1, total)

        logger.info(
            f"Spatial smoothing complete: {total} masks -> {self._output_dir}"
        )
        self.finished.emit(self._output_dir)


class TemporalSmoothWorker(QThread):
    """Background worker for temporal smoothing of mask sequence."""
    progress = pyqtSignal(int, int, str)   # current, total, step_name
    finished = pyqtSignal(str)              # output directory
    error = pyqtSignal(str)

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        sigma: float = 2.0,
        num_neighbors: int = 2,
        variance_threshold: Optional[float] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._input_dir = input_dir
        self._output_dir = output_dir
        self._sigma = sigma
        self._num_neighbors = num_neighbors
        self._variance_threshold = variance_threshold
        self._stop_flag = False

    def stop(self):
        self._stop_flag = True

    def run(self):
        try:
            self._run_smoothing()
        except Exception as e:
            logger.exception("Temporal smoothing failed")
            self.error.emit(str(e))

    def _run_smoothing(self):
        import cv2

        mask_files = get_image_files(self._input_dir)
        if not mask_files:
            self.error.emit(f"No mask files found in {self._input_dir}")
            return

        os.makedirs(self._output_dir, exist_ok=True)

        # Load all masks
        frames = []
        for i, path in enumerate(mask_files):
            if self._stop_flag:
                return
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Could not read mask: {path}, using zeros")
                if frames:
                    mask = np.zeros_like(frames[0])
                else:
                    self.error.emit(f"First mask unreadable: {path}")
                    return
            frames.append(mask)

        def on_progress(step_name, current, total):
            self.progress.emit(current, total, step_name)

        smoothed_frames = temporal_smooth_sequence(
            frames,
            variance_threshold=self._variance_threshold,
            num_neighbors=self._num_neighbors,
            sigma=self._sigma,
            progress_callback=on_progress,
        )

        # Save results
        for i, smoothed in enumerate(smoothed_frames):
            if self._stop_flag:
                return
            out_name = os.path.basename(mask_files[i])
            out_path = os.path.join(self._output_dir, out_name)
            cv2.imwrite(out_path, smoothed)

        logger.info(
            f"Temporal smoothing complete: "
            f"{len(smoothed_frames)} masks -> {self._output_dir}"
        )
        self.finished.emit(self._output_dir)


class SmoothingController(QObject):
    """Coordinates spatial and temporal smoothing workers.

    Ensures only one smoothing operation runs at a time.
    """
    progress = pyqtSignal(int, int, str)        # current, total, message
    smoothing_finished = pyqtSignal(str)         # output directory
    smoothing_error = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker: Optional[QThread] = None

    @property
    def is_running(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    def start_spatial(
        self,
        input_dir: str,
        output_dir: str,
        num_iterations: int = 50,
        dt: float = 0.1,
        kappa: float = 30.0,
        option: int = 1,
        post_gaussian_sigma: float = 0.0,
    ) -> None:
        if self.is_running:
            logger.warning("Smoothing already in progress")
            return

        worker = SpatialSmoothWorker(
            input_dir=input_dir,
            output_dir=output_dir,
            num_iterations=num_iterations,
            dt=dt,
            kappa=kappa,
            option=option,
            post_gaussian_sigma=post_gaussian_sigma,
        )
        self._connect_worker_spatial(worker)
        self._worker = worker
        worker.start()
        logger.info(f"Spatial smoothing started: {input_dir} -> {output_dir}")

    def start_temporal(
        self,
        input_dir: str,
        output_dir: str,
        sigma: float = 2.0,
        num_neighbors: int = 2,
        variance_threshold: Optional[float] = None,
    ) -> None:
        if self.is_running:
            logger.warning("Smoothing already in progress")
            return

        worker = TemporalSmoothWorker(
            input_dir=input_dir,
            output_dir=output_dir,
            sigma=sigma,
            num_neighbors=num_neighbors,
            variance_threshold=variance_threshold,
        )
        self._connect_worker_temporal(worker)
        self._worker = worker
        worker.start()
        logger.info(f"Temporal smoothing started: {input_dir} -> {output_dir}")

    def stop(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            logger.info("Smoothing stop requested")

    def _connect_worker_spatial(self, worker: SpatialSmoothWorker) -> None:
        worker.progress.connect(
            lambda cur, tot: self.progress.emit(
                cur, tot, f"Spatial smoothing ({cur}/{tot})"
            )
        )
        worker.finished.connect(self._on_finished)
        worker.error.connect(self.smoothing_error)

    def _connect_worker_temporal(self, worker: TemporalSmoothWorker) -> None:
        worker.progress.connect(self.progress)
        worker.finished.connect(self._on_finished)
        worker.error.connect(self.smoothing_error)

    def _on_finished(self, output_dir: str) -> None:
        self._worker = None
        self.smoothing_finished.emit(output_dir)
        logger.info(f"Smoothing finished: {output_dir}")
