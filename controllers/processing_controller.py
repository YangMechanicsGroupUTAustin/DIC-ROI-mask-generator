"""Processing controller for mask generation pipeline.

Orchestrates the full mask generation workflow via QThread workers.
Worker classes are defined in controllers/workers/ for modularity.
"""

import logging
from typing import Optional

from PyQt6.QtCore import QObject, QThread, pyqtSignal

from core.mask_generator import MaskGenerator
from core.preprocessing import PreprocessingConfig
from controllers.workers import (
    CorrectionWorker,
    PreprocessingSaveWorker,
    ProcessingWorker,
)

logger = logging.getLogger("sam2studio.processing_controller")


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

    @staticmethod
    def _check_disk_space(output_dir: str, min_mb: int = 100) -> str | None:
        """Return error message if disk space is insufficient, else None."""
        import shutil
        try:
            usage = shutil.disk_usage(output_dir)
            free_mb = usage.free / (1024 * 1024)
            if free_mb < min_mb:
                return (
                    f"Insufficient disk space: {free_mb:.0f} MB free "
                    f"(need at least {min_mb} MB)"
                )
        except OSError as e:
            logger.warning(f"Could not check disk space: {e}")
        return None

    def start_processing(self, skip_existing: bool = False) -> None:
        """Start the full mask generation pipeline.

        Args:
            skip_existing: If True, don't overwrite masks that already exist
                           in the output directory (resume mode).
        """
        if self.is_running:
            logger.warning("Processing already in progress")
            return

        # Validate output directory and disk space
        output_dir = self._state.output_dir
        if output_dir:
            import os
            os.makedirs(output_dir, exist_ok=True)
            space_err = self._check_disk_space(output_dir)
            if space_err:
                self.processing_error.emit(space_err)
                return

        model_cfg, checkpoint = self._state.get_model_config()

        # Convert UI 1-based anchor frame to internal 0-based for the worker.
        refine_anchor_zero_based = max(0, self._state.refine_anchor_frame - 1)

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
            skip_existing=skip_existing,
            mask_output_format=self._state.mask_output_format,
            refine_enabled=self._state.refine_enabled,
            refine_anchor_frame=refine_anchor_zero_based,
            refine_overwrite_count=self._state.refine_overwrite_count,
        )
        self._connect_worker(worker)
        self._worker = worker
        worker.start()
        logger.info("Processing started")

    def start_correction(
        self,
        anchor_frame_idx: int,
        range_start: int,
        range_end: int,
        points: list,
        labels: list,
    ) -> None:
        """Start range-based correction re-propagation.

        Args:
            anchor_frame_idx: 0-based frame where correction points are placed.
            range_start: 0-based inclusive range start.
            range_end: 0-based inclusive range end.
            points: [[x, y], ...] in original image coordinates.
            labels: [1 | 0, ...] — foreground/background per point.
        """
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
            anchor_frame_idx=anchor_frame_idx,
            range_start=range_start,
            range_end=range_end,
            points=points,
            labels=labels,
            threshold=self._state.threshold,
            image_files=self._state.image_files,
            output_dir=self._state.output_dir,
            intermediate_format=self._state.intermediate_format,
            mask_output_format=self._state.mask_output_format,
        )
        self._connect_worker(worker)
        self._worker = worker
        worker.start()
        logger.info(
            "Correction started: anchor=%d range=[%d, %d]",
            anchor_frame_idx, range_start, range_end,
        )

    def start_save_preprocessed(
        self, config: PreprocessingConfig,
    ) -> None:
        """Start standalone preprocessing + save workflow."""
        if self.is_running:
            logger.warning("Processing already in progress")
            return

        worker = PreprocessingSaveWorker(
            image_files=self._state.image_files,
            output_dir=self._state.output_dir,
            preprocessing_config=config,
        )
        # PreprocessingSaveWorker has different signals than other workers.
        # Connect them to the controller's unified signals.
        worker.progress.connect(self.progress)
        worker.finished.connect(
            lambda path: self.processing_finished.emit()
        )
        worker.error.connect(self.processing_error)
        self._worker = worker
        worker.start()
        logger.info("Preprocessing save started")

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
