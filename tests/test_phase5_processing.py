"""Phase 5 verification: processing and smoothing controllers.

Tests background workers (ProcessingWorker, CorrectionWorker,
SpatialSmoothWorker, TemporalSmoothWorker) and their controller
orchestrators (ProcessingController, SmoothingController).

Uses unittest.mock to avoid SAM2 model dependencies.
Requires PyQt6 and QApplication for signal tests.
"""
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch, PropertyMock

import cv2
import numpy as np
import pytest
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication


# --- Fixtures ---

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    return app


@pytest.fixture
def state(qapp):
    from controllers.app_state import AppState
    s = AppState()
    return s


@pytest.fixture
def mock_mask_generator():
    """Create a MaskGenerator mock with standard behavior."""
    mg = MagicMock()
    mg.is_initialized = False
    mg.has_inference_state = False
    mg.initialize = MagicMock()
    mg.set_video = MagicMock()
    mg.add_points = MagicMock()
    mg.add_original_points = MagicMock()
    mg.cleanup = MagicMock()

    def fake_propagate(threshold=0.0, progress_callback=None, frame_callback=None,
                       stop_check=None, start_frame_idx=None):
        segments = {}
        start = start_frame_idx or 0
        for i in range(start, start + 3):
            if stop_check and stop_check():
                break
            mask = np.zeros((50, 50), dtype=np.uint8)
            mask[10:40, 10:40] = 255
            if frame_callback:
                frame_callback(i, mask)
            else:
                segments[i] = mask
            if progress_callback:
                progress_callback(i - start + 1, 3)
        return segments

    mg.propagate = MagicMock(side_effect=fake_propagate)

    def fake_propagate_from(start_frame_idx=0, threshold=0.0, progress_callback=None,
                            frame_callback=None, stop_check=None):
        return fake_propagate(threshold, progress_callback, frame_callback,
                              stop_check, start_frame_idx)

    mg.propagate_from = MagicMock(side_effect=fake_propagate_from)
    mg.add_correction = MagicMock()

    def fake_propagate_range(anchor_frame_idx, range_start, range_end,
                             threshold=0.0, progress_callback=None,
                             frame_callback=None, stop_check=None):
        # Emit the anchor once, then walk outward toward each end.
        indices = list(range(int(range_start), int(range_end) + 1))
        for i in indices:
            if stop_check and stop_check():
                break
            if frame_callback:
                mask = np.zeros((50, 50), dtype=np.uint8)
                mask[10:40, 10:40] = 255
                frame_callback(i, mask)

    mg.propagate_range = MagicMock(side_effect=fake_propagate_range)
    mg.reset_corrections = MagicMock()

    def fake_refine_early_frames(
        anchor_frame_idx,
        anchor_mask,
        overwrite_count,
        threshold=0.0,
        frame_callback=None,
        progress_callback=None,
        stop_check=None,
    ):
        # Emit the earliest K frames [0, K-1] as the real method does.
        for i in range(int(overwrite_count)):
            if stop_check and stop_check():
                break
            if frame_callback:
                mask = np.zeros((50, 50), dtype=np.uint8)
                mask[15:35, 15:35] = 255
                frame_callback(i, mask)
            if progress_callback:
                progress_callback(i + 1, int(overwrite_count))

    mg.refine_early_frames = MagicMock(side_effect=fake_refine_early_frames)

    return mg


@pytest.fixture
def tmp_image_dir():
    """Create a temporary directory with small test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            path = os.path.join(tmpdir, f"img_{i:03d}.png")
            cv2.imwrite(path, img)
        yield tmpdir


@pytest.fixture
def tmp_mask_dir():
    """Create a temporary directory with small binary mask files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            mask = np.zeros((50, 50), dtype=np.uint8)
            mask[15:35, 15:35] = 255
            path = os.path.join(tmpdir, f"mask_{i:03d}.tiff")
            cv2.imwrite(path, mask)
        yield tmpdir


# =============================================================================
# ProcessingWorker Tests
# =============================================================================

class TestProcessingWorker:
    """Tests for the full mask generation pipeline worker."""

    def test_worker_copies_data_at_construction(self, qapp, mock_mask_generator):
        """Worker should snapshot input data for thread safety."""
        from controllers.processing_controller import ProcessingWorker

        original_points = [[10.0, 20.0], [30.0, 40.0]]
        original_labels = [1, 0]
        original_files = ["/a.png", "/b.png"]

        worker = ProcessingWorker(
            mask_generator=mock_mask_generator,
            image_files=original_files,
            output_dir="/tmp/out",
            model_cfg="test.yaml",
            checkpoint="test.pt",
            device="cpu",
            points=original_points,
            labels=original_labels,
            threshold=0.0,
            start_frame=1,
            end_frame=2,
            intermediate_format="JPEG (fast)",
        )

        # Mutate originals -- worker copies should be unaffected
        original_points.append([99.0, 99.0])
        original_labels.append(1)
        original_files.append("/c.png")

        assert len(worker._points) == 2
        assert len(worker._labels) == 2
        assert len(worker._image_files) == 2

    def test_worker_stop_event(self, qapp, mock_mask_generator):
        """Stop event should be settable (thread-safe)."""
        from controllers.processing_controller import ProcessingWorker

        worker = ProcessingWorker(
            mask_generator=mock_mask_generator,
            image_files=[],
            output_dir="/tmp/out",
            model_cfg="test.yaml",
            checkpoint="test.pt",
            device="cpu",
            points=[],
            labels=[],
            threshold=0.0,
            start_frame=1,
            end_frame=1,
            intermediate_format="JPEG (fast)",
        )
        assert not worker._stop_event.is_set()
        worker.stop()
        assert worker._stop_event.is_set()

    def test_worker_emits_error_on_empty_files(self, qapp, mock_mask_generator):
        """Worker should emit error when no image files provided."""
        from controllers.processing_controller import ProcessingWorker

        worker = ProcessingWorker(
            mask_generator=mock_mask_generator,
            image_files=[],
            output_dir="/tmp/out",
            model_cfg="test.yaml",
            checkpoint="test.pt",
            device="cpu",
            points=[],
            labels=[],
            threshold=0.0,
            start_frame=1,
            end_frame=1,
            intermediate_format="JPEG (fast)",
        )

        errors = []
        finished_signals = []
        worker.error.connect(lambda msg: errors.append(msg))
        worker.finished.connect(lambda: finished_signals.append(True))

        worker.run()  # run synchronously for testing

        assert len(errors) == 1
        assert "No image files" in errors[0]
        assert len(finished_signals) == 1  # finished always emits

    def test_worker_seeds_original_conditioning(
        self, qapp, mock_mask_generator, tmp_image_dir,
    ):
        """Initial annotation must be registered via `add_original_points`
        so that subsequent `reset_corrections()` can rebuild the clean
        baseline. The worker must NOT bypass this by calling `add_points`
        directly — that would leave `_original_conditioning` empty and
        break the Re-run Range workflow.
        """
        from controllers.processing_controller import ProcessingWorker

        image_files = sorted(
            [os.path.join(tmp_image_dir, f) for f in os.listdir(tmp_image_dir)
             if f.endswith(".png")]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            worker = ProcessingWorker(
                mask_generator=mock_mask_generator,
                image_files=image_files,
                output_dir=output_dir,
                model_cfg="test.yaml",
                checkpoint="test.pt",
                device="cpu",
                points=[[25.0, 25.0]],
                labels=[1],
                threshold=0.0,
                start_frame=1,
                end_frame=3,
                intermediate_format="JPEG (fast)",
                force_reprocess=True,
            )
            worker.run()

            mock_mask_generator.add_original_points.assert_called_once()
            mock_mask_generator.add_points.assert_not_called()

    def test_worker_full_pipeline(self, qapp, mock_mask_generator, tmp_image_dir):
        """Worker should run all 5 stages with mocked model."""
        from controllers.processing_controller import ProcessingWorker

        image_files = sorted(
            [os.path.join(tmp_image_dir, f) for f in os.listdir(tmp_image_dir)
             if f.endswith(".png")]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            worker = ProcessingWorker(
                mask_generator=mock_mask_generator,
                image_files=image_files,
                output_dir=output_dir,
                model_cfg="test.yaml",
                checkpoint="test.pt",
                device="cpu",
                points=[[25.0, 25.0]],
                labels=[1],
                threshold=0.0,
                start_frame=1,
                end_frame=3,
                intermediate_format="JPEG (fast)",
                force_reprocess=True,
            )

            progress_msgs = []
            frame_signals = []
            finished_signals = []
            worker.progress.connect(
                lambda c, t, m: progress_msgs.append((c, t, m))
            )
            worker.frame_processed.connect(
                lambda idx, mask: frame_signals.append(idx)
            )
            worker.finished.connect(lambda: finished_signals.append(True))

            worker.run()

            # Model methods should have been called
            mock_mask_generator.initialize.assert_called_once()
            mock_mask_generator.set_video.assert_called_once()
            mock_mask_generator.add_original_points.assert_called_once()
            mock_mask_generator.propagate.assert_called_once()

            # Should have progress messages
            assert len(progress_msgs) > 0

            # Should have frame signals for each propagated mask
            assert len(frame_signals) == 3

            # Masks should be saved
            mask_dir = os.path.join(output_dir, "masks")
            assert os.path.isdir(mask_dir)
            mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".tiff")]
            assert len(mask_files) == 3

            # finished should fire
            assert len(finished_signals) == 1

    # ------------------------------------------------------------------
    #  Early-Frame Refinement (Phase B of the refine feature)
    # ------------------------------------------------------------------

    def test_worker_skips_refine_when_disabled(
        self, qapp, mock_mask_generator, tmp_image_dir,
    ):
        """When `refine_enabled=False` the worker must NOT invoke
        `refine_early_frames` — it's a pure opt-in post-stage."""
        from controllers.processing_controller import ProcessingWorker

        image_files = sorted(
            [os.path.join(tmp_image_dir, f) for f in os.listdir(tmp_image_dir)
             if f.endswith(".png")]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            worker = ProcessingWorker(
                mask_generator=mock_mask_generator,
                image_files=image_files,
                output_dir=output_dir,
                model_cfg="test.yaml",
                checkpoint="test.pt",
                device="cpu",
                points=[[25.0, 25.0]],
                labels=[1],
                threshold=0.0,
                start_frame=1,
                end_frame=3,
                intermediate_format="JPEG (fast)",
                force_reprocess=True,
                refine_enabled=False,
                refine_anchor_frame=2,
                refine_overwrite_count=1,
            )
            worker.run()
            mock_mask_generator.refine_early_frames.assert_not_called()

    def test_worker_calls_refine_early_frames_when_enabled(
        self, qapp, mock_mask_generator, tmp_image_dir,
    ):
        """When enabled, worker must call refine_early_frames with the
        configured anchor & overwrite count (both 0-based internally)."""
        from controllers.processing_controller import ProcessingWorker

        image_files = sorted(
            [os.path.join(tmp_image_dir, f) for f in os.listdir(tmp_image_dir)
             if f.endswith(".png")]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            worker = ProcessingWorker(
                mask_generator=mock_mask_generator,
                image_files=image_files,
                output_dir=output_dir,
                model_cfg="test.yaml",
                checkpoint="test.pt",
                device="cpu",
                points=[[25.0, 25.0]],
                labels=[1],
                threshold=0.0,
                start_frame=1,
                end_frame=3,
                intermediate_format="JPEG (fast)",
                force_reprocess=True,
                refine_enabled=True,
                refine_anchor_frame=2,   # 0-based: third frame
                refine_overwrite_count=1,  # overwrite only frame 0
            )
            worker.run()

            mock_mask_generator.refine_early_frames.assert_called_once()
            kwargs = mock_mask_generator.refine_early_frames.call_args.kwargs
            assert kwargs["anchor_frame_idx"] == 2
            assert kwargs["overwrite_count"] == 1
            # Captured anchor mask should have been passed through
            assert kwargs["anchor_mask"] is not None
            assert hasattr(kwargs["anchor_mask"], "shape")

    def test_worker_captures_anchor_mask_at_forward_pass(
        self, qapp, mock_mask_generator, tmp_image_dir,
    ):
        """The anchor mask passed to refine must be the SAME instance the
        forward pass emitted for the anchor frame (in-memory capture, not
        disk round-trip)."""
        from controllers.processing_controller import ProcessingWorker

        image_files = sorted(
            [os.path.join(tmp_image_dir, f) for f in os.listdir(tmp_image_dir)
             if f.endswith(".png")]
        )

        # Customize the fake propagate so we can identify the anchor mask
        # by a unique pixel value per frame.
        def fake_propagate(threshold=0.0, progress_callback=None,
                           frame_callback=None, stop_check=None,
                           start_frame_idx=None):
            start = start_frame_idx or 0
            for i in range(start, start + 3):
                mask = np.full((50, 50), fill_value=i + 1, dtype=np.uint8)
                if frame_callback:
                    frame_callback(i, mask)

        mock_mask_generator.propagate.side_effect = fake_propagate

        with tempfile.TemporaryDirectory() as output_dir:
            worker = ProcessingWorker(
                mask_generator=mock_mask_generator,
                image_files=image_files,
                output_dir=output_dir,
                model_cfg="test.yaml",
                checkpoint="test.pt",
                device="cpu",
                points=[[25.0, 25.0]],
                labels=[1],
                threshold=0.0,
                start_frame=1,
                end_frame=3,
                intermediate_format="JPEG (fast)",
                force_reprocess=True,
                refine_enabled=True,
                refine_anchor_frame=2,   # third frame, marker value = 3
                refine_overwrite_count=1,
            )
            worker.run()

            captured = mock_mask_generator.refine_early_frames.call_args.kwargs[
                "anchor_mask"
            ]
            # The fake forward pass fills frame 2 with all-3s
            assert captured is not None
            assert (np.asarray(captured) == 3).all()

    def test_worker_refine_overwrites_disk_files(
        self, qapp, mock_mask_generator, tmp_image_dir,
    ):
        """Refine stage must write its output masks to the same masks/ dir
        as the forward pass (overwrite in place)."""
        from controllers.processing_controller import ProcessingWorker

        image_files = sorted(
            [os.path.join(tmp_image_dir, f) for f in os.listdir(tmp_image_dir)
             if f.endswith(".png")]
        )

        # Marker values: forward pass writes 100, refine writes 200.
        def fake_propagate(threshold=0.0, progress_callback=None,
                           frame_callback=None, stop_check=None,
                           start_frame_idx=None):
            start = start_frame_idx or 0
            for i in range(start, start + 3):
                mask = np.full((50, 50), fill_value=100, dtype=np.uint8)
                if frame_callback:
                    frame_callback(i, mask)

        def fake_refine(anchor_frame_idx, anchor_mask, overwrite_count,
                        threshold=0.0, frame_callback=None,
                        progress_callback=None, stop_check=None):
            for i in range(int(overwrite_count)):
                if frame_callback:
                    mask = np.full((50, 50), fill_value=200, dtype=np.uint8)
                    frame_callback(i, mask)

        mock_mask_generator.propagate.side_effect = fake_propagate
        mock_mask_generator.refine_early_frames.side_effect = fake_refine

        with tempfile.TemporaryDirectory() as output_dir:
            worker = ProcessingWorker(
                mask_generator=mock_mask_generator,
                image_files=image_files,
                output_dir=output_dir,
                model_cfg="test.yaml",
                checkpoint="test.pt",
                device="cpu",
                points=[[25.0, 25.0]],
                labels=[1],
                threshold=0.0,
                start_frame=1,
                end_frame=3,
                intermediate_format="JPEG (fast)",
                force_reprocess=True,
                refine_enabled=True,
                refine_anchor_frame=2,
                refine_overwrite_count=1,
            )
            worker.run()

            mask_dir = os.path.join(output_dir, "masks")
            # Frame 0 should have been overwritten with value 200 (refine)
            path0 = os.path.join(mask_dir, "mask_000000.tiff")
            assert os.path.exists(path0)
            img0 = cv2.imread(path0, cv2.IMREAD_GRAYSCALE)
            assert img0 is not None and (img0 == 200).all()
            # Frame 1 should still be 100 (forward pass, not refined)
            path1 = os.path.join(mask_dir, "mask_000001.tiff")
            img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
            assert img1 is not None and (img1 == 100).all()

    def test_worker_refine_skipped_if_stopped_mid_forward(
        self, qapp, mock_mask_generator, tmp_image_dir,
    ):
        """If the forward pass is cancelled mid-stream, refine must not run
        (the captured anchor mask may be missing or stale)."""
        from controllers.processing_controller import ProcessingWorker

        image_files = sorted(
            [os.path.join(tmp_image_dir, f) for f in os.listdir(tmp_image_dir)
             if f.endswith(".png")]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            worker = ProcessingWorker(
                mask_generator=mock_mask_generator,
                image_files=image_files,
                output_dir=output_dir,
                model_cfg="test.yaml",
                checkpoint="test.pt",
                device="cpu",
                points=[[25.0, 25.0]],
                labels=[1],
                threshold=0.0,
                start_frame=1,
                end_frame=3,
                intermediate_format="JPEG (fast)",
                force_reprocess=True,
                refine_enabled=True,
                refine_anchor_frame=2,
                refine_overwrite_count=1,
            )
            # Stop the worker before running — conversion stage bails out
            # before forward pass even starts.
            worker.stop()
            worker.run()
            mock_mask_generator.refine_early_frames.assert_not_called()

    def test_worker_png_format(self, qapp, mock_mask_generator, tmp_image_dir):
        """Worker should use PNG conversion when format specifies PNG."""
        from controllers.processing_controller import ProcessingWorker

        image_files = sorted(
            [os.path.join(tmp_image_dir, f) for f in os.listdir(tmp_image_dir)
             if f.endswith(".png")]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            worker = ProcessingWorker(
                mask_generator=mock_mask_generator,
                image_files=image_files,
                output_dir=output_dir,
                model_cfg="test.yaml",
                checkpoint="test.pt",
                device="cpu",
                points=[],
                labels=[],
                threshold=0.0,
                start_frame=1,
                end_frame=3,
                intermediate_format="PNG (lossless)",
                force_reprocess=True,
            )

            worker.run()

            converted_dir = os.path.join(output_dir, "converted_png")
            assert os.path.isdir(converted_dir)
            converted_files = os.listdir(converted_dir)
            assert all(f.endswith(".png") for f in converted_files)

    def test_worker_cancellation_during_conversion(
        self, qapp, mock_mask_generator, tmp_image_dir
    ):
        """Worker should stop when stop flag is set during conversion."""
        from controllers.processing_controller import ProcessingWorker

        image_files = sorted(
            [os.path.join(tmp_image_dir, f) for f in os.listdir(tmp_image_dir)
             if f.endswith(".png")]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            worker = ProcessingWorker(
                mask_generator=mock_mask_generator,
                image_files=image_files,
                output_dir=output_dir,
                model_cfg="test.yaml",
                checkpoint="test.pt",
                device="cpu",
                points=[],
                labels=[],
                threshold=0.0,
                start_frame=1,
                end_frame=3,
                intermediate_format="JPEG (fast)",
                force_reprocess=True,
            )

            # Set stop before running
            worker.stop()
            worker.run()

            # Model should NOT have been initialized
            mock_mask_generator.initialize.assert_not_called()

    def test_worker_emits_finished_on_exception(self, qapp):
        """Worker should emit finished even when an exception occurs."""
        from controllers.processing_controller import ProcessingWorker

        failing_mg = MagicMock()
        failing_mg.initialize = MagicMock(
            side_effect=RuntimeError("GPU out of memory")
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy image so we pass the empty-files check
            img = np.zeros((10, 10, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(tmpdir, "img.png"), img)

            image_files = [os.path.join(tmpdir, "img.png")]

            worker = ProcessingWorker(
                mask_generator=failing_mg,
                image_files=image_files,
                output_dir=tmpdir,
                model_cfg="test.yaml",
                checkpoint="test.pt",
                device="cpu",
                points=[],
                labels=[],
                threshold=0.0,
                start_frame=1,
                end_frame=1,
                intermediate_format="JPEG (fast)",
                force_reprocess=True,
            )

            errors = []
            finished_signals = []
            worker.error.connect(lambda msg: errors.append(msg))
            worker.finished.connect(lambda: finished_signals.append(True))

            worker.run()

            assert len(errors) == 1
            assert "GPU out of memory" in errors[0]
            assert len(finished_signals) == 1


# =============================================================================
# CorrectionWorker Tests
# =============================================================================

class TestCorrectionWorker:
    """Tests for the range-based correction re-propagation worker."""

    def test_correction_uninitialized_model(self, qapp):
        """Should emit error if model not initialized."""
        from controllers.processing_controller import CorrectionWorker

        mg = MagicMock()
        mg.is_initialized = False

        worker = CorrectionWorker(
            mask_generator=mg,
            anchor_frame_idx=5,
            range_start=5,
            range_end=8,
            points=[[10.0, 20.0]],
            labels=[1],
            threshold=0.0,
            image_files=[],
            output_dir="/tmp/out",
            intermediate_format="JPEG (fast)",
        )

        errors = []
        finished_signals = []
        worker.error.connect(lambda msg: errors.append(msg))
        worker.finished.connect(lambda: finished_signals.append(True))

        worker.run()

        assert len(errors) == 1
        assert "not initialized" in errors[0].lower()
        assert len(finished_signals) == 1

    def test_correction_success(self, qapp, mock_mask_generator, tmp_image_dir):
        """Correction worker should re-propagate over the selected range and save masks."""
        from controllers.processing_controller import CorrectionWorker

        mock_mask_generator.is_initialized = True

        image_files = sorted(
            [os.path.join(tmp_image_dir, f) for f in os.listdir(tmp_image_dir)
             if f.endswith(".png")]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            # Create converted_jpeg dir with dummy files so scale factor = 1.0
            converted_dir = os.path.join(output_dir, "converted_jpeg")
            os.makedirs(converted_dir, exist_ok=True)
            for i in range(len(image_files)):
                dummy = np.zeros((50, 50, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(converted_dir, f"{i:06d}.jpg"), dummy)

            worker = CorrectionWorker(
                mask_generator=mock_mask_generator,
                anchor_frame_idx=1,
                range_start=0,
                range_end=2,
                points=[[25.0, 25.0]],
                labels=[1],
                threshold=0.0,
                image_files=image_files,
                output_dir=output_dir,
                intermediate_format="JPEG (fast)",
            )

            frame_signals = []
            finished_signals = []
            worker.frame_processed.connect(
                lambda idx, mask: frame_signals.append(idx)
            )
            worker.finished.connect(lambda: finished_signals.append(True))

            worker.run()

            mock_mask_generator.add_correction.assert_called_once()
            mock_mask_generator.propagate_range.assert_called_once()

            # propagate_range should be called with the new 3-arg signature
            call_kwargs = mock_mask_generator.propagate_range.call_args
            assert call_kwargs.kwargs.get("anchor_frame_idx") == 1
            assert call_kwargs.kwargs.get("range_start") == 0
            assert call_kwargs.kwargs.get("range_end") == 2
            assert call_kwargs.kwargs.get("stop_check") is not None
            assert call_kwargs.kwargs.get("frame_callback") is not None

            # All frames should be inside [range_start, range_end]
            assert all(0 <= idx <= 2 for idx in frame_signals)

            # Masks saved for each emitted frame
            mask_dir = os.path.join(output_dir, "masks")
            mask_files = [f for f in os.listdir(mask_dir) if f.endswith(".tiff")]
            assert len(mask_files) == len(frame_signals)

            # On successful completion, reset_corrections() is called
            mock_mask_generator.reset_corrections.assert_called_once()

            assert len(finished_signals) == 1

    def test_correction_stop_during_propagation(self, qapp, tmp_image_dir):
        """Stop button should halt correction propagation via stop_check,
        and reset_corrections should NOT be called on an aborted run."""
        from controllers.processing_controller import CorrectionWorker

        mg = MagicMock()
        mg.is_initialized = True

        call_count = 0
        def fake_propagate_range(anchor_frame_idx, range_start, range_end,
                                 threshold=0.0, progress_callback=None,
                                 frame_callback=None, stop_check=None):
            nonlocal call_count
            for i in range(range_start, range_end + 1):
                if stop_check and stop_check():
                    break
                mask = np.zeros((50, 50), dtype=np.uint8)
                if frame_callback:
                    frame_callback(i, mask)
                call_count += 1

        mg.propagate_range = MagicMock(side_effect=fake_propagate_range)
        mg.add_correction = MagicMock()
        mg.reset_corrections = MagicMock()

        image_files = sorted(
            [os.path.join(tmp_image_dir, f) for f in os.listdir(tmp_image_dir)
             if f.endswith(".png")]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            converted_dir = os.path.join(output_dir, "converted_jpeg")
            os.makedirs(converted_dir, exist_ok=True)
            for i in range(len(image_files)):
                dummy = np.zeros((50, 50, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(converted_dir, f"{i:06d}.jpg"), dummy)

            worker = CorrectionWorker(
                mask_generator=mg,
                anchor_frame_idx=0,
                range_start=0,
                range_end=2,
                points=[[10.0, 20.0]],
                labels=[1],
                threshold=0.0,
                image_files=image_files,
                output_dir=output_dir,
                intermediate_format="JPEG (fast)",
            )

            worker.stop()
            worker.run()

            assert call_count == 0
            # Stopped runs must NOT call reset_corrections — per todo.md
            mg.reset_corrections.assert_not_called()

    def test_correction_scales_points(self, qapp, tmp_image_dir):
        """Correction should scale points when images are downsampled."""
        from controllers.processing_controller import CorrectionWorker

        mg = MagicMock()
        mg.is_initialized = True
        mg.add_correction = MagicMock()
        mg.propagate_range = MagicMock(return_value=None)
        mg.reset_corrections = MagicMock()

        image_files = sorted(
            [os.path.join(tmp_image_dir, f) for f in os.listdir(tmp_image_dir)
             if f.endswith(".png")]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            # Create converted images at half size (25x25 vs 50x50 originals)
            converted_dir = os.path.join(output_dir, "converted_jpeg")
            os.makedirs(converted_dir, exist_ok=True)
            for i in range(len(image_files)):
                dummy = np.zeros((25, 25, 3), dtype=np.uint8)
                cv2.imwrite(os.path.join(converted_dir, f"{i:06d}.jpg"), dummy)

            worker = CorrectionWorker(
                mask_generator=mg,
                anchor_frame_idx=0,
                range_start=0,
                range_end=2,
                points=[[40.0, 30.0]],  # original coords
                labels=[1],
                threshold=0.0,
                image_files=image_files,
                output_dir=output_dir,
                intermediate_format="JPEG (fast)",
            )

            worker.run()

            call_args = mg.add_correction.call_args
            scaled_points = call_args[0][1]  # second positional arg
            # scale_x = 25/50 = 0.5, scale_y = 25/50 = 0.5
            np.testing.assert_allclose(scaled_points[0, 0], 20.0, atol=0.1)
            np.testing.assert_allclose(scaled_points[0, 1], 15.0, atol=0.1)

    def test_correction_copies_data(self, qapp, mock_mask_generator):
        """Correction worker should copy points/labels at construction."""
        from controllers.processing_controller import CorrectionWorker

        points = [[1.0, 2.0]]
        labels = [1]
        files = ["/a.png"]

        worker = CorrectionWorker(
            mask_generator=mock_mask_generator,
            anchor_frame_idx=0,
            range_start=0,
            range_end=0,
            points=points,
            labels=labels,
            threshold=0.0,
            image_files=files,
            output_dir="/tmp/out",
            intermediate_format="JPEG (fast)",
        )

        points.append([99.0, 99.0])
        labels.append(0)
        files.append("/z.png")

        assert len(worker._points) == 1
        assert len(worker._labels) == 1
        assert len(worker._image_files) == 1


# =============================================================================
# ProcessingController Tests
# =============================================================================

class TestProcessingController:
    """Tests for the ProcessingController orchestrator."""

    def test_initial_state(self, qapp, state, mock_mask_generator):
        from controllers.processing_controller import ProcessingController
        ctrl = ProcessingController(state, mock_mask_generator)
        assert ctrl.is_running is False

    def test_start_correction_uninitialized(self, qapp, state, mock_mask_generator):
        """Should emit error signal if model not initialized."""
        from controllers.processing_controller import ProcessingController

        mock_mask_generator.is_initialized = False
        ctrl = ProcessingController(state, mock_mask_generator)

        errors = []
        ctrl.processing_error.connect(lambda msg: errors.append(msg))

        ctrl.start_correction(
            anchor_frame_idx=0,
            range_start=0,
            range_end=2,
            points=[[10.0, 20.0]],
            labels=[1],
        )
        assert len(errors) == 1
        assert "not initialized" in errors[0].lower()

    def test_stop_when_no_worker(self, qapp, state, mock_mask_generator):
        """Stop should be safe when no worker is running."""
        from controllers.processing_controller import ProcessingController
        ctrl = ProcessingController(state, mock_mask_generator)
        ctrl.stop_processing()  # Should not raise

    def test_controller_signals_connected(self, qapp, state, mock_mask_generator):
        """Controller should forward worker signals."""
        from controllers.processing_controller import ProcessingController

        ctrl = ProcessingController(state, mock_mask_generator)

        # Verify signal types exist
        assert hasattr(ctrl, "progress")
        assert hasattr(ctrl, "frame_processed")
        assert hasattr(ctrl, "processing_finished")
        assert hasattr(ctrl, "processing_error")

    def test_on_finished_clears_worker(self, qapp, state, mock_mask_generator):
        """_on_finished should clear the worker reference."""
        from controllers.processing_controller import ProcessingController
        ctrl = ProcessingController(state, mock_mask_generator)

        # Simulate a worker existing
        ctrl._worker = MagicMock()

        finished_signals = []
        ctrl.processing_finished.connect(lambda: finished_signals.append(True))

        ctrl._on_finished()
        assert ctrl._worker is None
        assert len(finished_signals) == 1


# =============================================================================
# SpatialSmoothWorker Tests
# =============================================================================

class TestSpatialSmoothWorker:
    """Tests for spatial smoothing background worker."""

    def test_spatial_smooth_basic(self, qapp, tmp_mask_dir):
        """Should smooth all masks and save to output dir."""
        from controllers.smoothing_controller import SpatialSmoothWorker

        with tempfile.TemporaryDirectory() as output_dir:
            worker = SpatialSmoothWorker(
                input_dir=tmp_mask_dir,
                output_dir=output_dir,
                num_iterations=5,
                dt=0.1,
                kappa=30.0,
                option=1,
            )

            progress_signals = []
            finished_dirs = []
            worker.progress.connect(
                lambda c, t: progress_signals.append((c, t))
            )
            worker.finished.connect(lambda d: finished_dirs.append(d))

            worker.run()

            # All 5 masks should be processed
            assert len(progress_signals) == 5
            assert progress_signals[-1] == (5, 5)

            # finished should emit with the output directory
            assert len(finished_dirs) == 1
            assert finished_dirs[0] == output_dir

            # Output files should exist
            output_files = [
                f for f in os.listdir(output_dir) if f.endswith(".tiff")
            ]
            assert len(output_files) == 5

    def test_spatial_smooth_empty_dir(self, qapp):
        """Should emit error for empty input directory."""
        from controllers.smoothing_controller import SpatialSmoothWorker

        with tempfile.TemporaryDirectory() as empty_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                worker = SpatialSmoothWorker(
                    input_dir=empty_dir,
                    output_dir=output_dir,
                )

                errors = []
                worker.error.connect(lambda msg: errors.append(msg))

                worker.run()

                assert len(errors) == 1
                assert "No mask files" in errors[0]

    def test_spatial_smooth_cancellation(self, qapp, tmp_mask_dir):
        """Should stop when stop flag is set."""
        from controllers.smoothing_controller import SpatialSmoothWorker

        with tempfile.TemporaryDirectory() as output_dir:
            worker = SpatialSmoothWorker(
                input_dir=tmp_mask_dir,
                output_dir=output_dir,
                num_iterations=5,
            )
            worker.stop()
            worker.run()

            # Output should be empty or very few files
            output_files = os.listdir(output_dir)
            assert len(output_files) == 0

    def test_spatial_smooth_option_2(self, qapp, tmp_mask_dir):
        """Should work with diffusivity option 2."""
        from controllers.smoothing_controller import SpatialSmoothWorker

        with tempfile.TemporaryDirectory() as output_dir:
            worker = SpatialSmoothWorker(
                input_dir=tmp_mask_dir,
                output_dir=output_dir,
                num_iterations=3,
                option=2,
            )

            finished_dirs = []
            worker.finished.connect(lambda d: finished_dirs.append(d))

            worker.run()

            assert len(finished_dirs) == 1
            output_files = [
                f for f in os.listdir(output_dir) if f.endswith(".tiff")
            ]
            assert len(output_files) == 5

    def test_spatial_smooth_with_gaussian(self, qapp, tmp_mask_dir):
        """Should apply per-iteration Gaussian smoothing."""
        from controllers.smoothing_controller import SpatialSmoothWorker

        with tempfile.TemporaryDirectory() as output_dir:
            worker = SpatialSmoothWorker(
                input_dir=tmp_mask_dir,
                output_dir=output_dir,
                num_iterations=3,
                gaussian_sigma=1.0,
            )

            finished_dirs = []
            worker.finished.connect(lambda d: finished_dirs.append(d))

            worker.run()

            assert len(finished_dirs) == 1


# =============================================================================
# TemporalSmoothWorker Tests
# =============================================================================

class TestTemporalSmoothWorker:
    """Tests for temporal smoothing background worker."""

    def test_temporal_smooth_basic(self, qapp, tmp_mask_dir):
        """Should temporally smooth mask sequence and save output."""
        from controllers.smoothing_controller import TemporalSmoothWorker

        with tempfile.TemporaryDirectory() as output_dir:
            worker = TemporalSmoothWorker(
                input_dir=tmp_mask_dir,
                output_dir=output_dir,
                sigma=1.0,
                num_neighbors=1,
            )

            progress_signals = []
            finished_dirs = []
            worker.progress.connect(
                lambda c, t, s: progress_signals.append((c, t, s))
            )
            worker.finished.connect(lambda d: finished_dirs.append(d))

            worker.run()

            assert len(finished_dirs) == 1
            assert finished_dirs[0] == output_dir

            output_files = [
                f for f in os.listdir(output_dir) if f.endswith(".tiff")
            ]
            assert len(output_files) == 5

    def test_temporal_smooth_empty_dir(self, qapp):
        """Should emit error for empty input directory."""
        from controllers.smoothing_controller import TemporalSmoothWorker

        with tempfile.TemporaryDirectory() as empty_dir:
            with tempfile.TemporaryDirectory() as output_dir:
                worker = TemporalSmoothWorker(
                    input_dir=empty_dir,
                    output_dir=output_dir,
                )

                errors = []
                worker.error.connect(lambda msg: errors.append(msg))

                worker.run()

                assert len(errors) == 1
                assert "No mask files" in errors[0]

    def test_temporal_smooth_cancellation(self, qapp, tmp_mask_dir):
        """Should stop when stop flag is set during loading."""
        from controllers.smoothing_controller import TemporalSmoothWorker

        with tempfile.TemporaryDirectory() as output_dir:
            worker = TemporalSmoothWorker(
                input_dir=tmp_mask_dir,
                output_dir=output_dir,
            )
            worker.stop()
            worker.run()

            output_files = os.listdir(output_dir)
            assert len(output_files) == 0

    def test_temporal_smooth_with_variance_threshold(self, qapp, tmp_mask_dir):
        """Should work with explicit variance threshold."""
        from controllers.smoothing_controller import TemporalSmoothWorker

        with tempfile.TemporaryDirectory() as output_dir:
            worker = TemporalSmoothWorker(
                input_dir=tmp_mask_dir,
                output_dir=output_dir,
                sigma=1.0,
                variance_threshold=50000.0,
            )

            finished_dirs = []
            worker.finished.connect(lambda d: finished_dirs.append(d))

            worker.run()

            assert len(finished_dirs) == 1

    def test_temporal_smooth_progress_callback(self, qapp, tmp_mask_dir):
        """Should emit progress signals from the temporal pipeline."""
        from controllers.smoothing_controller import TemporalSmoothWorker

        with tempfile.TemporaryDirectory() as output_dir:
            worker = TemporalSmoothWorker(
                input_dir=tmp_mask_dir,
                output_dir=output_dir,
                sigma=1.0,
            )

            progress_signals = []
            worker.progress.connect(
                lambda c, t, s: progress_signals.append(s)
            )

            worker.run()

            # Should have progress from temporal_smooth_sequence callback
            assert len(progress_signals) > 0


# =============================================================================
# SmoothingController Tests
# =============================================================================

class TestSmoothingController:
    """Tests for the SmoothingController orchestrator."""

    def test_initial_state(self, qapp):
        from controllers.smoothing_controller import SmoothingController
        ctrl = SmoothingController()
        assert ctrl.is_running is False

    def test_stop_when_no_worker(self, qapp):
        """Stop should be safe when no worker is running."""
        from controllers.smoothing_controller import SmoothingController
        ctrl = SmoothingController()
        ctrl.stop()  # Should not raise

    def test_controller_spatial_signals(self, qapp):
        """Controller should have expected signals."""
        from controllers.smoothing_controller import SmoothingController
        ctrl = SmoothingController()
        assert hasattr(ctrl, "progress")
        assert hasattr(ctrl, "smoothing_finished")
        assert hasattr(ctrl, "smoothing_error")

    def test_on_finished_clears_worker(self, qapp):
        """_on_finished should clear the worker reference and emit signal."""
        from controllers.smoothing_controller import SmoothingController
        ctrl = SmoothingController()
        ctrl._worker = MagicMock()

        finished_signals = []
        ctrl.smoothing_finished.connect(lambda d: finished_signals.append(d))

        ctrl._on_finished("/some/output/dir")

        assert ctrl._worker is None
        assert len(finished_signals) == 1
        assert finished_signals[0] == "/some/output/dir"

    def test_spatial_start_creates_worker(self, qapp, tmp_mask_dir):
        """start_spatial should create and start a SpatialSmoothWorker."""
        from controllers.smoothing_controller import SmoothingController

        with tempfile.TemporaryDirectory() as output_dir:
            ctrl = SmoothingController()

            finished_signals = []
            ctrl.smoothing_finished.connect(
                lambda d: finished_signals.append(d)
            )

            ctrl.start_spatial(
                input_dir=tmp_mask_dir,
                output_dir=output_dir,
                num_iterations=2,
            )

            # Worker should have been created
            assert ctrl._worker is not None

            # Wait for it to finish
            ctrl._worker.wait(10000)

            # Give event loop time to process
            qapp.processEvents()

    def test_temporal_start_creates_worker(self, qapp, tmp_mask_dir):
        """start_temporal should create and start a TemporalSmoothWorker."""
        from controllers.smoothing_controller import SmoothingController

        with tempfile.TemporaryDirectory() as output_dir:
            ctrl = SmoothingController()

            ctrl.start_temporal(
                input_dir=tmp_mask_dir,
                output_dir=output_dir,
                sigma=1.0,
            )

            assert ctrl._worker is not None

            ctrl._worker.wait(10000)
            qapp.processEvents()


# =============================================================================
# Integration-level tests
# =============================================================================

class TestProcessingIntegration:
    """Higher-level integration tests combining controller + state."""

    def test_controller_reads_state_config(self, qapp, state, mock_mask_generator):
        """ProcessingController should read model config from state."""
        from controllers.processing_controller import ProcessingController

        state.set_model_name("SAM2 Hiera Tiny")
        state.set_device("cpu")
        state.set_threshold(0.5)

        ctrl = ProcessingController(state, mock_mask_generator)

        cfg, ckpt = state.get_model_config()
        assert cfg == "sam2.1_hiera_t.yaml"
        assert "tiny" in ckpt

    def test_worker_skip_conversion_if_exists(
        self, qapp, mock_mask_generator, tmp_image_dir
    ):
        """Worker should skip conversion when file exists and not force."""
        from controllers.processing_controller import ProcessingWorker

        image_files = sorted(
            [os.path.join(tmp_image_dir, f) for f in os.listdir(tmp_image_dir)
             if f.endswith(".png")]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            # Pre-create converted directory with files
            converted_dir = os.path.join(output_dir, "converted_jpeg")
            os.makedirs(converted_dir, exist_ok=True)
            for i in range(len(image_files)):
                dummy_path = os.path.join(converted_dir, f"{i:06d}.jpg")
                cv2.imwrite(
                    dummy_path,
                    np.zeros((10, 10, 3), dtype=np.uint8),
                )

            worker = ProcessingWorker(
                mask_generator=mock_mask_generator,
                image_files=image_files,
                output_dir=output_dir,
                model_cfg="test.yaml",
                checkpoint="test.pt",
                device="cpu",
                points=[],
                labels=[],
                threshold=0.0,
                start_frame=1,
                end_frame=3,
                intermediate_format="JPEG (fast)",
                force_reprocess=False,  # Should skip existing
            )

            progress_msgs = []
            worker.progress.connect(
                lambda c, t, m: progress_msgs.append(m)
            )

            worker.run()

            # No conversion progress should be emitted (all skipped)
            conversion_msgs = [
                m for m in progress_msgs if "Converting" in m
            ]
            assert len(conversion_msgs) == 0
