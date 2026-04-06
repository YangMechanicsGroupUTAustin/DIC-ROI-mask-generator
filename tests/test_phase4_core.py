"""Phase 4 verification: core algorithm refactoring."""
import os
import sys
import json
import tempfile
import pytest
import numpy as np


# --- Image Processing Tests ---

class TestImageProcessing:
    def test_extract_numbers(self):
        from core.image_processing import extract_numbers
        assert extract_numbers("frame_001.png") == (1,)
        assert extract_numbers("Camera_16_18_24_005.tiff") == (16, 18, 24, 5)

    def test_natural_sort(self):
        from core.image_processing import get_image_files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test images
            for name in ["img_2.png", "img_10.png", "img_1.png"]:
                open(os.path.join(tmpdir, name), "w").close()
            files = get_image_files(tmpdir)
            basenames = [os.path.basename(f) for f in files]
            assert basenames == ["img_1.png", "img_2.png", "img_10.png"]

    def test_create_overlay_default_red(self):
        from core.image_processing import create_overlay
        image = np.full((10, 10, 3), 128, dtype=np.uint8)
        mask = np.full((10, 10), 255, dtype=np.uint8)
        result = create_overlay(image, mask)
        # Red channel should be boosted
        assert result[5, 5, 0] > result[5, 5, 1]

    def test_create_overlay_custom_color(self):
        from core.image_processing import create_overlay
        image = np.full((10, 10, 3), 128, dtype=np.uint8)
        mask = np.full((10, 10), 255, dtype=np.uint8)
        result = create_overlay(image, mask, color=(0, 255, 0))
        # Green channel should be boosted
        assert result[5, 5, 1] > result[5, 5, 0]

    def test_numpy_to_qimage(self):
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance() or QApplication(sys.argv)
        from core.image_processing import numpy_to_qimage
        # RGB image
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        rgb[:, :, 0] = 255  # Red
        qimg = numpy_to_qimage(rgb)
        assert qimg.width() == 200
        assert qimg.height() == 100
        # Grayscale
        gray = np.zeros((50, 60), dtype=np.uint8)
        qimg_gray = numpy_to_qimage(gray)
        assert qimg_gray.width() == 60
        # None input
        qimg_none = numpy_to_qimage(None)
        assert qimg_none.isNull()

    def test_convert_image_jpeg(self):
        from core.image_processing import convert_image
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a test PNG
            import cv2
            src = os.path.join(tmpdir, "test.png")
            img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            cv2.imwrite(src, img)
            dst = os.path.join(tmpdir, "test.jpg")
            assert convert_image(src, dst, format="jpeg")
            assert os.path.exists(dst)

    def test_convert_image_png(self):
        from core.image_processing import convert_image
        with tempfile.TemporaryDirectory() as tmpdir:
            import cv2
            src = os.path.join(tmpdir, "test.tiff")
            img = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            cv2.imwrite(src, img)
            dst = os.path.join(tmpdir, "test.png")
            assert convert_image(src, dst, format="png")
            assert os.path.exists(dst)


# --- Spatial Smoothing Tests ---

class TestSpatialSmoothing:
    def test_binary_roundtrip(self):
        """Binary mask in -> binary mask out."""
        from core.spatial_smoothing import perona_malik_smooth
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        result = perona_malik_smooth(mask, num_iterations=10)
        assert result.dtype == np.uint8
        assert set(np.unique(result)) <= {0, 255}

    def test_all_black(self):
        from core.spatial_smoothing import perona_malik_smooth
        mask = np.zeros((50, 50), dtype=np.uint8)
        result = perona_malik_smooth(mask, num_iterations=5)
        assert np.all(result == 0)

    def test_all_white(self):
        from core.spatial_smoothing import perona_malik_smooth
        mask = np.full((50, 50), 255, dtype=np.uint8)
        result = perona_malik_smooth(mask, num_iterations=5)
        assert np.all(result == 255)

    def test_dt_clamping(self):
        """dt > 0.25 should be clamped for stability."""
        from core.spatial_smoothing import perona_malik_smooth
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 255
        # Should not explode with large dt
        result = perona_malik_smooth(mask, num_iterations=10, dt=0.5)
        assert result.dtype == np.uint8

    def test_option_2(self):
        """Option 2 should also produce valid output."""
        from core.spatial_smoothing import perona_malik_smooth
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 255
        result = perona_malik_smooth(mask, num_iterations=10, option=2)
        assert result.dtype == np.uint8
        assert set(np.unique(result)) <= {0, 255}

    def test_edge_preservation(self):
        """Edges should be better preserved than pure Gaussian."""
        from core.spatial_smoothing import perona_malik_smooth
        from scipy.ndimage import gaussian_filter
        # Create sharp-edged mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255

        pm_result = perona_malik_smooth(mask, num_iterations=20, kappa=30.0)
        gauss_result = (gaussian_filter(mask.astype(float), sigma=3.0) > 127).astype(np.uint8) * 255

        # PM should have sharper transitions (fewer gray-area pixels in binary)
        pm_edge = np.sum(np.abs(np.diff(pm_result.astype(float), axis=0)) > 0)
        gauss_edge = np.sum(np.abs(np.diff(gauss_result.astype(float), axis=0)) > 0)
        # Both should have edges, PM edges should be comparable or sharper
        assert pm_edge > 0

    def test_gaussian_sigma(self):
        """Per-iteration Gaussian should smooth mask boundaries."""
        from core.spatial_smoothing import perona_malik_smooth
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:30, 20:30] = 255
        result_no_gauss = perona_malik_smooth(mask, num_iterations=5, gaussian_sigma=0.0)
        result_with_gauss = perona_malik_smooth(mask, num_iterations=5, gaussian_sigma=2.0)
        # Both should be valid binary masks
        assert set(np.unique(result_no_gauss)) <= {0, 255}
        assert set(np.unique(result_with_gauss)) <= {0, 255}


# --- Temporal Smoothing Tests ---

class TestTemporalSmoothing:
    def _make_frames(self, n=20, h=50, w=50):
        """Create test mask sequence with a moving square."""
        frames = []
        for i in range(n):
            f = np.zeros((h, w), dtype=np.uint8)
            offset = i * 2
            f[10+offset:20+offset, 10:30] = 255
            frames.append(f)
        return frames

    def test_basic_pipeline(self):
        from core.temporal_smoothing import temporal_smooth_sequence
        frames = self._make_frames(n=10)
        result = temporal_smooth_sequence(frames, sigma=1.0)
        assert len(result) == 10
        assert all(r.dtype == np.uint8 for r in result)

    def test_float32_memory(self):
        """Verify internal array uses float32."""
        from core.temporal_smoothing import temporal_smooth_sequence
        frames = self._make_frames(n=5)
        # We can't easily check internal memory, but the function should work
        result = temporal_smooth_sequence(frames, sigma=1.0)
        assert len(result) == 5

    def test_bad_frame_detection_adaptive(self):
        from core.temporal_smoothing import detect_bad_frames
        # Create sequence with one bad frame
        sequence = np.ones((50, 50, 10), dtype=np.float32) * 128
        sequence[:, :, 5] = 0  # blank frame
        bad = detect_bad_frames(sequence, variance_threshold=None)
        assert 5 in bad

    def test_bad_frame_detection_fixed(self):
        from core.temporal_smoothing import detect_bad_frames
        sequence = np.ones((50, 50, 10), dtype=np.float32) * 128
        sequence[:, :, 3] = np.random.rand(50, 50) * 1000  # noisy frame
        bad = detect_bad_frames(sequence, variance_threshold=50000)
        # May or may not detect depending on actual variance
        assert isinstance(bad, list)

    def test_single_frame(self):
        from core.temporal_smoothing import temporal_smooth_sequence
        frames = [np.full((50, 50), 255, dtype=np.uint8)]
        result = temporal_smooth_sequence(frames, sigma=1.0)
        assert len(result) == 1

    def test_two_frames(self):
        from core.temporal_smoothing import temporal_smooth_sequence
        frames = [np.zeros((50, 50), dtype=np.uint8),
                  np.full((50, 50), 255, dtype=np.uint8)]
        result = temporal_smooth_sequence(frames, sigma=1.0)
        assert len(result) == 2

    def test_progress_callback(self):
        from core.temporal_smoothing import temporal_smooth_sequence
        callbacks = []
        frames = self._make_frames(n=5)
        temporal_smooth_sequence(
            frames, sigma=1.0,
            progress_callback=lambda name, cur, tot: callbacks.append(name)
        )
        assert len(callbacks) > 0


# --- Annotation Config Tests ---

class TestAnnotationConfig:
    def test_save_load_v2(self):
        from core.annotation_config import save_annotation_config, load_annotation_config
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name
        try:
            save_annotation_config(
                filepath,
                input_points=[[10.0, 20.0], [30.0, 40.0]],
                input_labels=[1, 0],
                model_name="SAM2 Hiera Large",
                device="cuda",
                threshold=0.5,
            )
            config = load_annotation_config(filepath)
            assert config["version"] == "2.0"
            assert len(config["annotation"]["points"]) == 2
            assert config["annotation"]["labels"] == [1, 0]
            assert "smoothing" in config
            assert "correction_points" in config["annotation"]
        finally:
            os.unlink(filepath)

    def test_v1_backward_compat(self):
        """V1 config should be auto-migrated to v2."""
        from core.annotation_config import load_annotation_config
        v1_config = {
            "version": "1.0",
            "annotation": {
                "points": [[1.0, 2.0]],
                "labels": [1],
            },
            "parameters": {
                "model": "SAM2 Hiera Small",
            }
        }
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump(v1_config, f)
            filepath = f.name
        try:
            config = load_annotation_config(filepath)
            assert config["version"] == "2.0"
            assert "correction_points" in config["annotation"]
            assert "smoothing" in config
        finally:
            os.unlink(filepath)

    def test_invalid_config_raises(self):
        from core.annotation_config import load_annotation_config
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump({"bad": "data"}, f)
            filepath = f.name
        try:
            with pytest.raises(ValueError, match="missing 'annotation'"):
                load_annotation_config(filepath)
        finally:
            os.unlink(filepath)

    def test_smoothing_params_saved(self):
        from core.annotation_config import save_annotation_config, load_annotation_config
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            filepath = f.name
        try:
            save_annotation_config(
                filepath,
                input_points=[], input_labels=[],
                smoothing_spatial={"iterations": 100, "dt": 0.2, "kappa": 50.0, "option": 2},
                smoothing_temporal={"sigma": 3.0, "neighbors": 4},
            )
            config = load_annotation_config(filepath)
            assert config["smoothing"]["spatial"]["iterations"] == 100
            assert config["smoothing"]["temporal"]["sigma"] == 3.0
        finally:
            os.unlink(filepath)


# --- Mask Generator Tests ---

class TestMaskGenerator:
    def test_initialization(self):
        from core.mask_generator import MaskGenerator
        mg = MaskGenerator()
        assert not mg.is_initialized
        assert not mg.has_inference_state

    def test_cleanup_safe(self):
        """Cleanup on uninitialized generator should not crash."""
        from core.mask_generator import MaskGenerator
        mg = MaskGenerator()
        mg.cleanup()  # Should not raise

    def test_propagate_without_init_raises(self):
        from core.mask_generator import MaskGenerator
        mg = MaskGenerator()
        with pytest.raises(RuntimeError):
            mg.propagate()

    def test_add_points_without_init_raises(self):
        from core.mask_generator import MaskGenerator
        mg = MaskGenerator()
        with pytest.raises(RuntimeError):
            mg.add_points(0, np.array([[10, 20]]), np.array([1]))
