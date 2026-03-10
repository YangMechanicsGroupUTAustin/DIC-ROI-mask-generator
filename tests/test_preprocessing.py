"""Tests for image preprocessing pipeline."""
import numpy as np
import pytest

from core.preprocessing import (
    PreprocessingConfig,
    adjust_brightness,
    adjust_contrast,
    adjust_gain,
    apply_clahe,
    apply_pipeline,
    bilateral_filter,
    binary_threshold,
    clip_min_max,
    gaussian_smooth,
)


class TestPreprocessingConfig:
    def test_default_is_identity(self):
        config = PreprocessingConfig()
        assert config.is_identity()

    def test_non_default_not_identity(self):
        assert not PreprocessingConfig(brightness=10).is_identity()
        assert not PreprocessingConfig(gain=2.0).is_identity()
        assert not PreprocessingConfig(clahe_enabled=True).is_identity()

    def test_frozen(self):
        config = PreprocessingConfig()
        with pytest.raises(AttributeError):
            config.gain = 2.0  # type: ignore


class TestGain:
    def test_gain_doubles(self):
        img = np.full((10, 10, 3), 50, dtype=np.uint8)
        result = adjust_gain(img, factor=2.0)
        assert np.all(result == 100)

    def test_gain_clamp(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = adjust_gain(img, factor=2.0)
        assert np.all(result == 255)

    def test_unity_gain_no_change(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = adjust_gain(img, factor=1.0)
        np.testing.assert_array_equal(result, img)

    def test_does_not_mutate_input(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        original = img.copy()
        adjust_gain(img, factor=2.0)
        np.testing.assert_array_equal(img, original)


class TestBrightness:
    def test_increase_brightness(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        result = adjust_brightness(img, offset=50)
        assert np.all(result == 150)

    def test_brightness_clamp_upper(self):
        img = np.full((10, 10, 3), 200, dtype=np.uint8)
        result = adjust_brightness(img, offset=100)
        assert np.all(result == 255)

    def test_brightness_clamp_lower(self):
        img = np.full((10, 10, 3), 50, dtype=np.uint8)
        result = adjust_brightness(img, offset=-100)
        assert np.all(result == 0)

    def test_zero_offset_no_change(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = adjust_brightness(img, offset=0)
        np.testing.assert_array_equal(result, img)

    def test_does_not_mutate_input(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        original = img.copy()
        adjust_brightness(img, offset=50)
        np.testing.assert_array_equal(img, original)


class TestContrast:
    def test_increase_contrast(self):
        img = np.full((10, 10, 3), 128, dtype=np.uint8)
        result = adjust_contrast(img, factor=2.0)
        assert result.dtype == np.uint8

    def test_zero_contrast(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = adjust_contrast(img, factor=0.0)
        assert np.all(result == 128)

    def test_unity_factor_no_change(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = adjust_contrast(img, factor=1.0)
        np.testing.assert_array_equal(result, img)


class TestMinMaxClip:
    def test_clip_range(self):
        img = np.array([[[0, 50, 100], [150, 200, 255]]], dtype=np.uint8)
        result = clip_min_max(img, min_val=50, max_val=200)
        assert result.dtype == np.uint8
        assert result.min() == 0
        assert result.max() == 255

    def test_full_range_no_change(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = clip_min_max(img, min_val=0, max_val=255)
        np.testing.assert_array_equal(result, img)


class TestCLAHE:
    def test_clahe_runs(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = apply_clahe(img, clip_limit=2.0, tile_size=8)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestGaussianSmooth:
    def test_smooth_reduces_noise(self):
        rng = np.random.default_rng(42)
        img = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
        result = gaussian_smooth(img, sigma=2.0)
        assert result.std() < img.std()

    def test_zero_sigma_no_change(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        result = gaussian_smooth(img, sigma=0.0)
        np.testing.assert_array_equal(result, img)


class TestBilateralFilter:
    def test_bilateral_runs(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = bilateral_filter(img, d=9, sigma_color=75, sigma_space=75)
        assert result.shape == img.shape
        assert result.dtype == np.uint8


class TestBinaryThreshold:
    def test_fixed_threshold(self):
        img = np.array([[[50, 50, 50], [200, 200, 200]]], dtype=np.uint8)
        result = binary_threshold(img, threshold=127, method="fixed")
        assert result[0, 0, 0] == 0
        assert result[0, 1, 0] == 255

    def test_otsu_threshold(self):
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = binary_threshold(img, method="otsu")
        unique = np.unique(result)
        assert len(unique) <= 2


class TestApplyPipeline:
    def test_empty_pipeline(self):
        img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        config = PreprocessingConfig()
        result = apply_pipeline(img, config)
        np.testing.assert_array_equal(result, img)

    def test_pipeline_with_brightness_and_contrast(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        config = PreprocessingConfig(brightness=50, contrast=1.5)
        result = apply_pipeline(img, config)
        assert result.dtype == np.uint8
        assert not np.array_equal(result, img)

    def test_pipeline_immutability(self):
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        original = img.copy()
        config = PreprocessingConfig(brightness=50, gain=2.0)
        apply_pipeline(img, config)
        np.testing.assert_array_equal(img, original)

    def test_full_pipeline(self):
        img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
        config = PreprocessingConfig(
            gain=1.2,
            brightness=10,
            contrast=1.3,
            clip_min=20,
            clip_max=240,
            clahe_enabled=True,
            gaussian_sigma=1.0,
        )
        result = apply_pipeline(img, config)
        assert result.shape == img.shape
        assert result.dtype == np.uint8
