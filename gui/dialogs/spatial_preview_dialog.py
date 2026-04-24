"""Interactive spatial smoothing preview dialog.

Lets the user switch between strength presets and frames,
showing a side-by-side before/after comparison with pixel-diff stats.
Single-frame smoothing is fast enough for real-time interaction.
"""

import os
from typing import Optional

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.image_processing import imread_safe, numpy_to_qimage
from core.spatial_smoothing import perona_malik_smooth
from gui.theme import Colors, Fonts


# Preset definitions (same as sidebar — kept in sync manually)
_PRESETS = {
    "Light (5 iterations)": {"iterations": 5, "gaussian_sigma": 2.0},
    "Moderate (10 iterations)": {"iterations": 10, "gaussian_sigma": 2.0},
    "Standard (20 iterations)": {"iterations": 20, "gaussian_sigma": 2.0},
    "Strong (50 iterations)": {"iterations": 50, "gaussian_sigma": 2.0},
}
_PRESET_NAMES = list(_PRESETS.keys())
_DEFAULT_PRESET = "Standard (20 iterations)"


class SpatialPreviewDialog(QDialog):
    """Side-by-side spatial smoothing preview with live preset/frame switching."""

    def __init__(
        self,
        masks_dir: str,
        initial_frame: int = 0,
        initial_preset: str = _DEFAULT_PRESET,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Spatial Smoothing Preview")
        self.setMinimumSize(800, 500)

        self._masks_dir = masks_dir
        self._mask_files = sorted(
            f for f in os.listdir(masks_dir)
            if f.lower().endswith((".tiff", ".tif", ".png"))
        )
        self._cache: dict[tuple[int, str], np.ndarray] = {}

        # --- Layout ---
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # ── Controls row ──
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(16)

        # Strength selector
        ctrl_row.addWidget(self._make_label("Strength:"))
        self._strength_combo = QComboBox()
        self._strength_combo.addItems(_PRESET_NAMES)
        idx = _PRESET_NAMES.index(initial_preset) if initial_preset in _PRESET_NAMES else 2
        self._strength_combo.setCurrentIndex(idx)
        self._strength_combo.currentIndexChanged.connect(self._on_setting_changed)
        ctrl_row.addWidget(self._strength_combo)

        ctrl_row.addSpacing(24)

        # Frame selector
        ctrl_row.addWidget(self._make_label("Frame:"))
        self._frame_spin = QSpinBox()
        self._frame_spin.setMinimum(1)
        self._frame_spin.setMaximum(max(1, len(self._mask_files)))
        self._frame_spin.setValue(min(initial_frame + 1, len(self._mask_files)))
        self._frame_spin.valueChanged.connect(self._on_setting_changed)
        ctrl_row.addWidget(self._frame_spin)

        self._frame_total_label = QLabel(f"/ {len(self._mask_files)}")
        self._frame_total_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_SM}px;"
        )
        ctrl_row.addWidget(self._frame_total_label)

        # Prev / Next buttons
        prev_btn = QPushButton("<")
        prev_btn.setFixedWidth(30)
        prev_btn.clicked.connect(lambda: self._frame_spin.setValue(self._frame_spin.value() - 1))
        next_btn = QPushButton(">")
        next_btn.setFixedWidth(30)
        next_btn.clicked.connect(lambda: self._frame_spin.setValue(self._frame_spin.value() + 1))
        ctrl_row.addWidget(prev_btn)
        ctrl_row.addWidget(next_btn)

        ctrl_row.addStretch()
        root.addLayout(ctrl_row)

        # ── Image row ──
        img_row = QHBoxLayout()
        img_row.setSpacing(12)

        # Original column
        orig_col = QVBoxLayout()
        orig_title = QLabel("Original")
        orig_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        orig_title.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-weight: bold; "
            f"font-size: {Fonts.SIZE_BASE}px;"
        )
        orig_col.addWidget(orig_title)
        self._orig_img_label = QLabel()
        self._orig_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._orig_img_label.setMinimumHeight(300)
        orig_col.addWidget(self._orig_img_label, 1)
        img_row.addLayout(orig_col)

        # Smoothed column
        smooth_col = QVBoxLayout()
        smooth_title = QLabel("Smoothed")
        smooth_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        smooth_title.setStyleSheet(
            f"color: {Colors.TEXT_PRIMARY}; font-weight: bold; "
            f"font-size: {Fonts.SIZE_BASE}px;"
        )
        smooth_col.addWidget(smooth_title)
        self._smooth_img_label = QLabel()
        self._smooth_img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._smooth_img_label.setMinimumHeight(300)
        smooth_col.addWidget(self._smooth_img_label, 1)
        img_row.addLayout(smooth_col)

        root.addLayout(img_row, 1)

        # ── Stats row ──
        self._stats_label = QLabel()
        self._stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._stats_label.setStyleSheet(
            f"color: {Colors.TEXT_MUTED}; font-size: {Fonts.SIZE_SM}px;"
        )
        root.addWidget(self._stats_label)

        # ── Close button ──
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        close_btn.setFixedWidth(100)
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_row.addWidget(close_btn)
        btn_row.addStretch()
        root.addLayout(btn_row)

        # Initial render
        self._update_preview()

    # ── Helpers ──

    @staticmethod
    def _make_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_SM}px;"
        )
        return lbl

    def _current_preset_name(self) -> str:
        return self._strength_combo.currentText()

    def _current_frame_idx(self) -> int:
        return self._frame_spin.value() - 1  # 0-based

    def _load_mask(self, frame_idx: int) -> Optional[np.ndarray]:
        if frame_idx < 0 or frame_idx >= len(self._mask_files):
            return None
        path = os.path.join(self._masks_dir, self._mask_files[frame_idx])
        return imread_safe(path, cv2.IMREAD_GRAYSCALE)

    def _smooth_mask(self, mask: np.ndarray, preset_name: str) -> np.ndarray:
        """Run spatial smoothing with the given preset. Uses a cache."""
        frame_idx = self._current_frame_idx()
        key = (frame_idx, preset_name)
        if key in self._cache:
            return self._cache[key]

        preset = _PRESETS.get(preset_name, _PRESETS[_DEFAULT_PRESET])
        result = perona_malik_smooth(
            mask,
            num_iterations=preset["iterations"],
            dt=0.1,
            kappa=30.0,
            option=1,
            gaussian_sigma=preset["gaussian_sigma"],
        )
        self._cache[key] = result
        return result

    def _set_pixmap(self, label: QLabel, img: np.ndarray) -> None:
        """Scale image to fit the label area and set as pixmap."""
        qimg = numpy_to_qimage(img)
        max_h = max(200, label.height() - 10)
        max_w = max(200, label.width() - 10)
        pix = QPixmap.fromImage(qimg).scaled(
            max_w, max_h,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pix)

    # ── Update logic ──

    def _on_setting_changed(self) -> None:
        self._update_preview()

    def _update_preview(self) -> None:
        frame_idx = self._current_frame_idx()
        preset_name = self._current_preset_name()

        mask = self._load_mask(frame_idx)
        if mask is None:
            self._stats_label.setText("Failed to load mask")
            return

        smoothed = self._smooth_mask(mask, preset_name)

        self._set_pixmap(self._orig_img_label, mask)
        self._set_pixmap(self._smooth_img_label, smoothed)

        # Stats
        diff_count = int(np.count_nonzero(mask != smoothed))
        total_px = mask.shape[0] * mask.shape[1]
        pct = diff_count / total_px * 100 if total_px > 0 else 0
        preset = _PRESETS.get(preset_name, {})
        self._stats_label.setText(
            f"Changed: {diff_count:,} / {total_px:,} px ({pct:.2f}%)  |  "
            f"Iterations: {preset.get('iterations', '?')}  |  "
            f"Sigma: {preset.get('gaussian_sigma', '?')}"
        )

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Re-render images when dialog is resized."""
        super().resizeEvent(event)
        self._update_preview()
