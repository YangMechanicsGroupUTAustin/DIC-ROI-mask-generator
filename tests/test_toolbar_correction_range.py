"""Phase C tests: range spin boxes on the correction toolbar."""

import sys

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication(sys.argv)


@pytest.fixture
def toolbar(qapp):
    from gui.panels.toolbar import Toolbar
    return Toolbar()


# ---------------------------------------------------------- Button labels

class TestButtonLabels:
    def test_add_correction_button_renamed(self, toolbar):
        assert toolbar._add_correction_btn.text() == "Add Correction"

    def test_apply_correction_button_renamed(self, toolbar):
        assert toolbar._apply_correction_btn.text() == "Re-run Range"


# ---------------------------------------------------------- Spin boxes

class TestRangeSpinBoxes:
    def test_range_start_spin_exists(self, toolbar):
        assert hasattr(toolbar, "_range_start_spin")
        assert toolbar._range_start_spin.minimum() == 1

    def test_range_end_spin_exists(self, toolbar):
        assert hasattr(toolbar, "_range_end_spin")
        assert toolbar._range_end_spin.minimum() == 1

    def test_spin_boxes_initially_disabled(self, toolbar):
        # Correction workflow is inactive on startup → disabled
        assert not toolbar._range_start_spin.isEnabled()
        assert not toolbar._range_end_spin.isEnabled()


# ---------------------------------------------------------- Public API

class TestPublicAPI:
    def test_set_correction_frame_count_updates_max(self, toolbar):
        toolbar.set_correction_frame_count(50)
        assert toolbar._range_start_spin.maximum() == 50
        assert toolbar._range_end_spin.maximum() == 50

    def test_set_correction_range_updates_values(self, toolbar):
        toolbar.set_correction_frame_count(100)
        toolbar.set_correction_range(10, 25)
        assert toolbar._range_start_spin.value() == 10
        assert toolbar._range_end_spin.value() == 25

    def test_set_correction_range_does_not_re_emit(self, toolbar):
        """Programmatic set should not cause the toolbar to echo back a signal."""
        toolbar.set_correction_frame_count(100)
        emissions: list[tuple] = []
        toolbar.correction_range_changed.connect(
            lambda s, e: emissions.append((s, e))
        )
        toolbar.set_correction_range(10, 25)
        assert emissions == []

    def test_set_correction_range_enabled_toggles(self, toolbar):
        toolbar.set_correction_range_enabled(True)
        assert toolbar._range_start_spin.isEnabled()
        assert toolbar._range_end_spin.isEnabled()
        toolbar.set_correction_range_enabled(False)
        assert not toolbar._range_start_spin.isEnabled()
        assert not toolbar._range_end_spin.isEnabled()


# ---------------------------------------------------------- User-edit signal

class TestRangeChangedSignal:
    def test_signal_defined(self, toolbar):
        assert hasattr(toolbar, "correction_range_changed")

    def test_user_edit_emits_signal(self, toolbar):
        toolbar.set_correction_frame_count(100)
        toolbar.set_correction_range(5, 10)

        emissions: list[tuple] = []
        toolbar.correction_range_changed.connect(
            lambda s, e: emissions.append((s, e))
        )

        # Simulate a user changing a spin box directly (not via set_correction_range)
        toolbar._range_end_spin.setValue(30)

        assert len(emissions) >= 1
        last = emissions[-1]
        assert last == (5, 30)

    def test_user_edit_of_start_emits_signal(self, toolbar):
        toolbar.set_correction_frame_count(100)
        toolbar.set_correction_range(5, 10)

        emissions: list[tuple] = []
        toolbar.correction_range_changed.connect(
            lambda s, e: emissions.append((s, e))
        )

        toolbar._range_start_spin.setValue(3)
        assert len(emissions) >= 1
        assert emissions[-1] == (3, 10)
