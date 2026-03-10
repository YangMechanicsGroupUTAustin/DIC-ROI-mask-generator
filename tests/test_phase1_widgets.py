"""Phase 1 verification: theme, icons, and widgets load correctly."""

import sys

import pytest
from PyQt6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    return app


def test_theme_loads(qapp):
    from gui.theme import Colors, Fonts, generate_stylesheet

    ss = generate_stylesheet()
    assert len(ss) > 1000
    assert Colors.PRIMARY == "#6366f1"
    assert Fonts.FAMILY == "Segoe UI"


def test_icons_render(qapp):
    from gui.icons import get_icon, get_pixmap

    icon = get_icon("cursor")
    assert not icon.isNull()
    pixmap = get_pixmap("brain", "#ffffff", 24)
    assert not pixmap.isNull()
    assert pixmap.width() == 24


def test_collapsible_section(qapp):
    from gui.widgets.collapsible_section import CollapsibleSection

    section = CollapsibleSection("Test Section", "settings", default_open=True)
    assert section.is_open()
    section.set_open(False)
    assert not section.is_open()


def test_select_field(qapp):
    from gui.widgets.select_field import SelectField

    sf = SelectField("Device", ["CUDA", "CPU", "MPS"], default="CUDA")
    assert sf.value() == "CUDA"
    sf.set_value("CPU")
    assert sf.value() == "CPU"


def test_number_input(qapp):
    from gui.widgets.number_input import NumberInput

    ni = NumberInput("Iterations", default=50, min_val=1, max_val=500, step=1, decimals=0)
    assert ni.value() == 50
    ni.set_value(100)
    assert ni.value() == 100


def test_slider_input(qapp):
    from gui.widgets.slider_input import SliderInput

    si = SliderInput("Threshold", default=0.0, min_val=-5.0, max_val=5.0, step=0.01, decimals=2)
    assert si.value() == 0.0
    si.set_value(1.5)
    assert abs(si.value() - 1.5) < 0.1  # slider has discrete steps


def test_tool_button(qapp):
    from gui.widgets.tool_button import ToolButton

    tb = ToolButton("cursor", "Select", shortcut="V", checkable=True)
    assert not tb.isChecked()
    tb.setChecked(True)
    assert tb.isChecked()


def test_path_selector(qapp):
    from gui.widgets.path_selector import PathSelector

    ps = PathSelector("Input Directory")
    assert ps.path() == ""
    ps.set_path("C:/test/path")
    assert ps.path() == "C:/test/path"
