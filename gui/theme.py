"""Dark theme stylesheet system for SAM2 Studio.

Color palette derived from Figma reference design.
All colors, fonts, and QSS styling defined here.
"""

from typing import Final


class Colors:
    """Application color constants matching Figma design."""

    # Backgrounds (darkest -> lightest)
    BG_DARKEST: Final = "#0c0d12"   # Canvas background
    BG_DARK: Final = "#0f1117"       # Main window background
    BG_MEDIUM: Final = "#13141b"     # Sidebar, toolbar, panels
    BG_LIGHT: Final = "#16171f"      # Panel headers
    BG_INPUT: Final = "#1a1b23"      # Input fields, dropdowns

    # Borders
    BORDER: Final = "rgba(255, 255, 255, 0.06)"
    BORDER_HOVER: Final = "rgba(255, 255, 255, 0.12)"
    BORDER_SUBTLE: Final = "rgba(255, 255, 255, 0.04)"

    # Primary (Indigo)
    PRIMARY: Final = "#6366f1"
    PRIMARY_HOVER: Final = "#818cf8"
    PRIMARY_GLOW: Final = "rgba(99, 102, 241, 0.20)"
    PRIMARY_BG: Final = "rgba(99, 102, 241, 0.15)"
    PRIMARY_BORDER: Final = "rgba(99, 102, 241, 0.20)"

    # Semantic colors
    SUCCESS: Final = "#10b981"        # Emerald
    SUCCESS_BG: Final = "rgba(16, 185, 129, 0.20)"
    SUCCESS_BORDER: Final = "rgba(16, 185, 129, 0.30)"
    DANGER: Final = "#ef4444"         # Red
    DANGER_BG: Final = "rgba(239, 68, 68, 0.10)"
    DANGER_BORDER: Final = "rgba(239, 68, 68, 0.20)"
    WARNING: Final = "#f59e0b"        # Amber
    ROSE: Final = "#f43f5e"           # Rose (background points)
    ROSE_BG: Final = "rgba(244, 63, 94, 0.20)"
    ROSE_BORDER: Final = "rgba(244, 63, 94, 0.30)"

    # Text
    TEXT_PRIMARY: Final = "#e4e4e7"   # Zinc-200
    TEXT_SECONDARY: Final = "#a1a1aa"  # Zinc-400
    TEXT_MUTED: Final = "#71717a"     # Zinc-500
    TEXT_DIM: Final = "#52525b"       # Zinc-600

    # Annotation points
    FG_POINT: Final = "#10b981"       # Foreground (emerald)
    BG_POINT: Final = "#f43f5e"       # Background (rose)


class Fonts:
    """Font configuration."""

    FAMILY: Final = "Segoe UI"
    MONO: Final = "Consolas"  # JetBrains Mono fallback
    SIZE_XS: Final = 10       # 10px (status bar, badges)
    SIZE_SM: Final = 11       # 11px (labels, uppercase)
    SIZE_BASE: Final = 12     # 12px (body text)
    SIZE_MD: Final = 13       # 13px (buttons, inputs)
    SIZE_LG: Final = 15       # 15px (titles)


def generate_stylesheet() -> str:
    """Generate complete QSS stylesheet for the application."""
    c = Colors
    f = Fonts

    return f"""
    /* === Global === */
    QMainWindow, QWidget {{
        background-color: {c.BG_DARK};
        color: {c.TEXT_PRIMARY};
        font-family: '{f.FAMILY}';
        font-size: {f.SIZE_BASE}px;
    }}

    /* === Scrollbars === */
    QScrollBar:vertical {{
        background: {c.BG_MEDIUM};
        width: 8px;
        margin: 0;
        border: none;
    }}
    QScrollBar::handle:vertical {{
        background: {c.TEXT_DIM};
        min-height: 30px;
        border-radius: 4px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {c.TEXT_MUTED};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0;
    }}
    QScrollBar:horizontal {{
        background: {c.BG_MEDIUM};
        height: 8px;
        margin: 0;
        border: none;
    }}
    QScrollBar::handle:horizontal {{
        background: {c.TEXT_DIM};
        min-width: 30px;
        border-radius: 4px;
    }}
    QScrollBar::handle:horizontal:hover {{
        background: {c.TEXT_MUTED};
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
        width: 0;
    }}

    /* === Labels === */
    QLabel {{
        color: {c.TEXT_PRIMARY};
        background: transparent;
    }}
    QLabel[cssClass="label-uppercase"] {{
        color: {c.TEXT_MUTED};
        font-size: {f.SIZE_SM}px;
        text-transform: uppercase;
    }}
    QLabel[cssClass="label-dim"] {{
        color: {c.TEXT_DIM};
        font-size: {f.SIZE_SM}px;
    }}
    QLabel[cssClass="label-value"] {{
        color: {c.PRIMARY};
        font-family: '{f.MONO}';
        font-size: {f.SIZE_BASE}px;
    }}

    /* === Push Buttons === */
    QPushButton {{
        background-color: transparent;
        color: {c.TEXT_SECONDARY};
        border: 1px solid transparent;
        border-radius: 8px;
        padding: 6px 12px;
        font-size: {f.SIZE_MD}px;
    }}
    QPushButton:hover {{
        background-color: rgba(255, 255, 255, 0.05);
        color: {c.TEXT_PRIMARY};
    }}
    QPushButton:pressed {{
        background-color: rgba(255, 255, 255, 0.08);
    }}
    QPushButton:disabled {{
        color: {c.TEXT_DIM};
    }}

    /* Button variants via dynamic properties */
    QPushButton[cssClass="btn-primary"] {{
        background-color: {c.PRIMARY};
        color: white;
        border: 1px solid rgba(99, 102, 241, 0.50);
    }}
    QPushButton[cssClass="btn-primary"]:hover {{
        background-color: {c.PRIMARY_HOVER};
    }}
    QPushButton[cssClass="btn-primary"]:disabled {{
        background-color: {c.TEXT_DIM};
        border-color: transparent;
    }}

    QPushButton[cssClass="btn-danger"] {{
        color: {c.DANGER};
    }}
    QPushButton[cssClass="btn-danger"]:hover {{
        background-color: {c.DANGER_BG};
    }}

    QPushButton[cssClass="btn-success"] {{
        color: {c.SUCCESS};
    }}
    QPushButton[cssClass="btn-success"]:hover {{
        background-color: {c.SUCCESS_BG};
    }}

    QPushButton[cssClass="btn-accent"] {{
        background-color: {c.PRIMARY_BG};
        color: {c.PRIMARY};
        border: 1px solid {c.PRIMARY_BORDER};
    }}
    QPushButton[cssClass="btn-accent"]:hover {{
        background-color: {c.PRIMARY_GLOW};
    }}

    QPushButton[cssClass="btn-tool"] {{
        padding: 6px 10px;
        border-radius: 8px;
    }}
    QPushButton[cssClass="btn-tool"]:checked {{
        background-color: rgba(255, 255, 255, 0.08);
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.12);
    }}

    /* === Combo Box === */
    QComboBox {{
        background-color: {c.BG_INPUT};
        color: {c.TEXT_PRIMARY};
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 6px 12px;
        font-size: {f.SIZE_MD}px;
        min-height: 20px;
    }}
    QComboBox:hover {{
        border-color: rgba(255, 255, 255, 0.12);
    }}
    QComboBox:focus {{
        border-color: rgba(99, 102, 241, 0.50);
    }}
    QComboBox::drop-down {{
        border: none;
        width: 24px;
    }}
    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {c.TEXT_MUTED};
        margin-right: 8px;
    }}
    QComboBox QAbstractItemView {{
        background-color: {c.BG_INPUT};
        color: {c.TEXT_PRIMARY};
        border: 1px solid rgba(255, 255, 255, 0.08);
        selection-background-color: {c.PRIMARY_BG};
        selection-color: {c.PRIMARY};
        outline: none;
    }}

    /* === Spin Box / Double Spin Box === */
    QSpinBox, QDoubleSpinBox {{
        background-color: {c.BG_INPUT};
        color: {c.TEXT_PRIMARY};
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 6px 8px;
        font-family: '{f.MONO}';
        font-size: {f.SIZE_MD}px;
    }}
    QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: rgba(255, 255, 255, 0.12);
    }}
    QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: rgba(99, 102, 241, 0.50);
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button,
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        width: 0;
        border: none;
    }}

    /* === Line Edit === */
    QLineEdit {{
        background-color: {c.BG_INPUT};
        color: {c.TEXT_PRIMARY};
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 6px 12px;
        font-size: {f.SIZE_MD}px;
    }}
    QLineEdit:hover {{
        border-color: rgba(255, 255, 255, 0.12);
    }}
    QLineEdit:focus {{
        border-color: rgba(99, 102, 241, 0.50);
    }}
    QLineEdit:read-only {{
        color: {c.TEXT_MUTED};
    }}

    /* === Slider === */
    QSlider::groove:horizontal {{
        background: {c.BG_INPUT};
        height: 6px;
        border-radius: 3px;
    }}
    QSlider::handle:horizontal {{
        background: {c.PRIMARY};
        width: 14px;
        height: 14px;
        margin: -4px 0;
        border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover {{
        background: {c.PRIMARY_HOVER};
    }}
    QSlider::sub-page:horizontal {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #4f46e5, stop:1 {c.PRIMARY});
        border-radius: 3px;
    }}

    /* === Progress Bar === */
    QProgressBar {{
        background-color: {c.BG_INPUT};
        border: none;
        border-radius: 3px;
        height: 6px;
        text-align: center;
        color: transparent;
    }}
    QProgressBar::chunk {{
        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
            stop:0 #4f46e5, stop:1 {c.PRIMARY});
        border-radius: 3px;
    }}

    /* === Check Box === */
    QCheckBox {{
        color: {c.TEXT_MUTED};
        font-size: {f.SIZE_BASE}px;
        spacing: 6px;
    }}
    QCheckBox::indicator {{
        width: 14px;
        height: 14px;
        border-radius: 4px;
        border: 1px solid rgba(255, 255, 255, 0.10);
        background-color: {c.BG_INPUT};
    }}
    QCheckBox::indicator:checked {{
        background-color: {c.PRIMARY};
        border-color: {c.PRIMARY};
    }}

    /* === Splitter === */
    QSplitter::handle {{
        background: transparent;
        width: 3px;
        height: 3px;
    }}
    QSplitter::handle:hover {{
        background: rgba(99, 102, 241, 0.30);
    }}

    /* === Tool Tip === */
    QToolTip {{
        background-color: {c.BG_LIGHT};
        color: {c.TEXT_PRIMARY};
        border: 1px solid rgba(255, 255, 255, 0.06);
        padding: 4px 8px;
        border-radius: 6px;
        font-size: {f.SIZE_SM}px;
    }}

    /* === Menu Bar (future) === */
    QMenuBar {{
        background-color: {c.BG_MEDIUM};
        color: {c.TEXT_SECONDARY};
    }}
    QMenuBar::item:selected {{
        background-color: rgba(255, 255, 255, 0.05);
    }}
    QMenu {{
        background-color: {c.BG_INPUT};
        color: {c.TEXT_PRIMARY};
        border: 1px solid rgba(255, 255, 255, 0.08);
    }}
    QMenu::item:selected {{
        background-color: {c.PRIMARY_BG};
    }}
    """
