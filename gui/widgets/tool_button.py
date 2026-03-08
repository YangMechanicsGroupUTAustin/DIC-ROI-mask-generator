"""Toolbar icon button with label, shortcut badge, and tooltip.

Used in the top toolbar for tool selection (cursor, pencil, eraser, etc.).
"""

from PyQt6.QtWidgets import QPushButton, QWidget

from gui.icons import get_icon
from gui.theme import Colors


# Map variant names to cssClass values
_VARIANT_CSS: dict[str, str] = {
    "default": "btn-tool",
    "primary": "btn-primary",
    "danger": "btn-danger",
    "success": "btn-success",
}


class ToolButton(QPushButton):
    """Toolbar icon button with label, shortcut badge, and tooltip."""

    def __init__(
        self,
        icon_name: str,
        label: str = "",
        shortcut: str = "",
        variant: str = "default",
        checkable: bool = False,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)

        # Icon
        self.setIcon(get_icon(icon_name, Colors.TEXT_SECONDARY, 16))

        # Text
        if label:
            self.setText(label)

        # Checkable
        self.setCheckable(checkable)

        # Variant styling
        css_class = _VARIANT_CSS.get(variant, "btn-tool")
        self.setProperty("cssClass", css_class)

        # Tooltip with shortcut hint
        tooltip_parts: list[str] = []
        if label:
            tooltip_parts.append(label)
        if shortcut:
            tooltip_parts.append(f"[{shortcut}]")
        if tooltip_parts:
            self.setToolTip(" ".join(tooltip_parts))

        # Shortcut
        if shortcut:
            self.setShortcut(shortcut)
