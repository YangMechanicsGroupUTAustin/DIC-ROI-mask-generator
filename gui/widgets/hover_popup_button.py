"""Hover popup button widget for DIC Mask Generator.

A button that shows a floating popup panel on hover, used to group
preprocessing controls or other settings into a compact dropdown.
"""

from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSize, QTimer, Qt
from PyQt6.QtGui import QColor, QCursor, QEnterEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui.icons import get_icon
from gui.theme import Colors, Fonts

_HIDE_DELAY_MS = 400
_POPUP_MAX_WIDTH = 280

# Extra pixels of tolerance around the combined button+popup area
# when deciding whether to hide. Prevents accidental dismissal.
_CURSOR_PADDING = 8


class _PopupPanel(QWidget):
    """Floating panel that appears below the hover button."""

    def __init__(self, parent_button: HoverPopupButton) -> None:
        # Use Tool window type instead of ToolTip for reliable mouse
        # enter/leave events on Windows.
        super().__init__(
            None,
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint,
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setMouseTracking(True)

        self._parent_button = parent_button

        self.setMaximumWidth(_POPUP_MAX_WIDTH)
        self.setStyleSheet(
            f"_PopupPanel {{"
            f"  background: {Colors.BG_MEDIUM};"
            f"  border: 1px solid {Colors.BORDER};"
            f"  border-radius: 8px;"
            f"}}"
        )

        # Drop shadow
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(16)
        shadow.setOffset(0, 4)
        shadow.setColor(QColor(0, 0, 0, 80))
        self.setGraphicsEffect(shadow)

        # Content layout
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(8, 8, 8, 8)
        self._layout.setSpacing(6)

    def add_widget(self, widget: QWidget) -> None:
        """Add a child control to the popup panel."""
        self._layout.addWidget(widget)

    def enterEvent(self, event: QEnterEvent) -> None:  # noqa: N802
        self._parent_button._cancel_hide()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._parent_button._schedule_hide()
        super().leaveEvent(event)


class HoverPopupButton(QWidget):
    """Button that shows a floating popup panel on hover.

    Args:
        label: Button text.
        icon_name: Optional Lucide icon name to show before the label.
        parent: Parent widget.
    """

    def __init__(
        self,
        label: str,
        icon_name: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._active = False
        self._icon_name = icon_name

        # Hide timer with delay to allow crossing gaps between button and popup
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.setInterval(_HIDE_DELAY_MS)
        self._hide_timer.timeout.connect(self._do_hide)

        # Layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Button
        self._button = QPushButton()
        self._button.setCursor(Qt.CursorShape.PointingHandCursor)
        if icon_name:
            self._button.setIcon(get_icon(icon_name, Colors.TEXT_SECONDARY, 14))
        self._button.setText(f"{label} \u25BC")
        self._apply_button_style()
        layout.addWidget(self._button)

        # Popup panel
        self._popup = _PopupPanel(self)

    @property
    def active(self) -> bool:
        """Whether the button is in the active (highlighted) state."""
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self.set_active(value)

    @property
    def popup(self) -> _PopupPanel:
        """Access the popup panel for connecting signals or adding widgets."""
        return self._popup

    def set_active(self, active: bool) -> None:
        """Update button appearance based on active state."""
        if self._active == active:
            return
        self._active = active
        self._apply_button_style()
        if self._icon_name:
            icon_color = Colors.PRIMARY if active else Colors.TEXT_SECONDARY
            self._button.setIcon(get_icon(self._icon_name, icon_color, 14))

    def add_popup_widget(self, widget: QWidget) -> None:
        """Add a widget to the popup panel."""
        self._popup.add_widget(widget)

    # -- Event overrides --

    def enterEvent(self, event: QEnterEvent) -> None:  # noqa: N802
        self._cancel_hide()
        self._show_popup()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._schedule_hide()
        super().leaveEvent(event)

    # -- Private helpers --

    def _show_popup(self) -> None:
        """Position and show the popup panel, respecting screen boundaries.

        Tries to place the popup to the right of the button; if that
        would overflow the screen, falls back to the left side.  Also
        clamps vertically so the popup never extends beyond the screen.
        """
        from PyQt6.QtWidgets import QApplication

        button_rect = self._button.rect()
        btn_top_right = self._button.mapToGlobal(
            QPoint(button_rect.right(), button_rect.top())
        )
        btn_top_left = self._button.mapToGlobal(
            QPoint(button_rect.left(), button_rect.top())
        )

        # Measure popup (force layout so sizeHint is accurate)
        self._popup.adjustSize()
        popup_size: QSize = self._popup.sizeHint()
        pw, ph = popup_size.width(), popup_size.height()

        # Find the screen that contains this button
        screen = QApplication.screenAt(btn_top_right)
        if screen is None:
            screen = QApplication.primaryScreen()
        screen_rect: QRect = screen.availableGeometry()

        # --- Horizontal placement ---
        # Prefer right side of button
        x = btn_top_right.x()
        if x + pw > screen_rect.right():
            # Doesn't fit on right → try left side
            x = btn_top_left.x() - pw

        # Clamp to screen left edge
        if x < screen_rect.left():
            x = screen_rect.left()

        # --- Vertical placement ---
        y = btn_top_right.y()
        if y + ph > screen_rect.bottom():
            y = screen_rect.bottom() - ph
        if y < screen_rect.top():
            y = screen_rect.top()

        self._popup.move(QPoint(x, y))
        self._popup.show()

    def _schedule_hide(self) -> None:
        if not self._hide_timer.isActive():
            self._hide_timer.start()

    def _cancel_hide(self) -> None:
        self._hide_timer.stop()

    def _do_hide(self) -> None:
        """Hide popup only if cursor is outside both button AND popup.

        Uses QCursor.pos() (global coordinates) instead of underMouse()
        which is unreliable for Tool/ToolTip windows on Windows.
        Also keeps popup open while any child QComboBox dropdown is visible.
        """
        # Never hide while a combo dropdown is open inside the popup
        for combo in self._popup.findChildren(QComboBox):
            if combo.view().isVisible():
                return

        cursor = QCursor.pos()

        # Check button area (global)
        btn_global_tl = self._button.mapToGlobal(QPoint(0, 0))
        btn_rect = QRect(btn_global_tl, self._button.size())
        btn_rect.adjust(-_CURSOR_PADDING, -_CURSOR_PADDING,
                        _CURSOR_PADDING, _CURSOR_PADDING)

        if btn_rect.contains(cursor):
            return

        # Check popup area (already in global coords since it's a top-level)
        if self._popup.isVisible():
            popup_rect = self._popup.geometry()
            popup_rect.adjust(-_CURSOR_PADDING, -_CURSOR_PADDING,
                              _CURSOR_PADDING, _CURSOR_PADDING)
            if popup_rect.contains(cursor):
                return

        self._popup.hide()

    def _apply_button_style(self) -> None:
        text_color = Colors.PRIMARY if self._active else Colors.TEXT_SECONDARY
        text_hover = Colors.PRIMARY_HOVER if self._active else Colors.TEXT_PRIMARY
        self._button.setStyleSheet(
            f"QPushButton {{"
            f"  background: transparent;"
            f"  color: {text_color};"
            f"  border: none;"
            f"  border-radius: 6px;"
            f"  padding: 4px 8px;"
            f"  font-size: {Fonts.SIZE_SM}px;"
            f"  font-weight: 500;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background: rgba(255, 255, 255, 0.05);"
            f"  color: {text_hover};"
            f"}}"
        )
