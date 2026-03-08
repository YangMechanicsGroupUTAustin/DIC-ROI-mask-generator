"""Main application window assembling all GUI panels.

Layout matches the Figma App.tsx reference:
  QVBoxLayout:
    QHBoxLayout:
      Sidebar (fixed 300px)
      QVBoxLayout:
        Toolbar (fixed 56px)
        CanvasArea (expand)
        FrameNavigator (fixed 48px)
    StatusBar (fixed 28px)
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QWidget,
)

from gui.icons import get_icon
from gui.panels.canvas_area import CanvasArea
from gui.panels.frame_navigator import FrameNavigator
from gui.panels.sidebar import Sidebar
from gui.panels.status_bar import StatusBar
from gui.panels.toolbar import Toolbar
from gui.theme import Colors


class MainWindow(QMainWindow):
    """Main application window for SAM2 Studio."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("SAM2 Studio")
        self.setMinimumSize(1200, 700)

        # Application icon
        self.setWindowIcon(get_icon("brain", Colors.PRIMARY, 32))

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)

        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # --- Main content row: sidebar + right content ---
        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        # Sidebar
        self._sidebar = Sidebar()
        content_layout.addWidget(self._sidebar)

        # Right column: toolbar + canvas + frame nav
        right_layout = QVBoxLayout()
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._toolbar = Toolbar()
        right_layout.addWidget(self._toolbar)

        self._canvas_area = CanvasArea()
        right_layout.addWidget(self._canvas_area, 1)

        self._frame_navigator = FrameNavigator()
        right_layout.addWidget(self._frame_navigator)

        content_layout.addLayout(right_layout, 1)

        root_layout.addLayout(content_layout, 1)

        # Status bar
        self._status_bar = StatusBar()
        root_layout.addWidget(self._status_bar)

        # Start maximized
        self.showMaximized()

    # --- Properties for external access ---

    @property
    def sidebar(self) -> Sidebar:
        """Access the sidebar panel."""
        return self._sidebar

    @property
    def toolbar(self) -> Toolbar:
        """Access the toolbar panel."""
        return self._toolbar

    @property
    def canvas_area(self) -> CanvasArea:
        """Access the canvas area panel."""
        return self._canvas_area

    @property
    def frame_navigator(self) -> FrameNavigator:
        """Access the frame navigator panel."""
        return self._frame_navigator

    @property
    def status_bar_widget(self) -> StatusBar:
        """Access the status bar widget.

        Named status_bar_widget to avoid conflict with QMainWindow.statusBar().
        """
        return self._status_bar
