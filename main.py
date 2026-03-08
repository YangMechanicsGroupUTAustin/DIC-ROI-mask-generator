"""SAM2 Studio v2.0 -- Entry Point."""
import sys
import os

# Set environment before any torch imports
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from gui.main_window import MainWindow
from gui.theme import generate_stylesheet


def main():
    # Enable high DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("SAM2 Studio")
    app.setApplicationVersion("2.0.0")
    app.setStyleSheet(generate_stylesheet())

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
