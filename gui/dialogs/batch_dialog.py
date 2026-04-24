"""Batch multi-directory processing dialog.

Allows users to queue multiple input directories for sequential
processing using the current annotation and model configuration.
"""

import os

from PyQt6.QtCore import pyqtSignal, Qt
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
)

from gui.theme import Colors, Fonts


class BatchProcessingDialog(QDialog):
    """Dialog for configuring batch multi-directory processing."""

    batch_start = pyqtSignal(list)  # list of (input_dir, output_dir) tuples

    def __init__(self, default_output: str = "", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Processing")
        self.setMinimumSize(600, 400)
        self.setStyleSheet(
            f"QDialog {{ background: {Colors.BG_MEDIUM}; "
            f"color: {Colors.TEXT_PRIMARY}; }}"
        )

        self._default_output = default_output

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Instructions
        info = QLabel(
            "Add input directories to process sequentially.\n"
            "Each directory will use the current annotation points and model config.\n"
            "Output masks will be saved to a 'masks' subfolder within each output directory."
        )
        info.setStyleSheet(
            f"color: {Colors.TEXT_SECONDARY}; font-size: {Fonts.SIZE_BASE}px;"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # Directory list
        self._dir_list = QListWidget()
        self._dir_list.setStyleSheet(
            f"QListWidget {{ background: {Colors.BG_DARK}; "
            f"color: {Colors.TEXT_PRIMARY}; border: 1px solid {Colors.BORDER}; "
            f"border-radius: 6px; }}"
            f"QListWidget::item {{ padding: 6px; }}"
            f"QListWidget::item:selected {{ background: {Colors.PRIMARY_BG}; }}"
        )
        layout.addWidget(self._dir_list, 1)

        # Add/Remove buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)

        add_btn = QPushButton("Add Directory...")
        add_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        add_btn.clicked.connect(self._add_directory)
        btn_row.addWidget(add_btn)

        remove_btn = QPushButton("Remove Selected")
        remove_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(remove_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        clear_btn.clicked.connect(self._dir_list.clear)
        btn_row.addWidget(clear_btn)

        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Count label
        self._count_label = QLabel("0 directories queued")
        self._count_label.setStyleSheet(
            f"color: {Colors.TEXT_DIM}; font-size: {Fonts.SIZE_SM}px;"
        )
        layout.addWidget(self._count_label)

        # Start / Cancel buttons
        action_row = QHBoxLayout()
        action_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        cancel_btn.clicked.connect(self.reject)
        action_row.addWidget(cancel_btn)

        start_btn = QPushButton("Start Batch Processing")
        start_btn.setProperty("cssClass", "btn-accent")
        start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        start_btn.clicked.connect(self._start_batch)
        action_row.addWidget(start_btn)

        layout.addLayout(action_row)

    def _add_directory(self) -> None:
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Input Directory"
        )
        if dir_path and os.path.isdir(dir_path):
            # Check for duplicates
            for i in range(self._dir_list.count()):
                if self._dir_list.item(i).data(Qt.ItemDataRole.UserRole) == dir_path:
                    return

            item = QListWidgetItem(dir_path)
            item.setData(Qt.ItemDataRole.UserRole, dir_path)
            self._dir_list.addItem(item)
            self._update_count()

    def _remove_selected(self) -> None:
        for item in self._dir_list.selectedItems():
            self._dir_list.takeItem(self._dir_list.row(item))
        self._update_count()

    def _update_count(self) -> None:
        count = self._dir_list.count()
        self._count_label.setText(f"{count} director{'y' if count == 1 else 'ies'} queued")

    def _start_batch(self) -> None:
        dirs = []
        for i in range(self._dir_list.count()):
            input_dir = self._dir_list.item(i).data(Qt.ItemDataRole.UserRole)
            # Output dir mirrors input structure under default output
            base_name = os.path.basename(input_dir)
            if self._default_output:
                output_dir = os.path.join(self._default_output, f"batch_{base_name}")
            else:
                output_dir = os.path.join(input_dir, "output")
            dirs.append((input_dir, output_dir))

        if dirs:
            self.batch_start.emit(dirs)
            self.accept()
