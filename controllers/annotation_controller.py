"""Annotation controller with command-pattern undo/redo.

Manages point annotations with full undo/redo support.
Each user action is wrapped in a Command object.
"""

from abc import ABC, abstractmethod

from PyQt6.QtCore import QObject, pyqtSignal


class Command(ABC):
    """Base class for undoable commands."""

    @abstractmethod
    def execute(self, state) -> None:
        """Execute the command."""

    @abstractmethod
    def undo(self, state) -> None:
        """Reverse the command."""

    @abstractmethod
    def description(self) -> str:
        """Human-readable description."""


class AddPointCommand(Command):
    def __init__(self, x: float, y: float, label: int):
        self._x = x
        self._y = y
        self._label = label

    def execute(self, state) -> None:
        state.add_point(self._x, self._y, self._label)

    def undo(self, state) -> None:
        # Remove the last point (the one we added)
        if state.points:
            state.remove_point(len(state.points) - 1)

    def description(self) -> str:
        mode = "foreground" if self._label == 1 else "background"
        return f"Add {mode} point at ({self._x:.1f}, {self._y:.1f})"


class MovePointCommand(Command):
    def __init__(self, index: int, old_x: float, old_y: float, new_x: float, new_y: float):
        self._index = index
        self._old_x = old_x
        self._old_y = old_y
        self._new_x = new_x
        self._new_y = new_y

    def execute(self, state) -> None:
        state.move_point(self._index, self._new_x, self._new_y)

    def undo(self, state) -> None:
        state.move_point(self._index, self._old_x, self._old_y)

    def description(self) -> str:
        return f"Move point {self._index}"


class RemovePointCommand(Command):
    def __init__(self, index: int, point: list[float], label: int):
        self._index = index
        self._point = point
        self._label = label

    def execute(self, state) -> None:
        state.remove_point(self._index)

    def undo(self, state) -> None:
        # Re-insert at original position via public API
        new_points = list(state.points)
        new_labels = list(state.labels)
        new_points.insert(self._index, self._point)
        new_labels.insert(self._index, self._label)
        state.set_points(new_points, new_labels)

    def description(self) -> str:
        return f"Remove point {self._index}"


class ClearPointsCommand(Command):
    def __init__(self):
        self._snapshot_points: list[list[float]] = []
        self._snapshot_labels: list[int] = []

    def execute(self, state) -> None:
        self._snapshot_points = list(state.points)
        self._snapshot_labels = list(state.labels)
        state.clear_points()

    def undo(self, state) -> None:
        state.set_points(self._snapshot_points, self._snapshot_labels)

    def description(self) -> str:
        return f"Clear {len(self._snapshot_points)} points"


class AnnotationController(QObject):
    """Manages point annotations with undo/redo support."""

    can_undo_changed = pyqtSignal(bool)
    can_redo_changed = pyqtSignal(bool)

    MAX_HISTORY = 100

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state
        self._undo_stack: list[Command] = []
        self._redo_stack: list[Command] = []
        self._point_mode = "foreground"   # "foreground" or "background"

    def _execute(self, command: Command) -> None:
        """Execute command, push to undo stack, clear redo stack."""
        command.execute(self._state)
        self._undo_stack.append(command)
        if len(self._undo_stack) > self.MAX_HISTORY:
            self._undo_stack.pop(0)
        self._redo_stack.clear()
        self._emit_state()

    def add_point(self, x: float, y: float) -> None:
        """Add point with current mode (foreground=1, background=0)."""
        label = 1 if self._point_mode == "foreground" else 0
        self._execute(AddPointCommand(x, y, label))

    def move_point(self, index: int, new_x: float, new_y: float) -> None:
        """Move existing point. Captures old position for undo."""
        if 0 <= index < len(self._state.points):
            old = self._state.points[index]
            self._execute(MovePointCommand(index, old[0], old[1], new_x, new_y))

    def remove_point(self, index: int) -> None:
        """Remove point at index."""
        if 0 <= index < len(self._state.points):
            point = self._state.points[index]
            label = self._state.labels[index]
            self._execute(RemovePointCommand(index, point, label))

    def clear_points(self) -> None:
        """Clear all points (undoable)."""
        if self._state.points:
            self._execute(ClearPointsCommand())

    def undo(self) -> None:
        """Undo last command."""
        if self._undo_stack:
            command = self._undo_stack.pop()
            command.undo(self._state)
            self._redo_stack.append(command)
            self._emit_state()

    def redo(self) -> None:
        """Redo last undone command."""
        if self._redo_stack:
            command = self._redo_stack.pop()
            command.execute(self._state)
            self._undo_stack.append(command)
            self._emit_state()

    @property
    def can_undo(self) -> bool:
        return len(self._undo_stack) > 0

    @property
    def can_redo(self) -> bool:
        return len(self._redo_stack) > 0

    @property
    def point_mode(self) -> str:
        return self._point_mode

    def set_point_mode(self, mode: str) -> None:
        """Set point mode: 'foreground' or 'background'."""
        self._point_mode = mode

    def _emit_state(self) -> None:
        self.can_undo_changed.emit(self.can_undo)
        self.can_redo_changed.emit(self.can_redo)

    def save_config(self, filepath: str) -> None:
        """Save current annotations to file."""
        from core.annotation_config import save_annotation_config
        save_annotation_config(
            filepath,
            input_points=self._state.points,
            input_labels=self._state.labels,
            model_name=self._state.model_name,
            device=self._state.device,
            threshold=self._state.threshold,
            start_frame=self._state.start_frame,
            end_frame=self._state.end_frame,
            intermediate_format=self._state.intermediate_format,
        )

    def load_config(self, filepath: str) -> None:
        """Load annotations from file, replacing current state."""
        from core.annotation_config import load_annotation_config
        config = load_annotation_config(filepath)
        annotation = config["annotation"]
        self._state.set_points(annotation["points"], annotation["labels"])
        # Load parameters if present
        params = config.get("parameters", {})
        if params.get("model"):
            self._state.set_model_name(params["model"])
        if params.get("device"):
            self._state.set_device(params["device"])
        if params.get("threshold") is not None:
            self._state.set_threshold(params["threshold"])
        if params.get("start_frame"):
            start = params["start_frame"]
            end = params.get("end_frame", self._state.total_frames)
            self._state.set_frame_range(start, end)
        # Clear undo/redo after load
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._emit_state()
