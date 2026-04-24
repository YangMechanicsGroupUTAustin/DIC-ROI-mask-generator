"""Structured logging configuration for DIC Mask Generator."""

import logging
import os
from datetime import datetime
from collections import deque


class MemoryHandler(logging.Handler):
    """Keeps last N log records in memory for in-app display."""

    def __init__(self, capacity: int = 500):
        super().__init__()
        self.buffer: deque[logging.LogRecord] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        self.buffer.append(record)

    def get_messages(self) -> list[str]:
        return [self.format(r) for r in self.buffer]


_memory_handler: MemoryHandler | None = None


def setup_logging(log_dir: str = "logs") -> logging.Logger:
    """Configure application logging.

    Returns the root 'sam2studio' logger.
    Logs to file and in-memory buffer.
    """
    global _memory_handler

    logger = logging.getLogger("sam2studio")
    if logger.handlers:
        return logger  # Already configured

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"sam2studio_{timestamp}.log"),
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Memory handler
    _memory_handler = MemoryHandler(capacity=500)
    _memory_handler.setLevel(logging.INFO)
    _memory_handler.setFormatter(formatter)
    logger.addHandler(_memory_handler)

    # Console handler (WARNING+ only)
    console = logging.StreamHandler()
    console.setLevel(logging.WARNING)
    console.setFormatter(formatter)
    logger.addHandler(console)

    return logger


def get_memory_handler() -> MemoryHandler | None:
    return _memory_handler
