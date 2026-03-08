"""GPU/CPU device detection and VRAM monitoring."""

import torch


class DeviceManager:
    """Utility class for device detection and monitoring."""

    @staticmethod
    def detect_available_devices() -> list[str]:
        """Return list of available compute devices."""
        devices = []
        if torch.cuda.is_available():
            devices.append("CUDA")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            devices.append("MPS")
        devices.append("CPU")
        return devices

    @staticmethod
    def get_device_string(choice: str) -> str:
        """Convert display name to torch device string."""
        mapping = {"CUDA": "cuda", "MPS": "mps", "CPU": "cpu"}
        return mapping.get(choice, "cpu")

    @staticmethod
    def get_gpu_name() -> str:
        """Return GPU name or 'CPU'."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "Apple Silicon (MPS)"
        return "CPU"

    @staticmethod
    def get_vram_usage() -> tuple[float, float]:
        """Return (used_gb, total_gb). Returns (0, 0) for CPU/MPS."""
        if torch.cuda.is_available():
            used = torch.cuda.memory_allocated(0) / (1024 ** 3)
            total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            return round(used, 1), round(total, 1)
        return 0.0, 0.0

    @staticmethod
    def get_torch_version() -> str:
        """Return PyTorch version string."""
        return torch.__version__

    @staticmethod
    def empty_cache() -> None:
        """Clear GPU cache if available."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
