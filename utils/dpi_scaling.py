"""DPI-aware scaling utilities for high-resolution displays."""

import tkinter as tk
from tkinter import ttk


class DPIScaler:
    """Centralized DPI scaling for 4K, 2K, and standard displays."""

    THRESHOLD_4K = 3840
    THRESHOLD_2K = 2560

    SCALE_LIMITS = {
        "4k": (1.5, 3.0),
        "2k": (1.2, 2.0),
        "standard": (1.0, 1.5),
    }

    FONT_SIZES = {
        "4k": {"font": 12, "title": 14, "padding": 12},
        "2k": {"font": 10, "title": 12, "padding": 11},
        "standard": {"font": 9, "title": 10, "padding": 10},
    }

    def __init__(self, root: tk.Tk):
        self.root = root
        self.screen_width = root.winfo_screenwidth()
        self.screen_height = root.winfo_screenheight()
        self.tier = self._detect_tier()
        self.scale = self._compute_scale()

    def _detect_tier(self) -> str:
        if self.screen_width >= self.THRESHOLD_4K or self.screen_height >= 2160:
            return "4k"
        elif self.screen_width >= self.THRESHOLD_2K or self.screen_height >= 1440:
            return "2k"
        return "standard"

    def _compute_scale(self) -> float:
        try:
            dpi = self.root.winfo_fpixels('1i')
            raw_scale = dpi / 96.0
        except Exception:
            return 1.25

        lo, hi = self.SCALE_LIMITS[self.tier]
        scale = max(lo, min(raw_scale, hi))
        print(f"Screen: {self.screen_width}x{self.screen_height} ({self.tier})")
        print(f"DPI: {dpi:.1f}, Scale: {scale:.2f}")
        return scale

    def font_size(self, role: str = "font") -> int:
        """Get scaled font size. role: 'font', 'title', or 'padding'."""
        base = self.FONT_SIZES[self.tier].get(role, 10)
        minimums = {"font": 10, "title": 12, "padding": 10}
        return max(minimums.get(role, 8), int(base * self.scale))

    def point_size(self) -> int:
        """Get scaled scatter point size for annotations."""
        bases = {"4k": 120, "2k": 100, "standard": 90}
        mins = {"4k": 100, "2k": 80, "standard": 70}
        return max(mins[self.tier], int(bases[self.tier] * self.scale))

    def matplotlib_sizes(self) -> dict:
        """Return dict of matplotlib font sizes for current resolution."""
        return {
            "title": self.font_size("title"),
            "label": self.font_size("font"),
            "tick": max(7, int(9 * self.scale)),
            "legend": max(8, int(10 * self.scale)),
            "text": max(10, int(12 * self.scale)),
            "point": self.point_size(),
        }

    def configure_ttk_style(self, style: ttk.Style):
        """Apply DPI-scaled fonts and sizes to ttk Style."""
        font_sz = self.font_size("font")
        title_sz = self.font_size("title")
        pad = self.font_size("padding")
        bar_thickness = max(20, int(25 * self.scale))

        style.configure('TLabel', font=('Segoe UI', font_sz))
        style.configure('TButton', font=('Segoe UI', font_sz), padding=(pad, pad // 2))
        style.configure('TEntry', font=('Segoe UI', font_sz), fieldbackground='white')
        style.configure('TCombobox', font=('Segoe UI', font_sz))
        style.configure('TCheckbutton', font=('Segoe UI', font_sz))
        style.configure('TLabelFrame', font=('Segoe UI', font_sz, 'bold'))
        style.configure('TLabelFrame.Label', font=('Segoe UI', title_sz, 'bold'))
        style.configure('TProgressbar', thickness=bar_thickness)

    def scaled_window_size(self, base_w=1400, base_h=900, min_w=1200, min_h=700):
        """Return (width, height, min_width, min_height) scaled by DPI."""
        return (
            int(base_w * self.scale),
            int(base_h * self.scale),
            int(min_w * self.scale),
            int(min_h * self.scale),
        )
