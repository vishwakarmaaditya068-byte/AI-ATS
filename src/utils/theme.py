from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal

# ── Dark palette — VSCode Default Dark+ ───────────────────────────────────────
DARK_COLORS: dict[str, str] = {
    # Brand / Interactive
    "primary":              "#007ACC",   # VSCode blue  (4.6:1 on #252526)
    "primary_dark":         "#005F9E",
    "primary_glow":         "#1A3F5C",
    "accent":               "#C586C0",   # VSCode purple
    "accent_dark":          "#9A4090",
    "accent_dim":           "#2D1F2D",
    "secondary":            "#858585",

    # Status
    "success":              "#4EC9B0",   # VSCode teal
    "success_dark":         "#3AA890",
    "success_dim":          "#1A3028",
    "warning":              "#CCA700",
    "warning_dim":          "#2D2800",
    "error":                "#F14C4C",
    "error_dim":            "#3D1010",
    "info":                 "#9CDCFE",   # VSCode light-blue

    # Surfaces (L0 → L3)
    "background":           "#1E1E1E",   # Editor / content area
    "surface":              "#252526",   # Sidebar / panel
    "surface_elevated":     "#2D2D30",   # Input, tab bar, card bg
    "surface_overlay":      "#3C3C3C",   # Hover, dropdown, overlay

    # Borders
    "border_subtle":        "#474747",
    "border_muted":         "#5A5A5A",

    # Typography
    "text_primary":         "#D4D4D4",   # VSCode editor foreground
    "text_secondary":       "#858585",
    "text_tertiary":        "#4A4A4A",
    "text_on_primary":      "#FFFFFF",

    # VSCode chrome tokens
    "statusbar_bg":         "#007ACC",
    "statusbar_fg":         "#FFFFFF",
    "activitybar_bg":       "#333333",
    "activitybar_active":   "#FFFFFF",
    "activitybar_inactive": "#858585",
}

# ── Light palette — VSCode Light Modern ───────────────────────────────────────
LIGHT_COLORS: dict[str, str] = {
    # Brand / Interactive
    "primary":              "#005FB8",   # VSCode light-mode blue
    "primary_dark":         "#004A94",
    "primary_glow":         "#CCE4FF",
    "accent":               "#7B3FB0",
    "accent_dark":          "#5D2F85",
    "accent_dim":           "#EDE1F8",
    "secondary":            "#6C6C6C",

    # Status
    "success":              "#267F99",
    "success_dark":         "#1B6879",
    "success_dim":          "#D4EEF4",
    "warning":              "#805000",   # 4.5:1 on white
    "warning_dim":          "#FEF5D4",
    "error":                "#A1260D",
    "error_dim":            "#FFE4DE",
    "info":                 "#1672C2",

    # Surfaces (L0 → L3)
    "background":           "#FFFFFF",
    "surface":              "#F3F3F3",
    "surface_elevated":     "#FFFFFF",
    "surface_overlay":      "#E8E8E8",

    # Borders
    "border_subtle":        "#E4E4E4",
    "border_muted":         "#D4D4D4",

    # Typography
    "text_primary":         "#1B1B1B",
    "text_secondary":       "#6C6C6C",
    "text_tertiary":        "#C8C8C8",
    "text_on_primary":      "#FFFFFF",

    # VSCode chrome tokens
    "statusbar_bg":         "#007ACC",
    "statusbar_fg":         "#FFFFFF",
    "activitybar_bg":       "#2C2C2C",   # Activity bar is always dark
    "activitybar_active":   "#FFFFFF",
    "activitybar_inactive": "#858585",
}


class ThemeManager(QObject):
    """Singleton that owns the active color palette.

    Mutates COLORS in-place on switch so every widget reading
    COLORS['key'] at call-time automatically picks up the new value.
    """

    theme_changed = pyqtSignal(str)   # emits "dark" or "light"

    def __init__(self) -> None:
        super().__init__()
        self._mode: str = "dark"

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_dark(self) -> bool:
        return self._mode == "dark"

    def set_dark(self) -> None:
        self._apply("dark", DARK_COLORS)

    def set_light(self) -> None:
        self._apply("light", LIGHT_COLORS)

    def toggle(self) -> None:
        if self._mode == "dark":
            self.set_light()
        else:
            self.set_dark()

    def _apply(self, mode: str, palette: dict[str, str]) -> None:
        if self._mode == mode:
            return
        from src.utils.constants import COLORS
        COLORS.clear()
        COLORS.update(palette)
        self._mode = mode
        self.theme_changed.emit(mode)


_theme: Optional[ThemeManager] = None


def get_theme() -> ThemeManager:
    global _theme
    if _theme is None:
        _theme = ThemeManager()
    return _theme
