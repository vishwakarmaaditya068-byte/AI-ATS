"""
Button widgets — VSCode button hierarchy.

  PrimaryButton   — filled blue  (primary CTA: Save, Run, Import)
  SecondaryButton — ghost/outline (Cancel, Back)
  DangerButton    — filled red    (Delete, Reject)
  SuccessButton   — filled teal   (Hire, Approve)
  IconButton      — square icon   (toolbars, 36×36 px)
  TextButton      — link-style    ("View all", "Details")
  ButtonGroup     — horizontal container with spacing helpers

All classes implement refresh_styles() so the theme cascade from MainWindow
re-applies their colour tokens without requiring recreation.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QPushButton, QHBoxLayout, QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.utils.constants import COLORS


def _base_font() -> QFont:
    f = QFont("Segoe UI", 10)
    f.setWeight(QFont.Weight.Medium)
    return f


# ── Primary ────────────────────────────────────────────────────────────────────

class PrimaryButton(QPushButton):
    """Filled VSCode-blue CTA button."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setFont(_base_font())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(30)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: {COLORS['text_on_primary']};
                border: none;
                border-radius: 2px;
                padding: 6px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_dark']};
                border: 1px solid {COLORS['primary']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['primary_dark']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_tertiary']};
                border: 1px solid {COLORS['border_subtle']};
            }}
            """
        )

    def refresh_styles(self) -> None:
        self._apply_style()


# ── Secondary ──────────────────────────────────────────────────────────────────

class SecondaryButton(QPushButton):
    """Ghost/outline button — transparent fill, muted border."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setFont(_base_font())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(30)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 2px;
                padding: 6px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['surface_overlay']};
                border-color: {COLORS['text_secondary']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['surface_elevated']};
            }}
            QPushButton:disabled {{
                color: {COLORS['text_tertiary']};
                border-color: {COLORS['border_subtle']};
            }}
            """
        )

    def refresh_styles(self) -> None:
        self._apply_style()


# ── Danger ─────────────────────────────────────────────────────────────────────

class DangerButton(QPushButton):
    """Destructive action button — error-red palette."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setFont(_base_font())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(30)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['error_dim']};
                color: {COLORS['error']};
                border: 1px solid {COLORS['error']};
                border-radius: 2px;
                padding: 6px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['error']};
                color: {COLORS['text_on_primary']};
                border-color: {COLORS['error']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['error']};
                color: {COLORS['text_on_primary']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_tertiary']};
                border-color: {COLORS['border_subtle']};
            }}
            """
        )

    def refresh_styles(self) -> None:
        self._apply_style()


# ── Success ────────────────────────────────────────────────────────────────────

class SuccessButton(QPushButton):
    """Positive action button — VSCode teal palette."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setFont(_base_font())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(30)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['success_dim']};
                color: {COLORS['success']};
                border: 1px solid {COLORS['success']};
                border-radius: 2px;
                padding: 6px 16px;
                font-weight: 500;
            }}
            QPushButton:hover {{
                background-color: {COLORS['success']};
                color: {COLORS['text_on_primary']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['success_dark']};
                color: {COLORS['text_on_primary']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_tertiary']};
                border-color: {COLORS['border_subtle']};
            }}
            """
        )

    def refresh_styles(self) -> None:
        self._apply_style()


# ── Icon button ────────────────────────────────────────────────────────────────

class IconButton(QPushButton):
    """Square 36×36 px icon-only toolbar button."""

    def __init__(self, icon_text: str = "", parent=None) -> None:
        super().__init__(icon_text, parent)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setFixedSize(36, 36)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFont(QFont("Segoe UI Symbol", 13))
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: none;
                border-radius: 2px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['surface_overlay']};
                color: {COLORS['text_primary']};
            }}
            QPushButton:pressed {{
                background-color: {COLORS['primary_glow']};
            }}
            """
        )

    def refresh_styles(self) -> None:
        self._apply_style()


# ── Text / link button ─────────────────────────────────────────────────────────

class TextButton(QPushButton):
    """Inline text-link button — no background, primary colour."""

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(text, parent)
        self._apply_style()

    def _apply_style(self) -> None:
        self.setFont(_base_font())
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['primary']};
                border: none;
                padding: 3px 4px;
            }}
            QPushButton:hover {{
                color: {COLORS['info']};
                text-decoration: underline;
            }}
            QPushButton:pressed {{
                color: {COLORS['primary_dark']};
            }}
            """
        )

    def refresh_styles(self) -> None:
        self._apply_style()


# ── Button group ───────────────────────────────────────────────────────────────

class ButtonGroup(QWidget):
    """Horizontal container for grouping related buttons."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(6)

    def add_button(self, button: QPushButton) -> None:
        self.layout.addWidget(button)

    def add_stretch(self) -> None:
        self.layout.addStretch()

    def add_spacing(self, width: int = 12) -> None:
        self.layout.addSpacing(width)
