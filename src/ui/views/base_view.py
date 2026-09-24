from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from src.utils.constants import COLORS


class BaseView(QWidget):
    """
    Base class for all application views.

    Provides a scrollable content area with a VSCode-style section header
    (title + optional description + hairline separator).  Subclasses call
    add_widget() / add_layout() to populate the scroll area.
    """

    refresh_requested = pyqtSignal()

    def __init__(self, title: str, description: str = "", parent=None) -> None:
        super().__init__(parent)
        self.title = title
        self.description = description
        self._setup_ui()
        self._apply_styles()

    # ── UI construction ────────────────────────────────────────────────────────

    def _setup_ui(self) -> None:
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(24, 20, 24, 20)
        self.main_layout.setSpacing(14)

        self._header_widget = self._create_header()
        self.main_layout.addWidget(self._header_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 8, 0)
        self.content_layout.setSpacing(14)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        self.scroll_area.setWidget(self.content_widget)
        self.main_layout.addWidget(self.scroll_area)

    def _create_header(self) -> QWidget:
        header = QWidget()
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 6)
        header_layout.setSpacing(3)

        self.title_label = QLabel(self.title)
        title_font = QFont("Segoe UI", 16, QFont.Weight.DemiBold)
        self.title_label.setFont(title_font)
        header_layout.addWidget(self.title_label)

        if self.description:
            self.description_label = QLabel(self.description)
            self.description_label.setWordWrap(True)
            header_layout.addWidget(self.description_label)

        self._separator = QFrame()
        self._separator.setFrameShape(QFrame.Shape.HLine)
        self._separator.setFixedHeight(1)
        header_layout.addWidget(self._separator)

        return header

    # ── Style application ──────────────────────────────────────────────────────

    def _apply_styles(self) -> None:
        self.title_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        if hasattr(self, "description_label"):
            self.description_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 12px;"
            )
        self._separator.setStyleSheet(
            f"background-color: {COLORS['border_subtle']}; border: none;"
        )
        self.content_widget.setStyleSheet("background-color: transparent;")
        self.scroll_area.setStyleSheet(
            "QScrollArea { background-color: transparent; border: none; }"
        )

    def refresh_styles(self) -> None:
        """Re-apply all inline styles after a theme change."""
        self._apply_styles()

    # ── Public content API ─────────────────────────────────────────────────────

    def add_widget(self, widget: QWidget) -> None:
        self.content_layout.addWidget(widget)

    def add_layout(self, layout) -> None:
        self.content_layout.addLayout(layout)

    def add_stretch(self) -> None:
        self.content_layout.addStretch()

    def clear_content(self) -> None:
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def set_title(self, title: str) -> None:
        self.title = title
        self.title_label.setText(title)

    def refresh(self) -> None:
        self.refresh_requested.emit()
