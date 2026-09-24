"""
Card widgets — VSCode surface elevation system.

Elevation model (matches COLORS surface tokens):
  Card          → surface (L1)        base card, no hover border
  StatCard      → surface_elevated (L2) KPI metric card with accent stripe
  InfoCard      → surface (L1)        content card with title + slot
  ScoreCard     → surface (L1)        score display with colour-coded progress bar
  CandidateCard → surface_elevated (L2) candidate result row with skill chips (hover border)

All classes implement refresh_styles() for live theme switching.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QFrame,
    QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor

from src.utils.constants import COLORS, SHADOWS


def _make_shadow(level: str = "md") -> QGraphicsDropShadowEffect:
    blur, x, y, alpha = SHADOWS[level]
    fx = QGraphicsDropShadowEffect()
    fx.setBlurRadius(blur)
    fx.setXOffset(x)
    fx.setYOffset(y)
    fx.setColor(QColor(0, 0, 0, alpha))
    return fx


# ── Card ───────────────────────────────────────────────────────────────────────

class Card(QFrame):
    """
    Base card — surface (L1) elevation.
    No hover border — subclasses that are interactive add their own.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._setup_base()

    def _setup_base(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(16, 14, 16, 14)
        self.layout.setSpacing(10)
        self._apply_card_style()

    def _apply_card_style(self) -> None:
        # No type selector — applies to this widget only, no cascade to child QFrames
        self.setStyleSheet(
            f"background-color: {COLORS['surface']}; border: none; border-radius: 4px;"
        )

    def refresh_styles(self) -> None:
        self._apply_card_style()


# ── StatCard ───────────────────────────────────────────────────────────────────

class StatCard(Card):
    """
    KPI / statistics card — surface_elevated (L2).

    Shows a large metric value, an UPPERCASE title label, optional trend
    subtitle, and a 2 px coloured left-accent stripe.
    """

    def __init__(
        self,
        title: str,
        value: str,
        subtitle: str = "",
        color: str | None = None,
        parent=None,
    ) -> None:
        self._stat_title = title
        self._stat_value = value
        self._stat_subtitle = subtitle
        self._accent_color: str = color or ""   # resolved after super().__init__
        super().__init__(parent)
        if not self._accent_color:
            self._accent_color = COLORS["primary"]
        self._setup_content()

    def _setup_base(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(14, 12, 14, 12)
        self.layout.setSpacing(6)
        self._apply_card_style()

    def _apply_card_style(self) -> None:
        accent = getattr(self, "_accent_color", COLORS["primary"])
        # No type selector — scoped to this widget only
        self.setStyleSheet(
            f"background-color: {COLORS['surface_elevated']};"
            f" border: 1px solid {COLORS['border_subtle']};"
            f" border-left: 3px solid {accent};"
            f" border-radius: 4px;"
        )

    def _setup_content(self) -> None:
        self.title_label = QLabel(self._stat_title.upper())
        self.title_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 10px;"
            f" font-weight: 700; letter-spacing: 1px;"
        )
        self.layout.addWidget(self.title_label)

        self.value_label = QLabel(self._stat_value)
        self.value_label.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.layout.addWidget(self.value_label)

        if self._stat_subtitle:
            self.subtitle_label = QLabel(self._stat_subtitle)
            self.subtitle_label.setStyleSheet(
                f"color: {self._accent_color}; font-size: 11px;"
            )
            self.layout.addWidget(self.subtitle_label)

        self.layout.addStretch()

    def set_value(self, value: str) -> None:
        self._stat_value = value
        self.value_label.setText(value)

    def set_subtitle(self, subtitle: str) -> None:
        self._stat_subtitle = subtitle
        if hasattr(self, "subtitle_label"):
            self.subtitle_label.setText(subtitle)

    def refresh_styles(self) -> None:
        self._apply_card_style()
        if hasattr(self, "title_label"):
            self.title_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 10px;"
                f" font-weight: 700; letter-spacing: 1px;"
            )
            self.value_label.setStyleSheet(f"color: {COLORS['text_primary']};")
            if hasattr(self, "subtitle_label"):
                self.subtitle_label.setStyleSheet(
                    f"color: {self._accent_color}; font-size: 11px;"
                )


# ── InfoCard ───────────────────────────────────────────────────────────────────

class InfoCard(Card):
    """
    Content card with title, optional description, and a generic content slot.
    Always borderless — background-color contrast provides visual grouping.
    """

    def __init__(
        self,
        title: str,
        description: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._info_title = title
        self._info_desc = description
        self._setup_content()

    def _apply_card_style(self) -> None:
        self.setStyleSheet(
            f"background-color: {COLORS['surface']}; border: none; border-radius: 4px;"
        )

    def _setup_content(self) -> None:
        if self._info_title:
            self.title_label = QLabel(self._info_title)
            self.title_label.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
            self.title_label.setStyleSheet(f"color: {COLORS['text_primary']};")
            self.layout.addWidget(self.title_label)

        if self._info_desc:
            self.description_label = QLabel(self._info_desc)
            self.description_label.setWordWrap(True)
            self.description_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 12px; line-height: 1.5;"
            )
            self.layout.addWidget(self.description_label)

        self.content_area = QVBoxLayout()
        self.content_area.setSpacing(6)
        self.layout.addLayout(self.content_area)

    def add_content(self, widget: QWidget) -> None:
        self.content_area.addWidget(widget)

    def refresh_styles(self) -> None:
        self._apply_card_style()
        if hasattr(self, "title_label"):
            self.title_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        if hasattr(self, "description_label"):
            self.description_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 12px; line-height: 1.5;"
            )


# ── ScoreCard ──────────────────────────────────────────────────────────────────

class ScoreCard(Card):
    """
    Score display card with colour-coded progress bar.
    Green ≥ 85 %, blue ≥ 70 %, amber ≥ 50 %, red < 50 %.
    """

    def __init__(
        self,
        title: str,
        score: float,
        max_score: float = 1.0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._score_title = title
        self.score = score
        self.max_score = max_score
        self._setup_content()

    def _setup_content(self) -> None:
        header = QHBoxLayout()

        self.title_label = QLabel(self._score_title)
        self.title_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px;"
        )
        header.addWidget(self.title_label)
        header.addStretch()

        self.score_label = QLabel(f"{self.score * 100:.0f}%")
        self.score_label.setFont(QFont("Segoe UI", 13, QFont.Weight.Bold))
        self.score_label.setStyleSheet(f"color: {self._score_color()};")
        header.addWidget(self.score_label)
        self.layout.addLayout(header)

        # Progress track
        self.progress_bar = QFrame()
        self.progress_bar.setFixedHeight(4)
        self.progress_bar.setStyleSheet(
            f"background-color: {COLORS['border_subtle']}; border-radius: 2px;"
        )
        self.progress_fill = QFrame(self.progress_bar)
        self._update_progress()
        self.layout.addWidget(self.progress_bar)

    def _score_color(self) -> str:
        if self.max_score == 0:
            return COLORS["text_secondary"]
        pct = self.score / self.max_score
        if pct >= 0.85:
            return COLORS["success"]
        if pct >= 0.70:
            return COLORS["primary"]
        if pct >= 0.50:
            return COLORS["warning"]
        return COLORS["error"]

    def _update_progress(self) -> None:
        pct = min(self.score / self.max_score, 1.0) if self.max_score else 0.0
        w = int(self.progress_bar.width() * pct) if self.progress_bar.width() > 0 else 0
        self.progress_fill.setGeometry(0, 0, max(w, 4), 4)
        self.progress_fill.setStyleSheet(
            f"background-color: {self._score_color()}; border-radius: 2px;"
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_progress()

    def set_score(self, score: float) -> None:
        self.score = score
        self.score_label.setText(f"{score * 100:.0f}%")
        self.score_label.setStyleSheet(f"color: {self._score_color()};")
        self._update_progress()

    def refresh_styles(self) -> None:
        self._apply_card_style()
        if hasattr(self, "title_label"):
            self.title_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 12px;"
            )
            self.score_label.setStyleSheet(f"color: {self._score_color()};")
            self.progress_bar.setStyleSheet(
                f"background-color: {COLORS['border_subtle']}; border-radius: 2px;"
            )
            self._update_progress()


# ── CandidateCard ──────────────────────────────────────────────────────────────

class CandidateCard(Card):
    """
    Candidate result card — surface_elevated (L2), most prominent.
    Shows name, AI match score badge, experience, and skill chips.
    Hovering brightens the border to VSCode focus-blue (interactive card).
    """

    def __init__(
        self,
        name: str,
        score: float,
        skills: list[str] | None = None,
        experience: str = "",
        parent=None,
    ) -> None:
        self._candidate_name = name
        self._candidate_score = score
        self._candidate_skills = skills or []
        self._candidate_exp = experience
        self._hovered: bool = False
        super().__init__(parent)
        self._setup_content()

    def _setup_base(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(14, 12, 14, 12)
        self.layout.setSpacing(8)
        self._apply_card_style()

    def _apply_card_style(self) -> None:
        border = COLORS["primary"] if self._hovered else COLORS["border_muted"]
        self.setStyleSheet(
            f"background-color: {COLORS['surface_elevated']};"
            f" border: 1px solid {border}; border-radius: 4px;"
        )

    def enterEvent(self, event) -> None:
        self._hovered = True
        self._apply_card_style()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self._apply_card_style()
        super().leaveEvent(event)

    def _setup_content(self) -> None:
        header = QHBoxLayout()

        self.name_label = QLabel(self._candidate_name)
        self.name_label.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
        self.name_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        header.addWidget(self.name_label)
        header.addStretch()

        score_pct = int((self._candidate_score or 0) * 100)
        score_color = self._score_color()
        self.score_badge = QLabel(f"{score_pct}%")
        self.score_badge.setFixedSize(42, 22)
        self.score_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._score_badge_color = score_color
        self._apply_badge_style()
        header.addWidget(self.score_badge)
        self.layout.addLayout(header)

        if self._candidate_exp:
            exp_label = QLabel(self._candidate_exp)
            exp_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 12px;"
            )
            self.layout.addWidget(exp_label)

        if self._candidate_skills:
            chips_row = QHBoxLayout()
            chips_row.setSpacing(4)
            for skill in self._candidate_skills[:4]:
                chip = QLabel(skill)
                chip.setStyleSheet(
                    f"""
                    QLabel {{
                        background-color: {COLORS['primary_glow']};
                        color: {COLORS['primary']};
                        border: 1px solid {COLORS['border_muted']};
                        padding: 2px 6px;
                        border-radius: 2px;
                        font-size: 10px;
                    }}
                    """
                )
                chips_row.addWidget(chip)
            if len(self._candidate_skills) > 4:
                more = QLabel(f"+{len(self._candidate_skills) - 4}")
                more.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: 10px;")
                chips_row.addWidget(more)
            chips_row.addStretch()
            self.layout.addLayout(chips_row)

    def _apply_badge_style(self) -> None:
        c = self._score_badge_color
        self.score_badge.setStyleSheet(
            f"""
            QLabel {{
                background-color: {c}22;
                color: {c};
                border: 1px solid {c};
                border-radius: 2px;
                font-weight: bold;
                font-size: 10px;
            }}
            """
        )

    def _score_color(self) -> str:
        s = self._candidate_score
        if s is None:
            return COLORS["text_secondary"]
        if s >= 0.85:
            return COLORS["success"]
        if s >= 0.70:
            return COLORS["primary"]
        if s >= 0.50:
            return COLORS["warning"]
        return COLORS["error"]

    def refresh_styles(self) -> None:
        self._score_badge_color = self._score_color()
        self._apply_card_style()
        if hasattr(self, "name_label"):
            self.name_label.setStyleSheet(f"color: {COLORS['text_primary']};")
            self._apply_badge_style()
