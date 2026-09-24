import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QFrame,
    QSizePolicy,
    QGraphicsDropShadowEffect,
)
from PyQt6.QtCore import Qt, QSize, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QIcon, QColor

from src.utils.config import get_settings
from src.utils.constants import APP_DISPLAY_NAME, VERSION, COLORS
from src.utils.theme import get_theme

from src.ui.views.dashboard_view import DashboardView
from src.ui.views.jobs_view import JobsView
from src.ui.views.matching_view import MatchingView
from src.ui.views.settings_view import SettingsView
from src.ui.views.candidates_view import CandidatesView
from src.ui.views.analytics_view import AnalyticsView


# ── Nav items ──────────────────────────────────────────────────────────────────

_NAV_ITEMS: list[tuple[str, str, str]] = [
    ("Dashboard", "dashboard", "⊟"),
    ("Candidates", "candidates", "◈"),
    ("Job Postings", "jobs", "⊡"),
    ("AI Matching", "matching", "⊕"),
    ("Analytics", "analytics", "⊘"),
    ("Settings", "settings", "⊗"),
]


# ── Unified nav button (icon + label, single panel) ────────────────────────────


class NavButton(QWidget):
    """
    Single nav row showing a glyph icon + text label side by side.

    States:
      • inactive  — dim text, transparent background
      • hover     — brighter text, subtle bg highlight
      • active    — primary text + 2 px left accent stripe
    """

    def __init__(self, glyph: str, text: str, parent=None) -> None:
        super().__init__(parent)
        self._checked: bool = False
        self._glyph = glyph
        self._text = text
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(32)
        self._clicked_callback = lambda: None
        self._build_ui()

    def _build_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # 2 px left accent stripe
        self._accent_bar = QFrame()
        self._accent_bar.setFixedWidth(2)
        outer.addWidget(self._accent_bar)

        outer.addSpacing(10)

        # Glyph icon
        self._icon_label = QLabel(self._glyph)
        self._icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._icon_label.setFixedWidth(16)
        self._icon_label.setFont(QFont("Segoe UI Symbol", 11))
        outer.addWidget(self._icon_label)

        outer.addSpacing(8)

        # Text label
        self._text_label = QLabel(self._text)
        self._text_label.setFont(QFont("Segoe UI", 10))
        outer.addWidget(self._text_label)
        outer.addStretch()

        self._apply_style()

    def _apply_style(self) -> None:
        if self._checked:
            self.setStyleSheet(f"QWidget {{ background-color: {COLORS['primary_glow']}; }}")
            self._accent_bar.setStyleSheet(f"background-color: {COLORS['primary']}; border: none;")
            self._icon_label.setStyleSheet(
                f"color: {COLORS['text_primary']}; background-color: transparent;"
            )
            self._text_label.setStyleSheet(
                f"color: {COLORS['text_primary']}; font-weight: 600;"
                f" background-color: transparent;"
            )
        else:
            self.setStyleSheet("QWidget { background-color: transparent; }")
            self._accent_bar.setStyleSheet("background-color: transparent; border: none;")
            self._icon_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; background-color: transparent;"
            )
            self._text_label.setStyleSheet(
                f"color: {COLORS['text_secondary']}; background-color: transparent;"
            )

    def setChecked(self, checked: bool) -> None:
        self._checked = checked
        self._apply_style()

    def isChecked(self) -> bool:
        return self._checked

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._clicked_callback()

    def enterEvent(self, event) -> None:
        if not self._checked:
            self.setStyleSheet(f"QWidget {{ background-color: {COLORS['surface_overlay']}; }}")
            self._icon_label.setStyleSheet(
                f"color: {COLORS['text_primary']}; background-color: transparent;"
            )
            self._text_label.setStyleSheet(
                f"color: {COLORS['text_primary']}; background-color: transparent;"
            )
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if not self._checked:
            self._apply_style()
        super().leaveEvent(event)

    def set_click_callback(self, cb) -> None:
        self._clicked_callback = cb

    def refresh_styles(self) -> None:
        self._apply_style()


# ── Nav panel (unified sidebar, icon + label) ──────────────────────────────────


class NavPanel(QFrame):
    """
    Single unified sidebar panel — 210 px wide, shows icon + label on each row.
    Replaces the old split ActivityBar (48 px icons) + SidebarPanel (162 px text).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(210)
        self.nav_buttons: list[NavButton] = []
        self._apply_style()
        self._build_ui()

    def _apply_style(self) -> None:
        # Use ID selector so border-right applies only to NavPanel, not child QFrames
        self.setObjectName("navPanel")
        self.setStyleSheet(
            f"""
            QFrame#navPanel {{
                background-color: {COLORS['surface']};
                border-right: 1px solid {COLORS['border_subtle']};
            }}
            QFrame#navPanel QWidget {{
                background-color: {COLORS['surface']};
            }}
            """
        )

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Section header label (VSCode "EXPLORER" pattern) — stored for theme refresh
        self._section_header = QLabel("EXPLORER")
        self._section_header.setFixedHeight(26)
        self._section_header.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 10px;"
            f" font-weight: 700; letter-spacing: 1.5px; padding: 0 14px;"
            f" background-color: transparent;"
        )
        layout.addWidget(self._section_header)

        nav_frame = QWidget()
        nav_layout = QVBoxLayout(nav_frame)
        nav_layout.setContentsMargins(0, 4, 0, 4)
        nav_layout.setSpacing(1)

        for text, name, glyph in _NAV_ITEMS[:-1]:
            btn = NavButton(glyph, text)
            btn.setObjectName(name)
            self.nav_buttons.append(btn)
            nav_layout.addWidget(btn)

        nav_layout.addStretch()

        self._divider = QFrame()
        self._divider.setFrameShape(QFrame.Shape.HLine)
        self._divider.setFixedHeight(1)
        self._divider.setStyleSheet(f"background-color: {COLORS['border_subtle']}; border: none;")
        nav_layout.addWidget(self._divider)
        nav_layout.addSpacing(2)

        text, name, glyph = _NAV_ITEMS[-1]
        settings_btn = NavButton(glyph, text)
        settings_btn.setObjectName(name)
        self.nav_buttons.append(settings_btn)
        nav_layout.addWidget(settings_btn)
        nav_layout.addSpacing(8)

        self._ver_label = QLabel(f"v{VERSION}")
        self._ver_label.setStyleSheet(
            f"color: {COLORS['text_tertiary']}; font-size: 10px; padding: 4px 14px;"
            f" background-color: transparent;"
        )
        nav_layout.addWidget(self._ver_label)

        layout.addWidget(nav_frame)

        if self.nav_buttons:
            self.nav_buttons[0].setChecked(True)

    def refresh_styles(self) -> None:
        self._apply_style()
        if hasattr(self, "_section_header"):
            self._section_header.setStyleSheet(
                f"color: {COLORS['text_secondary']}; font-size: 10px;"
                f" font-weight: 700; letter-spacing: 1.5px; padding: 0 14px;"
                f" background-color: transparent;"
            )
        if hasattr(self, "_divider"):
            self._divider.setStyleSheet(
                f"background-color: {COLORS['border_subtle']}; border: none;"
            )
        if hasattr(self, "_ver_label"):
            self._ver_label.setStyleSheet(
                f"color: {COLORS['text_tertiary']}; font-size: 10px; padding: 4px 14px;"
                f" background-color: transparent;"
            )
        for btn in self.nav_buttons:
            btn.refresh_styles()


# ── Top header bar ─────────────────────────────────────────────────────────────


class ContentHeader(QFrame):
    """
    Thin header above the content stack.
    Shows current view title, workspace pill, theme toggle, and app tag.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(36)
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(8)

        self._title = QLabel("Dashboard")
        self._title.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        layout.addWidget(self._title)

        layout.addStretch()

        self._workspace_pill = QPushButton("No Workspace  ▾")
        self._workspace_pill.setCursor(Qt.CursorShape.PointingHandCursor)
        self._workspace_pill.clicked.connect(self._open_workspace_selector)
        layout.addWidget(self._workspace_pill)

        layout.addSpacing(4)

        self._theme_btn = QPushButton("☽")
        self._theme_btn.setFixedSize(28, 28)
        self._theme_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._theme_btn.setToolTip("Toggle light / dark theme")
        self._theme_btn.clicked.connect(self._toggle_theme)
        layout.addWidget(self._theme_btn)

        layout.addSpacing(8)

        self._app_tag = QLabel(APP_DISPLAY_NAME)
        layout.addWidget(self._app_tag)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QFrame {{
                background-color: {COLORS['surface_elevated']};
                border-bottom: 1px solid {COLORS['border_subtle']};
            }}
            """
        )
        self._title.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self._workspace_pill.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 3px;
                padding: 2px 8px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['surface_overlay']};
                color: {COLORS['text_primary']};
                border-color: {COLORS['primary']};
            }}
            """
        )
        is_dark = get_theme().is_dark
        self._theme_btn.setText("☽" if is_dark else "☀")
        self._theme_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
                font-size: 16px;
                padding: 0px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['surface_overlay']};
                color: {COLORS['text_primary']};
                border-color: {COLORS['primary']};
            }}
            """
        )
        self._app_tag.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: 10px;")

    def set_title(self, title: str) -> None:
        self._title.setText(title)

    def refresh_styles(self) -> None:
        self._apply_style()

    def update_workspace(self, workspace: object) -> None:
        if workspace is None:
            self._workspace_pill.setText("No Workspace  ▾")
        else:
            name: str = getattr(workspace, "name", "Workspace")
            display = name if len(name) <= 24 else name[:22] + "…"
            self._workspace_pill.setText(f"⊡ {display}  ▾")

    def _toggle_theme(self) -> None:
        get_theme().toggle()

    def _open_workspace_selector(self) -> None:
        from src.ui.dialogs.workspace_dialogs import WorkspaceSelectorDialog
        from src.utils.workspace_state import get_workspace_state

        dlg = WorkspaceSelectorDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.selected_workspace:
            get_workspace_state().set_workspace(dlg.selected_workspace)


# ── Status bar (VSCode blue, always) ──────────────────────────────────────────


class StatusBar(QFrame):
    """
    24 px status bar at the very bottom of the window.
    Uses the statusbar_bg token which stays VSCode blue in both themes.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(24)
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        layout.setSpacing(0)

        self._left = QLabel("⬡  AI-ATS")
        layout.addWidget(self._left)
        layout.addStretch()
        self._right = QLabel(f"v{VERSION}   Python   UTF-8")
        layout.addWidget(self._right)

    def _apply_style(self) -> None:
        bg = COLORS.get("statusbar_bg", COLORS["primary"])
        fg = COLORS.get("statusbar_fg", COLORS["text_on_primary"])
        label_qss = f"color: {fg}; background-color: transparent; font-size: 11px;"
        self.setStyleSheet(f"QFrame {{ background-color: {bg}; }}")
        self._left.setStyleSheet(label_qss)
        self._right.setStyleSheet(label_qss)

    def refresh_styles(self) -> None:
        self._apply_style()

    def update_workspace(self, workspace: object) -> None:
        if workspace is None:
            self._left.setText("⬡  AI-ATS")
        else:
            name: str = getattr(workspace, "name", "Workspace")
            display = name if len(name) <= 20 else name[:18] + "…"
            self._left.setText(f"⬡  {display}")


# ── Placeholder view ───────────────────────────────────────────────────────────


class PlaceholderView(QWidget):
    def __init__(self, title: str, description: str, parent=None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"background-color: {COLORS['background']};")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        t = QLabel(title)
        t.setFont(QFont("Segoe UI", 22, QFont.Weight.Bold))
        t.setStyleSheet(f"color: {COLORS['text_primary']};")
        layout.addWidget(t, alignment=Qt.AlignmentFlag.AlignCenter)

        d = QLabel(description)
        d.setFont(QFont("Segoe UI", 13))
        d.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(d, alignment=Qt.AlignmentFlag.AlignCenter)

    def refresh_styles(self) -> None:
        self.setStyleSheet(f"background-color: {COLORS['background']};")


# ── Global QSS ────────────────────────────────────────────────────────────────


def _build_global_qss() -> str:
    """Full QSS stylesheet built from current COLORS. Applied to MainWindow."""
    C = COLORS
    return f"""
        /* ── Base ──────────────────────────────────────────────────────── */
        QMainWindow, QWidget {{
            background-color: {C['background']};
            color: {C['text_primary']};
            font-family: "Segoe UI", "Inter", "Ubuntu", sans-serif;
            font-size: 13px;
        }}

        /* ── Scrollbars — VSCode thin style ─────────────────────────────── */
        QScrollBar:vertical {{
            background: transparent;
            width: 10px;
            border: none;
        }}
        QScrollBar::handle:vertical {{
            background: {C['border_muted']};
            border-radius: 5px;
            min-height: 20px;
            margin: 2px 2px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {C['text_secondary']};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar:horizontal {{
            background: transparent;
            height: 10px;
            border: none;
        }}
        QScrollBar::handle:horizontal {{
            background: {C['border_muted']};
            border-radius: 5px;
            min-width: 20px;
            margin: 2px 2px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {C['text_secondary']};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

        /* ── Tooltip ─────────────────────────────────────────────────────── */
        QToolTip {{
            background-color: {C['surface_overlay']};
            color: {C['text_primary']};
            border: 1px solid {C['border_muted']};
            border-radius: 4px;
            padding: 4px 8px;
            font-size: 12px;
        }}

        /* ── Inputs ──────────────────────────────────────────────────────── */
        QLineEdit, QTextEdit, QPlainTextEdit {{
            background-color: {C['surface_elevated']};
            color: {C['text_primary']};
            border: 1px solid {C['border_muted']};
            border-radius: 2px;
            padding: 5px 8px;
            selection-background-color: {C['primary_glow']};
        }}
        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus {{
            border-color: {C['primary']};
            outline: none;
        }}
        QLineEdit:hover:!focus, QTextEdit:hover:!focus {{
            border-color: {C['text_secondary']};
        }}
        QLineEdit::placeholder {{ color: {C['text_tertiary']}; }}

        /* ── ComboBox ─────────────────────────────────────────────────────── */
        QComboBox {{
            background-color: {C['surface_elevated']};
            color: {C['text_primary']};
            border: 1px solid {C['border_muted']};
            border-radius: 2px;
            padding: 5px 8px;
            min-height: 28px;
        }}
        QComboBox:focus {{ border-color: {C['primary']}; }}
        QComboBox:hover:!focus {{ border-color: {C['text_secondary']}; }}
        QComboBox::drop-down {{ border: none; width: 20px; }}
        QComboBox QAbstractItemView {{
            background-color: {C['surface_overlay']};
            color: {C['text_primary']};
            border: 1px solid {C['border_muted']};
            selection-background-color: {C['primary_glow']};
            selection-color: {C['primary']};
            padding: 2px;
            outline: none;
        }}

        /* ── Labels ───────────────────────────────────────────────────────── */
        QLabel {{ color: {C['text_primary']}; background-color: transparent; }}

        /* ── Default QPushButton ──────────────────────────────────────────── */
        QPushButton {{
            background-color: {C['surface_elevated']};
            color: {C['text_primary']};
            border: 1px solid {C['border_muted']};
            border-radius: 2px;
            padding: 5px 12px;
            font-size: 12px;
        }}
        QPushButton:hover {{
            background-color: {C['surface_overlay']};
            border-color: {C['text_secondary']};
        }}
        QPushButton:pressed {{
            background-color: {C['primary_glow']};
        }}
        QPushButton:disabled {{
            color: {C['text_tertiary']};
            border-color: {C['border_subtle']};
        }}

        /* ── Splitter ─────────────────────────────────────────────────────── */
        QSplitter::handle {{ background-color: {C['border_subtle']}; }}
        QSplitter::handle:horizontal {{ width: 1px; }}
        QSplitter::handle:vertical   {{ height: 1px; }}

        /* ── Tabs ─────────────────────────────────────────────────────────── */
        QTabWidget::pane {{
            border: 1px solid {C['border_subtle']};
            background-color: {C['surface']};
        }}
        QTabBar::tab {{
            background-color: {C['surface_elevated']};
            color: {C['text_secondary']};
            padding: 7px 16px;
            border: 1px solid {C['border_subtle']};
            border-bottom: none;
            margin-right: 1px;
        }}
        QTabBar::tab:selected {{
            background-color: {C['surface']};
            color: {C['text_primary']};
            border-bottom: 2px solid {C['primary']};
        }}
        QTabBar::tab:hover:!selected {{
            background-color: {C['surface_overlay']};
            color: {C['text_primary']};
        }}

        /* ── CheckBox ─────────────────────────────────────────────────────── */
        QCheckBox {{
            color: {C['text_primary']};
            spacing: 8px;
        }}
        QCheckBox::indicator {{
            width: 15px;
            height: 15px;
            border-radius: 2px;
            border: 1px solid {C['border_muted']};
            background-color: {C['surface_elevated']};
        }}
        QCheckBox::indicator:checked {{
            background-color: {C['primary']};
            border-color: {C['primary']};
        }}
        QCheckBox::indicator:hover {{
            border-color: {C['primary']};
        }}

        /* ── SpinBox ──────────────────────────────────────────────────────── */
        QSpinBox, QDoubleSpinBox {{
            background-color: {C['surface_elevated']};
            color: {C['text_primary']};
            border: 1px solid {C['border_muted']};
            border-radius: 2px;
            padding: 4px 6px;
        }}
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border-color: {C['primary']};
        }}
        QSpinBox::up-button, QSpinBox::down-button,
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            background-color: {C['surface_overlay']};
            border: none;
            width: 16px;
        }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: {C['primary_glow']};
        }}

        /* ── GroupBox ─────────────────────────────────────────────────────── */
        QGroupBox {{
            color: {C['text_secondary']};
            border: 1px solid {C['border_subtle']};
            border-radius: 4px;
            margin-top: 14px;
            padding: 14px 10px 10px 10px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            left: 10px;
            padding: 0 6px;
            background-color: {C['surface']};
            color: {C['text_secondary']};
        }}

        /* ── ProgressBar ──────────────────────────────────────────────────── */
        QProgressBar {{
            background-color: {C['surface_elevated']};
            border: none;
            border-radius: 2px;
            height: 4px;
            text-align: center;
            color: transparent;
        }}
        QProgressBar::chunk {{
            background-color: {C['primary']};
            border-radius: 2px;
        }}

        /* ── Slider ───────────────────────────────────────────────────────── */
        QSlider::groove:horizontal {{
            background: {C['surface_elevated']};
            height: 4px;
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {C['primary']};
            width: 12px;
            height: 12px;
            margin: -4px 0;
            border-radius: 6px;
        }}
        QSlider::sub-page:horizontal {{
            background: {C['primary']};
            border-radius: 2px;
        }}

        /* ── Dialog & MessageBox ──────────────────────────────────────────── */
        QDialog, QMessageBox {{
            background-color: {C['surface']};
        }}
        QDialog QLabel, QMessageBox QLabel {{
            color: {C['text_primary']};
        }}

        /* ── ListWidget ───────────────────────────────────────────────────── */
        QListWidget {{
            background-color: {C['surface_elevated']};
            color: {C['text_primary']};
            border: 1px solid {C['border_muted']};
            border-radius: 2px;
            outline: none;
        }}
        QListWidget::item {{ padding: 6px 10px; }}
        QListWidget::item:selected {{
            background-color: {C['primary_glow']};
            color: {C['primary']};
        }}
        QListWidget::item:hover:!selected {{
            background-color: {C['surface_overlay']};
        }}

        /* ── TableWidget ──────────────────────────────────────────────────── */
        QTableWidget {{
            background-color: {C['surface']};
            color: {C['text_primary']};
            gridline-color: {C['border_subtle']};
            border: none;
            outline: none;
        }}
        QTableWidget::item {{ padding: 6px 10px; }}
        QTableWidget::item:selected {{
            background-color: {C['primary_glow']};
            color: {C['text_primary']};
        }}
        QTableWidget::item:hover:!selected {{
            background-color: {C['surface_overlay']};
        }}
        QHeaderView::section {{
            background-color: {C['surface_elevated']};
            color: {C['text_secondary']};
            border: none;
            border-right: 1px solid {C['border_subtle']};
            border-bottom: 1px solid {C['border_muted']};
            padding: 6px 10px;
            font-weight: 600;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        /* ── ScrollArea ───────────────────────────────────────────────────── */
        QScrollArea {{
            background-color: transparent;
            border: none;
        }}
        QScrollArea > QWidget > QWidget {{
            background-color: transparent;
        }}

        /* ── FormLayout labels ────────────────────────────────────────────── */
        QFormLayout QLabel {{
            color: {C['text_secondary']};
            font-size: 12px;
        }}

        /* ── DialogButtonBox ──────────────────────────────────────────────── */
        QDialogButtonBox QPushButton {{
            min-width: 72px;
            padding: 5px 12px;
        }}
    """


# ── Main window ────────────────────────────────────────────────────────────────


class MainWindow(QMainWindow):
    """Main application window — VSCode-style chrome."""

    def __init__(self) -> None:
        super().__init__()
        self.settings = get_settings()
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setWindowTitle(APP_DISPLAY_NAME)
        self.setMinimumSize(1100, 680)
        self.resize(
            self.settings.ui.window_width,
            self.settings.ui.window_height,
        )

        self.setStyleSheet(_build_global_qss())

        central = QWidget()
        self.setCentralWidget(central)

        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Main body (sidebar + content) ──────────────────────────────────
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        # Unified nav panel (210px, icon + label per row)
        self.nav_panel = NavPanel()
        body.addWidget(self.nav_panel)

        # Right column: header + content stack
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(0)

        self.header = ContentHeader()
        right_col.addWidget(self.header)

        self.content_stack = QStackedWidget()
        self.content_stack.setStyleSheet(f"background-color: {COLORS['background']};")
        right_col.addWidget(self.content_stack)

        right_wrapper = QWidget()
        right_wrapper.setLayout(right_col)
        body.addWidget(right_wrapper)

        body_widget = QWidget()
        body_widget.setLayout(body)
        root.addWidget(body_widget)

        # Status bar (24px)
        self.status_bar = StatusBar()
        root.addWidget(self.status_bar)

        # ── Lazy refresh tracking ──────────────────────────────────────────
        self._view_needs_refresh: dict[int, bool] = {}

        # ── Instantiate views ──────────────────────────────────────────────
        self.dashboard_view = DashboardView()
        self.content_stack.addWidget(self.dashboard_view)

        self.candidates_view = CandidatesView()
        self.content_stack.addWidget(self.candidates_view)

        self.jobs_view = JobsView()
        self.content_stack.addWidget(self.jobs_view)

        self.matching_view = MatchingView()
        self.content_stack.addWidget(self.matching_view)

        self.analytics_view = AnalyticsView()
        self.content_stack.addWidget(self.analytics_view)

        try:
            self.settings_view = SettingsView()
        except Exception:
            self.settings_view = PlaceholderView("Settings", "Application configuration")
        self.content_stack.addWidget(self.settings_view)

        for i in range(self.content_stack.count()):
            self._view_needs_refresh[i] = True
        self._view_needs_refresh[0] = False

        # ── Wire nav callbacks ─────────────────────────────────────────────
        for i, btn in enumerate(self.nav_panel.nav_buttons):
            btn.set_click_callback(lambda idx=i: self.switch_view(idx))

        # ── Cross-view signals ─────────────────────────────────────────────
        self.dashboard_view.navigate_to_view.connect(self.switch_view)
        self.jobs_view.job_created.connect(lambda: self.mark_view_dirty(3))

        # ── Workspace state ────────────────────────────────────────────────
        from src.utils.workspace_state import get_workspace_state

        self._ws_state = get_workspace_state()
        self._ws_state.workspace_changed.connect(self.header.update_workspace)
        self._ws_state.workspace_changed.connect(self.status_bar.update_workspace)
        self._bootstrap_workspace()

        # ── Theme switching ────────────────────────────────────────────────
        self._theme = get_theme()
        self._theme.theme_changed.connect(self._apply_theme)

        # ── Load dashboard ─────────────────────────────────────────────────
        self.dashboard_view.refresh()

    def switch_view(self, index: int) -> None:
        self.content_stack.setCurrentIndex(index)

        for i, btn in enumerate(self.nav_panel.nav_buttons):
            btn.setChecked(i == index)

        if 0 <= index < len(_NAV_ITEMS):
            self.header.set_title(_NAV_ITEMS[index][0])

        if self._view_needs_refresh.get(index, False):
            widget = self.content_stack.widget(index)
            if hasattr(widget, "refresh"):
                widget.refresh()
            self._view_needs_refresh[index] = False

    def mark_view_dirty(self, index: int) -> None:
        self._view_needs_refresh[index] = True

    def _bootstrap_workspace(self) -> None:
        try:
            from src.data.sql.repositories import get_workspace_repository

            workspaces = get_workspace_repository().list_recent(limit=1)
            if workspaces:
                self._ws_state.set_workspace(workspaces[0])
        except Exception:
            pass

    def _apply_theme(self, _mode: str = "") -> None:
        """Rebuild global QSS and cascade refresh_styles() to every child widget."""
        # 1. Global window QSS
        self.setStyleSheet(_build_global_qss())

        # 2. Direct chrome widgets
        self.nav_panel.refresh_styles()
        self.header.refresh_styles()
        self.status_bar.refresh_styles()
        self.content_stack.setStyleSheet(f"background-color: {COLORS['background']};")

        # 3. Cascade to ALL child widgets that implement refresh_styles()
        for widget in self.findChildren(QWidget):
            try:
                if callable(getattr(widget, "refresh_styles", None)):
                    widget.refresh_styles()
            except Exception:
                pass


# ── Entry point ────────────────────────────────────────────────────────────────


def run_application() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setApplicationVersion(VERSION)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    return app.exec()


def main() -> None:
    sys.exit(run_application())


if __name__ == "__main__":
    main()
