from __future__ import annotations

import getpass
from datetime import datetime, timezone
from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QColor
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
    QFrame,
)

from src.utils.constants import COLORS
from src.ui.widgets.buttons import DangerButton, PrimaryButton, SecondaryButton


def _dialog_qss() -> str:
    return f"""
        QDialog {{
            background-color: {COLORS['surface']};
        }}
        QLabel {{
            color: {COLORS['text_primary']};
            background-color: transparent;
        }}
        QLineEdit, QPlainTextEdit {{
            background-color: {COLORS['surface_elevated']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border_muted']};
            border-radius: 2px;
            padding: 5px 8px;
        }}
        QLineEdit:focus, QPlainTextEdit:focus {{
            border-color: {COLORS['primary']};
        }}
        QListWidget {{
            background-color: {COLORS['surface_elevated']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border_muted']};
            border-radius: 2px;
            outline: none;
        }}
        QListWidget::item {{
            padding: 8px 10px;
            border-bottom: 1px solid {COLORS['border_subtle']};
        }}
        QListWidget::item:selected {{
            background-color: {COLORS['primary_glow']};
            color: {COLORS['primary']};
        }}
        QListWidget::item:hover:!selected {{
            background-color: {COLORS['surface_overlay']};
        }}
        QDialogButtonBox QPushButton {{
            min-width: 72px;
            padding: 5px 14px;
        }}
    """


# ── Workspace Selector ─────────────────────────────────────────────────────────

class WorkspaceSelectorDialog(QDialog):
    """List recent workspaces — switch, create new, or archive."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Switch Workspace")
        self.setMinimumSize(560, 460)
        self._selected_workspace: Optional[object] = None
        self._workspaces: list[object] = []
        self._build_ui()
        self._load()

    def _build_ui(self) -> None:
        self.setStyleSheet(_dialog_qss())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Title
        title_label = QLabel("Switch Workspace")
        title_label.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        layout.addWidget(title_label)

        subtitle = QLabel("Select a workspace or create a new one.")
        subtitle.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        layout.addWidget(subtitle)

        # Hairline separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(
            f"background-color: {COLORS['border_subtle']}; border: none;"
        )
        layout.addWidget(sep)

        self._list = QListWidget()
        self._list.itemDoubleClicked.connect(self._on_switch)
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()

        self._new_btn = SecondaryButton("+ New Workspace")
        self._new_btn.clicked.connect(self._on_new)
        btn_row.addWidget(self._new_btn)

        self._archive_btn = SecondaryButton("Archive Selected")
        self._archive_btn.setEnabled(False)
        self._archive_btn.clicked.connect(self._on_archive)
        btn_row.addWidget(self._archive_btn)

        btn_row.addStretch()

        self._switch_btn = PrimaryButton("Switch")
        self._switch_btn.setEnabled(False)
        self._switch_btn.clicked.connect(self._on_switch)
        btn_row.addWidget(self._switch_btn)

        layout.addLayout(btn_row)

    def _load(self) -> None:
        try:
            from src.data.sql.repositories import (
                get_job_record_repository,
                get_workspace_repository,
            )
            ws_repo = get_workspace_repository()
            job_repo = get_job_record_repository()
            self._workspaces = ws_repo.list_recent(limit=20)
        except Exception:
            self._workspaces = []
            job_repo = None

        self._list.clear()

        if not self._workspaces:
            placeholder = QListWidgetItem(
                "No workspaces yet — click '+ New Workspace' to get started"
            )
            placeholder.setFlags(placeholder.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            placeholder.setForeground(
                QColor(COLORS["text_tertiary"])
            )
            self._list.addItem(placeholder)
            return

        for ws in self._workspaces:
            job_count = 0
            if job_repo is not None:
                try:
                    job_count = job_repo.count_for_workspace(ws.id)
                except Exception:
                    pass

            last_opened = ws.last_opened_at
            age_str = ""
            if last_opened is not None:
                delta = datetime.now(timezone.utc) - last_opened
                age_str = " · today" if delta.days == 0 else f" · {delta.days}d ago"

            jobs_str = f"{job_count} job" + ("s" if job_count != 1 else "")
            label = (
                f"{ws.name}   "
                f"[{ws.status.value}]{age_str}   ({jobs_str})"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, ws)
            self._list.addItem(item)

    def _on_selection_changed(self) -> None:
        items = self._list.selectedItems()
        if not items:
            self._switch_btn.setEnabled(False)
            self._archive_btn.setEnabled(False)
            return
        ws = items[0].data(Qt.ItemDataRole.UserRole)
        if ws is None:
            self._switch_btn.setEnabled(False)
            self._archive_btn.setEnabled(False)
            return
        from src.data.sql.models.workspace import WorkspaceStatus
        self._switch_btn.setEnabled(True)
        self._archive_btn.setEnabled(ws.status == WorkspaceStatus.ACTIVE)

    def _on_switch(self) -> None:
        items = self._list.selectedItems()
        if not items:
            return
        ws = items[0].data(Qt.ItemDataRole.UserRole)
        if ws is None:
            return
        try:
            from src.data.sql.repositories import get_workspace_repository
            get_workspace_repository().touch(ws.id)
        except Exception:
            pass
        self._selected_workspace = ws
        self.accept()

    def _on_new(self) -> None:
        dlg = CreateWorkspaceDialog(parent=self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            from src.data.sql.repositories import get_workspace_repository
            ws = get_workspace_repository().create_workspace(
                name=dlg.workspace_name,
                description=dlg.workspace_description or None,
                created_by=_current_user(),
            )
            self._selected_workspace = ws
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not create workspace:\n{exc}")

    def _on_archive(self) -> None:
        items = self._list.selectedItems()
        if not items:
            return
        ws = items[0].data(Qt.ItemDataRole.UserRole)
        if ws is None:
            return
        reply = QMessageBox.question(
            self,
            "Archive Workspace",
            f'Archive "{ws.name}"?\n\n'
            "Archived workspaces are read-only. You can restore them later from Settings.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        try:
            from src.data.sql.repositories import get_workspace_repository
            get_workspace_repository().archive(ws.id)
            self._load()
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not archive workspace:\n{exc}")

    @property
    def selected_workspace(self) -> Optional[object]:
        return self._selected_workspace

    def refresh_styles(self) -> None:
        self.setStyleSheet(_dialog_qss())


# ── Create Workspace ───────────────────────────────────────────────────────────

class CreateWorkspaceDialog(QDialog):
    """Name + optional description for a new workspace."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("New Workspace")
        self.setFixedWidth(460)
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet(_dialog_qss())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title_label = QLabel("Create Workspace")
        title_label.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        layout.addWidget(title_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(
            f"background-color: {COLORS['border_subtle']}; border: none;"
        )
        layout.addWidget(sep)

        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        name_label = QLabel("Name *")
        name_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("e.g. Senior Engineers — Q3 2026")
        self._name_edit.setMinimumHeight(30)
        self._name_edit.textChanged.connect(self._validate)
        form.addRow(name_label, self._name_edit)

        desc_label = QLabel("Description")
        desc_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")

        self._desc_edit = QPlainTextEdit()
        self._desc_edit.setPlaceholderText("Optional description…")
        self._desc_edit.setFixedHeight(72)
        form.addRow(desc_label, self._desc_edit)

        layout.addLayout(form)

        self._error_label = QLabel("")
        self._error_label.setStyleSheet(
            f"color: {COLORS['error']}; font-size: 11px;"
        )
        self._error_label.setVisible(False)
        layout.addWidget(self._error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self._save_btn = buttons.button(QDialogButtonBox.StandardButton.Save)
        self._save_btn.setEnabled(False)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _validate(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            self._error_label.setText("Name is required.")
            self._error_label.setVisible(True)
            self._save_btn.setEnabled(False)
        elif len(name) > 200:
            self._error_label.setText("Name must be 200 characters or fewer.")
            self._error_label.setVisible(True)
            self._save_btn.setEnabled(False)
        else:
            self._error_label.setVisible(False)
            self._save_btn.setEnabled(True)

    @property
    def workspace_name(self) -> str:
        return self._name_edit.text().strip()

    @property
    def workspace_description(self) -> str:
        return self._desc_edit.toPlainText().strip()

    def refresh_styles(self) -> None:
        self.setStyleSheet(_dialog_qss())


# ── Purge Confirmation ─────────────────────────────────────────────────────────

class PurgeWorkspaceDialog(QDialog):
    """User must type the exact workspace name to confirm permanent deletion."""

    def __init__(
        self,
        workspace_name: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Purge Workspace")
        self.setFixedWidth(500)
        self._workspace_name = workspace_name
        self._build_ui()

    def _build_ui(self) -> None:
        self.setStyleSheet(_dialog_qss())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        title_label = QLabel("Purge Workspace")
        title_label.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        title_label.setStyleSheet(f"color: {COLORS['error']};")
        layout.addWidget(title_label)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(
            f"background-color: {COLORS['border_subtle']}; border: none;"
        )
        layout.addWidget(sep)

        warning = QLabel(
            f"⚠  This will permanently delete all data for\n"
            f'"{self._workspace_name}":\n\n'
            "  • All job records\n"
            "  • All match scores (including manual overrides)\n\n"
            "Audit logs are preserved (workspace reference set to null).\n"
            "This cannot be undone."
        )
        warning.setWordWrap(True)
        warning.setStyleSheet(
            f"color: {COLORS['warning']}; font-size: 12px; padding: 12px;"
            f" background-color: {COLORS['warning_dim']}; border-radius: 4px;"
        )
        layout.addWidget(warning)

        confirm_label = QLabel("Type the workspace name to confirm:")
        confirm_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px;"
        )
        layout.addWidget(confirm_label)

        self._confirm_edit = QLineEdit()
        self._confirm_edit.setPlaceholderText(self._workspace_name)
        self._confirm_edit.setMinimumHeight(30)
        self._confirm_edit.textChanged.connect(self._validate)
        layout.addWidget(self._confirm_edit)

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = SecondaryButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        self._purge_btn = DangerButton("Purge Permanently")
        self._purge_btn.setEnabled(False)
        self._purge_btn.clicked.connect(self.accept)
        btn_row.addWidget(self._purge_btn)

        layout.addLayout(btn_row)

    def _validate(self) -> None:
        self._purge_btn.setEnabled(
            self._confirm_edit.text().strip() == self._workspace_name
        )

    def refresh_styles(self) -> None:
        self.setStyleSheet(_dialog_qss())


# ── Helpers ────────────────────────────────────────────────────────────────────

def _current_user() -> str:
    try:
        return getpass.getuser()
    except Exception:
        return "unknown"
