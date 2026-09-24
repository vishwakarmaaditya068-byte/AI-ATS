"""
Settings view for AI-ATS application.

Provides interface for configuring application settings.
"""

from PyQt6.QtWidgets import (
    QFileDialog,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QGroupBox,
    QFormLayout,
    QMessageBox,
    QTabWidget,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from src.utils.constants import COLORS, DEFAULT_SCORING_WEIGHTS
from src.ui.views.base_view import BaseView
from src.ui.widgets import (
    Card,
    InfoCard,
    PrimaryButton,
    SecondaryButton,
)
from src.ui.widgets.buttons import DangerButton


class SettingsView(BaseView):
    """
    Application settings view.

    Provides configuration for:
    - General settings (theme, language)
    - Matching settings (weights, thresholds)
    - Database settings
    - AI/ML settings
    """

    def __init__(self, parent=None):
        """Initialize the settings view."""
        super().__init__(
            title="Settings",
            description="Configure application settings and preferences",
            parent=parent,
        )
        self._setup_settings_view()
        self._load_current_settings()

    def _tab_widget_qss(self) -> str:
        return f"""
            QTabWidget::pane {{
                border: 1px solid {COLORS['border_subtle']};
                background-color: {COLORS['surface']};
                padding: 14px;
            }}
            QTabBar::tab {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_secondary']};
                padding: 7px 16px;
                border: 1px solid {COLORS['border_subtle']};
                border-bottom: none;
                margin-right: 1px;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['surface']};
                color: {COLORS['primary']};
                font-weight: 500;
                border-bottom: 2px solid {COLORS['primary']};
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {COLORS['surface_overlay']};
                color: {COLORS['text_primary']};
            }}
        """

    def refresh_styles(self) -> None:
        """Re-apply inline styles after theme change."""
        super().refresh_styles()
        if hasattr(self, "_tabs"):
            self._tabs.setStyleSheet(self._tab_widget_qss())

    def _setup_settings_view(self):
        """Set up the settings view content."""
        tabs = QTabWidget()
        self._tabs = tabs
        tabs.setStyleSheet(self._tab_widget_qss())

        # General settings tab
        general_tab = self._create_general_tab()
        tabs.addTab(general_tab, "General")

        # Matching settings tab
        matching_tab = self._create_matching_tab()
        tabs.addTab(matching_tab, "Matching")

        # AI/ML settings tab
        ml_tab = self._create_ml_tab()
        tabs.addTab(ml_tab, "AI / ML")

        # Database settings tab
        db_tab = self._create_database_tab()
        tabs.addTab(db_tab, "Database")

        # Data management tab
        data_tab = self._create_data_management_tab()
        tabs.addTab(data_tab, "Data Management")

        self.add_widget(tabs)

        # Save button
        save_layout = QHBoxLayout()
        save_layout.addStretch()

        reset_btn = SecondaryButton("Reset to Defaults")
        reset_btn.clicked.connect(self._reset_defaults)
        save_layout.addWidget(reset_btn)

        save_btn = PrimaryButton("Save Settings")
        save_btn.clicked.connect(self._save_settings)
        save_layout.addWidget(save_btn)

        self.add_layout(save_layout)

    def _create_general_tab(self) -> QWidget:
        """Create general settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # Appearance group
        appearance_group = QGroupBox("Appearance")
        appearance_group.setStyleSheet(self._group_style())
        appearance_layout = QFormLayout(appearance_group)
        appearance_layout.setSpacing(12)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "System"])
        self.theme_combo.setMinimumHeight(32)
        appearance_layout.addRow("Theme:", self.theme_combo)

        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(10, 18)
        self.font_size_spin.setValue(12)
        self.font_size_spin.setSuffix(" pt")
        appearance_layout.addRow("Font Size:", self.font_size_spin)

        layout.addWidget(appearance_group)

        # Language group
        lang_group = QGroupBox("Language & Region")
        lang_group.setStyleSheet(self._group_style())
        lang_layout = QFormLayout(lang_group)
        lang_layout.setSpacing(12)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "Spanish", "French", "German"])
        self.lang_combo.setMinimumHeight(32)
        lang_layout.addRow("Language:", self.lang_combo)

        layout.addWidget(lang_group)

        # Window group
        window_group = QGroupBox("Window")
        window_group.setStyleSheet(self._group_style())
        window_layout = QFormLayout(window_group)
        window_layout.setSpacing(12)

        self.width_spin = QSpinBox()
        self.width_spin.setRange(800, 2560)
        self.width_spin.setValue(1400)
        self.width_spin.setSuffix(" px")
        window_layout.addRow("Default Width:", self.width_spin)

        self.height_spin = QSpinBox()
        self.height_spin.setRange(600, 1440)
        self.height_spin.setValue(900)
        self.height_spin.setSuffix(" px")
        window_layout.addRow("Default Height:", self.height_spin)

        layout.addWidget(window_group)

        layout.addStretch()
        return tab

    def _create_matching_tab(self) -> QWidget:
        """Create matching settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # Scoring weights group
        weights_group = QGroupBox("Scoring Weights")
        weights_group.setStyleSheet(self._group_style())
        weights_layout = QFormLayout(weights_group)
        weights_layout.setSpacing(12)

        info_label = QLabel("Adjust how different factors contribute to the match score. Total should equal 100%.")
        info_label.setWordWrap(True)
        info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; margin-bottom: 8px;")
        weights_layout.addRow(info_label)

        self.skills_weight = QSpinBox()
        self.skills_weight.setRange(0, 100)
        self.skills_weight.setValue(35)
        self.skills_weight.setSuffix(" %")
        weights_layout.addRow("Skills Match:", self.skills_weight)

        self.experience_weight = QSpinBox()
        self.experience_weight.setRange(0, 100)
        self.experience_weight.setValue(25)
        self.experience_weight.setSuffix(" %")
        weights_layout.addRow("Experience Match:", self.experience_weight)

        self.education_weight = QSpinBox()
        self.education_weight.setRange(0, 100)
        self.education_weight.setValue(15)
        self.education_weight.setSuffix(" %")
        weights_layout.addRow("Education Match:", self.education_weight)

        self.semantic_weight = QSpinBox()
        self.semantic_weight.setRange(0, 100)
        self.semantic_weight.setValue(20)
        self.semantic_weight.setSuffix(" %")
        weights_layout.addRow("Semantic Similarity:", self.semantic_weight)

        self.keyword_weight = QSpinBox()
        self.keyword_weight.setRange(0, 100)
        self.keyword_weight.setValue(5)
        self.keyword_weight.setSuffix(" %")
        weights_layout.addRow("Keyword Match:", self.keyword_weight)

        layout.addWidget(weights_group)

        # Thresholds group
        threshold_group = QGroupBox("Score Thresholds")
        threshold_group.setStyleSheet(self._group_style())
        threshold_layout = QFormLayout(threshold_group)
        threshold_layout.setSpacing(12)

        self.excellent_threshold = QDoubleSpinBox()
        self.excellent_threshold.setRange(0, 1)
        self.excellent_threshold.setSingleStep(0.05)
        self.excellent_threshold.setValue(0.85)
        threshold_layout.addRow("Excellent (≥):", self.excellent_threshold)

        self.good_threshold = QDoubleSpinBox()
        self.good_threshold.setRange(0, 1)
        self.good_threshold.setSingleStep(0.05)
        self.good_threshold.setValue(0.70)
        threshold_layout.addRow("Good (≥):", self.good_threshold)

        self.fair_threshold = QDoubleSpinBox()
        self.fair_threshold.setRange(0, 1)
        self.fair_threshold.setSingleStep(0.05)
        self.fair_threshold.setValue(0.50)
        threshold_layout.addRow("Fair (≥):", self.fair_threshold)

        layout.addWidget(threshold_group)

        layout.addStretch()
        return tab

    def _create_ml_tab(self) -> QWidget:
        """Create AI/ML settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # Embedding model group
        embedding_group = QGroupBox("Embedding Model")
        embedding_group.setStyleSheet(self._group_style())
        embedding_layout = QFormLayout(embedding_group)
        embedding_layout.setSpacing(12)

        self.embedding_model_combo = QComboBox()
        self.embedding_model_combo.addItems([
            "all-MiniLM-L6-v2 (Fast)",
            "all-mpnet-base-v2 (Balanced)",
            "all-MiniLM-L12-v2 (Quality)",
        ])
        embedding_layout.addRow("Model:", self.embedding_model_combo)

        self.embedding_device_combo = QComboBox()
        self.embedding_device_combo.addItems(["Auto", "CPU", "CUDA", "MPS"])
        embedding_layout.addRow("Device:", self.embedding_device_combo)

        layout.addWidget(embedding_group)

        # Bias detection group
        bias_group = QGroupBox("Bias Detection")
        bias_group.setStyleSheet(self._group_style())
        bias_layout = QFormLayout(bias_group)
        bias_layout.setSpacing(12)

        self.bias_detection_check = QCheckBox("Enable bias detection")
        self.bias_detection_check.setChecked(True)
        bias_layout.addRow(self.bias_detection_check)

        self.auto_mitigate_check = QCheckBox("Auto-apply bias mitigation")
        self.auto_mitigate_check.setChecked(False)
        bias_layout.addRow(self.auto_mitigate_check)

        layout.addWidget(bias_group)

        # Explainability group
        explain_group = QGroupBox("Explainability")
        explain_group.setStyleSheet(self._group_style())
        explain_layout = QFormLayout(explain_group)
        explain_layout.setSpacing(12)

        self.lime_check = QCheckBox("Generate LIME explanations")
        self.lime_check.setChecked(True)
        explain_layout.addRow(self.lime_check)

        self.shap_check = QCheckBox("Generate SHAP values")
        self.shap_check.setChecked(True)
        explain_layout.addRow(self.shap_check)

        layout.addWidget(explain_group)

        layout.addStretch()
        return tab

    def _create_database_tab(self) -> QWidget:
        """Create database settings tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # MongoDB settings group
        mongo_group = QGroupBox("MongoDB Connection")
        mongo_group.setStyleSheet(self._group_style())
        mongo_layout = QFormLayout(mongo_group)
        mongo_layout.setSpacing(12)

        self.db_host_input = QLineEdit()
        self.db_host_input.setText("localhost")
        self.db_host_input.setMinimumHeight(32)
        mongo_layout.addRow("Host:", self.db_host_input)

        self.db_port_spin = QSpinBox()
        self.db_port_spin.setRange(1, 65535)
        self.db_port_spin.setValue(27017)
        mongo_layout.addRow("Port:", self.db_port_spin)

        self.db_name_input = QLineEdit()
        self.db_name_input.setText("ai_ats")
        self.db_name_input.setMinimumHeight(32)
        mongo_layout.addRow("Database:", self.db_name_input)

        self.db_user_input = QLineEdit()
        self.db_user_input.setPlaceholderText("Optional")
        self.db_user_input.setMinimumHeight(32)
        mongo_layout.addRow("Username:", self.db_user_input)

        self.db_pass_input = QLineEdit()
        self.db_pass_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.db_pass_input.setPlaceholderText("Optional")
        self.db_pass_input.setMinimumHeight(32)
        mongo_layout.addRow("Password:", self.db_pass_input)

        test_btn = SecondaryButton("Test Connection")
        test_btn.clicked.connect(self._test_db_connection)
        mongo_layout.addRow("", test_btn)

        layout.addWidget(mongo_group)

        # Vector store settings
        vector_group = QGroupBox("Vector Store")
        vector_group.setStyleSheet(self._group_style())
        vector_layout = QFormLayout(vector_group)
        vector_layout.setSpacing(12)

        self.vector_provider_combo = QComboBox()
        self.vector_provider_combo.addItems(["ChromaDB", "FAISS"])
        vector_layout.addRow("Provider:", self.vector_provider_combo)

        layout.addWidget(vector_group)

        layout.addStretch()
        return tab

    def _create_data_management_tab(self) -> QWidget:
        """Export, import, and purge controls for the active workspace."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(16)

        # ── Active workspace info ──────────────────────────────────────────────
        ws_group = QGroupBox("Active Workspace")
        ws_group.setStyleSheet(self._group_style())
        ws_layout = QVBoxLayout(ws_group)
        ws_layout.setSpacing(10)

        self._ws_info_label = QLabel("No workspace selected")
        self._ws_info_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px;"
        )
        ws_layout.addWidget(self._ws_info_label)

        ws_btn_row = QHBoxLayout()
        switch_btn = SecondaryButton("Switch Workspace…")
        switch_btn.clicked.connect(self._open_workspace_selector)
        ws_btn_row.addWidget(switch_btn)

        self._archive_ws_btn = SecondaryButton("Archive Workspace")
        self._archive_ws_btn.setEnabled(False)
        self._archive_ws_btn.clicked.connect(self._on_archive_workspace)
        ws_btn_row.addWidget(self._archive_ws_btn)

        ws_btn_row.addStretch()
        ws_layout.addLayout(ws_btn_row)
        layout.addWidget(ws_group)

        # Connect workspace state signal so label stays current
        try:
            from src.utils.workspace_state import get_workspace_state
            self._ws_state = get_workspace_state()
            self._ws_state.workspace_changed.connect(self._on_workspace_changed)
            self._on_workspace_changed(self._ws_state.workspace)
        except Exception:
            pass

        # ── Export ────────────────────────────────────────────────────────────
        export_group = QGroupBox("Export")
        export_group.setStyleSheet(self._group_style())
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(10)

        export_desc = QLabel(
            "Export the active workspace to a portable file. "
            "ZIP bundles all data as JSON; SQLite gives you a single query-able file."
        )
        export_desc.setWordWrap(True)
        export_desc.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        export_layout.addWidget(export_desc)

        export_btn_row = QHBoxLayout()
        zip_btn = PrimaryButton("Export to ZIP")
        zip_btn.clicked.connect(lambda: self._export("zip"))
        export_btn_row.addWidget(zip_btn)

        sqlite_btn = SecondaryButton("Export to SQLite")
        sqlite_btn.clicked.connect(lambda: self._export("sqlite"))
        export_btn_row.addWidget(sqlite_btn)

        jsonl_btn = SecondaryButton("Export Training Data (JSONL)")
        jsonl_btn.clicked.connect(lambda: self._export("jsonl"))
        export_btn_row.addWidget(jsonl_btn)

        export_btn_row.addStretch()
        export_layout.addLayout(export_btn_row)
        layout.addWidget(export_group)

        # ── Import ────────────────────────────────────────────────────────────
        import_group = QGroupBox("Import")
        import_group.setStyleSheet(self._group_style())
        import_layout = QVBoxLayout(import_group)
        import_layout.setSpacing(10)

        import_desc = QLabel(
            "Import a previously exported workspace. A new workspace is created "
            "with fresh IDs — your existing data is never overwritten."
        )
        import_desc.setWordWrap(True)
        import_desc.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        import_layout.addWidget(import_desc)

        import_btn_row = QHBoxLayout()
        import_zip_btn = PrimaryButton("Import from ZIP")
        import_zip_btn.clicked.connect(lambda: self._import("zip"))
        import_btn_row.addWidget(import_zip_btn)

        import_sqlite_btn = SecondaryButton("Import from SQLite")
        import_sqlite_btn.clicked.connect(lambda: self._import("sqlite"))
        import_btn_row.addWidget(import_sqlite_btn)

        import_btn_row.addStretch()
        import_layout.addLayout(import_btn_row)
        layout.addWidget(import_group)

        # ── Danger zone ───────────────────────────────────────────────────────
        danger_group = QGroupBox("Danger Zone")
        danger_group.setStyleSheet(
            self._group_style().replace(
                f"border: 1px solid {COLORS['border_subtle']}",
                f"border: 1px solid {COLORS['error']}",
            )
        )
        danger_layout = QVBoxLayout(danger_group)
        danger_layout.setSpacing(10)

        danger_desc = QLabel(
            "Purge permanently deletes all workspace data (jobs + match scores). "
            "An automatic ZIP backup is created first. "
            "The workspace must be archived before it can be purged."
        )
        danger_desc.setWordWrap(True)
        danger_desc.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        danger_layout.addWidget(danger_desc)

        self._purge_btn = DangerButton("Purge Workspace…")
        self._purge_btn.setEnabled(False)
        self._purge_btn.setFixedWidth(200)
        self._purge_btn.clicked.connect(self._on_purge_workspace)
        danger_layout.addWidget(self._purge_btn)

        layout.addWidget(danger_group)
        layout.addStretch()
        return tab

    # ── Data management event handlers ─────────────────────────────────────────

    def _on_workspace_changed(self, workspace: object) -> None:
        """Update workspace info label and button states when active WS changes."""
        if workspace is None:
            self._ws_info_label.setText("No workspace selected")
            self._archive_ws_btn.setEnabled(False)
            self._purge_btn.setEnabled(False)
            return

        from src.data.sql.models.workspace import WorkspaceStatus
        name: str = getattr(workspace, "name", "")
        status_val: str = getattr(workspace, "status", WorkspaceStatus.ACTIVE).value
        self._ws_info_label.setText(
            f"Name: {name}   Status: {status_val}"
        )
        is_active = getattr(workspace, "status", None) == WorkspaceStatus.ACTIVE
        is_archived = getattr(workspace, "status", None) == WorkspaceStatus.ARCHIVED
        self._archive_ws_btn.setEnabled(is_active)
        self._purge_btn.setEnabled(is_archived)

    def _open_workspace_selector(self) -> None:
        from PyQt6.QtWidgets import QDialog
        from src.ui.dialogs.workspace_dialogs import WorkspaceSelectorDialog
        from src.utils.workspace_state import get_workspace_state
        dlg = WorkspaceSelectorDialog(parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.selected_workspace:
            get_workspace_state().set_workspace(dlg.selected_workspace)

    def _on_archive_workspace(self) -> None:
        try:
            from src.utils.workspace_state import get_workspace_state
            ws = get_workspace_state().workspace
            if ws is None:
                return
            reply = QMessageBox.question(
                self,
                "Archive Workspace",
                f'Archive "{ws.name}"?\n\nArchived workspaces are read-only.',  # type: ignore[attr-defined]
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            from src.data.sql.repositories import get_workspace_repository
            updated = get_workspace_repository().archive(ws.id)  # type: ignore[attr-defined]
            if updated:
                get_workspace_state().set_workspace(updated)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not archive workspace:\n{exc}")

    def _export(self, fmt: str) -> None:
        try:
            from src.utils.workspace_state import get_workspace_state
            ws = get_workspace_state().workspace
            if ws is None:
                QMessageBox.warning(
                    self, "No Workspace", "Please select a workspace first."
                )
                return

            from pathlib import Path
            import datetime as _dt
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(
                c if c.isalnum() or c in "-_" else "_"
                for c in getattr(ws, "name", "workspace")
            )

            if fmt == "zip":
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Export Workspace to ZIP",
                    f"workspace_{safe_name}_{ts}.zip",
                    "ZIP Archives (*.zip)",
                )
                if not path:
                    return
                from src.services.export_service import get_export_service
                get_export_service().export_to_zip(ws.id, Path(path))  # type: ignore[attr-defined]

            elif fmt == "sqlite":
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Export Workspace to SQLite",
                    f"workspace_{safe_name}_{ts}.db",
                    "SQLite Databases (*.db *.sqlite)",
                )
                if not path:
                    return
                from src.services.export_service import get_export_service
                get_export_service().export_to_sqlite(ws.id, Path(path))  # type: ignore[attr-defined]

            elif fmt == "jsonl":
                path, _ = QFileDialog.getSaveFileName(
                    self,
                    "Export Training Data",
                    f"training_{safe_name}_{ts}.jsonl",
                    "JSON Lines (*.jsonl)",
                )
                if not path:
                    return
                from src.services.export_service import get_export_service
                result = get_export_service().export_training_jsonl(ws.id, Path(path))  # type: ignore[attr-defined]
                QMessageBox.information(
                    self, "Export Complete", f"Training data exported to:\n{result}"
                )
                return

            QMessageBox.information(
                self, "Export Complete", f"Workspace exported to:\n{path}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export Failed", f"Export error:\n{exc}")

    def _import(self, fmt: str) -> None:
        try:
            if fmt == "zip":
                path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Import Workspace from ZIP",
                    "",
                    "ZIP Archives (*.zip)",
                )
            else:
                path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Import Workspace from SQLite",
                    "",
                    "SQLite Databases (*.db *.sqlite)",
                )
            if not path:
                return

            from pathlib import Path
            from src.services.import_service import get_import_service
            svc = get_import_service()
            if fmt == "zip":
                ws = svc.import_from_zip(Path(path))
            else:
                ws = svc.import_from_sqlite(Path(path))

            from src.utils.workspace_state import get_workspace_state
            get_workspace_state().set_workspace(ws)

            QMessageBox.information(
                self,
                "Import Complete",
                f'Workspace "{ws.name}" imported successfully.\n'  # type: ignore[attr-defined]
                "It is now your active workspace.",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Import Failed", f"Import error:\n{exc}")

    def _on_purge_workspace(self) -> None:
        try:
            from src.utils.workspace_state import get_workspace_state
            ws = get_workspace_state().workspace
            if ws is None:
                return

            # Step 1: auto-export backup
            import datetime as _dt
            from pathlib import Path
            ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = "".join(
                c if c.isalnum() or c in "-_" else "_"
                for c in getattr(ws, "name", "workspace")
            )
            backup_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Backup Before Purge (required)",
                f"workspace_{safe_name}_{ts}_backup.zip",
                "ZIP Archives (*.zip)",
            )
            if not backup_path:
                return

            from src.services.export_service import get_export_service
            get_export_service().export_to_zip(ws.id, Path(backup_path))  # type: ignore[attr-defined]

            # Step 2: typed confirmation
            from src.ui.dialogs.workspace_dialogs import PurgeWorkspaceDialog
            dlg = PurgeWorkspaceDialog(workspace_name=ws.name, parent=self)  # type: ignore[attr-defined]
            from PyQt6.QtWidgets import QDialog
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return

            # Step 3: delete workspace row (CASCADE removes jobs + matches)
            from src.data.sql.repositories import get_workspace_repository
            get_workspace_repository().delete(ws.id)  # type: ignore[attr-defined]
            get_workspace_state().clear()

            QMessageBox.information(
                self,
                "Workspace Purged",
                f'"{ws.name}" has been permanently deleted.\n'  # type: ignore[attr-defined]
                f"Backup saved at:\n{backup_path}",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Purge Failed", f"Purge error:\n{exc}")

    def _group_style(self) -> str:
        """Return common group box style."""
        return f"""
            QGroupBox {{
                font-weight: 600;
                font-size: 13px;
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 8px;
                margin-top: 12px;
                padding: 16px;
                padding-top: 24px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 8px;
                background-color: {COLORS['surface']};
            }}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 6px;
                padding: 6px 10px;
                min-height: 28px;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border-color: {COLORS['primary']};
            }}
            QCheckBox {{
                color: {COLORS['text_primary']};
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid {COLORS['border_muted']};
                background-color: {COLORS['surface_elevated']};
            }}
            QCheckBox::indicator:checked {{
                background-color: {COLORS['primary']};
                border-color: {COLORS['primary']};
            }}
        """

    def _load_current_settings(self):
        """Load current settings from AppSettings into the UI."""
        try:
            from src.utils.config import get_settings
            settings = get_settings()

            # General tab
            theme_map = {"light": 0, "dark": 1, "system": 2}
            self.theme_combo.setCurrentIndex(theme_map.get(settings.ui.theme, 2))
            self.font_size_spin.setValue(settings.ui.font_size)
            self.width_spin.setValue(settings.ui.window_width)
            self.height_spin.setValue(settings.ui.window_height)

            # Matching tab - weights from DEFAULT_SCORING_WEIGHTS
            self.skills_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("skills_match", 0.35) * 100))
            self.experience_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("experience_match", 0.25) * 100))
            self.education_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("education_match", 0.15) * 100))
            self.semantic_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("semantic_similarity", 0.20) * 100))
            self.keyword_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("keyword_match", 0.05) * 100))

            # ML tab
            model_map = {
                "sentence-transformers/all-MiniLM-L6-v2": 0,
                "sentence-transformers/all-mpnet-base-v2": 1,
                "sentence-transformers/all-MiniLM-L12-v2": 2,
            }
            self.embedding_model_combo.setCurrentIndex(
                model_map.get(settings.ml.embedding_model, 0)
            )
            device_map = {"auto": 0, "cpu": 1, "cuda": 2, "mps": 3}
            self.embedding_device_combo.setCurrentIndex(
                device_map.get(settings.ml.device, 0)
            )

            # Database tab
            self.db_host_input.setText(settings.database.host)
            self.db_port_spin.setValue(settings.database.port)
            self.db_name_input.setText(settings.database.name)
            if settings.database.username:
                self.db_user_input.setText(settings.database.username)
            if settings.database.password:
                self.db_pass_input.setText(settings.database.password)

            # Vector store
            provider_map = {"chromadb": 0, "faiss": 1}
            self.vector_provider_combo.setCurrentIndex(
                provider_map.get(settings.vector_store.provider, 0)
            )

        except Exception as e:
            from src.utils.logger import get_logger
            get_logger(__name__).error(f"Error loading settings: {e}")

    def _save_settings(self):
        """Save current settings to environment and reload."""
        # Validate weight totals
        total_weight = (
            self.skills_weight.value() +
            self.experience_weight.value() +
            self.education_weight.value() +
            self.semantic_weight.value() +
            self.keyword_weight.value()
        )

        if total_weight != 100:
            QMessageBox.warning(
                self,
                "Invalid Weights",
                f"Scoring weights must total 100%.\nCurrent total: {total_weight}%",
            )
            return

        import os
        from src.utils.config import reload_settings, write_env_settings

        theme_values = ["light", "dark", "system"]
        model_values = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/all-MiniLM-L12-v2",
        ]
        device_values = ["auto", "cpu", "cuda", "mps"]
        provider_values = ["chromadb", "faiss"]

        # Build the full env update dict
        env_updates: dict[str, str] = {
            "UI_THEME":            theme_values[self.theme_combo.currentIndex()],
            "UI_FONT_SIZE":        str(self.font_size_spin.value()),
            "UI_WINDOW_WIDTH":     str(self.width_spin.value()),
            "UI_WINDOW_HEIGHT":    str(self.height_spin.value()),
            "DB_HOST":             self.db_host_input.text().strip(),
            "DB_PORT":             str(self.db_port_spin.value()),
            "DB_NAME":             self.db_name_input.text().strip(),
            "ML_EMBEDDING_MODEL":  model_values[self.embedding_model_combo.currentIndex()],
            "ML_DEVICE":           device_values[self.embedding_device_combo.currentIndex()],
            "VECTOR_PROVIDER":     provider_values[self.vector_provider_combo.currentIndex()],
        }
        if self.db_user_input.text().strip():
            env_updates["DB_USERNAME"] = self.db_user_input.text().strip()
        if self.db_pass_input.text().strip():
            env_updates["DB_PASSWORD"] = self.db_pass_input.text().strip()

        # 1. Write to .env for persistence across restarts
        try:
            write_env_settings(env_updates)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not write settings file: {e}")
            return

        # 2. Apply to current process environment so the reload picks them up
        for key, value in env_updates.items():
            os.environ[key] = value

        # 3. Reload settings singleton
        try:
            reload_settings()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not reload settings: {e}")
            return

        QMessageBox.information(
            self,
            "Settings Saved",
            "Your settings have been saved and applied.",
        )

    def _reset_defaults(self):
        """Reset settings to defaults."""
        reply = QMessageBox.question(
            self,
            "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Reset weights
            self.skills_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("skills_match", 0.35) * 100))
            self.experience_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("experience_match", 0.25) * 100))
            self.education_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("education_match", 0.15) * 100))
            self.semantic_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("semantic_similarity", 0.20) * 100))
            self.keyword_weight.setValue(int(DEFAULT_SCORING_WEIGHTS.get("keyword_match", 0.05) * 100))

            # Reset thresholds
            self.excellent_threshold.setValue(0.85)
            self.good_threshold.setValue(0.70)
            self.fair_threshold.setValue(0.50)

            # Reset other settings
            self.theme_combo.setCurrentIndex(2)  # System
            self.font_size_spin.setValue(12)
            self.width_spin.setValue(1400)
            self.height_spin.setValue(900)

            # Reset DB
            self.db_host_input.setText("localhost")
            self.db_port_spin.setValue(27017)
            self.db_name_input.setText("ai_ats")
            self.db_user_input.clear()
            self.db_pass_input.clear()

            # Reset ML
            self.embedding_model_combo.setCurrentIndex(0)
            self.embedding_device_combo.setCurrentIndex(0)
            self.vector_provider_combo.setCurrentIndex(0)

    def _test_db_connection(self):
        """Test database connection using current field values."""
        host = self.db_host_input.text().strip()
        port = self.db_port_spin.value()
        db_name = self.db_name_input.text().strip()

        try:
            from urllib.parse import quote_plus
            from pymongo import MongoClient
            from pymongo.errors import ConnectionFailure

            # Build URI with URL-encoded credentials
            username = self.db_user_input.text().strip()
            password = self.db_pass_input.text().strip()
            if username and password:
                uri = f"mongodb://{quote_plus(username)}:{quote_plus(password)}@{host}:{port}/{db_name}"
            else:
                uri = f"mongodb://{host}:{port}/{db_name}"

            client = MongoClient(uri, serverSelectionTimeoutMS=3000)
            client.admin.command("ping")
            client.close()

            QMessageBox.information(
                self,
                "Connection Successful",
                f"Successfully connected to MongoDB.\n\n"
                f"Host: {host}\n"
                f"Port: {port}\n"
                f"Database: {db_name}",
            )
        except ConnectionFailure:
            QMessageBox.critical(
                self,
                "Connection Failed",
                f"Could not connect to MongoDB at {host}:{port}.\n"
                "Please check the host, port, and credentials.",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Connection Error",
                f"Error testing connection: {e}",
            )
