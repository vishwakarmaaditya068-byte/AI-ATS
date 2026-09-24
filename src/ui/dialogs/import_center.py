"""
Import Center Dialog for AI-ATS.

Provides a unified interface for importing resumes from:
  • Local files and folders (with drag & drop)
  • Google Drive (including Google Forms uploads)
  • Google Sheets (for candidate metadata)
"""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field

from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QWidget,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QTextEdit,
    QProgressBar,
    QFileDialog,
    QMessageBox,
    QFrame,
    QScrollArea,
    QGridLayout,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QGroupBox,
    QTreeWidget,
    QTreeWidgetItem,
    QHeaderView,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer, QMimeData
from PyQt6.QtGui import QFont, QDragEnterEvent, QDropEvent, QIcon

from src.utils.constants import COLORS, SUPPORTED_RESUME_FORMATS
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ImportTask:
    source: str
    file_path: Optional[str] = None
    file_name: str = ""
    status: str = "pending"
    error_message: str = ""
    candidate_name: str = ""
    candidate_email: str = ""


class ImportWorker(QThread):
    """Background worker for importing resumes."""

    progress = pyqtSignal(int, int, str)
    file_processed = pyqtSignal(dict)
    finished = pyqtSignal(int, int)

    def __init__(self, file_paths: list[str], parent: object = None) -> None:
        super().__init__(parent)
        self.file_paths = file_paths
        self._cancelled: bool = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        from src.services.ingestion_service import IngestionService, IngestionResult
        svc: IngestionService = IngestionService()
        total: int = len(self.file_paths)
        success_count: int = 0
        error_count: int = 0

        for i, file_path in enumerate(self.file_paths):
            if self._cancelled:
                break
            self.progress.emit(i, total, f"Processing: {Path(file_path).name}")
            result: IngestionResult = svc.ingest_file(file_path)
            if result.status == "success":
                success_count += 1
            elif result.status != "duplicate":
                error_count += 1
            self.file_processed.emit({
                "file_path": file_path,
                "file_name": Path(file_path).name,
                "status": result.status,
                "candidate_name": result.candidate_name,
                "candidate_email": result.candidate_email,
                "error_message": result.error_message,
                "processing_time_ms": result.processing_time_ms,
            })

        self.finished.emit(success_count, error_count)


# ── Drop zone ──────────────────────────────────────────────────────────────────

def _dropzone_qss(active: bool = False) -> str:
    border_color = COLORS["primary"] if active else COLORS["border_muted"]
    bg_color = COLORS["primary_glow"] if active else COLORS["surface"]
    return f"""
        QFrame {{
            background-color: {bg_color};
            border: 2px dashed {border_color};
            border-radius: 8px;
        }}
    """


class DropZone(QFrame):
    """Drag and drop zone for files."""

    files_dropped = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._setup_ui()

    def _setup_ui(self) -> None:
        self.setMinimumHeight(180)
        self.setStyleSheet(_dropzone_qss())

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(8)

        icon_label = QLabel("⬆")
        icon_label.setFont(QFont("Segoe UI Symbol", 32))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label.setStyleSheet(f"color: {COLORS['text_tertiary']};")
        layout.addWidget(icon_label)

        text_label = QLabel("Drag & Drop Resume Files Here")
        text_label.setFont(QFont("Segoe UI", 13, QFont.Weight.DemiBold))
        text_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        text_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(text_label)

        subtext = QLabel("or click Browse to select files")
        subtext.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        subtext.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtext)

        formats = QLabel(f"Supported: {', '.join(SUPPORTED_RESUME_FORMATS)}")
        formats.setStyleSheet(f"color: {COLORS['text_tertiary']}; font-size: 11px;")
        formats.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(formats)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.setStyleSheet(_dropzone_qss(active=True))

    def dragLeaveEvent(self, event) -> None:
        self.setStyleSheet(_dropzone_qss())

    def dropEvent(self, event: QDropEvent) -> None:
        self.setStyleSheet(_dropzone_qss())
        files = []
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if os.path.isfile(path):
                if Path(path).suffix.lower() in SUPPORTED_RESUME_FORMATS:
                    files.append(path)
            elif os.path.isdir(path):
                for ext in SUPPORTED_RESUME_FORMATS:
                    files.extend([str(p) for p in Path(path).rglob(f"*{ext}")])

        if files:
            self.files_dropped.emit(files)
        else:
            QMessageBox.warning(
                self,
                "No Valid Files",
                "No supported resume files found.\n\n"
                f"Supported formats: {', '.join(SUPPORTED_RESUME_FORMATS)}"
            )

    def refresh_styles(self) -> None:
        self._setup_ui()


# ── Local import tab ───────────────────────────────────────────────────────────

class LocalImportTab(QWidget):
    import_requested = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._selected_files: list[str] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        self.drop_zone = DropZone()
        self.drop_zone.files_dropped.connect(self._on_files_dropped)
        layout.addWidget(self.drop_zone)

        btn_layout = QHBoxLayout()

        browse_files_btn = QPushButton("Browse Files")
        browse_files_btn.setMinimumHeight(32)
        browse_files_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_files_btn.setStyleSheet(
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
            }}
            """
        )
        browse_files_btn.clicked.connect(self._browse_files)
        btn_layout.addWidget(browse_files_btn)

        browse_folder_btn = QPushButton("Browse Folder")
        browse_folder_btn.setMinimumHeight(32)
        browse_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        browse_folder_btn.setStyleSheet(
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
            """
        )
        browse_folder_btn.clicked.connect(self._browse_folder)
        btn_layout.addWidget(browse_folder_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.files_list = QListWidget()
        self.files_list.setMinimumHeight(130)
        self.files_list.setStyleSheet(
            f"""
            QListWidget {{
                background-color: {COLORS['surface_elevated']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 2px;
                outline: none;
            }}
            QListWidget::item {{
                padding: 6px 10px;
                border-bottom: 1px solid {COLORS['border_subtle']};
                color: {COLORS['text_primary']};
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['primary_glow']};
                color: {COLORS['primary']};
            }}
            QListWidget::item:hover:!selected {{
                background-color: {COLORS['surface_overlay']};
            }}
            """
        )
        layout.addWidget(QLabel("Selected Files:"))
        layout.addWidget(self.files_list)

        action_layout = QHBoxLayout()

        clear_btn = QPushButton("Clear All")
        clear_btn.setStyleSheet(
            f"QPushButton {{ background-color: transparent; color: {COLORS['text_secondary']};"
            f" border: 1px solid {COLORS['border_subtle']}; border-radius: 2px; padding: 4px 10px; }}"
            f" QPushButton:hover {{ color: {COLORS['text_primary']}; }}"
        )
        clear_btn.clicked.connect(self._clear_files)
        action_layout.addWidget(clear_btn)

        action_layout.addStretch()

        self.import_btn = QPushButton("Import Selected Files")
        self.import_btn.setMinimumHeight(32)
        self.import_btn.setEnabled(False)
        self.import_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.import_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: {COLORS['text_on_primary']};
                border: none;
                border-radius: 2px;
                padding: 6px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['primary_dark']};
            }}
            QPushButton:disabled {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_tertiary']};
                border: 1px solid {COLORS['border_subtle']};
            }}
            """
        )
        self.import_btn.clicked.connect(self._import_files)
        action_layout.addWidget(self.import_btn)
        layout.addLayout(action_layout)

    def _browse_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Resume Files", "",
            "Documents (*.pdf *.docx *.doc *.txt);;All Files (*)",
        )
        if files:
            self._add_files(files)

    def _browse_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Select Folder Containing Resumes")
        if folder:
            files = []
            for ext in SUPPORTED_RESUME_FORMATS:
                files.extend([str(p) for p in Path(folder).rglob(f"*{ext}")])
            if files:
                self._add_files(files)
            else:
                QMessageBox.information(
                    self, "No Files Found",
                    f"No resume files found in the selected folder.\n\n"
                    f"Supported formats: {', '.join(SUPPORTED_RESUME_FORMATS)}"
                )

    def _on_files_dropped(self, files: list) -> None:
        self._add_files(files)

    def _add_files(self, files: list) -> None:
        for file_path in files:
            if file_path not in self._selected_files:
                self._selected_files.append(file_path)
                item = QListWidgetItem(f"  {Path(file_path).name}")
                item.setData(Qt.ItemDataRole.UserRole, file_path)
                self.files_list.addItem(item)
        self.import_btn.setEnabled(len(self._selected_files) > 0)
        self.import_btn.setText(f"Import {len(self._selected_files)} File(s)")

    def _clear_files(self) -> None:
        self._selected_files.clear()
        self.files_list.clear()
        self.import_btn.setEnabled(False)
        self.import_btn.setText("Import Selected Files")

    def _import_files(self) -> None:
        if self._selected_files:
            self.import_requested.emit(self._selected_files.copy())


# ── Google Drive tab ───────────────────────────────────────────────────────────

class GoogleDriveTab(QWidget):
    import_requested = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._service = None
        self._current_folder_id = "root"
        self._folder_history: list[str] = []
        self._gdrive_files: list = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        self.status_frame = QFrame()
        self.status_frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: {COLORS['warning_dim']};
                border: 1px solid {COLORS['warning']};
                border-radius: 4px;
                padding: 10px;
            }}
            """
        )
        status_layout = QHBoxLayout(self.status_frame)

        self.status_label = QLabel("⚠  Not connected to Google Drive")
        self.status_label.setStyleSheet(f"color: {COLORS['warning']}; font-weight: 500;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        self.connect_btn = QPushButton("Connect to Google Drive")
        self.connect_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.connect_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4285f4;
                color: white;
                border: none;
                border-radius: 2px;
                padding: 6px 14px;
                font-weight: 500;
            }
            QPushButton:hover { background-color: #3367d6; }
            """
        )
        self.connect_btn.clicked.connect(self._connect_gdrive)
        status_layout.addWidget(self.connect_btn)
        layout.addWidget(self.status_frame)

        search_group = QGroupBox("Quick Find: Google Form Responses")
        search_layout = QHBoxLayout(search_group)
        self.form_name_input = QLineEdit()
        self.form_name_input.setPlaceholderText("Enter Google Form name…")
        self.form_name_input.setMinimumHeight(30)
        search_layout.addWidget(self.form_name_input)
        self.find_form_btn = QPushButton("Find Form Folder")
        self.find_form_btn.setEnabled(False)
        self.find_form_btn.clicked.connect(self._find_form_folder)
        search_layout.addWidget(self.find_form_btn)
        layout.addWidget(search_group)

        nav_layout = QHBoxLayout()
        self.back_btn = QPushButton("← Back")
        self.back_btn.setEnabled(False)
        self.back_btn.clicked.connect(self._go_back)
        nav_layout.addWidget(self.back_btn)
        self.path_label = QLabel("/ My Drive")
        self.path_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        nav_layout.addWidget(self.path_label)
        nav_layout.addStretch()
        self.refresh_btn = QPushButton("↺ Refresh")
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.clicked.connect(self._refresh_folder)
        nav_layout.addWidget(self.refresh_btn)
        layout.addLayout(nav_layout)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        folders_widget = QWidget()
        folders_layout = QVBoxLayout(folders_widget)
        folders_layout.setContentsMargins(0, 0, 0, 0)
        folders_layout.addWidget(QLabel("Folders:"))
        self.folders_tree = QTreeWidget()
        self.folders_tree.setHeaderHidden(True)
        self.folders_tree.setMinimumWidth(220)
        self.folders_tree.itemDoubleClicked.connect(self._on_folder_double_clicked)
        self.folders_tree.setStyleSheet(
            f"""
            QTreeWidget {{
                background-color: {COLORS['surface_elevated']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 2px;
                outline: none;
            }}
            QTreeWidget::item {{ padding: 5px; color: {COLORS['text_primary']}; }}
            QTreeWidget::item:selected {{
                background-color: {COLORS['primary_glow']};
                color: {COLORS['primary']};
            }}
            QTreeWidget::item:hover:!selected {{
                background-color: {COLORS['surface_overlay']};
            }}
            """
        )
        folders_layout.addWidget(self.folders_tree)
        splitter.addWidget(folders_widget)

        files_widget = QWidget()
        files_layout = QVBoxLayout(files_widget)
        files_layout.setContentsMargins(0, 0, 0, 0)
        files_layout.addWidget(QLabel("Resume Files:"))
        self.files_table = QTableWidget()
        self.files_table.setColumnCount(3)
        self.files_table.setHorizontalHeaderLabels(["Name", "Type", "Size"])
        self.files_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.files_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.files_table.setStyleSheet(
            f"""
            QTableWidget {{
                background-color: {COLORS['surface_elevated']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 2px;
                outline: none;
            }}
            QTableWidget::item {{
                padding: 6px 10px;
                color: {COLORS['text_primary']};
            }}
            QTableWidget::item:selected {{
                background-color: {COLORS['primary_glow']};
                color: {COLORS['primary']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['surface_overlay']};
                color: {COLORS['text_secondary']};
                border: none;
                border-bottom: 1px solid {COLORS['border_muted']};
                padding: 6px 10px;
                font-weight: 600;
            }}
            """
        )
        files_layout.addWidget(self.files_table)
        splitter.addWidget(files_widget)
        splitter.setSizes([280, 500])
        layout.addWidget(splitter)

        import_layout = QHBoxLayout()
        self.select_all_btn = QPushButton("Select All")
        self.select_all_btn.setEnabled(False)
        self.select_all_btn.clicked.connect(self._select_all_files)
        import_layout.addWidget(self.select_all_btn)
        import_layout.addStretch()
        self.import_gdrive_btn = QPushButton("Download & Import Selected")
        self.import_gdrive_btn.setMinimumHeight(32)
        self.import_gdrive_btn.setEnabled(False)
        self.import_gdrive_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: {COLORS['text_on_primary']};
                border: none;
                border-radius: 2px;
                padding: 6px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {COLORS['primary_dark']}; }}
            QPushButton:disabled {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_tertiary']};
                border: 1px solid {COLORS['border_subtle']};
            }}
            """
        )
        self.import_gdrive_btn.clicked.connect(self._import_selected)
        import_layout.addWidget(self.import_gdrive_btn)
        layout.addLayout(import_layout)

    def _connect_gdrive(self) -> None:
        try:
            from src.services.google_drive_service import get_drive_service
            self._service = get_drive_service()

            if not self._service.is_available():
                QMessageBox.warning(
                    self, "Missing Dependencies",
                    "Google API libraries not installed.\n\n"
                    "Install with:\npip install google-auth-oauthlib google-api-python-client"
                )
                return

            if not self._service.has_credentials():
                QMessageBox.warning(
                    self, "Missing Credentials",
                    "Google Drive credentials not found.\n\n"
                    "1. Go to console.cloud.google.com\n"
                    "2. Create project & enable Drive API\n"
                    "3. Create OAuth credentials\n"
                    "4. Download as credentials.json"
                )
                return

            self.status_label.setText("↻ Connecting…")
            self.connect_btn.setEnabled(False)

            if self._service.authenticate():
                self._on_connected()
            else:
                self.status_label.setText("✕ Connection failed")
                self.connect_btn.setEnabled(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Connection failed: {e}")
            self.connect_btn.setEnabled(True)

    def _on_connected(self) -> None:
        self.status_frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: {COLORS['success_dim']};
                border: 1px solid {COLORS['success']};
                border-radius: 4px;
                padding: 10px;
            }}
            """
        )
        self.status_label.setText("✓ Connected to Google Drive")
        self.status_label.setStyleSheet(f"color: {COLORS['success']}; font-weight: 500;")
        self.connect_btn.setText("Reconnect")
        self.connect_btn.setEnabled(True)
        self.find_form_btn.setEnabled(True)
        self.refresh_btn.setEnabled(True)
        self.select_all_btn.setEnabled(True)
        self._refresh_folder()

    def _refresh_folder(self) -> None:
        if not self._service or not self._service.is_authenticated():
            return
        self.folders_tree.clear()
        for folder in self._service.list_folders(self._current_folder_id):
            item = QTreeWidgetItem([f"  {folder.name}"])
            item.setData(0, Qt.ItemDataRole.UserRole, folder.id)
            self.folders_tree.addTopLevelItem(item)

        self.files_table.setRowCount(0)
        self._gdrive_files = self._service.list_resume_files(self._current_folder_id)

        for file in self._gdrive_files:
            row = self.files_table.rowCount()
            self.files_table.insertRow(row)
            name_item = QTableWidgetItem(file.name)
            name_item.setData(Qt.ItemDataRole.UserRole, file)
            self.files_table.setItem(row, 0, name_item)
            self.files_table.setItem(row, 1, QTableWidgetItem(file.mime_type.split("/")[-1].upper()))
            size = f"{file.size / 1024:.1f} KB" if file.size else "N/A"
            self.files_table.setItem(row, 2, QTableWidgetItem(size))

        self.import_gdrive_btn.setEnabled(len(self._gdrive_files) > 0)

    def _on_folder_double_clicked(self, item: QTreeWidgetItem, column: int) -> None:
        folder_id = item.data(0, Qt.ItemDataRole.UserRole)
        if folder_id:
            self._folder_history.append(self._current_folder_id)
            self._current_folder_id = folder_id
            self.back_btn.setEnabled(True)
            self.path_label.setText(f"/ {item.text(0).strip()}")
            self._refresh_folder()

    def _go_back(self) -> None:
        if self._folder_history:
            self._current_folder_id = self._folder_history.pop()
            self.back_btn.setEnabled(len(self._folder_history) > 0)
            self.path_label.setText("/ My Drive" if self._current_folder_id == "root" else "/ …")
            self._refresh_folder()

    def _find_form_folder(self) -> None:
        form_name = self.form_name_input.text().strip()
        if not form_name:
            QMessageBox.warning(self, "Input Required", "Please enter the Google Form name.")
            return
        folder_id = self._service.get_forms_response_folder(form_name)
        if folder_id:
            self._folder_history.append(self._current_folder_id)
            self._current_folder_id = folder_id
            self.back_btn.setEnabled(True)
            self.path_label.setText(f"/ {form_name} (File responses)")
            self._refresh_folder()
            QMessageBox.information(
                self, "Folder Found",
                f"Found form responses folder!\nFiles: {len(self._gdrive_files)} resume(s) found"
            )
        else:
            QMessageBox.warning(
                self, "Not Found",
                f"Could not find folder for form: '{form_name}'\n\n"
                "Make sure:\n• The form name is exact\n"
                "• You have access to the folder\n"
                "• The form has file upload responses"
            )

    def _select_all_files(self) -> None:
        self.files_table.selectAll()

    def _import_selected(self) -> None:
        selected_rows = set(item.row() for item in self.files_table.selectedItems())
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select files to import.")
            return
        from src.utils.config import DATA_DIR
        import_dir = DATA_DIR / "imports" / "gdrive"
        import_dir.mkdir(parents=True, exist_ok=True)
        downloaded_files = []
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog("Downloading files…", "Cancel", 0, len(selected_rows), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        for i, row in enumerate(selected_rows):
            if progress.wasCanceled():
                break
            file = self._gdrive_files[row]
            progress.setLabelText(f"Downloading: {file.name}")
            progress.setValue(i)
            output_path = import_dir / file.name
            if self._service.download_file(file.id, output_path):
                downloaded_files.append(str(output_path))
        progress.setValue(len(selected_rows))
        if downloaded_files:
            self.import_requested.emit(downloaded_files)


# ── Google Sheets tab ──────────────────────────────────────────────────────────

class GoogleSheetsTab(QWidget):
    metadata_imported = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._sheet_data: list = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)

        info_frame = QFrame()
        info_frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: {COLORS['primary_glow']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 4px;
                padding: 12px;
            }}
            """
        )
        info_layout = QVBoxLayout(info_frame)
        info_title = QLabel("Google Sheets Integration")
        info_title.setFont(QFont("Segoe UI", 11, QFont.Weight.DemiBold))
        info_title.setStyleSheet(f"color: {COLORS['primary']};")
        info_layout.addWidget(info_title)
        info_text = QLabel(
            "Import candidate metadata from Google Sheets linked to Google Forms.\n"
            "Fields: Name, Roll Number, Branch/Department, Email, Phone, CGPA, Preferred Roles."
        )
        info_text.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        info_text.setWordWrap(True)
        info_layout.addWidget(info_text)
        layout.addWidget(info_frame)

        url_group = QGroupBox("Google Sheet URL")
        url_layout = QVBoxLayout(url_group)
        self.sheet_url_input = QLineEdit()
        self.sheet_url_input.setPlaceholderText("Paste Google Sheets URL here…")
        self.sheet_url_input.setMinimumHeight(32)
        url_layout.addWidget(self.sheet_url_input)
        url_hint = QLabel("Example: https://docs.google.com/spreadsheets/d/SHEET_ID/edit")
        url_hint.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        url_layout.addWidget(url_hint)
        layout.addWidget(url_group)

        mapping_group = QGroupBox("Column Mapping")
        mapping_layout = QGridLayout(mapping_group)
        fields = [
            ("Name Column:", "name_col", "A"),
            ("Email Column:", "email_col", "B"),
            ("Roll Number:", "roll_col", "C"),
            ("Branch/Dept:", "branch_col", "D"),
            ("Phone Column:", "phone_col", "E"),
            ("CGPA Column:", "cgpa_col", "F"),
        ]
        self.column_inputs: dict[str, QLineEdit] = {}
        for i, (label, key, default) in enumerate(fields):
            row = i // 2
            col = (i % 2) * 2
            mapping_layout.addWidget(QLabel(label), row, col)
            inp = QLineEdit(default)
            inp.setMaximumWidth(56)
            self.column_inputs[key] = inp
            mapping_layout.addWidget(inp, row, col + 1)
        layout.addWidget(mapping_group)

        preview_label = QLabel("Data Preview:")
        preview_label.setFont(QFont("Segoe UI", 10, QFont.Weight.DemiBold))
        layout.addWidget(preview_label)

        self.preview_table = QTableWidget()
        self.preview_table.setMinimumHeight(180)
        self.preview_table.setStyleSheet(
            f"""
            QTableWidget {{
                background-color: {COLORS['surface_elevated']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 2px;
                outline: none;
            }}
            QTableWidget::item {{
                padding: 5px 8px;
                color: {COLORS['text_primary']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['surface_overlay']};
                color: {COLORS['text_secondary']};
                border: none;
                border-bottom: 1px solid {COLORS['border_muted']};
                padding: 5px 8px;
                font-weight: 600;
            }}
            """
        )
        layout.addWidget(self.preview_table)

        btn_layout = QHBoxLayout()
        fetch_btn = QPushButton("↻ Fetch Data")
        fetch_btn.setMinimumHeight(32)
        fetch_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #4285f4;
                color: white;
                border: none;
                border-radius: 2px;
                padding: 6px 16px;
                font-weight: 500;
            }
            QPushButton:hover { background-color: #3367d6; }
            """
        )
        fetch_btn.clicked.connect(self._fetch_sheet_data)
        btn_layout.addWidget(fetch_btn)
        btn_layout.addStretch()
        self.import_meta_btn = QPushButton("Import Metadata")
        self.import_meta_btn.setMinimumHeight(32)
        self.import_meta_btn.setEnabled(False)
        self.import_meta_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: {COLORS['text_on_primary']};
                border: none;
                border-radius: 2px;
                padding: 6px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {COLORS['primary_dark']}; }}
            QPushButton:disabled {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_tertiary']};
                border: 1px solid {COLORS['border_subtle']};
            }}
            """
        )
        self.import_meta_btn.clicked.connect(self._import_metadata)
        btn_layout.addWidget(self.import_meta_btn)
        layout.addLayout(btn_layout)

    def _fetch_sheet_data(self) -> None:
        url = self.sheet_url_input.text().strip()
        if not url:
            QMessageBox.warning(self, "Input Required", "Please enter the Google Sheets URL.")
            return
        import re
        match = re.search(r'/spreadsheets/d/([a-zA-Z0-9-_]+)', url)
        if not match:
            QMessageBox.warning(
                self, "Invalid URL",
                "Could not extract Sheet ID from URL.\n"
                "Make sure you're using the full Google Sheets URL."
            )
            return
        sheet_id = match.group(1)
        try:
            from src.services.google_drive_service import get_drive_service
            service = get_drive_service()
            if not service.is_authenticated():
                if not service.authenticate():
                    QMessageBox.warning(self, "Auth Required", "Please connect to Google Drive first.")
                    return
            csv_content = service.export_sheet_as_csv(sheet_id)
            if csv_content is None:
                QMessageBox.critical(
                    self, "Export Failed",
                    "Could not export the sheet. Make sure:\n"
                    "• The URL points to a Google Sheet\n"
                    "• You have read access to this sheet"
                )
                return
            import csv
            rows = list(csv.reader(csv_content.splitlines()))
            if not rows:
                QMessageBox.information(self, "Empty Sheet", "The sheet has no data.")
                return
            headers = rows[0]
            data_rows = rows[1:]
            preview_rows = data_rows[:50]
            self.preview_table.setColumnCount(len(headers))
            self.preview_table.setHorizontalHeaderLabels(headers)
            self.preview_table.setRowCount(len(preview_rows))
            for row_idx, row in enumerate(preview_rows):
                for col_idx, cell in enumerate(row[:len(headers)]):
                    self.preview_table.setItem(row_idx, col_idx, QTableWidgetItem(cell))
            self.preview_table.resizeColumnsToContents()
            self._sheet_data = []
            for row in data_rows:
                entry: dict[str, str] = {}
                for field_key, input_widget in self.column_inputs.items():
                    col_letter = input_widget.text().strip().upper()
                    if len(col_letter) == 1 and col_letter.isalpha():
                        col_idx = ord(col_letter) - ord('A')
                        entry[field_key] = row[col_idx] if col_idx < len(row) else ""
                self._sheet_data.append(entry)
            self.import_meta_btn.setEnabled(len(self._sheet_data) > 0)
            QMessageBox.information(
                self, "Data Loaded",
                f"Loaded {len(data_rows)} row(s) from the sheet.\n"
                "Click 'Import Metadata' to update matching candidates."
            )
        except Exception as e:
            logger.error(f"Failed to fetch Google Sheet: {e}")
            QMessageBox.critical(self, "Error", f"Failed to fetch sheet data: {e}")

    def _import_metadata(self) -> None:
        if self._sheet_data:
            self.metadata_imported.emit(self._sheet_data)


# ── Import Center Dialog ───────────────────────────────────────────────────────

class ImportCenterDialog(QDialog):
    """Main Import Center dialog."""

    candidates_imported = pyqtSignal(list)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Import Center")
        self.setMinimumSize(880, 680)
        self._imported_candidates: list = []
        self._setup_ui()
        self._apply_style()

    def _apply_style(self) -> None:
        self.setStyleSheet(
            f"""
            QDialog {{ background-color: {COLORS['surface']}; }}
            QLabel {{ color: {COLORS['text_primary']}; background-color: transparent; }}
            """
        )

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Header
        header = QFrame()
        header.setStyleSheet(
            f"""
            QFrame {{
                background-color: {COLORS['surface_elevated']};
                border-bottom: 1px solid {COLORS['border_subtle']};
            }}
            """
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 12, 20, 12)

        title = QLabel("Import Center")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.DemiBold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        header_layout.addWidget(title)
        header_layout.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(28, 28)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(
            f"""
            QPushButton {{
                background: transparent;
                border: none;
                font-size: 14px;
                color: {COLORS['text_secondary']};
                border-radius: 2px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['surface_overlay']};
                color: {COLORS['text_primary']};
            }}
            """
        )
        close_btn.clicked.connect(self.close)
        header_layout.addWidget(close_btn)
        layout.addWidget(header)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            f"""
            QTabWidget::pane {{
                border: none;
                background-color: {COLORS['surface']};
            }}
            QTabBar::tab {{
                padding: 10px 20px;
                margin-right: 1px;
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border_subtle']};
                border-bottom: none;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['surface']};
                color: {COLORS['primary']};
                border-bottom: 2px solid {COLORS['primary']};
                font-weight: 500;
            }}
            QTabBar::tab:hover:!selected {{
                background-color: {COLORS['surface_overlay']};
                color: {COLORS['text_primary']};
            }}
            """
        )

        self.local_tab = LocalImportTab()
        self.local_tab.import_requested.connect(self._process_files)
        self.tabs.addTab(self.local_tab, "Local Files")

        self.gdrive_tab = GoogleDriveTab()
        self.gdrive_tab.import_requested.connect(self._process_files)
        self.tabs.addTab(self.gdrive_tab, "Google Drive")

        self.gsheets_tab = GoogleSheetsTab()
        self.gsheets_tab.metadata_imported.connect(self._handle_metadata)
        self.tabs.addTab(self.gsheets_tab, "Google Sheets")

        layout.addWidget(self.tabs)

        # Progress frame
        self.progress_frame = QFrame()
        self.progress_frame.setVisible(False)
        self.progress_frame.setStyleSheet(
            f"""
            QFrame {{
                background-color: {COLORS['surface_elevated']};
                border-top: 1px solid {COLORS['border_subtle']};
            }}
            """
        )
        progress_layout = QVBoxLayout(self.progress_frame)
        progress_layout.setContentsMargins(20, 12, 20, 12)
        self.progress_label = QLabel("Processing…")
        self.progress_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        progress_layout.addWidget(self.progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                border: none;
                background-color: {COLORS['border_subtle']};
                border-radius: 2px;
                height: 6px;
                text-align: center;
                color: transparent;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 2px;
            }}
            """
        )
        progress_layout.addWidget(self.progress_bar)
        layout.addWidget(self.progress_frame)

        # Footer
        self.footer = QFrame()
        self.footer.setStyleSheet(
            f"""
            QFrame {{
                background-color: {COLORS['surface_elevated']};
                border-top: 1px solid {COLORS['border_subtle']};
            }}
            """
        )
        footer_layout = QHBoxLayout(self.footer)
        footer_layout.setContentsMargins(20, 12, 20, 12)
        self.result_label = QLabel("")
        self.result_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        footer_layout.addWidget(self.result_label)
        footer_layout.addStretch()
        self.done_btn = QPushButton("Done")
        self.done_btn.setMinimumHeight(30)
        self.done_btn.setMinimumWidth(100)
        self.done_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.done_btn.setStyleSheet(
            f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: {COLORS['text_on_primary']};
                border: none;
                border-radius: 2px;
                padding: 6px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background-color: {COLORS['primary_dark']}; }}
            """
        )
        self.done_btn.clicked.connect(self._finish_import)
        footer_layout.addWidget(self.done_btn)
        layout.addWidget(self.footer)

    def _process_files(self, file_paths: list) -> None:
        self.progress_frame.setVisible(True)
        self.progress_bar.setMaximum(len(file_paths))
        self.progress_bar.setValue(0)
        self._worker = ImportWorker(file_paths)
        self._worker.progress.connect(self._on_progress)
        self._worker.file_processed.connect(self._on_file_processed)
        self._worker.finished.connect(self._on_import_finished)
        self._worker.start()

    def _on_progress(self, current: int, total: int, message: str) -> None:
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"Processing ({current}/{total}): {message}")

    def _on_file_processed(self, result: dict) -> None:
        if result.get("status") == "success":
            self._imported_candidates.append(result)

    def _on_import_finished(self, success_count: int, error_count: int) -> None:
        self.progress_frame.setVisible(False)
        self.result_label.setText(f"✓ Imported: {success_count}   ✕ Errors: {error_count}")
        self.result_label.setStyleSheet(f"color: {COLORS['success']}; font-weight: 500;")
        if success_count > 0:
            QMessageBox.information(
                self, "Import Complete",
                f"Successfully imported {success_count} candidate(s).\n"
                f"Errors: {error_count}\n\nClick 'Done' to add them to the database."
            )

    def _handle_metadata(self, metadata: list) -> None:
        if not metadata:
            return
        try:
            from src.data.repositories.candidate_repository import get_candidate_repository
            repo = get_candidate_repository()
            updated = 0
            for row in metadata:
                email = row.get("email_col", "").strip().lower()
                if not email:
                    continue
                candidate = repo.get_by_email(email)
                if not candidate:
                    continue
                new_tags = []
                if row.get("roll_col", "").strip():
                    new_tags.append(f"roll:{row['roll_col'].strip()}")
                if row.get("branch_col", "").strip():
                    new_tags.append(f"dept:{row['branch_col'].strip()}")
                if new_tags:
                    existing_tags = candidate.metadata.tags if candidate.metadata else []
                    repo.update(candidate.id, {"metadata.tags": list(set(existing_tags + new_tags))})
                    updated += 1
            QMessageBox.information(
                self, "Metadata Imported",
                f"Updated metadata for {updated} candidate(s) matched by email."
            )
        except Exception as e:
            logger.error(f"Metadata import error: {e}")
            QMessageBox.critical(self, "Error", f"Failed to update candidates: {e}")

    def _finish_import(self) -> None:
        if self._imported_candidates:
            self.candidates_imported.emit(self._imported_candidates)
        self.accept()

    def get_imported_candidates(self) -> list:
        return self._imported_candidates

    def refresh_styles(self) -> None:
        self._apply_style()
