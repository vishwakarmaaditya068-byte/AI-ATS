"""
Table widgets — VSCode-style table components.

StyledTable  — bare QTableWidget with VSCode styling + refresh_styles()
DataTable    — StyledTable + search bar + row-count label + refresh_styles()
"""

from __future__ import annotations

from typing import Any, Optional

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QLabel,
    QLineEdit,
    QAbstractItemView,
    QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QColor

from src.utils.constants import COLORS


def _table_qss() -> str:
    return f"""
        QTableWidget {{
            background-color: {COLORS['surface']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border_subtle']};
            border-radius: 4px;
            gridline-color: transparent;
            outline: none;
        }}
        QTableWidget::item {{
            padding: 7px 12px;
            color: {COLORS['text_primary']};
            border-bottom: 1px solid {COLORS['border_subtle']};
        }}
        QTableWidget::item:selected {{
            background-color: {COLORS['primary_glow']};
            color: {COLORS['text_primary']};
        }}
        QTableWidget::item:hover:!selected {{
            background-color: {COLORS['surface_overlay']};
        }}
        QHeaderView::section {{
            background-color: {COLORS['surface_elevated']};
            color: {COLORS['text_secondary']};
            font-weight: 700;
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            padding: 10px 12px;
            border: none;
            border-bottom: 1px solid {COLORS['border_muted']};
            border-right: 1px solid {COLORS['border_subtle']};
        }}
        QTableWidget QTableCornerButton::section {{
            background-color: {COLORS['surface_elevated']};
            border: none;
        }}
        QScrollBar:vertical {{
            background: transparent;
            width: 10px;
        }}
        QScrollBar::handle:vertical {{
            background: {COLORS['border_muted']};
            border-radius: 5px;
            min-height: 20px;
            margin: 2px 2px;
        }}
        QScrollBar::handle:vertical:hover {{ background: {COLORS['text_tertiary']}; }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
        QScrollBar:horizontal {{
            background: transparent;
            height: 10px;
        }}
        QScrollBar::handle:horizontal {{
            background: {COLORS['border_muted']};
            border-radius: 5px;
            min-width: 20px;
            margin: 2px 2px;
        }}
        QScrollBar::handle:horizontal:hover {{ background: {COLORS['text_tertiary']}; }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    """


def _search_qss() -> str:
    return f"""
        QLineEdit {{
            background-color: {COLORS['surface_elevated']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border_muted']};
            border-radius: 2px;
            padding: 6px 10px;
            font-size: 12px;
        }}
        QLineEdit:focus {{
            border-color: {COLORS['primary']};
        }}
        QLineEdit:hover:!focus {{
            border-color: {COLORS['text_tertiary']};
        }}
    """


class StyledTable(QTableWidget):
    """
    VSCode-styled table widget with alternating row colours disabled
    (VSCode uses flat rows) and a thin-line selection highlight.
    """

    row_clicked = pyqtSignal(int)
    row_double_clicked = pyqtSignal(int)
    selection_count_changed = pyqtSignal(int)

    def __init__(self, multi_select: bool = False, parent=None) -> None:
        super().__init__(parent)
        self._multi_select = multi_select
        self._setup_style()
        self._connect_signals()

    def _setup_style(self) -> None:
        self.setAlternatingRowColors(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        if self._multi_select:
            self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setShowGrid(False)
        self.setFrameShape(QFrame.Shape.NoFrame)

        header = self.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft)
        header.setHighlightSections(False)
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        vert = self.verticalHeader()
        vert.setVisible(False)
        vert.setDefaultSectionSize(44)

        self.setStyleSheet(_table_qss())

    def _connect_signals(self) -> None:
        self.cellClicked.connect(lambda row, _: self.row_clicked.emit(row))
        self.cellDoubleClicked.connect(lambda row, _: self.row_double_clicked.emit(row))
        self.itemSelectionChanged.connect(self._on_selection_changed)

    def _on_selection_changed(self) -> None:
        count = len(set(item.row() for item in self.selectedItems()))
        self.selection_count_changed.emit(count)

    def set_columns(self, columns: list[str]) -> None:
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)

    def add_row(self, values: list[Any], data: Any = None) -> None:
        row = self.rowCount()
        self.insertRow(row)
        for col, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setData(Qt.ItemDataRole.UserRole, data)
            self.setItem(row, col, item)

    def clear_rows(self) -> None:
        self.setRowCount(0)

    def get_row_data(self, row: int) -> Any:
        item = self.item(row, 0)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def get_selected_row_data(self) -> Any:
        selected = self.selectedItems()
        return selected[0].data(Qt.ItemDataRole.UserRole) if selected else None

    def get_all_selected_rows_data(self) -> list[Any]:
        rows = sorted(set(item.row() for item in self.selectedItems()))
        return [d for r in rows if (d := self.get_row_data(r)) is not None]

    def refresh_styles(self) -> None:
        self.setStyleSheet(_table_qss())


class DataTable(QWidget):
    """
    Full data table: search bar + StyledTable + row-count footer.
    """

    row_selected = pyqtSignal(object)
    selection_changed = pyqtSignal(int)

    def __init__(
        self,
        columns: list[str],
        searchable: bool = True,
        multi_select: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.columns = columns
        self.searchable = searchable
        self._multi_select = multi_select
        self._all_data: list[dict] = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        if self.searchable:
            self.search_input = QLineEdit()
            self.search_input.setPlaceholderText("Search…")
            self.search_input.setMinimumHeight(32)
            self.search_input.setStyleSheet(_search_qss())
            self.search_input.textChanged.connect(self._filter_data)
            layout.addWidget(self.search_input)

        self.table = StyledTable(multi_select=self._multi_select)
        self.table.set_columns(self.columns)
        self.table.row_clicked.connect(self._on_row_clicked)
        self.table.selection_count_changed.connect(self.selection_changed)
        layout.addWidget(self.table)

        self.count_label = QLabel("0 items")
        self.count_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
        layout.addWidget(self.count_label)

    def set_data(
        self,
        data: list[dict],
        columns_map: Optional[dict[str, str]] = None,
    ) -> None:
        self._all_data = data
        self._display_data(data, columns_map)

    def _display_data(
        self,
        data: list[dict],
        columns_map: Optional[dict[str, str]] = None,
    ) -> None:
        self.table.clear_rows()
        if columns_map is None:
            columns_map = {col: col.lower().replace(" ", "_") for col in self.columns}
        for row_data in data:
            values = [row_data.get(columns_map.get(col, col), "") for col in self.columns]
            self.table.add_row(values, row_data)
        self._update_count()

    def _filter_data(self, search_text: str) -> None:
        if not search_text:
            self._display_data(self._all_data)
            return
        search_lower = search_text.lower()
        filtered = [
            row for row in self._all_data
            if any(
                search_lower in str(v).lower()
                for v in row.values()
                if isinstance(v, (str, int, float))
            )
        ]
        self._display_data(filtered)

    def _on_row_clicked(self, row: int) -> None:
        data = self.table.get_row_data(row)
        if data:
            self.row_selected.emit(data)

    def _update_count(self) -> None:
        count = self.table.rowCount()
        self.count_label.setText(f"{count} item{'s' if count != 1 else ''}")

    def clear(self) -> None:
        self._all_data = []
        self.table.clear_rows()
        self._update_count()

    def get_selected_data(self) -> Any:
        return self.table.get_selected_row_data()

    def get_all_selected_data(self) -> list[Any]:
        return self.table.get_all_selected_rows_data()

    def refresh_styles(self) -> None:
        self.table.refresh_styles()
        if hasattr(self, "search_input"):
            self.search_input.setStyleSheet(_search_qss())
        self.count_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px;"
        )
