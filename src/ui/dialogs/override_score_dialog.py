from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QTextEdit,
    QVBoxLayout,
)

from src.utils.constants import COLORS


MIN_REASON_LENGTH: int = 10
SCORE_STEP: float = 0.01


def _dialog_qss() -> str:
    return f"""
        QDialog {{
            background-color: {COLORS['surface']};
        }}
        QLabel {{
            color: {COLORS['text_primary']};
            background-color: transparent;
        }}
        QTextEdit {{
            background-color: {COLORS['surface_elevated']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border_muted']};
            border-radius: 2px;
            padding: 6px 8px;
        }}
        QTextEdit:focus {{
            border-color: {COLORS['primary']};
        }}
        QDoubleSpinBox {{
            background-color: {COLORS['surface_elevated']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border_muted']};
            border-radius: 2px;
            padding: 4px 6px;
        }}
        QDoubleSpinBox:focus {{
            border-color: {COLORS['primary']};
        }}
        QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
            background-color: {COLORS['surface_overlay']};
            border: none;
            width: 16px;
        }}
        QSlider::groove:horizontal {{
            background: {COLORS['surface_elevated']};
            height: 4px;
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {COLORS['primary']};
            width: 12px;
            height: 12px;
            margin: -4px 0;
            border-radius: 6px;
        }}
        QSlider::sub-page:horizontal {{
            background: {COLORS['primary']};
            border-radius: 2px;
        }}
        QDialogButtonBox QPushButton {{
            min-width: 80px;
            padding: 5px 14px;
            background-color: {COLORS['surface_elevated']};
            color: {COLORS['text_primary']};
            border: 1px solid {COLORS['border_muted']};
            border-radius: 2px;
        }}
        QDialogButtonBox QPushButton:hover {{
            background-color: {COLORS['surface_overlay']};
        }}
        QDialogButtonBox QPushButton[text="Save override"] {{
            background-color: {COLORS['primary']};
            color: {COLORS['text_on_primary']};
            border: none;
        }}
        QDialogButtonBox QPushButton[text="Save override"]:hover {{
            background-color: {COLORS['primary_dark']};
        }}
        QDialogButtonBox QPushButton[text="Save override"]:disabled {{
            background-color: {COLORS['surface_elevated']};
            color: {COLORS['text_tertiary']};
            border: 1px solid {COLORS['border_subtle']};
        }}
    """


class OverrideScoreDialog(QDialog):
    """
    Modal for capturing a manual score override + reason.

    Returns via .accepted() — callers read .new_score and .reason after
    exec() returns QDialog.DialogCode.Accepted.
    """

    def __init__(
        self,
        *,
        candidate_name: str,
        job_title: str,
        current_score: float,
        ai_score: float,
        existing_reason: Optional[str] = None,
        parent: Optional[object] = None,
    ) -> None:
        super().__init__(parent)  # type: ignore[arg-type]
        self.setWindowTitle("Override Match Score")
        self.setModal(True)
        self.setMinimumWidth(480)
        self.setStyleSheet(_dialog_qss())

        self._ai_score: float = max(0.0, min(1.0, float(ai_score)))
        initial_score: float = max(0.0, min(1.0, float(current_score)))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)

        # Header
        header = QLabel(f"<b>{candidate_name}</b>  —  {job_title}")
        header.setWordWrap(True)
        header.setFont(QFont("Segoe UI", 12))
        layout.addWidget(header)

        ai_line = QLabel(
            f"AI score: <b>{self._ai_score:.0%}</b>"
            f"  ·  Current effective score: <b>{initial_score:.0%}</b>"
        )
        ai_line.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        layout.addWidget(ai_line)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setFixedHeight(1)
        sep.setStyleSheet(
            f"background-color: {COLORS['border_subtle']}; border: none;"
        )
        layout.addWidget(sep)

        # Score controls
        score_row = QFormLayout()
        score_row.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        score_row.setSpacing(8)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(100)
        self._slider.setValue(int(round(initial_score * 100)))
        self._slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._slider.setTickInterval(10)

        self._spin = QDoubleSpinBox()
        self._spin.setRange(0.0, 1.0)
        self._spin.setDecimals(2)
        self._spin.setSingleStep(SCORE_STEP)
        self._spin.setValue(initial_score)
        self._spin.setFixedWidth(72)

        slider_wrap = QHBoxLayout()
        slider_wrap.addWidget(self._slider, 1)
        slider_wrap.addWidget(self._spin, 0)

        score_label = QLabel("New score:")
        score_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        score_row.addRow(score_label, slider_wrap)
        layout.addLayout(score_row)

        # Reason
        reason_label = QLabel(
            f"Reason (min {MIN_REASON_LENGTH} chars) — logged in audit trail:"
        )
        reason_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px;"
        )
        layout.addWidget(reason_label)

        self._reason_edit = QTextEdit()
        self._reason_edit.setFont(QFont("Consolas", 11))
        self._reason_edit.setMinimumHeight(100)
        if existing_reason:
            self._reason_edit.setPlainText(existing_reason)
        layout.addWidget(self._reason_edit)

        self._validation_label = QLabel("")
        self._validation_label.setStyleSheet(
            f"color: {COLORS['error']}; font-size: 11px;"
        )
        layout.addWidget(self._validation_label)

        self._buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        save_btn = self._buttons.button(QDialogButtonBox.StandardButton.Save)
        if save_btn is not None:
            save_btn.setText("Save override")
        layout.addWidget(self._buttons)

        self._slider.valueChanged.connect(self._on_slider_changed)
        self._spin.valueChanged.connect(self._on_spin_changed)
        self._reason_edit.textChanged.connect(self._revalidate)
        self._buttons.accepted.connect(self._on_accept)
        self._buttons.rejected.connect(self.reject)

        self._revalidate()

    @property
    def new_score(self) -> float:
        return float(self._spin.value())

    @property
    def reason(self) -> str:
        return self._reason_edit.toPlainText().strip()

    def _on_slider_changed(self, value: int) -> None:
        new: float = value / 100.0
        if abs(new - self._spin.value()) > 1e-6:
            self._spin.blockSignals(True)
            self._spin.setValue(new)
            self._spin.blockSignals(False)
        self._revalidate()

    def _on_spin_changed(self, value: float) -> None:
        slider_val = int(round(value * 100))
        if slider_val != self._slider.value():
            self._slider.blockSignals(True)
            self._slider.setValue(slider_val)
            self._slider.blockSignals(False)
        self._revalidate()

    def _revalidate(self) -> None:
        reason = self._reason_edit.toPlainText().strip()
        errors: list[str] = []
        if len(reason) < MIN_REASON_LENGTH:
            errors.append(
                f"Reason must be at least {MIN_REASON_LENGTH} characters "
                f"({len(reason)}/{MIN_REASON_LENGTH})"
            )
        score = self._spin.value()
        if not (0.0 <= score <= 1.0):
            errors.append("Score must be between 0.00 and 1.00")
        self._validation_label.setText(" · ".join(errors))

        save_btn = self._buttons.button(QDialogButtonBox.StandardButton.Save)
        if save_btn is not None:
            save_btn.setEnabled(len(errors) == 0)

    def _on_accept(self) -> None:
        self._revalidate()
        if self._validation_label.text():
            return
        self.accept()

    def refresh_styles(self) -> None:
        self.setStyleSheet(_dialog_qss())
