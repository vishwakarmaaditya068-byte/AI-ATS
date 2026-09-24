"""
Analytics view for AI-ATS application.

Provides charts and insights about recruitment metrics,
candidate pipeline, and matching performance.
"""

from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGridLayout,
    QFrame,
    QComboBox,
    QScrollArea,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QRectF
from PyQt6.QtGui import QFont, QPainter, QColor, QPen, QBrush, QPainterPath

from src.utils.constants import COLORS
from src.utils.logger import get_logger
from src.ui.views.base_view import BaseView

logger = get_logger(__name__)
from src.ui.widgets import StatCard, InfoCard, Card


class BarChart(QWidget):
    """Custom bar chart widget using QPainter."""

    def __init__(
        self,
        data: dict[str, int],
        title: str = "",
        colors: list[str] = None,
        parent=None,
    ):
        """
        Initialize the bar chart.

        Args:
            data: Dictionary of label -> value pairs.
            title: Chart title.
            colors: List of colors for bars.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.data = data
        self.title = title
        self.colors = colors or [
            COLORS["primary"],
            COLORS["success"],
            "#8b5cf6",
            COLORS["warning"],
            "#ec4899",
            "#06b6d4",
        ]
        self.setMinimumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, data: dict[str, int]):
        """Update chart data."""
        self.data = data
        self.update()

    def paintEvent(self, event):
        """Paint the bar chart."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(0, 0, width, height, QColor(COLORS["surface_elevated"]))

        if not self.data:
            return

        # Margins
        left_margin = 60
        right_margin = 20
        top_margin = 40 if self.title else 20
        bottom_margin = 60

        chart_width = width - left_margin - right_margin
        chart_height = height - top_margin - bottom_margin

        # Title
        if self.title:
            painter.setPen(QColor(COLORS["text_primary"]))
            font = QFont("Segoe UI", 12)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(left_margin, 25, self.title)

        # Get max value
        max_value = max(self.data.values()) if self.data.values() else 1

        # Draw Y-axis labels
        painter.setPen(QColor(COLORS["text_secondary"]))
        painter.setFont(QFont("Segoe UI", 9))

        num_y_labels = 5
        for i in range(num_y_labels + 1):
            y = top_margin + chart_height - (i * chart_height / num_y_labels)
            value = int(max_value * i / num_y_labels)
            painter.drawText(5, int(y) + 4, 50, 20, Qt.AlignmentFlag.AlignRight, str(value))

            # Grid line
            painter.setPen(QPen(QColor(COLORS["border_subtle"]), 1, Qt.PenStyle.DashLine))
            painter.drawLine(left_margin, int(y), width - right_margin, int(y))
            painter.setPen(QColor(COLORS["text_secondary"]))

        # Draw bars
        bar_count = len(self.data)
        bar_spacing = 10
        bar_width = (chart_width - (bar_count + 1) * bar_spacing) / bar_count

        for i, (label, value) in enumerate(self.data.items()):
            x = left_margin + bar_spacing + i * (bar_width + bar_spacing)
            bar_height = (value / max_value) * chart_height if max_value > 0 else 0
            y = top_margin + chart_height - bar_height

            # Bar
            color = QColor(self.colors[i % len(self.colors)])
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)

            # Rounded top corners
            path = QPainterPath()
            radius = 4
            path.moveTo(x, y + bar_height)
            path.lineTo(x, y + radius)
            path.quadTo(x, y, x + radius, y)
            path.lineTo(x + bar_width - radius, y)
            path.quadTo(x + bar_width, y, x + bar_width, y + radius)
            path.lineTo(x + bar_width, y + bar_height)
            path.closeSubpath()
            painter.drawPath(path)

            # Value label on top of bar
            painter.setPen(QColor(COLORS["text_primary"]))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(
                int(x), int(y) - 18, int(bar_width), 16,
                Qt.AlignmentFlag.AlignCenter, str(value)
            )

            # X-axis label
            painter.setPen(QColor(COLORS["text_secondary"]))
            painter.setFont(QFont("Segoe UI", 9))

            # Truncate long labels
            display_label = label[:10] + "..." if len(label) > 10 else label
            painter.drawText(
                int(x) - 5, height - bottom_margin + 10, int(bar_width) + 10, 40,
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                display_label
            )


class DonutChart(QWidget):
    """Custom donut/pie chart widget using QPainter."""

    def __init__(
        self,
        data: dict[str, int],
        title: str = "",
        colors: list[str] = None,
        parent=None,
    ):
        """
        Initialize the donut chart.

        Args:
            data: Dictionary of label -> value pairs.
            title: Chart title.
            colors: List of colors for segments.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.data = data
        self.title = title
        self.colors = colors or [
            COLORS["primary"],
            COLORS["success"],
            "#8b5cf6",
            COLORS["warning"],
            "#ec4899",
            "#06b6d4",
            "#f97316",
            "#84cc16",
        ]
        self.setMinimumHeight(280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, data: dict[str, int]):
        """Update chart data."""
        self.data = data
        self.update()

    def paintEvent(self, event):
        """Paint the donut chart."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(0, 0, width, height, QColor(COLORS["surface_elevated"]))

        if not self.data:
            return

        # Title
        top_offset = 0
        if self.title:
            painter.setPen(QColor(COLORS["text_primary"]))
            font = QFont("Segoe UI", 12)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(20, 25, self.title)
            top_offset = 30

        # Calculate total
        total = sum(self.data.values())
        if total == 0:
            return

        # Chart dimensions
        chart_size = min(width - 150, height - top_offset - 20) - 20
        chart_x = 20
        chart_y = top_offset + 10
        inner_radius = chart_size * 0.35

        # Draw segments
        start_angle = 90 * 16  # Start from top (90 degrees, in 1/16th degree units)

        for i, (label, value) in enumerate(self.data.items()):
            span_angle = int((value / total) * 360 * 16)

            color = QColor(self.colors[i % len(self.colors)])
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(QColor(COLORS["surface"]), 2))

            # Draw pie segment
            painter.drawPie(
                int(chart_x), int(chart_y),
                int(chart_size), int(chart_size),
                start_angle, -span_angle
            )

            start_angle -= span_angle

        # Draw inner circle to create donut effect
        center_x = chart_x + chart_size / 2
        center_y = chart_y + chart_size / 2
        painter.setBrush(QBrush(QColor(COLORS["surface_elevated"])))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(
            int(center_x - inner_radius),
            int(center_y - inner_radius),
            int(inner_radius * 2),
            int(inner_radius * 2)
        )

        # Draw total in center
        painter.setPen(QColor(COLORS["text_primary"]))
        font = QFont("Segoe UI", 20)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            int(center_x - inner_radius),
            int(center_y - 15),
            int(inner_radius * 2),
            30,
            Qt.AlignmentFlag.AlignCenter,
            str(total)
        )
        painter.setFont(QFont("Segoe UI", 10))
        painter.setPen(QColor(COLORS["text_secondary"]))
        painter.drawText(
            int(center_x - inner_radius),
            int(center_y + 10),
            int(inner_radius * 2),
            20,
            Qt.AlignmentFlag.AlignCenter,
            "Total"
        )

        # Draw legend
        legend_x = chart_x + chart_size + 20
        legend_y = chart_y + 20

        for i, (label, value) in enumerate(self.data.items()):
            color = QColor(self.colors[i % len(self.colors)])
            percentage = (value / total) * 100

            # Color square
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(int(legend_x), int(legend_y + i * 25), 12, 12, 2, 2)

            # Label
            painter.setPen(QColor(COLORS["text_primary"]))
            painter.setFont(QFont("Segoe UI", 10))
            display_label = label[:12] + "..." if len(label) > 12 else label
            painter.drawText(
                int(legend_x + 18), int(legend_y + i * 25),
                120, 16,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"{display_label} ({percentage:.0f}%)"
            )


class HorizontalBarChart(QWidget):
    """Horizontal bar chart for ranking/comparison data."""

    def __init__(
        self,
        data: dict[str, float],
        title: str = "",
        max_value: float = 100,
        show_percentage: bool = True,
        parent=None,
    ):
        """
        Initialize the horizontal bar chart.

        Args:
            data: Dictionary of label -> value pairs.
            title: Chart title.
            max_value: Maximum value for the scale.
            show_percentage: Show values as percentages.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.data = data
        self.title = title
        self.max_value = max_value
        self.show_percentage = show_percentage
        self.setMinimumHeight(50 + len(data) * 35)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_data(self, data: dict[str, float]):
        """Update chart data."""
        self.data = data
        self.setMinimumHeight(50 + len(data) * 35)
        self.update()

    def paintEvent(self, event):
        """Paint the horizontal bar chart."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # Background
        painter.fillRect(0, 0, width, height, QColor(COLORS["surface_elevated"]))

        if not self.data:
            return

        # Title
        top_offset = 0
        if self.title:
            painter.setPen(QColor(COLORS["text_primary"]))
            font = QFont("Segoe UI", 12)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(20, 25, self.title)
            top_offset = 35

        # Margins
        left_margin = 120
        right_margin = 60
        bar_height = 20
        bar_spacing = 15

        chart_width = width - left_margin - right_margin

        # Sort data by value descending
        sorted_data = sorted(self.data.items(), key=lambda x: x[1], reverse=True)

        for i, (label, value) in enumerate(sorted_data):
            y = top_offset + 10 + i * (bar_height + bar_spacing)

            # Label
            painter.setPen(QColor(COLORS["text_primary"]))
            painter.setFont(QFont("Segoe UI", 10))
            display_label = label[:15] + "..." if len(label) > 15 else label
            painter.drawText(
                5, int(y), left_margin - 10, bar_height,
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                display_label
            )

            # Background bar
            painter.setBrush(QBrush(QColor(COLORS["surface_overlay"])))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(
                left_margin, int(y),
                int(chart_width), bar_height,
                4, 4
            )

            # Value bar
            bar_width = (value / self.max_value) * chart_width if self.max_value > 0 else 0

            # Color based on value
            ratio = value / self.max_value if self.max_value > 0 else 0
            if ratio >= 0.7:
                color = QColor(COLORS["success"])
            elif ratio >= 0.4:
                color = QColor(COLORS["primary"])
            else:
                color = QColor(COLORS["warning"])

            painter.setBrush(QBrush(color))
            painter.drawRoundedRect(
                left_margin, int(y),
                int(bar_width), bar_height,
                4, 4
            )

            # Value label
            painter.setPen(QColor(COLORS["text_primary"]))
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            value_text = f"{value:.0f}%" if self.show_percentage else f"{value:.0f}"
            painter.drawText(
                left_margin + int(chart_width) + 5, int(y),
                50, bar_height,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                value_text
            )


class MetricTrendCard(Card):
    """Card showing a metric with trend indicator."""

    def __init__(
        self,
        title: str,
        value: str,
        change: float,
        change_label: str = "vs last month",
        parent=None,
    ):
        """
        Initialize the metric trend card.

        Args:
            title: Metric name.
            value: Current value.
            change: Percentage change (positive = improvement).
            change_label: Label for the change period.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.metric_title = title
        self.value = value
        self.change = change
        self.change_label = change_label

        self._setup_content()

    def _setup_content(self):
        """Set up card content."""
        # Title
        title_label = QLabel(self.metric_title)
        title_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 12px;
        """)
        self.layout.addWidget(title_label)

        # Value row
        value_row = QHBoxLayout()

        self.value_label = QLabel(self.value)
        value_font = QFont("Segoe UI", 24)
        value_font.setBold(True)
        self.value_label.setFont(value_font)
        self.value_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        value_row.addWidget(self.value_label)

        value_row.addStretch()

        # Trend indicator
        if self.change != 0:
            trend_color = COLORS["success"] if self.change > 0 else COLORS["error"]
            trend_arrow = "↑" if self.change > 0 else "↓"
            trend_label = QLabel(f"{trend_arrow} {abs(self.change):.1f}%")
            trend_label.setStyleSheet(f"""
                color: {trend_color};
                font-size: 14px;
                font-weight: bold;
            """)
            value_row.addWidget(trend_label)

        self.layout.addLayout(value_row)

        # Change label
        change_label = QLabel(self.change_label)
        change_label.setStyleSheet(f"""
            color: {COLORS['text_secondary']};
            font-size: 11px;
        """)
        self.layout.addWidget(change_label)

        self.layout.addStretch()

    def set_value(self, value: str):
        """Update the displayed value."""
        self.value_label.setText(value)

    def refresh_styles(self) -> None:
        self.value_label.setStyleSheet(f"color: {COLORS['text_primary']};")


class AnalyticsView(BaseView):
    """
    Analytics view showing recruitment metrics and insights.

    Provides:
    - Key performance metrics
    - Recruitment funnel visualization
    - Source effectiveness analysis
    - Matching performance stats
    - Skills distribution
    """

    def __init__(self, parent=None):
        """Initialize the analytics view."""
        super().__init__(
            title="Analytics",
            description="Recruitment metrics, insights, and performance analysis",
            parent=parent,
        )
        self._setup_analytics_view()

    def _setup_analytics_view(self):
        """Set up the analytics view content."""
        # Time period selector
        self._create_toolbar()

        # Key metrics row
        self._create_metrics_section()

        # Charts grid
        charts_layout = QGridLayout()
        charts_layout.setSpacing(16)

        # Recruitment funnel (bar chart)
        self.funnel_chart = self._create_funnel_chart()
        funnel_card = self._create_chart_card(
            "Recruitment Funnel",
            self.funnel_chart,
        )
        charts_layout.addWidget(funnel_card, 0, 0)

        # Candidate sources (donut chart)
        self.sources_chart = self._create_sources_chart()
        sources_card = self._create_chart_card(
            "Candidate Sources",
            self.sources_chart,
        )
        charts_layout.addWidget(sources_card, 0, 1)

        self.add_layout(charts_layout)

        # Second row of charts
        charts_row2 = QHBoxLayout()
        charts_row2.setSpacing(16)

        # Top skills chart
        self.skills_chart = self._create_skills_chart()
        skills_card = self._create_chart_card(
            "Top Skills in Demand",
            self.skills_chart,
        )
        charts_row2.addWidget(skills_card)

        # Matching performance
        self.matching_chart = self._create_matching_chart()
        matching_card = self._create_chart_card(
            "Matching Score Distribution",
            self.matching_chart,
        )
        charts_row2.addWidget(matching_card)

        self.add_layout(charts_row2)

        # Job performance table
        self._create_job_performance_section()

        self.add_stretch()

    def _create_toolbar(self):
        """Create the toolbar with filters."""
        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)

        # Time period selector
        period_label = QLabel("Time Period:")
        period_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        toolbar.addWidget(period_label)

        self.period_combo = QComboBox()
        self.period_combo.addItems([
            "Last 7 Days",
            "Last 30 Days",
            "Last 90 Days",
            "This Year",
            "All Time",
        ])
        self.period_combo.setCurrentIndex(1)  # Default to Last 30 Days
        self.period_combo.setMinimumWidth(140)
        self.period_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 2px;
                padding: 8px 12px;
            }}
            QComboBox:focus {{
                border-color: {COLORS['primary']};
            }}
        """)
        self.period_combo.currentIndexChanged.connect(self._on_period_changed)
        toolbar.addWidget(self.period_combo)

        toolbar.addStretch()

        # Export button
        export_label = QLabel("📊 Export Report")
        export_label.setStyleSheet(f"""
            color: {COLORS['primary']};
            font-size: 13px;
            padding: 8px 12px;
        """)
        export_label.setCursor(Qt.CursorShape.PointingHandCursor)
        toolbar.addWidget(export_label)

        self.add_layout(toolbar)

    def _create_metrics_section(self):
        """Create key metrics cards."""
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(16)

        # Total Candidates
        self.total_candidates_card = MetricTrendCard(
            title="Total Candidates",
            value="--",
            change=0,
            change_label="",
        )
        metrics_layout.addWidget(self.total_candidates_card)

        # Active Jobs
        self.active_jobs_card = MetricTrendCard(
            title="Active Jobs",
            value="--",
            change=0,
            change_label="",
        )
        metrics_layout.addWidget(self.active_jobs_card)

        # Avg Match Score
        self.avg_match_card = MetricTrendCard(
            title="Avg Match Score",
            value="--",
            change=0,
            change_label="",
        )
        metrics_layout.addWidget(self.avg_match_card)

        # Total Matches
        self.total_matches_card = MetricTrendCard(
            title="Total Matches",
            value="--",
            change=0,
            change_label="",
        )
        metrics_layout.addWidget(self.total_matches_card)

        self.add_layout(metrics_layout)

    def _create_chart_card(self, title: str, chart_widget: QWidget) -> QFrame:
        """Create a card containing a chart."""
        card = QFrame()
        card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface_elevated']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
            }}
        """)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addWidget(chart_widget)

        return card

    def _create_funnel_chart(self) -> BarChart:
        """Create the recruitment funnel bar chart (empty initial data)."""
        return BarChart(
            data={},
            title="Candidates by Status",
            colors=[
                COLORS["primary"],
                "#f59e0b",
                "#8b5cf6",
                COLORS["success"],
                "#10b981",
                COLORS["error"],
            ]
        )

    def _create_sources_chart(self) -> DonutChart:
        """Create the candidate sources donut chart (empty initial data)."""
        return DonutChart(
            data={},
            title="Candidates by Source",
        )

    def _create_skills_chart(self) -> HorizontalBarChart:
        """Create the top skills horizontal bar chart (empty initial data)."""
        return HorizontalBarChart(
            data={},
            title="Most Common Skills",
            max_value=100,
            show_percentage=False,
        )

    def _create_matching_chart(self) -> BarChart:
        """Create the matching score distribution chart (empty initial data)."""
        return BarChart(
            data={},
            title="Match Score Distribution",
            colors=[
                "#10b981",
                COLORS["success"],
                COLORS["primary"],
                "#f59e0b",
                COLORS["warning"],
                COLORS["error"],
            ]
        )

    def _create_job_performance_section(self):
        """Create the job performance table section."""
        # Section header
        header_layout = QHBoxLayout()

        header = QLabel("Job Posting Performance")
        header_font = QFont("Segoe UI", 14)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet(f"color: {COLORS['text_primary']};")
        header_layout.addWidget(header)

        header_layout.addStretch()

        self.add_layout(header_layout)

        # Performance table card (clearable)
        self.job_perf_card = QFrame()
        self.job_perf_card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface_elevated']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
            }}
        """)

        self.job_perf_layout = QVBoxLayout(self.job_perf_card)
        self.job_perf_layout.setContentsMargins(0, 0, 0, 0)
        self.job_perf_layout.setSpacing(0)

        # Placeholder
        placeholder = QLabel("Loading job performance...")
        placeholder.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 16px;")
        self.job_perf_layout.addWidget(placeholder)

        self.add_widget(self.job_perf_card)

    def _populate_job_performance(self, job_data: list[tuple]):
        """Populate the job performance table with data rows."""
        # Clear existing
        while self.job_perf_layout.count():
            item = self.job_perf_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Table header
        header_widget = QWidget()
        header_widget.setStyleSheet(f"background-color: {COLORS['surface_overlay']};")
        header_row = QHBoxLayout(header_widget)
        header_row.setContentsMargins(16, 12, 16, 12)

        headers = ["Job Title", "Candidates", "Avg Score", "Status"]
        widths = [2, 1, 1, 1]

        for header_text, width in zip(headers, widths):
            label = QLabel(header_text)
            label.setStyleSheet(f"""
                color: {COLORS['text_secondary']};
                font-size: 12px;
                font-weight: 600;
            """)
            header_row.addWidget(label, width)

        self.job_perf_layout.addWidget(header_widget)

        if not job_data:
            placeholder = QLabel("No job data available.")
            placeholder.setStyleSheet(
                f"color: {COLORS['text_secondary']}; padding: 16px;"
            )
            self.job_perf_layout.addWidget(placeholder)
            return

        for job in job_data:
            row_widget = QWidget()
            row_widget.setStyleSheet(f"""
                QWidget {{
                    border-bottom: 1px solid {COLORS['border_subtle']};
                }}
                QWidget:hover {{
                    background-color: {COLORS['surface_overlay']};
                }}
            """)
            row = QHBoxLayout(row_widget)
            row.setContentsMargins(16, 12, 16, 12)

            for i, (value, width) in enumerate(zip(job, widths)):
                label = QLabel(str(value))

                if i == 0:  # Job title
                    label.setStyleSheet(f"""
                        color: {COLORS['text_primary']};
                        font-weight: 500;
                    """)
                elif i == 3:  # Status
                    status_color = COLORS["success"] if value == "Open" else COLORS["text_secondary"]
                    label.setStyleSheet(f"""
                        color: {status_color};
                        font-weight: 500;
                    """)
                else:
                    label.setStyleSheet(f"color: {COLORS['text_secondary']};")

                row.addWidget(label, width)

            self.job_perf_layout.addWidget(row_widget)

    def refresh_styles(self) -> None:
        self.period_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 2px;
                padding: 8px 12px;
            }}
            QComboBox:focus {{
                border-color: {COLORS['primary']};
            }}
        """)
        self.job_perf_card.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface_elevated']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
            }}
        """)
        self.funnel_chart.update()
        self.sources_chart.update()
        self.skills_chart.update()
        self.matching_chart.update()

    def _on_period_changed(self, index: int):
        """Handle time period change."""
        self.refresh()

    def refresh(self):
        """Refresh all analytics data from MongoDB."""
        self._load_analytics_data()

    def _load_analytics_data(self):
        """Load all analytics data from MongoDB."""
        try:
            from src.data.database import get_database_manager

            db_manager = get_database_manager()
            if not db_manager.check_sync_connection():
                return

            from src.data.repositories import (
                get_candidate_repository,
                get_job_repository,
                get_match_repository,
            )

            candidate_repo = get_candidate_repository()
            job_repo = get_job_repository()
            match_repo = get_match_repository()

            # --- Metric cards ---
            total_candidates = candidate_repo.count({})
            open_jobs = job_repo.count({"status": "open"})
            total_matches = match_repo.count({})

            self.total_candidates_card.set_value(str(total_candidates))
            self.active_jobs_card.set_value(str(open_jobs))
            self.total_matches_card.set_value(str(total_matches))

            # Average match score
            try:
                all_matches = match_repo.find({}, limit=500)
                if all_matches:
                    scores = [m.overall_score for m in all_matches if m.overall_score is not None]
                    if scores:
                        avg = sum(scores) / len(scores)
                        self.avg_match_card.set_value(f"{avg * 100:.0f}%")
                    else:
                        self.avg_match_card.set_value("N/A")
                else:
                    self.avg_match_card.set_value("N/A")
            except Exception:
                self.avg_match_card.set_value("N/A")

            # --- Funnel chart: candidates by status ---
            try:
                status_counts = candidate_repo.get_status_counts()
                # Map status values to display labels
                status_labels = {
                    "new": "New",
                    "screening": "Screening",
                    "shortlisted": "Shortlisted",
                    "interviewing": "Interview",
                    "offered": "Offer",
                    "hired": "Hired",
                    "rejected": "Rejected",
                    "withdrawn": "Withdrawn",
                }
                funnel_data = {}
                for status_key, label in status_labels.items():
                    count = status_counts.get(status_key, 0)
                    if count > 0:
                        funnel_data[label] = count
                if funnel_data:
                    self.funnel_chart.set_data(funnel_data)
            except Exception:
                pass

            # --- Sources chart: job status distribution ---
            try:
                job_status_counts = job_repo.get_status_counts()
                status_labels = {
                    "draft": "Draft",
                    "open": "Open",
                    "paused": "Paused",
                    "closed": "Closed",
                    "filled": "Filled",
                }
                sources_data = {}
                for key, label in status_labels.items():
                    count = job_status_counts.get(key, 0)
                    if count > 0:
                        sources_data[label] = count
                if sources_data:
                    self.sources_chart.set_data(sources_data)
            except Exception:
                pass

            # --- Skills chart: top skills from candidates ---
            try:
                skill_dist = candidate_repo.get_skill_distribution(limit=8)
                if skill_dist:
                    skills_data = {}
                    for item in skill_dist:
                        name = item.get("_id", "Unknown")
                        # Capitalize skill name
                        name = name.title() if name else "Unknown"
                        skills_data[name] = item.get("count", 0)
                    if skills_data:
                        max_count = max(skills_data.values())
                        self.skills_chart.max_value = max_count
                        self.skills_chart.set_data(skills_data)
            except Exception:
                pass

            # --- Matching score distribution chart ---
            try:
                all_matches = match_repo.find({}, limit=500)
                if all_matches:
                    buckets = {
                        "90-100%": 0,
                        "80-89%": 0,
                        "70-79%": 0,
                        "60-69%": 0,
                        "50-59%": 0,
                        "<50%": 0,
                    }
                    for m in all_matches:
                        pct = (m.overall_score or 0) * 100
                        if pct >= 90:
                            buckets["90-100%"] += 1
                        elif pct >= 80:
                            buckets["80-89%"] += 1
                        elif pct >= 70:
                            buckets["70-79%"] += 1
                        elif pct >= 60:
                            buckets["60-69%"] += 1
                        elif pct >= 50:
                            buckets["50-59%"] += 1
                        else:
                            buckets["<50%"] += 1

                    # Only show non-zero buckets
                    score_data = {k: v for k, v in buckets.items() if v > 0}
                    if score_data:
                        self.matching_chart.set_data(score_data)
            except Exception:
                pass

            # --- Job performance table ---
            try:
                jobs = job_repo.find({}, limit=10, sort_by="created_at", sort_order=-1)
                job_rows = []
                for job in jobs:
                    title = job.title
                    app_count = str(job.metadata.applications_count) if job.metadata else "0"

                    # Get avg match score for this job
                    avg_score_str = "N/A"
                    try:
                        stats = match_repo.get_score_stats_for_job(str(job.id))
                        if stats and stats.get("avg_score") is not None:
                            avg_score_str = f"{stats['avg_score'] * 100:.0f}%"
                    except Exception:
                        pass

                    status_map = {
                        "open": "Open", "draft": "Draft", "paused": "Paused",
                        "closed": "Closed", "filled": "Filled",
                    }
                    status = status_map.get(
                        job.status.value if job.status else "open", "Open"
                    )

                    job_rows.append((title, app_count, avg_score_str, status))

                self._populate_job_performance(job_rows)
            except Exception:
                self._populate_job_performance([])

        except Exception as e:
            logger.error(f"Error loading analytics: {e}")
