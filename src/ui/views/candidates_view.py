"""
Candidates management view for AI-ATS application.

Provides interface for viewing, managing, and importing candidate profiles.
"""

from pathlib import Path
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QTextEdit,
    QComboBox,
    QSpinBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QMessageBox,
    QFileDialog,
    QScrollArea,
    QFrame,
    QSplitter,
    QListWidget,
    QListWidgetItem,
    QTabWidget,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from src.utils.constants import COLORS, CandidateStatus
from src.utils.logger import get_logger

logger = get_logger(__name__)
from src.ui.views.base_view import BaseView
from src.ui.widgets import (
    DataTable,
    PrimaryButton,
    SecondaryButton,
    DangerButton,
    SuccessButton,
    InfoCard,
    ScoreCard,
)
from src.ui.dialogs import ImportCenterDialog


class CandidateFormDialog(QDialog):
    """Dialog for creating/editing candidate profiles."""

    def __init__(self, candidate_data: dict = None, parent=None):
        """
        Initialize the candidate form dialog.

        Args:
            candidate_data: Existing candidate data for editing, None for new.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.candidate_data = candidate_data or {}
        self.is_edit = bool(candidate_data)

        self._setup_ui()

    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Edit Candidate" if self.is_edit else "Add New Candidate")
        self.setMinimumWidth(550)
        self.setMinimumHeight(500)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {COLORS['surface']};
            }}
            QLabel {{
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QLineEdit, QTextEdit, QComboBox, QSpinBox {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 2px;
                padding: 8px;
                font-size: 13px;
            }}
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus, QSpinBox:focus {{
                border-color: {COLORS['primary']};
            }}
            QTabWidget::pane {{
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 0px;
                background-color: {COLORS['surface']};
            }}
            QTabBar::tab {{
                padding: 8px 16px;
                margin-right: 4px;
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_secondary']};
                border: none;
                border-bottom: 2px solid transparent;
            }}
            QTabBar::tab:selected {{
                background-color: {COLORS['surface']};
                color: {COLORS['text_primary']};
                border-bottom: 2px solid {COLORS['primary']};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Tab widget for organizing fields
        tabs = QTabWidget()

        # Basic Info Tab
        basic_tab = QWidget()
        basic_layout = QFormLayout(basic_tab)
        basic_layout.setSpacing(12)
        basic_layout.setContentsMargins(16, 16, 16, 16)

        self.first_name_input = QLineEdit()
        self.first_name_input.setPlaceholderText("e.g., John")
        self.first_name_input.setText(self.candidate_data.get("first_name", ""))
        basic_layout.addRow("First Name *", self.first_name_input)

        self.last_name_input = QLineEdit()
        self.last_name_input.setPlaceholderText("e.g., Doe")
        self.last_name_input.setText(self.candidate_data.get("last_name", ""))
        basic_layout.addRow("Last Name *", self.last_name_input)

        self.email_input = QLineEdit()
        self.email_input.setPlaceholderText("e.g., john.doe@email.com")
        self.email_input.setText(self.candidate_data.get("email", ""))
        basic_layout.addRow("Email *", self.email_input)

        self.phone_input = QLineEdit()
        self.phone_input.setPlaceholderText("e.g., +1 234 567 8900")
        self.phone_input.setText(self.candidate_data.get("phone", ""))
        basic_layout.addRow("Phone", self.phone_input)

        self.headline_input = QLineEdit()
        self.headline_input.setPlaceholderText("e.g., Senior Software Engineer")
        self.headline_input.setText(self.candidate_data.get("headline", ""))
        basic_layout.addRow("Headline", self.headline_input)

        self.status_combo = QComboBox()
        for status in CandidateStatus:
            self.status_combo.addItem(status.value.replace("_", " ").title(), status.value)
        if self.candidate_data.get("status"):
            index = self.status_combo.findData(self.candidate_data.get("status"))
            if index >= 0:
                self.status_combo.setCurrentIndex(index)
        basic_layout.addRow("Status", self.status_combo)

        tabs.addTab(basic_tab, "Basic Info")

        # Professional Tab
        prof_tab = QWidget()
        prof_layout = QFormLayout(prof_tab)
        prof_layout.setSpacing(12)
        prof_layout.setContentsMargins(16, 16, 16, 16)

        self.experience_spin = QSpinBox()
        self.experience_spin.setRange(0, 50)
        self.experience_spin.setSuffix(" years")
        self.experience_spin.setValue(self.candidate_data.get("experience_years", 0))
        prof_layout.addRow("Experience", self.experience_spin)

        self.skills_input = QLineEdit()
        self.skills_input.setPlaceholderText("e.g., Python, SQL, Machine Learning (comma separated)")
        self.skills_input.setText(self.candidate_data.get("skills", ""))
        prof_layout.addRow("Skills", self.skills_input)

        self.education_input = QLineEdit()
        self.education_input.setPlaceholderText("e.g., B.S. Computer Science")
        self.education_input.setText(self.candidate_data.get("education", ""))
        prof_layout.addRow("Education", self.education_input)

        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText("e.g., San Francisco, CA")
        self.location_input.setText(self.candidate_data.get("location", ""))
        prof_layout.addRow("Location", self.location_input)

        self.summary_input = QTextEdit()
        self.summary_input.setPlaceholderText("Brief professional summary...")
        self.summary_input.setMinimumHeight(100)
        self.summary_input.setText(self.candidate_data.get("summary", ""))
        prof_layout.addRow("Summary", self.summary_input)

        tabs.addTab(prof_tab, "Professional")

        # Links Tab
        links_tab = QWidget()
        links_layout = QFormLayout(links_tab)
        links_layout.setSpacing(12)
        links_layout.setContentsMargins(16, 16, 16, 16)

        self.linkedin_input = QLineEdit()
        self.linkedin_input.setPlaceholderText("https://linkedin.com/in/...")
        self.linkedin_input.setText(self.candidate_data.get("linkedin", ""))
        links_layout.addRow("LinkedIn", self.linkedin_input)

        self.github_input = QLineEdit()
        self.github_input.setPlaceholderText("https://github.com/...")
        self.github_input.setText(self.candidate_data.get("github", ""))
        links_layout.addRow("GitHub", self.github_input)

        self.portfolio_input = QLineEdit()
        self.portfolio_input.setPlaceholderText("https://portfolio.com/...")
        self.portfolio_input.setText(self.candidate_data.get("portfolio", ""))
        links_layout.addRow("Portfolio", self.portfolio_input)

        tabs.addTab(links_tab, "Links")

        layout.addWidget(tabs)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = SecondaryButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        save_btn = PrimaryButton("Save Candidate")
        save_btn.clicked.connect(self._validate_and_accept)
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)

    def _validate_and_accept(self):
        """Validate form and accept if valid."""
        if not self.first_name_input.text().strip():
            QMessageBox.warning(self, "Validation Error", "First name is required.")
            return
        if not self.last_name_input.text().strip():
            QMessageBox.warning(self, "Validation Error", "Last name is required.")
            return
        if not self.email_input.text().strip():
            QMessageBox.warning(self, "Validation Error", "Email is required.")
            return
        self.accept()

    def get_candidate_data(self) -> dict:
        """Get the form data as a dictionary."""
        return {
            "first_name": self.first_name_input.text().strip(),
            "last_name": self.last_name_input.text().strip(),
            "email": self.email_input.text().strip(),
            "phone": self.phone_input.text().strip(),
            "headline": self.headline_input.text().strip(),
            "status": self.status_combo.currentData(),
            "experience_years": self.experience_spin.value(),
            "skills": self.skills_input.text().strip(),
            "education": self.education_input.text().strip(),
            "location": self.location_input.text().strip(),
            "summary": self.summary_input.toPlainText().strip(),
            "linkedin": self.linkedin_input.text().strip(),
            "github": self.github_input.text().strip(),
            "portfolio": self.portfolio_input.text().strip(),
        }


class CandidateDetailPanel(QFrame):
    """Panel showing detailed candidate information."""

    def __init__(self, parent=None):
        """Initialize the detail panel."""
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        """Set up the panel UI."""
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
            }}
        """)
        self.setMinimumWidth(300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Header
        self.name_label = QLabel("Select a candidate")
        self.name_label.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        self.name_label.setStyleSheet(f"color: {COLORS['text_primary']}; border: none;")
        layout.addWidget(self.name_label)

        self.headline_label = QLabel("")
        self.headline_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        layout.addWidget(self.headline_label)

        # Status badge
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"""
            background-color: {COLORS['primary']}20;
            color: {COLORS['primary']};
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
            border: none;
        """)
        self.status_label.setFixedHeight(24)
        layout.addWidget(self.status_label, alignment=Qt.AlignmentFlag.AlignLeft)

        # Divider
        divider = QFrame()
        divider.setFrameShape(QFrame.Shape.HLine)
        divider.setStyleSheet(f"background-color: {COLORS['border_subtle']}; border: none;")
        divider.setFixedHeight(1)
        layout.addWidget(divider)

        # Contact info
        contact_header = QLabel("Contact")
        contact_header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        contact_header.setStyleSheet(f"color: {COLORS['text_primary']}; border: none;")
        layout.addWidget(contact_header)

        self.email_label = QLabel("")
        self.email_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        layout.addWidget(self.email_label)

        self.phone_label = QLabel("")
        self.phone_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        layout.addWidget(self.phone_label)

        self.location_label = QLabel("")
        self.location_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        layout.addWidget(self.location_label)

        # Divider
        divider2 = QFrame()
        divider2.setFrameShape(QFrame.Shape.HLine)
        divider2.setStyleSheet(f"background-color: {COLORS['border_subtle']}; border: none;")
        divider2.setFixedHeight(1)
        layout.addWidget(divider2)

        # Experience & Education
        exp_header = QLabel("Experience & Education")
        exp_header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        exp_header.setStyleSheet(f"color: {COLORS['text_primary']}; border: none;")
        layout.addWidget(exp_header)

        self.experience_label = QLabel("")
        self.experience_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        layout.addWidget(self.experience_label)

        self.education_label = QLabel("")
        self.education_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        self.education_label.setWordWrap(True)
        layout.addWidget(self.education_label)

        # Divider
        divider3 = QFrame()
        divider3.setFrameShape(QFrame.Shape.HLine)
        divider3.setStyleSheet(f"background-color: {COLORS['border_subtle']}; border: none;")
        divider3.setFixedHeight(1)
        layout.addWidget(divider3)

        # Skills
        skills_header = QLabel("Skills")
        skills_header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        skills_header.setStyleSheet(f"color: {COLORS['text_primary']}; border: none;")
        layout.addWidget(skills_header)

        self.skills_container = QWidget()
        self.skills_layout = QHBoxLayout(self.skills_container)
        self.skills_layout.setContentsMargins(0, 0, 0, 0)
        self.skills_layout.setSpacing(6)
        self.skills_container.setStyleSheet("border: none;")
        layout.addWidget(self.skills_container)

        # Summary
        summary_header = QLabel("Summary")
        summary_header.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        summary_header.setStyleSheet(f"color: {COLORS['text_primary']}; border: none;")
        layout.addWidget(summary_header)

        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        layout.addStretch()

        # Links
        self.links_container = QWidget()
        self.links_layout = QHBoxLayout(self.links_container)
        self.links_layout.setContentsMargins(0, 0, 0, 0)
        self.links_layout.setSpacing(8)
        self.links_container.setStyleSheet("border: none;")
        layout.addWidget(self.links_container)

    def update_candidate(self, candidate: dict):
        """Update the panel with candidate data."""
        if not candidate:
            self.name_label.setText("Select a candidate")
            self.headline_label.setText("")
            self.status_label.setText("")
            self.email_label.setText("")
            self.phone_label.setText("")
            self.location_label.setText("")
            self.experience_label.setText("")
            self.education_label.setText("")
            self.summary_label.setText("")
            self._clear_skills()
            self._clear_links()
            return

        name = f"{candidate.get('first_name', '')} {candidate.get('last_name', '')}"
        self.name_label.setText(name.strip())
        self.headline_label.setText(candidate.get("headline", "No headline"))

        status = candidate.get("status", "new")
        status_display = status.replace("_", " ").title()
        self.status_label.setText(status_display)

        # Update status badge color based on status
        status_colors = {
            "new": COLORS["primary"],
            "screening": COLORS["warning"],
            "interview": COLORS["accent"],
            "offer": COLORS["success"],
            "hired": COLORS["success"],
            "rejected": COLORS["error"],
            "withdrawn": COLORS["text_secondary"],
        }
        color = status_colors.get(status, COLORS["primary"])
        self.status_label.setStyleSheet(f"""
            background-color: {color}20;
            color: {color};
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
            border: none;
        """)

        self.email_label.setText(f"📧 {candidate.get('email', 'N/A')}")
        self.phone_label.setText(f"📱 {candidate.get('phone', 'N/A')}" if candidate.get("phone") else "")
        self.location_label.setText(f"📍 {candidate.get('location', 'N/A')}" if candidate.get("location") else "")

        exp_years = candidate.get("experience_years", 0)
        self.experience_label.setText(f"💼 {exp_years} years of experience")
        self.education_label.setText(f"🎓 {candidate.get('education', 'N/A')}" if candidate.get("education") else "")

        self.summary_label.setText(candidate.get("summary", "No summary available."))

        # Update skills
        self._clear_skills()
        skills_str = candidate.get("skills", "")
        if skills_str:
            skills = [s.strip() for s in skills_str.split(",") if s.strip()]
            for skill in skills[:6]:  # Show max 6 skills
                skill_label = QLabel(skill)
                skill_label.setStyleSheet(f"""
                    background-color: {COLORS['primary_glow']};
                    color: {COLORS['primary']};
                    padding: 4px 8px;
                    border-radius: 2px;
                    font-size: 11px;
                    border: none;
                """)
                self.skills_layout.addWidget(skill_label)
            self.skills_layout.addStretch()

        # Update links
        self._clear_links()
        if candidate.get("linkedin"):
            linkedin_btn = QLabel("LinkedIn")
            linkedin_btn.setStyleSheet(f"""
                background-color: #0077b5;
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
                border: none;
            """)
            linkedin_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.links_layout.addWidget(linkedin_btn)

        if candidate.get("github"):
            github_btn = QLabel("GitHub")
            github_btn.setStyleSheet(f"""
                background-color: {COLORS['activitybar_bg']};
                color: {COLORS['text_on_primary']};
                padding: 6px 12px;
                border-radius: 2px;
                font-size: 11px;
                border: none;
            """)
            github_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.links_layout.addWidget(github_btn)

        if candidate.get("portfolio"):
            portfolio_btn = QLabel("Portfolio")
            portfolio_btn.setStyleSheet(f"""
                background-color: {COLORS['primary']};
                color: white;
                padding: 6px 12px;
                border-radius: 4px;
                font-size: 11px;
                border: none;
            """)
            portfolio_btn.setCursor(Qt.CursorShape.PointingHandCursor)
            self.links_layout.addWidget(portfolio_btn)

        self.links_layout.addStretch()

    def refresh_styles(self) -> None:
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['surface']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
            }}
        """)
        self.name_label.setStyleSheet(f"color: {COLORS['text_primary']}; border: none;")
        self.headline_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        self.email_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        self.phone_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        self.location_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        self.experience_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        self.education_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")
        self.summary_label.setStyleSheet(f"color: {COLORS['text_secondary']}; border: none;")

    def _clear_skills(self):
        """Clear all skill labels."""
        while self.skills_layout.count():
            item = self.skills_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _clear_links(self):
        """Clear all link buttons."""
        while self.links_layout.count():
            item = self.links_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()


class CandidatesView(BaseView):
    """
    Candidates management view.

    Provides:
    - List of all candidates
    - Add/edit/delete candidates
    - Import resumes
    - Filter by status and search
    - View candidate details
    """

    candidate_selected = pyqtSignal(dict)  # Emitted when a candidate is selected

    def __init__(self, parent=None):
        """Initialize the candidates view."""
        super().__init__(
            title="Candidates",
            description="Manage candidate profiles and resumes",
            parent=parent,
        )
        self._candidates = []  # Store candidate data
        self._current_candidate: dict | None = None  # Last clicked candidate
        self._setup_candidates_view()

    @staticmethod
    def _candidate_to_dict(candidate) -> dict:
        """Convert a Candidate model object to a flat dict for the UI."""
        contact = getattr(candidate, "contact", None)
        edu_str = ""
        if candidate.education:
            e = candidate.education[0]
            parts = [p for p in [e.degree, e.field_of_study, e.institution] if p]
            edu_str = ", ".join(parts)

        location_parts = []
        if contact:
            if contact.city:
                location_parts.append(contact.city)
            if contact.state:
                location_parts.append(contact.state)
        location = ", ".join(location_parts)

        return {
            "id": str(candidate.id),
            "first_name": candidate.first_name,
            "last_name": candidate.last_name,
            "email": contact.email if contact else "",
            "phone": contact.phone or "" if contact else "",
            "headline": candidate.headline or "",
            "status": getattr(candidate.status, "value", candidate.status) or "new",
            "experience_years": candidate.total_experience_years,
            "skills": ", ".join(candidate.skill_names) if candidate.skills else "",
            "education": edu_str,
            "location": location,
            "summary": candidate.summary or "",
            "linkedin": contact.linkedin_url or "" if contact else "",
            "github": contact.github_url or "" if contact else "",
            "portfolio": contact.portfolio_url or "" if contact else "",
        }

    @staticmethod
    def _dict_to_candidate_create(data: dict):
        """Convert a UI form dict to a CandidateCreate schema."""
        from src.data.models.candidate import (
            CandidateCreate, ContactInfo, Skill, Education,
        )

        skills = []
        if data.get("skills"):
            for s in data["skills"].split(","):
                s = s.strip()
                if s:
                    skills.append(Skill(name=s))

        education = []
        if data.get("education"):
            education.append(Education(
                degree=data["education"],
                field_of_study="",
                institution="",
            ))

        loc = data.get("location", "")
        parts = [p.strip() for p in loc.split(",") if p.strip()] if loc else []
        city = parts[0] if len(parts) > 0 else None
        state = parts[1] if len(parts) > 1 else None

        email = data.get("email") or ""
        if not email or "@" not in email:
            raise ValueError("No valid email address — skipping database save")

        contact = ContactInfo(
            email=email,
            phone=data.get("phone") or None,
            linkedin_url=data.get("linkedin") or None,
            github_url=data.get("github") or None,
            portfolio_url=data.get("portfolio") or None,
            city=city,
            state=state,
        )

        return CandidateCreate(
            first_name=data.get("first_name") or "Unknown",
            last_name=data.get("last_name") or "Candidate",
            contact=contact,
            headline=data.get("headline") or None,
            summary=data.get("summary") or None,
            skills=skills,
            education=education,
        )

    def _setup_candidates_view(self):
        """Set up the candidates view content."""
        # Toolbar
        self._create_toolbar()

        # Main content with splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {COLORS['border_subtle']};
                width: 1px;
            }}
        """)

        # Left side - Table
        table_container = QWidget()
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(0, 0, 8, 0)

        self.candidates_table = DataTable(
            columns=["Name", "Email", "Headline", "Status", "Experience", "Skills"],
            searchable=True,
            multi_select=True,
        )
        self.candidates_table.row_selected.connect(self._on_candidate_selected)
        self.candidates_table.selection_changed.connect(self._on_selection_changed)
        table_layout.addWidget(self.candidates_table)

        splitter.addWidget(table_container)

        # Right side - Detail panel
        self.detail_panel = CandidateDetailPanel()
        splitter.addWidget(self.detail_panel)

        # Set splitter sizes (70% table, 30% detail)
        splitter.setSizes([700, 300])

        self.add_widget(splitter)

        # Load data from database
        self._load_candidates_from_db()

    def _create_toolbar(self):
        """Create the toolbar with actions."""
        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)

        # Add candidate button
        add_btn = PrimaryButton("+ Add Candidate")
        add_btn.clicked.connect(self._add_candidate)
        toolbar.addWidget(add_btn)

        # Import Center button (main import functionality)
        import_center_btn = SuccessButton("📥 Import Center")
        import_center_btn.setMinimumWidth(140)
        import_center_btn.clicked.connect(self._open_import_center)
        toolbar.addWidget(import_center_btn)

        # Quick import button (simple file selection)
        import_btn = SecondaryButton("Quick Import")
        import_btn.clicked.connect(self._import_resume)
        toolbar.addWidget(import_btn)

        # Status filter
        filter_label = QLabel("Status:")
        filter_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        toolbar.addWidget(filter_label)

        self.status_filter = QComboBox()
        self.status_filter.addItem("All", "all")
        for status in CandidateStatus:
            self.status_filter.addItem(status.value.replace("_", " ").title(), status.value)
        self.status_filter.setMinimumWidth(120)
        self.status_filter.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 2px;
                padding: 6px 12px;
            }}
            QComboBox:focus {{
                border-color: {COLORS['primary']};
            }}
        """)
        self.status_filter.currentIndexChanged.connect(self._filter_candidates)
        toolbar.addWidget(self.status_filter)

        toolbar.addStretch()

        # Edit button
        self.edit_btn = SecondaryButton("Edit")
        self.edit_btn.clicked.connect(self._edit_candidate)
        self.edit_btn.setEnabled(False)
        toolbar.addWidget(self.edit_btn)

        # Select All / Deselect All toggle
        self.select_all_btn = SecondaryButton("Select All")
        self.select_all_btn.setToolTip("Select all rows (Ctrl+A), click again to deselect")
        self.select_all_btn.clicked.connect(self._toggle_select_all)
        toolbar.addWidget(self.select_all_btn)

        # Delete button
        self.delete_btn = DangerButton("Delete")
        self.delete_btn.clicked.connect(self._delete_candidate)
        self.delete_btn.setEnabled(False)
        toolbar.addWidget(self.delete_btn)

        self.add_layout(toolbar)

    def _load_candidates_from_db(self):
        """Load candidates from MongoDB."""
        try:
            from src.data.database import get_database_manager
            from src.data.repositories import get_candidate_repository

            db_manager = get_database_manager()
            if db_manager.check_sync_connection():
                candidate_repo = get_candidate_repository()
                candidates = candidate_repo.find(
                    {}, limit=200, sort_by="created_at", sort_order=-1
                )
                self._candidates = [self._candidate_to_dict(c) for c in candidates]
            else:
                self._candidates = []
        except Exception as e:
            self._candidates = []
            logger.error(f"Error loading candidates: {e}")

        self._refresh_table()

    def _refresh_table(self, candidates: list = None):
        """Refresh the candidates table."""
        data = candidates if candidates is not None else self._candidates

        # Transform data for table display
        table_data = []
        for c in data:
            table_data.append({
                "id": c["id"],
                "Name": f"{c['first_name']} {c['last_name']}",
                "Email": c["email"],
                "Headline": c.get("headline", ""),
                "Status": c.get("status", "new").replace("_", " ").title(),
                "Experience": f"{c.get('experience_years', 0)} yrs",
                "Skills": c.get("skills", "")[:30] + "..." if len(c.get("skills", "")) > 30 else c.get("skills", ""),
                # Store original data for retrieval
                "_original": c,
            })

        columns_map = {
            "Name": "Name",
            "Email": "Email",
            "Headline": "Headline",
            "Status": "Status",
            "Experience": "Experience",
            "Skills": "Skills",
        }
        self.candidates_table.set_data(table_data, columns_map)

    def _filter_candidates(self):
        """Filter candidates by status."""
        status = self.status_filter.currentData()
        if status == "all":
            self._refresh_table()
        else:
            filtered = [c for c in self._candidates if c.get("status") == status]
            self._refresh_table(filtered)

    def _toggle_select_all(self) -> None:
        """Select all rows if any are unselected; deselect all otherwise."""
        table = self.candidates_table.table
        total = table.rowCount()
        selected = len(set(item.row() for item in table.selectedItems()))
        if selected < total and total > 0:
            table.selectAll()
            self.select_all_btn.setText("Deselect All")
        else:
            table.clearSelection()
            self.select_all_btn.setText("Select All")

    def _on_selection_changed(self, count: int) -> None:
        """Update toolbar button states when the table selection changes."""
        self.edit_btn.setEnabled(count == 1)
        self.delete_btn.setEnabled(count >= 1)
        if count > 1:
            self.delete_btn.setText(f"Delete ({count})")
        else:
            self.delete_btn.setText("Delete")
        # Keep Select All label in sync
        total = self.candidates_table.table.rowCount()
        if count == 0:
            self._current_candidate = None
            self.select_all_btn.setText("Select All")
        elif total > 0 and count == total:
            self.select_all_btn.setText("Deselect All")

    def _on_candidate_selected(self, data: dict):
        """Handle candidate row click — update detail panel and track for editing."""
        # Get original candidate data
        original = data.get("_original", data)

        # Find full candidate data
        candidate = None
        for c in self._candidates:
            if c["id"] == original.get("id", data.get("id")):
                candidate = c
                break

        if candidate:
            self._current_candidate = candidate
            self.detail_panel.update_candidate(candidate)
            self.candidate_selected.emit(candidate)
        else:
            self._current_candidate = None

    def _add_candidate(self):
        """Open dialog to add a new candidate."""
        dialog = CandidateFormDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            candidate_data = dialog.get_candidate_data()
            try:
                from src.data.database import get_database_manager
                from src.data.repositories import get_candidate_repository

                db_manager = get_database_manager()
                if db_manager.check_sync_connection():
                    candidate_repo = get_candidate_repository()
                    schema = self._dict_to_candidate_create(candidate_data)
                    created = candidate_repo.create_from_schema(schema)
                    ui_dict = self._candidate_to_dict(created)
                    self._candidates.insert(0, ui_dict)
                    self._refresh_table()
                    self._filter_candidates()
                    return
            except Exception as e:
                QMessageBox.warning(self, "Database Error", f"Could not save to database: {e}")

            # Fallback: in-memory only
            candidate_data["id"] = str(len(self._candidates) + 1)
            self._candidates.append(candidate_data)
            self._refresh_table()
            self._filter_candidates()

    def _edit_candidate(self):
        """Edit the currently selected candidate."""
        candidate = self._current_candidate
        if candidate:
            dialog = CandidateFormDialog(candidate_data=candidate, parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                updated_data = dialog.get_candidate_data()
                updated_data["id"] = candidate["id"]

                # Persist to database
                try:
                    from src.data.database import get_database_manager
                    from src.data.repositories import get_candidate_repository
                    from src.data.models.candidate import (
                        CandidateUpdate, ContactInfo, Skill, Education,
                    )

                    db_manager = get_database_manager()
                    if db_manager.check_sync_connection():
                        candidate_repo = get_candidate_repository()

                        skills = []
                        if updated_data.get("skills"):
                            for s in updated_data["skills"].split(","):
                                s = s.strip()
                                if s:
                                    skills.append(Skill(name=s))

                        education = []
                        if updated_data.get("education"):
                            education.append(Education(
                                degree=updated_data["education"],
                                field_of_study="",
                                institution="",
                            ))

                        loc = updated_data.get("location", "")
                        parts = [p.strip() for p in loc.split(",") if p.strip()] if loc else []
                        city = parts[0] if len(parts) > 0 else None
                        state = parts[1] if len(parts) > 1 else None

                        contact = ContactInfo(
                            email=updated_data.get("email") or "unknown@import.local",
                            phone=updated_data.get("phone") or None,
                            linkedin_url=updated_data.get("linkedin") or None,
                            github_url=updated_data.get("github") or None,
                            portfolio_url=updated_data.get("portfolio") or None,
                            city=city or None,
                            state=state or None,
                        )

                        status_val = updated_data.get("status")
                        from src.utils.constants import CandidateStatus as CS
                        status = CS(status_val) if status_val else None

                        update_schema = CandidateUpdate(
                            first_name=updated_data.get("first_name"),
                            last_name=updated_data.get("last_name"),
                            contact=contact,
                            headline=updated_data.get("headline") or None,
                            summary=updated_data.get("summary") or None,
                            skills=skills if skills else None,
                            education=education if education else None,
                            status=status,
                        )

                        result = candidate_repo.update_from_schema(
                            candidate["id"], update_schema
                        )
                        if result:
                            updated_data = self._candidate_to_dict(result)
                except Exception as e:
                    QMessageBox.warning(
                        self, "Database Error",
                        f"Could not update in database: {e}\nChanges saved locally.",
                    )

                # Update in list
                for i, c in enumerate(self._candidates):
                    if c["id"] == candidate["id"]:
                        self._candidates[i] = updated_data
                        break
                # Keep current candidate up to date so the detail panel reflects edits
                self._current_candidate = updated_data
                self._refresh_table()
                self._filter_candidates()
                self.detail_panel.update_candidate(updated_data)

    def _delete_candidate(self):
        """Delete all currently selected candidates."""
        all_selected = self.candidates_table.get_all_selected_data()
        if not all_selected:
            return

        # Collect unique IDs and display names from the selection
        to_delete = []
        seen_ids: set[str] = set()
        for data in all_selected:
            original = data.get("_original", data)
            cid = original.get("id", data.get("id"))
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                name = (
                    f"{original.get('first_name', '')} {original.get('last_name', '')}".strip()
                    or original.get("Name", "Unknown")
                )
                to_delete.append({"id": cid, "name": name or "Unknown"})

        if not to_delete:
            return

        # Build confirmation message
        if len(to_delete) == 1:
            confirm_msg = f"Are you sure you want to delete '{to_delete[0]['name']}'?"
        else:
            sample = ", ".join(f"'{c['name']}'" for c in to_delete[:3])
            if len(to_delete) > 3:
                sample += f" and {len(to_delete) - 3} more"
            confirm_msg = f"Delete {len(to_delete)} candidates?\n\n{sample}"

        reply = QMessageBox.question(
            self,
            "Delete Candidates",
            confirm_msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # Delete from database
        db_errors: list[str] = []
        try:
            from src.data.database import get_database_manager
            from src.data.repositories import get_candidate_repository

            db_manager = get_database_manager()
            if db_manager.check_sync_connection():
                candidate_repo = get_candidate_repository()
                for c in to_delete:
                    try:
                        candidate_repo.delete(c["id"])
                    except Exception as e:
                        db_errors.append(f"{c['name']}: {e}")
        except Exception as e:
            QMessageBox.warning(self, "Database Error", f"Could not connect: {e}")

        if db_errors:
            QMessageBox.warning(
                self, "Database Error",
                "Some candidates could not be deleted from the database:\n"
                + "\n".join(db_errors[:5]),
            )

        # Remove from in-memory list regardless of DB errors (mirror original behaviour)
        ids_to_remove = {c["id"] for c in to_delete}
        self._candidates = [c for c in self._candidates if c["id"] not in ids_to_remove]
        self._current_candidate = None
        self._refresh_table()
        self._filter_candidates()
        self.detail_panel.update_candidate(None)

    def _import_resume(self):
        """Import resume files and create candidates using NLP pipeline."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Resumes",
            "",
            "Documents (*.pdf *.docx *.doc *.txt);;All Files (*)",
        )
        if not file_paths:
            return

        # Show progress dialog
        from PyQt6.QtWidgets import QProgressDialog
        from PyQt6.QtCore import Qt

        progress = QProgressDialog(
            "Importing resumes...",
            "Cancel",
            0,
            len(file_paths),
            self
        )
        progress.setWindowTitle("Import Progress")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)

        success_count = 0
        error_count = 0
        errors = []

        try:
            from src.ml.nlp import get_resume_parser
            parser = get_resume_parser()

            for i, file_path in enumerate(file_paths):
                if progress.wasCanceled():
                    break

                progress.setValue(i)
                progress.setLabelText(f"Processing: {Path(file_path).name}")

                try:
                    # Parse the resume
                    result = parser.parse_file(file_path)

                    if result.success and result.contact:
                        # Convert to candidate data
                        candidate_data = {
                            "first_name": result.contact.get("first_name") or "Unknown",
                            "last_name": result.contact.get("last_name") or "Candidate",
                            "email": result.contact.get("email") or "",
                            "phone": result.contact.get("phone") or "",
                            "headline": result.experience[0].get("job_title") if result.experience else "",
                            "status": "new",
                            "experience_years": int(result.total_experience_years),
                            "skills": ", ".join([s.get("name", "") for s in result.skills[:10]]),
                            "education": result.education[0].get("degree") if result.education else "",
                            "location": result.contact.get("city") or "",
                            "summary": result.contact.get("summary") or "",
                            "linkedin": result.contact.get("linkedin_url") or "",
                            "github": result.contact.get("github_url") or "",
                            "portfolio": result.contact.get("portfolio_url") or "",
                        }

                        # Persist to database
                        saved_to_db = False
                        try:
                            from src.data.database import get_database_manager
                            from src.data.repositories import get_candidate_repository

                            db_manager = get_database_manager()
                            if db_manager.check_sync_connection():
                                candidate_repo = get_candidate_repository()
                                schema = self._dict_to_candidate_create(candidate_data)
                                try:
                                    created = candidate_repo.create_from_schema(schema)
                                except Exception as db_err:
                                    if "11000" in str(db_err) or "duplicate key" in str(db_err).lower():
                                        # Email already exists — load existing record
                                        created = candidate_repo.get_by_email(candidate_data.get("email", ""))
                                    else:
                                        raise
                                if created:
                                    ui_dict = self._candidate_to_dict(created)
                                    self._candidates.insert(0, ui_dict)
                                    saved_to_db = True
                        except Exception as e:
                            logger.error(f"Could not save imported candidate to DB: {e}")

                        if not saved_to_db:
                            candidate_data["id"] = str(len(self._candidates) + success_count + 1)
                            self._candidates.append(candidate_data)

                        success_count += 1
                    else:
                        error_count += 1
                        errors.append(f"{Path(file_path).name}: Parsing failed")

                except Exception as e:
                    error_count += 1
                    errors.append(f"{Path(file_path).name}: {str(e)[:50]}")

            progress.setValue(len(file_paths))

        except ImportError:
            QMessageBox.warning(
                self,
                "Import Error",
                "NLP module not available. Please check dependencies.",
            )
            return

        # Refresh table
        self._refresh_table()
        self._filter_candidates()

        # Show summary
        summary = f"Import Complete!\n\n✓ Imported: {success_count}\n✗ Errors: {error_count}"
        if errors:
            summary += "\n\nErrors:\n" + "\n".join(errors[:5])
            if len(errors) > 5:
                summary += f"\n... and {len(errors) - 5} more"

        QMessageBox.information(self, "Import Summary", summary)

    def _open_import_center(self):
        """Open the Import Center dialog for comprehensive file imports."""
        dialog = ImportCenterDialog(parent=self)
        dialog.candidates_imported.connect(self._handle_candidates_imported)
        dialog.exec()

    def _handle_candidates_imported(self, imported_candidates: list):
        """
        Handle candidates imported from the Import Center.

        The Import Center already parses resumes and returns candidate data.
        Persists each candidate to MongoDB.

        Args:
            imported_candidates: List of parsed candidate dictionaries from the worker.
        """
        if not imported_candidates:
            return

        added_count = 0
        db_errors = 0

        # Try to get DB connection once
        candidate_repo = None
        try:
            from src.data.database import get_database_manager
            from src.data.repositories import get_candidate_repository

            db_manager = get_database_manager()
            if db_manager.check_sync_connection():
                candidate_repo = get_candidate_repository()
        except Exception:
            pass

        for candidate_data in imported_candidates:
            # Skip failed imports (they have success=False)
            if not candidate_data.get("success", True):
                continue

            # Handle skills - could be list or string
            skills = candidate_data.get("skills", "")
            if isinstance(skills, list):
                skills = ", ".join(str(s) for s in skills if s)

            # Transform the imported data to match our view's format
            candidate = {
                "first_name": candidate_data.get("first_name") or "Unknown",
                "last_name": candidate_data.get("last_name") or "Candidate",
                "email": candidate_data.get("email") or "",
                "phone": candidate_data.get("phone") or "",
                "headline": candidate_data.get("headline") or "",
                "status": "new",
                "experience_years": int(candidate_data.get("experience_years", 0)),
                "skills": skills,
                "education": candidate_data.get("education") or "",
                "location": candidate_data.get("location") or "",
                "summary": candidate_data.get("summary") or "",
                "linkedin": candidate_data.get("linkedin") or "",
                "github": candidate_data.get("github") or "",
                "portfolio": candidate_data.get("portfolio") or "",
            }

            # Persist to DB
            if candidate_repo:
                try:
                    schema = self._dict_to_candidate_create(candidate)
                    try:
                        created = candidate_repo.create_from_schema(schema)
                    except Exception as db_err:
                        if "11000" in str(db_err) or "duplicate key" in str(db_err).lower():
                            created = candidate_repo.get_by_email(candidate.get("email", ""))
                        else:
                            raise
                    if created:
                        ui_dict = self._candidate_to_dict(created)
                        self._candidates.insert(0, ui_dict)
                        added_count += 1
                        continue
                except Exception as e:
                    logger.error(f"Could not save imported candidate to DB: {e}")
                    db_errors += 1

            # Fallback: in-memory only
            candidate["id"] = str(len(self._candidates) + added_count + 1)
            self._candidates.append(candidate)
            added_count += 1

        # Refresh the table
        self._refresh_table()
        self._filter_candidates()

        if added_count > 0:
            msg = f"Successfully added {added_count} candidate(s)."
            if db_errors > 0:
                msg += f"\n{db_errors} could not be saved to database."
            QMessageBox.information(self, "Import Complete", msg)

    def refresh_styles(self) -> None:
        self.status_filter.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_muted']};
                border-radius: 2px;
                padding: 6px 12px;
            }}
            QComboBox:focus {{
                border-color: {COLORS['primary']};
            }}
        """)
        self.candidates_table.refresh_styles()
        self.detail_panel.refresh_styles()

    def refresh(self):
        """Reload candidates from the database."""
        self._load_candidates_from_db()

    def get_candidates(self) -> list[dict]:
        """Get all candidates."""
        return self._candidates.copy()

    def get_candidate_by_id(self, candidate_id: str) -> dict | None:
        """Get a specific candidate by ID."""
        for c in self._candidates:
            if c["id"] == candidate_id:
                return c.copy()
        return None
