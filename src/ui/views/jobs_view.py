"""
Jobs management view for AI-ATS application.

Provides interface for creating, viewing, and managing job postings.
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
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from src.utils.constants import COLORS
from src.utils.logger import get_logger
from src.ui.views.base_view import BaseView

logger = get_logger(__name__)
from src.ui.widgets import (
    DataTable,
    PrimaryButton,
    SecondaryButton,
    DangerButton,
    ButtonGroup,
    InfoCard,
)


class JobFormDialog(QDialog):
    """Dialog for creating/editing job postings."""

    def __init__(self, job_data: dict = None, parent=None):
        """
        Initialize the job form dialog.

        Args:
            job_data: Existing job data for editing, None for new job.
            parent: Parent widget.
        """
        super().__init__(parent)
        self.job_data = job_data or {}
        self.is_edit = bool(job_data)

        self._setup_ui()

    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Edit Job" if self.is_edit else "Create New Job")
        self.setMinimumWidth(500)
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
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Form
        form = QFormLayout()
        form.setSpacing(12)

        # Job Title
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("e.g., Senior Software Engineer")
        self.title_input.setText(self.job_data.get("title", ""))
        form.addRow("Job Title *", self.title_input)

        # Company
        self.company_input = QLineEdit()
        self.company_input.setPlaceholderText("e.g., Tech Company Inc.")
        self.company_input.setText(self.job_data.get("company", ""))
        form.addRow("Company", self.company_input)

        # Employment Type
        self.type_combo = QComboBox()
        self.type_combo.addItems([
            "Full-time",
            "Part-time",
            "Contract",
            "Internship",
            "Freelance",
        ])
        if self.job_data.get("type"):
            self.type_combo.setCurrentText(self.job_data.get("type"))
        form.addRow("Employment Type", self.type_combo)

        # Experience Level
        self.level_combo = QComboBox()
        self.level_combo.addItems([
            "Entry Level",
            "Junior",
            "Mid-Level",
            "Senior",
            "Lead",
            "Manager",
            "Director",
        ])
        if self.job_data.get("level"):
            self.level_combo.setCurrentText(self.job_data.get("level"))
        form.addRow("Experience Level", self.level_combo)

        # Min Experience Years
        self.exp_spin = QSpinBox()
        self.exp_spin.setRange(0, 30)
        self.exp_spin.setSuffix(" years")
        self.exp_spin.setValue(self.job_data.get("min_experience", 0))
        form.addRow("Min Experience", self.exp_spin)

        # Required Skills
        self.skills_input = QLineEdit()
        self.skills_input.setPlaceholderText("e.g., Python, SQL, Machine Learning (comma separated)")
        self.skills_input.setText(self.job_data.get("required_skills", ""))
        form.addRow("Required Skills", self.skills_input)

        # Description
        self.desc_input = QTextEdit()
        self.desc_input.setPlaceholderText("Enter job description...")
        self.desc_input.setMinimumHeight(150)
        self.desc_input.setText(self.job_data.get("description", ""))
        form.addRow("Description", self.desc_input)

        layout.addLayout(form)

        # Buttons
        button_box = QDialogButtonBox()
        save_btn = PrimaryButton("Save Job")
        save_btn.clicked.connect(self.accept)
        cancel_btn = SecondaryButton("Cancel")
        cancel_btn.clicked.connect(self.reject)

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(save_btn)
        layout.addLayout(button_layout)

    def get_job_data(self) -> dict:
        """Get the form data as a dictionary."""
        return {
            "title": self.title_input.text().strip(),
            "company": self.company_input.text().strip(),
            "type": self.type_combo.currentText(),
            "level": self.level_combo.currentText(),
            "min_experience": self.exp_spin.value(),
            "required_skills": self.skills_input.text().strip(),
            "description": self.desc_input.toPlainText().strip(),
            "status": self.job_data.get("status", "Open"),
        }


class JobsView(BaseView):
    """
    Jobs management view.

    Provides:
    - List of all job postings
    - Create/edit/delete jobs
    - Import job descriptions
    - Filter and search
    """

    job_selected = pyqtSignal(dict)  # Emitted when a job is selected
    job_created = pyqtSignal()       # Emitted whenever a job is saved to the DB

    # Mapping between UI display text and model enum values
    _TYPE_TO_ENUM = {
        "Full-time": "full_time",
        "Part-time": "part_time",
        "Contract": "contract",
        "Internship": "internship",
        "Freelance": "freelance",
    }
    _ENUM_TO_TYPE = {v: k for k, v in _TYPE_TO_ENUM.items()}

    _LEVEL_TO_ENUM = {
        "Entry Level": "entry",
        "Junior": "entry",
        "Mid-Level": "mid",
        "Senior": "senior",
        "Lead": "lead",
        "Manager": "executive",
        "Director": "executive",
    }
    _ENUM_TO_LEVEL = {
        "entry": "Entry Level",
        "mid": "Mid-Level",
        "senior": "Senior",
        "lead": "Lead",
        "executive": "Director",
    }

    _STATUS_TO_ENUM = {
        "Open": "open",
        "Draft": "draft",
        "Paused": "paused",
        "Closed": "closed",
        "Filled": "filled",
    }
    _ENUM_TO_STATUS = {v: k for k, v in _STATUS_TO_ENUM.items()}

    def __init__(self, parent=None):
        """Initialize the jobs view."""
        super().__init__(
            title="Job Postings",
            description="Manage job postings and requirements",
            parent=parent,
        )
        self._jobs = []  # Store job data
        self._setup_jobs_view()

    def _setup_jobs_view(self):
        """Set up the jobs view content."""
        # Toolbar
        self._create_toolbar()

        # Jobs table
        self.jobs_table = DataTable(
            columns=["Title", "Company", "Type", "Level", "Status", "Candidates"],
            searchable=True,
        )
        self.jobs_table.row_selected.connect(self._on_job_selected)
        self.add_widget(self.jobs_table)

        # Load data from database
        self._load_jobs_from_db()

    def _create_toolbar(self):
        """Create the toolbar with actions."""
        toolbar = QHBoxLayout()
        toolbar.setSpacing(12)

        # Create job button
        create_btn = PrimaryButton("+ Create Job")
        create_btn.clicked.connect(self._create_job)
        toolbar.addWidget(create_btn)

        # Import button
        import_btn = SecondaryButton("Import from File")
        import_btn.clicked.connect(self._import_job)
        toolbar.addWidget(import_btn)

        toolbar.addStretch()

        # Edit button
        self.edit_btn = SecondaryButton("Edit")
        self.edit_btn.clicked.connect(self._edit_job)
        self.edit_btn.setEnabled(False)
        toolbar.addWidget(self.edit_btn)

        # Delete button
        self.delete_btn = DangerButton("Delete")
        self.delete_btn.clicked.connect(self._delete_job)
        self.delete_btn.setEnabled(False)
        toolbar.addWidget(self.delete_btn)

        self.add_layout(toolbar)

    def _job_to_dict(self, job) -> dict:
        """Convert a Job model object to a flat dict for the UI."""
        emp_type = self._ENUM_TO_TYPE.get(
            getattr(job.employment_type, "value", job.employment_type) or "full_time",
            "Full-time",
        )
        exp_level = self._ENUM_TO_LEVEL.get(
            getattr(job.experience_level, "value", job.experience_level) or "mid",
            "Mid-Level",
        )
        status = self._ENUM_TO_STATUS.get(
            getattr(job.status, "value", job.status) or "open",
            "Open",
        )
        skills = ", ".join(job.all_skills) if job.skill_requirements else ""
        candidates_count = str(job.metadata.applications_count) if job.metadata else "0"

        min_exp = 0
        if job.experience_requirement and job.experience_requirement.minimum_years:
            min_exp = int(job.experience_requirement.minimum_years)

        return {
            "id": str(job.id),
            "title": job.title,
            "company": job.company_name or "",
            "type": emp_type,
            "level": exp_level,
            "status": status,
            "candidates": candidates_count,
            "min_experience": min_exp,
            "required_skills": skills,
            "description": job.description or "",
        }

    def _dict_to_job_create(self, data: dict):
        """Convert a UI form dict to a JobCreate schema."""
        from src.data.models.job import (
            JobCreate, EmploymentType, ExperienceLevel, SkillRequirement,
            ExperienceRequirement,
        )

        emp_type_val = self._TYPE_TO_ENUM.get(data.get("type", "Full-time")) or "full_time"
        exp_level_val = self._LEVEL_TO_ENUM.get(data.get("level", "Mid-Level")) or "mid"

        skills = []
        if data.get("required_skills"):
            for s in data["required_skills"].split(","):
                s = s.strip()
                if s:
                    skills.append(SkillRequirement(name=s, is_required=True))

        exp_req = None
        if data.get("min_experience", 0) > 0:
            exp_req = ExperienceRequirement(minimum_years=data["min_experience"])

        return JobCreate(
            title=data.get("title") or "Untitled Job",
            description=data.get("description") or "No description provided.",
            company_name=data.get("company") or "Unknown Company",
            employment_type=EmploymentType(emp_type_val),
            experience_level=ExperienceLevel(exp_level_val),
            skill_requirements=skills,
            experience_requirement=exp_req,
        )

    def _load_jobs_from_db(self):
        """Load jobs from MongoDB."""
        try:
            from src.data.database import get_database_manager
            from src.data.repositories import get_job_repository

            db_manager = get_database_manager()
            if db_manager.check_sync_connection():
                job_repo = get_job_repository()
                jobs = job_repo.find(
                    {}, limit=200, sort_by="created_at", sort_order=-1
                )
                self._jobs = [self._job_to_dict(j) for j in jobs]
            else:
                self._jobs = []
        except Exception as e:
            self._jobs = []
            logger.error(f"Error loading jobs: {e}")

        self._refresh_table()

    def _refresh_table(self):
        """Refresh the jobs table."""
        columns_map = {
            "Title": "title",
            "Company": "company",
            "Type": "type",
            "Level": "level",
            "Status": "status",
            "Candidates": "candidates",
        }
        self.jobs_table.set_data(self._jobs, columns_map)

    def _on_job_selected(self, job_data: dict):
        """Handle job selection."""
        self.edit_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)
        self.job_selected.emit(job_data)

    def _create_job(self):
        """Open dialog to create a new job."""
        dialog = JobFormDialog(parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            job_data = dialog.get_job_data()
            if job_data["title"]:
                try:
                    from src.data.database import get_database_manager
                    from src.data.repositories import get_job_repository

                    db_manager = get_database_manager()
                    if db_manager.check_sync_connection():
                        job_repo = get_job_repository()
                        schema = self._dict_to_job_create(job_data)
                        created = job_repo.create_from_schema(schema)
                        ui_dict = self._job_to_dict(created)
                        self._jobs.insert(0, ui_dict)
                        self._refresh_table()
                        self.job_created.emit()
                        return
                except Exception as e:
                    QMessageBox.warning(self, "Database Error", f"Could not save to database: {e}")

                # Fallback: in-memory only
                job_data["id"] = str(len(self._jobs) + 1)
                job_data["candidates"] = "0"
                self._jobs.append(job_data)
                self._refresh_table()

    def _edit_job(self):
        """Edit selected job."""
        job_data = self.jobs_table.get_selected_data()
        if job_data:
            dialog = JobFormDialog(job_data=job_data, parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                updated_data = dialog.get_job_data()
                updated_data["id"] = job_data["id"]
                updated_data["candidates"] = job_data.get("candidates", "0")

                # Persist to database
                try:
                    from src.data.database import get_database_manager
                    from src.data.repositories import get_job_repository
                    from src.data.models.job import (
                        JobUpdate, EmploymentType, ExperienceLevel,
                        SkillRequirement, ExperienceRequirement,
                    )
                    from src.utils.constants import JobStatus

                    db_manager = get_database_manager()
                    if db_manager.check_sync_connection():
                        job_repo = get_job_repository()

                        emp_type_val = self._TYPE_TO_ENUM.get(updated_data.get("type", "Full-time")) or "full_time"
                        exp_level_val = self._LEVEL_TO_ENUM.get(updated_data.get("level", "Mid-Level")) or "mid"

                        skills = []
                        if updated_data.get("required_skills"):
                            for s in updated_data["required_skills"].split(","):
                                s = s.strip()
                                if s:
                                    skills.append(SkillRequirement(name=s, is_required=True))

                        exp_req = None
                        if updated_data.get("min_experience", 0) > 0:
                            exp_req = ExperienceRequirement(minimum_years=updated_data["min_experience"])

                        status_val = self._STATUS_TO_ENUM.get(updated_data.get("status", "Open"), "open")

                        update_schema = JobUpdate(
                            title=updated_data.get("title"),
                            description=updated_data.get("description") or None,
                            company_name=updated_data.get("company"),
                            employment_type=EmploymentType(emp_type_val),
                            experience_level=ExperienceLevel(exp_level_val),
                            skill_requirements=skills if skills else None,
                            experience_requirement=exp_req,
                            status=JobStatus(status_val),
                        )

                        result = job_repo.update_from_schema(job_data["id"], update_schema)
                        if result:
                            updated_data = self._job_to_dict(result)
                except Exception as e:
                    QMessageBox.warning(
                        self, "Database Error",
                        f"Could not update in database: {e}\nChanges saved locally.",
                    )

                # Update in list
                for i, job in enumerate(self._jobs):
                    if job["id"] == job_data["id"]:
                        self._jobs[i] = updated_data
                        break
                self._refresh_table()

    def _delete_job(self):
        """Delete selected job."""
        job_data = self.jobs_table.get_selected_data()
        if job_data:
            reply = QMessageBox.question(
                self,
                "Delete Job",
                f"Are you sure you want to delete '{job_data['title']}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                # Delete from database
                try:
                    from src.data.database import get_database_manager
                    from src.data.repositories import get_job_repository

                    db_manager = get_database_manager()
                    if db_manager.check_sync_connection():
                        job_repo = get_job_repository()
                        job_repo.delete(job_data["id"])
                except Exception as e:
                    QMessageBox.warning(
                        self, "Database Error",
                        f"Could not delete from database: {e}",
                    )

                self._jobs = [j for j in self._jobs if j["id"] != job_data["id"]]
                self._refresh_table()
                self.edit_btn.setEnabled(False)
                self.delete_btn.setEnabled(False)

    def _import_job(self):
        """Import job description from file and pre-fill the create dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Job Description",
            "",
            "Documents (*.pdf *.docx *.txt);;All Files (*)",
        )
        if file_path:
            # Try to parse the JD file using NLP
            parsed_data = {}
            try:
                from src.ml.nlp import get_jd_parser
                parser = get_jd_parser()
                result = parser.parse_file(file_path)
                if result:
                    parsed_data = {
                        "title": getattr(result, "title", "") or "",
                        "company": getattr(result, "company_name", "") or "",
                        "description": getattr(result, "description", "") or "",
                        "required_skills": ", ".join(
                            getattr(result, "required_skills", []) or []
                        ),
                    }
            except Exception as e:
                # If parsing fails, just open empty dialog
                logger.warning(f"JD parsing failed, opening empty form: {e}")

            dialog = JobFormDialog(job_data=parsed_data if parsed_data else None, parent=self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                job_data = dialog.get_job_data()
                if job_data["title"]:
                    try:
                        from src.data.database import get_database_manager
                        from src.data.repositories import get_job_repository

                        db_manager = get_database_manager()
                        if db_manager.check_sync_connection():
                            job_repo = get_job_repository()
                            schema = self._dict_to_job_create(job_data)
                            created = job_repo.create_from_schema(schema)
                            ui_dict = self._job_to_dict(created)
                            self._jobs.insert(0, ui_dict)
                            self._refresh_table()
                            self.job_created.emit()
                            return
                    except Exception as e:
                        QMessageBox.warning(self, "Database Error", f"Could not save to database: {e}")

                    job_data["id"] = str(len(self._jobs) + 1)
                    job_data["candidates"] = "0"
                    self._jobs.append(job_data)
                    self._refresh_table()

    def refresh(self):
        """Reload jobs from the database."""
        self._load_jobs_from_db()

    def get_jobs(self) -> list[dict]:
        """Get all jobs."""
        return self._jobs.copy()
