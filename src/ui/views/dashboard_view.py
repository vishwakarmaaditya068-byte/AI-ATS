from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont

from src.utils.constants import COLORS
from src.utils.logger import get_logger
from src.ui.views.base_view import BaseView
from src.ui.widgets import StatCard, InfoCard, CandidateCard, PrimaryButton, SuccessButton, SecondaryButton

logger = get_logger(__name__)


class DashboardView(BaseView):
    """
    Main dashboard view showing recruitment overview.

    Displays:
    - Key statistics (jobs, candidates, matches)
    - Recent activity
    - Quick actions
    - Top candidates
    """

    navigate_to_view = pyqtSignal(int)  # Emitted to switch main window view

    def __init__(self, parent=None):
        """Initialize the dashboard view."""
        super().__init__(
            title="Dashboard",
            description="Overview of recruitment activities and key metrics",
            parent=parent,
        )
        self._setup_dashboard()

    def _setup_dashboard(self):
        """Set up the dashboard content."""
        # Statistics row
        self._create_stats_section()

        # Main content grid
        content_grid = QHBoxLayout()
        content_grid.setSpacing(16)

        # Left column - Recent Activity & Quick Actions
        left_column = QVBoxLayout()
        left_column.setSpacing(16)
        self._create_quick_actions(left_column)
        self._create_recent_activity(left_column)
        left_column.addStretch()
        content_grid.addLayout(left_column, stretch=1)

        # Right column - Top Candidates
        right_column = QVBoxLayout()
        right_column.setSpacing(16)
        self._create_top_candidates(right_column)
        right_column.addStretch()
        content_grid.addLayout(right_column, stretch=1)

        self.add_layout(content_grid)
        self.add_stretch()

    def _create_stats_section(self):
        """Create the statistics cards section."""
        stats_layout = QHBoxLayout()
        stats_layout.setSpacing(16)

        # Active Jobs
        self.jobs_card = StatCard(
            title="Active Jobs",
            value="--",
            subtitle="",
            color=COLORS["primary"],
        )
        stats_layout.addWidget(self.jobs_card)

        # Total Candidates
        self.candidates_card = StatCard(
            title="Total Candidates",
            value="--",
            subtitle="",
            color=COLORS["success"],
        )
        stats_layout.addWidget(self.candidates_card)

        # Matches Made
        self.matches_card = StatCard(
            title="Matches Made",
            value="--",
            subtitle="",
            color=COLORS["primary"],
        )
        stats_layout.addWidget(self.matches_card)

        # Interviews Scheduled
        self.interviews_card = StatCard(
            title="Interviews",
            value="--",
            subtitle="",
            color=COLORS["warning"],
        )
        stats_layout.addWidget(self.interviews_card)

        self.add_layout(stats_layout)

    def _create_quick_actions(self, parent_layout: QVBoxLayout):
        """Create quick actions section."""
        # Section header
        header = QLabel("Quick Actions")
        header_font = QFont("Segoe UI", 14)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet(f"color: {COLORS['text_primary']};")
        parent_layout.addWidget(header)

        # Action buttons
        actions_card = InfoCard(
            title="",
            description="",
        )

        # Create job button -> navigate to Jobs (index 2)
        create_job_btn = PrimaryButton("+ Create New Job Posting")
        create_job_btn.setMinimumHeight(44)
        create_job_btn.clicked.connect(lambda: self.navigate_to_view.emit(2))
        actions_card.add_content(create_job_btn)

        # Import resumes button -> navigate to Candidates (index 1)
        import_btn = SuccessButton("Import Resumes")
        import_btn.setMinimumHeight(44)
        import_btn.clicked.connect(lambda: self.navigate_to_view.emit(1))
        actions_card.add_content(import_btn)

        # Run matching button -> navigate to Matching (index 3)
        match_btn = SecondaryButton("Run AI Matching")
        match_btn.setMinimumHeight(44)
        match_btn.clicked.connect(lambda: self.navigate_to_view.emit(3))
        actions_card.add_content(match_btn)

        parent_layout.addWidget(actions_card)

    def _create_recent_activity(self, parent_layout: QVBoxLayout):
        """Create recent activity section."""
        # Section header
        header = QLabel("Recent Activity")
        header_font = QFont("Segoe UI", 14)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet(f"color: {COLORS['text_primary']};")
        parent_layout.addWidget(header)

        # Activity card (clearable)
        self.activity_card = InfoCard(title="", description="")

        # Placeholder
        placeholder = QLabel("Loading activity...")
        placeholder.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 16px;")
        self.activity_card.add_content(placeholder)

        parent_layout.addWidget(self.activity_card)

    def _create_activity_item(self, title: str, description: str, time: str) -> QWidget:
        """Create a single activity item widget."""
        item = QWidget()
        layout = QHBoxLayout(item)
        layout.setContentsMargins(0, 8, 0, 8)
        layout.setSpacing(12)

        # Activity dot
        dot = QLabel("●")
        dot.setStyleSheet(f"color: {COLORS['primary']}; font-size: 8px;")
        dot.setFixedWidth(12)
        layout.addWidget(dot)

        # Content
        content = QVBoxLayout()
        content.setSpacing(2)

        title_label = QLabel(title)
        title_label.setStyleSheet(
            f"""
            color: {COLORS['text_primary']};
            font-weight: 500;
            font-size: 13px;
        """
        )
        content.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setStyleSheet(
            f"""
            color: {COLORS['text_secondary']};
            font-size: 12px;
        """
        )
        content.addWidget(desc_label)

        layout.addLayout(content)
        layout.addStretch()

        # Time
        time_label = QLabel(time)
        time_label.setStyleSheet(
            f"""
            color: {COLORS['text_secondary']};
            font-size: 11px;
        """
        )
        layout.addWidget(time_label)

        return item

    def _create_top_candidates(self, parent_layout: QVBoxLayout):
        """Create top candidates section."""
        # Section header with view all link
        header_layout = QHBoxLayout()

        header = QLabel("Top Candidates")
        header_font = QFont("Segoe UI", 14)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet(f"color: {COLORS['text_primary']};")
        header_layout.addWidget(header)

        header_layout.addStretch()

        view_all = QLabel("View All →")
        view_all.setStyleSheet(
            f"""
            color: {COLORS['primary']};
            font-size: 12px;
        """
        )
        view_all.setCursor(Qt.CursorShape.PointingHandCursor)
        header_layout.addWidget(view_all)

        parent_layout.addLayout(header_layout)

        # Container for candidate cards (clearable)
        self.top_candidates_container = QWidget()
        self.top_candidates_layout = QVBoxLayout(self.top_candidates_container)
        self.top_candidates_layout.setContentsMargins(0, 0, 0, 0)
        self.top_candidates_layout.setSpacing(8)

        placeholder = QLabel("Loading top candidates...")
        placeholder.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 16px;")
        self.top_candidates_layout.addWidget(placeholder)

        parent_layout.addWidget(self.top_candidates_container)

    def refresh_styles(self) -> None:
        self.jobs_card.refresh_styles()
        self.candidates_card.refresh_styles()
        self.matches_card.refresh_styles()
        self.interviews_card.refresh_styles()

    def refresh(self):
        """Refresh dashboard data from MongoDB."""
        self._load_stats()
        self._load_recent_activity()
        self._load_top_candidates()

    def _load_stats(self):
        """Load statistics from the database."""
        try:
            from src.data.database import get_database_manager

            db_manager = get_database_manager()
            if not db_manager.check_sync_connection():
                self.jobs_card.set_value("0")
                self.candidates_card.set_value("0")
                self.matches_card.set_value("0")
                self.interviews_card.set_value("0")
                return

            from src.data.repositories import (
                get_job_repository,
                get_candidate_repository,
                get_match_repository,
            )

            job_repo = get_job_repository()
            candidate_repo = get_candidate_repository()
            match_repo = get_match_repository()

            open_jobs = job_repo.count({"status": "open"})
            total_candidates = candidate_repo.count({})
            total_matches = match_repo.count({})

            # Count interviews from candidate status
            interview_count = 0
            try:
                status_counts = candidate_repo.get_status_counts()
                interview_count = status_counts.get("interviewing", 0)
            except Exception as e:
                logger.debug(f"Error getting interview count: {e}")

            self.jobs_card.set_value(str(open_jobs))
            self.candidates_card.set_value(str(total_candidates))
            self.matches_card.set_value(str(total_matches))
            self.interviews_card.set_value(str(interview_count))

        except Exception as e:
            logger.error(f"Error loading dashboard stats: {e}")
            self.jobs_card.set_value("0")
            self.candidates_card.set_value("0")
            self.matches_card.set_value("0")
            self.interviews_card.set_value("0")

    def _load_recent_activity(self):
        """Load recent activity from audit logs."""
        # Clear existing content
        content_layout = self.activity_card.content_area
        while content_layout.count():
            item = content_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        activities = []
        try:
            # Read from PostgreSQL audit_logs (authoritative since Phase 2).
            from src.data.sql.engine import get_sql_manager
            from src.data.sql.repositories.audit_record_repo import (
                get_audit_record_repository,
            )
            from sqlalchemy import select

            sql_mgr = get_sql_manager()
            if sql_mgr.check_sync_connection():
                repo = get_audit_record_repository()
                from src.data.sql.models.audit_record import AuditRecord

                with sql_mgr.sync_session() as session:
                    stmt = select(AuditRecord).order_by(AuditRecord.occurred_at.desc()).limit(5)
                    logs = session.execute(stmt).scalars().all()
                    for r in logs:
                        session.expunge(r)

                from datetime import datetime, timezone

                for log in logs:
                    action = getattr(log, "action", "")
                    details = getattr(log, "action_description", "") or ""
                    occurred = getattr(log, "occurred_at", None)

                    time_str = ""
                    if occurred:
                        now = datetime.now(timezone.utc)
                        if occurred.tzinfo is None:
                            occurred = occurred.replace(tzinfo=timezone.utc)
                        delta = now - occurred
                        if delta.days > 0:
                            time_str = f"{delta.days}d ago"
                        elif delta.seconds > 3600:
                            time_str = f"{delta.seconds // 3600}h ago"
                        elif delta.seconds > 60:
                            time_str = f"{delta.seconds // 60}m ago"
                        else:
                            time_str = "Just now"

                    activities.append(
                        (
                            str(action).replace("_", " ").title(),
                            str(details)[:60],
                            time_str,
                        )
                    )
        except Exception as e:
            logger.debug(f"Error loading recent activity: {e}")

        if not activities:
            activities = [("No recent activity", "Start by adding jobs or candidates", "")]

        for title, desc, time in activities:
            activity_item = self._create_activity_item(title, desc, time)
            self.activity_card.add_content(activity_item)

    def _load_top_candidates(self):
        """Load top candidates from match results."""
        # Clear existing
        while self.top_candidates_layout.count():
            item = self.top_candidates_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        try:
            from src.data.database import get_database_manager
            from src.data.repositories import (
                get_match_repository,
                get_candidate_repository,
            )

            db_manager = get_database_manager()
            if db_manager.check_sync_connection():
                match_repo = get_match_repository()
                candidate_repo = get_candidate_repository()

                top_matches = match_repo.find({}, limit=4, sort_by="overall_score", sort_order=-1)

                found = False
                for match in top_matches:
                    candidate_id = getattr(match, "candidate_id", None)
                    score = getattr(match, "overall_score", 0) or 0
                    if not candidate_id:
                        continue

                    candidate = candidate_repo.get_by_id(str(candidate_id))
                    if not candidate:
                        continue

                    found = True
                    name = f"{candidate.first_name} {candidate.last_name}"
                    skills = candidate.skill_names[:4] if candidate.skills else []
                    exp = f"{candidate.total_experience_years} yrs experience"
                    if candidate.headline:
                        exp += f" - {candidate.headline}"

                    card = CandidateCard(
                        name=name,
                        score=score,
                        skills=skills,
                        experience=exp,
                    )
                    self.top_candidates_layout.addWidget(card)

                if not found:
                    placeholder = QLabel("No match results yet. Run AI Matching first.")
                    placeholder.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 16px;")
                    self.top_candidates_layout.addWidget(placeholder)
                return
        except Exception as e:
            logger.debug(f"Error loading top candidates: {e}")

        placeholder = QLabel("No match results yet. Run AI Matching first.")
        placeholder.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 16px;")
        self.top_candidates_layout.addWidget(placeholder)

    def update_stats(
        self,
        jobs: int = None,
        candidates: int = None,
        matches: int = None,
        interviews: int = None,
    ):
        """Update dashboard statistics."""
        if jobs is not None:
            self.jobs_card.set_value(str(jobs))
        if candidates is not None:
            self.candidates_card.set_value(str(candidates))
        if matches is not None:
            self.matches_card.set_value(str(matches))
        if interviews is not None:
            self.interviews_card.set_value(str(interviews))
