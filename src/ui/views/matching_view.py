from typing import Optional
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QProgressBar,
    QDialog,
    QFileDialog,
    QMessageBox,
    QSplitter,
    QTextEdit,
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QObject
from PyQt6.QtGui import QFont

from src.utils.constants import COLORS
from src.utils.logger import get_logger
from src.ui.views.base_view import BaseView

logger = get_logger(__name__)
from src.ui.widgets import (
    InfoCard,
    ScoreCard,
    PrimaryButton,
    SecondaryButton,
    DataTable,
)


class MatchingWorker(QObject):
    """Worker thread for running matching in background."""

    progress = pyqtSignal(int, str)  # progress percentage, status message
    finished = pyqtSignal(list)  # results list
    error = pyqtSignal(str)  # error message

    def __init__(self, job_data: dict, resume_files: list[str]):
        super().__init__()
        self.job_data = job_data
        self.resume_files = resume_files
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        """Run the matching process using AccurateJDParser + match_from_parsed pipeline."""
        try:
            from src.ml.nlp.accurate_jd_parser import AccurateJDParser
            from src.ml.nlp.accurate_resume_parser import AccurateResumeParser
            from src.core.matching import get_matching_engine
            from src.data.models.job import Job, SkillRequirement

            from src.core.ranking import rank_candidates, RankingConfig

            results: list[dict] = []
            match_results_list: list = []
            total: int = len(self.resume_files)

            # ── Initialise ──────────────────────────────────────────────────
            self.progress.emit(5, "Initializing matching engine...")
            jd_parser: AccurateJDParser = AccurateJDParser()
            resume_parser: AccurateResumeParser = AccurateResumeParser()
            matching_engine = get_matching_engine()

            workspace_id: Optional[str] = None
            job_record_id: Optional[str] = None
            persistence = None
            audit = None
            try:
                from src.services.match_persistence import get_match_persistence_service
                from src.services.audit_service import get_audit_service

                persistence = get_match_persistence_service()
                audit = get_audit_service()
                ws = persistence.ensure_default_workspace()
                workspace_id = str(ws.id)
                job_rec = persistence.ensure_job_record(
                    workspace_id=ws.id,
                    title=self.job_data.get("title") or "Untitled Position",
                    company_name=self.job_data.get("company"),
                    description=self.job_data.get("description"),
                    mongo_doc_id=self.job_data.get("id"),
                )
                job_record_id = str(job_rec.id)
            except Exception as exc:
                logger.warning(
                    f"Postgres persistence unavailable — matches will not be saved: {exc}"
                )

            # ── Parse job description ────────────────────────────────────────
            self.progress.emit(10, "Parsing job description...")
            jd_text: str = self.job_data.get("description", "")
            parsed_jd = jd_parser.parse(jd_text)

            # Override with explicit form-field values when provided
            if self.job_data.get("title"):
                parsed_jd.title = self.job_data["title"]
            if self.job_data.get("company"):
                parsed_jd.company_name = self.job_data["company"]
            extra_skills_str: str = self.job_data.get("required_skills", "")
            if extra_skills_str:
                extra: list[str] = [
                    s.strip().lower() for s in extra_skills_str.split(",") if s.strip()
                ]
                existing: set[str] = set(parsed_jd.required_skills)
                parsed_jd.required_skills = parsed_jd.required_skills + [
                    s for s in extra if s not in existing
                ]

            # Build a Job model for matching (no DB — purely in-memory)
            description_text: str = (
                parsed_jd.description or jd_text[:500] or "No description provided."
            )
            if len(description_text) < 10:
                description_text = description_text.ljust(10)

            job_for_matching: Job = Job(
                title=parsed_jd.title or "Untitled Position",
                description=description_text,
                responsibilities=parsed_jd.responsibilities,
                company_name=parsed_jd.company_name or "Unknown Company",
                skill_requirements=(
                    [SkillRequirement(name=s, is_required=True) for s in parsed_jd.required_skills]
                    + [
                        SkillRequirement(name=s, is_required=False)
                        for s in parsed_jd.preferred_skills
                    ]
                ),
            )

            # ── Process each resume ──────────────────────────────────────────
            for i, resume_file in enumerate(self.resume_files):
                if self._is_cancelled:
                    return

                progress_pct: int = 15 + int((i / total) * 80)
                self.progress.emit(progress_pct, f"Processing resume {i+1}/{total}...")

                try:
                    parsed = resume_parser.parse(resume_file)

                    # Run matching via new typed pipeline
                    match_result = matching_engine.match_from_parsed(parsed, job_for_matching)

                    # Candidate name
                    name: str = parsed.contact.name or "Unknown Candidate"

                    # Score level
                    score_level: str = (
                        match_result.score_level.value if match_result.score_level else "fair"
                    )

                    # Explanation text
                    explanation_parts: list[str] = []
                    if match_result.explanation:
                        if match_result.explanation.summary:
                            explanation_parts.append(match_result.explanation.summary)
                        if match_result.explanation.strengths:
                            explanation_parts.append("\n\nStrengths:")
                            for s in match_result.explanation.strengths[:3]:
                                explanation_parts.append(f"• {s}")
                        if match_result.explanation.gaps:
                            explanation_parts.append("\n\nAreas for improvement:")
                            for g in match_result.explanation.gaps[:3]:
                                explanation_parts.append(f"• {g}")

                    # Bias check
                    bias_detected: bool = False
                    bias_info: str = ""
                    if match_result.bias_check:
                        bias_detected = match_result.bias_check.potential_bias_detected
                        if bias_detected and match_result.bias_check.protected_attributes_found:
                            bias_info = (
                                f"Potential bias indicators: "
                                f"{', '.join(match_result.bias_check.protected_attributes_found)}"
                            )

                    # ── Persist match + emit audit (fail-soft) ─────────────
                    persisted_match_id: Optional[str] = None
                    effective_score: float = match_result.overall_score
                    is_override: bool = False
                    if persistence is not None and job_record_id is not None:
                        try:
                            candidate_mongo_id = persistence.resolve_candidate_mongo_id(
                                email=(parsed.contact.email or None),
                            )
                            if candidate_mongo_id:
                                bc = match_result.bias_check
                                bias_payload = (
                                    {
                                        "potential_bias_detected": bool(
                                            bc.potential_bias_detected if bc else False
                                        ),
                                        "protected_attributes_found": list(
                                            bc.protected_attributes_found if bc else []
                                        ),
                                    }
                                    if bc
                                    else {}
                                )
                                explanation_payload = (
                                    {
                                        "summary": (
                                            match_result.explanation.summary
                                            if match_result.explanation
                                            else None
                                        ),
                                        "strengths": (
                                            list(match_result.explanation.strengths)
                                            if match_result.explanation
                                            else []
                                        ),
                                        "gaps": (
                                            list(match_result.explanation.gaps)
                                            if match_result.explanation
                                            else []
                                        ),
                                    }
                                    if match_result.explanation
                                    else {}
                                )
                                persisted = persistence.persist_match(
                                    job_id=job_record_id,
                                    candidate_mongo_id=candidate_mongo_id,
                                    overall_score=float(match_result.overall_score),
                                    skills_score=match_result.skills_score,
                                    experience_score=match_result.experience_score,
                                    education_score=match_result.education_score,
                                    semantic_score=match_result.semantic_score,
                                    keyword_score=match_result.keyword_score,
                                    score_level=score_level,
                                    explanation=explanation_payload,
                                    bias_check=bias_payload,
                                )
                                persisted_match_id = str(persisted.id)
                                effective_score = persisted.effective_score
                                is_override = persisted.manual_score_override is not None

                                if audit is not None:
                                    audit.emit_candidate_scored(
                                        workspace_id=workspace_id,
                                        job_id=job_record_id,
                                        match_id=persisted_match_id,
                                        candidate_mongo_id=candidate_mongo_id,
                                        candidate_name=name,
                                        job_title=self.job_data.get("title") or "Untitled",
                                        score=float(match_result.overall_score),
                                        explanation_summary=(
                                            match_result.explanation.summary
                                            if match_result.explanation
                                            else None
                                        ),
                                    )
                                    if bias_detected:
                                        audit.emit_bias_detected(
                                            workspace_id=workspace_id,
                                            bias_types=(
                                                list(bc.protected_attributes_found)
                                                if bc and bc.protected_attributes_found
                                                else ["unspecified"]
                                            ),
                                            detection_method="protected_attribute_detector",
                                            affected_count=1,
                                            candidate_mongo_id=candidate_mongo_id,
                                            job_id=job_record_id,
                                            match_id=persisted_match_id,
                                        )
                            else:
                                logger.info(
                                    f"Candidate not yet ingested — skipping persistence "
                                    f"for {parsed.contact.email or resume_file}"
                                )
                        except Exception as exc:
                            logger.warning(f"Match persistence failed: {exc}")

                    result_dict: dict = {
                        "candidate": name,
                        "email": parsed.contact.email or "",
                        "score": effective_score,
                        "ai_score": match_result.overall_score,
                        "score_display": f"{effective_score:.0%}",
                        "match_level": score_level.capitalize(),
                        "skills_match": (
                            f"{len(match_result.matched_skills)}/{len(match_result.skill_matches)}"
                            if match_result.skill_matches
                            else "N/A"
                        ),
                        "skills_score": match_result.skills_score or 0,
                        "experience_score": match_result.experience_score or 0,
                        "education_score": match_result.education_score or 0,
                        "semantic_score": match_result.semantic_score or 0,
                        "explanation": (
                            "\n".join(explanation_parts)
                            if explanation_parts
                            else "No detailed explanation available."
                        ),
                        "bias_detected": bias_detected,
                        "bias_info": bias_info,
                        "file_path": resume_file,
                        "match_id": persisted_match_id,
                        "workspace_id": workspace_id,
                        "job_record_id": job_record_id,
                        "is_override": is_override,
                    }

                    # Append to both lists atomically — they must stay in 1:1 sync
                    # because the post-loop zip(match_results_list, results) relies on it.
                    results.append(result_dict)
                    match_results_list.append(match_result)

                except Exception as e:
                    logger.warning(f"Error processing {resume_file}: {e}")
                    continue

            # Rank using CandidateRanker + FairnessReranker
            ranking_config = RankingConfig(fairness_mode="flag")
            ranked = rank_candidates(match_results_list, ranking_config)

            # Map RankedResult back to the existing dict format (1:1 by object identity)
            mr_to_dict: dict[int, dict] = {id(mr): d for mr, d in zip(match_results_list, results)}
            final_results: list[dict] = []
            for rr in ranked:
                d = mr_to_dict[id(rr.match_result)]
                d["rank"] = str(rr.rank)
                d["fairness_flags"] = rr.fairness_flags
                final_results.append(d)
            results = final_results

            self.progress.emit(100, "Matching complete!")
            self.finished.emit(results)

        except Exception as e:
            self.error.emit(str(e))


class MatchingView(BaseView):
    """
    AI Matching view for candidate-job matching.

    Provides:
    - Job selection for matching
    - Resume import/selection
    - Run matching process
    - View match results with scores
    - Detailed match explanations
    """

    def __init__(self, parent=None):
        """Initialize the matching view."""
        super().__init__(
            title="AI Matching",
            description="Run AI-powered candidate-job matching and view results",
            parent=parent,
        )
        self._jobs = []  # Store job data
        self._resume_files = []
        self._results: list[dict] = []
        self._selected_result: Optional[dict] = None
        self._worker = None
        self._worker_thread = None
        self._setup_matching_view()
        self._load_jobs()

    def _cleanup_worker(self):
        """Clean up the worker thread if running."""
        if self._worker:
            self._worker.cancel()
            self._worker = None
        if self._worker_thread and self._worker_thread.isRunning():
            self._worker_thread.quit()
            self._worker_thread.wait(3000)
            self._worker_thread = None

    def closeEvent(self, event):
        """Clean up worker thread on view close."""
        self._cleanup_worker()
        super().closeEvent(event)

    def _setup_matching_view(self):
        """Set up the matching view content."""
        # Create splitter for left/right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left panel - Controls and Results
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 8, 0)
        left_layout.setSpacing(16)

        self._create_controls_section(left_layout)
        self._create_results_section(left_layout)

        splitter.addWidget(left_panel)

        # Right panel - Match Details
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(8, 0, 0, 0)
        right_layout.setSpacing(16)

        self._create_details_section(right_layout)

        splitter.addWidget(right_panel)

        # Set splitter proportions
        splitter.setSizes([500, 400])

        self.add_widget(splitter)

    def _load_jobs(self):
        """Load jobs from database."""
        try:
            from src.data.database import get_database_manager
            from src.data.repositories import get_job_repository

            db_manager = get_database_manager()
            if db_manager.check_sync_connection():
                job_repo = get_job_repository()
                jobs = job_repo.find({}, limit=50, sort_by="created_at", sort_order=-1)

                self._jobs = []
                self.job_combo.clear()

                for job in jobs:
                    job_data = {
                        "id": str(job.id),
                        "title": job.title,
                        "company": job.company_name,
                        "description": job.description or "",
                        "required_skills": (
                            ", ".join([s.name for s in job.skill_requirements if s.is_required])
                            if job.skill_requirements
                            else ""
                        ),
                    }
                    self._jobs.append(job_data)
                    self.job_combo.addItem(f"{job.title} - {job.company_name}")

                if not self._jobs:
                    self.job_combo.addItem("No jobs found - create jobs first")
            else:
                self.job_combo.addItem("Database not connected")
        except Exception as e:
            self.job_combo.addItem(f"Error loading jobs: {str(e)[:30]}")

    def _create_controls_section(self, parent_layout: QVBoxLayout):
        """Create matching controls section."""
        controls_card = InfoCard(
            title="Matching Configuration",
            description="Select a job and import resumes to run AI matching",
        )

        # Job selection
        job_layout = QHBoxLayout()
        job_label = QLabel("Select Job:")
        job_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 500;")
        job_layout.addWidget(job_label)

        self.job_combo = QComboBox()
        self.job_combo.setMinimumWidth(300)
        self.job_combo.setMinimumHeight(36)
        self.job_combo.setStyleSheet(
            f"""
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
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
        """
        )
        job_layout.addWidget(self.job_combo)

        # Refresh jobs button
        refresh_btn = SecondaryButton("↻")
        refresh_btn.setFixedWidth(40)
        refresh_btn.setToolTip("Refresh job list")
        refresh_btn.clicked.connect(self._load_jobs)
        job_layout.addWidget(refresh_btn)

        job_layout.addStretch()
        controls_card.add_content(self._wrap_layout(job_layout))

        # Resume import
        resume_layout = QHBoxLayout()
        resume_label = QLabel("Resumes:")
        resume_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 500;")
        resume_layout.addWidget(resume_label)

        self.resume_count_label = QLabel("0 resumes loaded")
        self.resume_count_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        resume_layout.addWidget(self.resume_count_label)

        resume_layout.addStretch()

        import_btn = SecondaryButton("Import Resumes")
        import_btn.clicked.connect(self._import_resumes)
        resume_layout.addWidget(import_btn)

        controls_card.add_content(self._wrap_layout(resume_layout))

        # Progress status
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-style: italic;")
        self.status_label.hide()
        controls_card.add_content(self.status_label)

        # Action buttons
        action_layout = QHBoxLayout()
        action_layout.addStretch()

        self.run_btn = PrimaryButton("▶ Run AI Matching")
        self.run_btn.setMinimumWidth(180)
        self.run_btn.setMinimumHeight(44)
        self.run_btn.clicked.connect(self._run_matching)
        action_layout.addWidget(self.run_btn)

        controls_card.add_content(self._wrap_layout(action_layout))

        # Progress bar (hidden by default)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(8)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet(
            f"""
            QProgressBar {{
                background-color: {COLORS['surface_overlay']};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 4px;
            }}
        """
        )
        self.progress_bar.hide()
        controls_card.add_content(self.progress_bar)

        parent_layout.addWidget(controls_card)

    def _create_results_section(self, parent_layout: QVBoxLayout):
        """Create matching results section."""
        # Section header
        header_layout = QHBoxLayout()
        header = QLabel("Match Results")
        header_font = QFont("Segoe UI", 14)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet(f"color: {COLORS['text_primary']};")
        header_layout.addWidget(header)

        header_layout.addStretch()

        self.results_count_label = QLabel("")
        self.results_count_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        header_layout.addWidget(self.results_count_label)

        parent_layout.addLayout(header_layout)

        # Results table
        self.results_table = DataTable(
            columns=["Rank", "Candidate", "Score", "Match Level", "Skills Match"],
            searchable=True,
        )
        self.results_table.row_selected.connect(self._on_result_selected)
        parent_layout.addWidget(self.results_table)

    def _create_details_section(self, parent_layout: QVBoxLayout):
        """Create match details section."""
        # Section header
        header = QLabel("Match Details")
        header_font = QFont("Segoe UI", 14)
        header_font.setBold(True)
        header.setFont(header_font)
        header.setStyleSheet(f"color: {COLORS['text_primary']};")
        parent_layout.addWidget(header)

        # Candidate info card
        self.candidate_info = InfoCard(
            title="Select a candidate to view details",
            description="Click on a match result to see detailed analysis",
        )
        parent_layout.addWidget(self.candidate_info)

        # Score breakdown
        self.score_section = QWidget()
        score_layout = QVBoxLayout(self.score_section)
        score_layout.setContentsMargins(0, 0, 0, 0)
        score_layout.setSpacing(8)

        score_header = QLabel("Score Breakdown")
        score_header.setStyleSheet(
            f"""
            color: {COLORS['text_primary']};
            font-weight: 600;
            font-size: 13px;
        """
        )
        score_layout.addWidget(score_header)

        self.skills_score = ScoreCard("Skills Match", 0.0)
        score_layout.addWidget(self.skills_score)

        self.experience_score = ScoreCard("Experience Match", 0.0)
        score_layout.addWidget(self.experience_score)

        self.education_score = ScoreCard("Education Match", 0.0)
        score_layout.addWidget(self.education_score)

        self.semantic_score = ScoreCard("Semantic Similarity", 0.0)
        score_layout.addWidget(self.semantic_score)

        self.score_section.hide()
        parent_layout.addWidget(self.score_section)

        # Explanation section
        self.explanation_section = QWidget()
        exp_layout = QVBoxLayout(self.explanation_section)
        exp_layout.setContentsMargins(0, 0, 0, 0)
        exp_layout.setSpacing(8)

        exp_header = QLabel("AI Explanation")
        exp_header.setStyleSheet(
            f"""
            color: {COLORS['text_primary']};
            font-weight: 600;
            font-size: 13px;
        """
        )
        exp_layout.addWidget(exp_header)

        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        self.explanation_text.setMinimumHeight(150)
        self.explanation_text.setStyleSheet(
            f"""
            QTextEdit {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
                padding: 12px;
                font-size: 13px;
                line-height: 1.5;
            }}
        """
        )
        exp_layout.addWidget(self.explanation_text)

        self.explanation_section.hide()
        parent_layout.addWidget(self.explanation_section)

        # Bias check section
        self.bias_section = QWidget()
        bias_layout = QVBoxLayout(self.bias_section)
        bias_layout.setContentsMargins(0, 0, 0, 0)
        bias_layout.setSpacing(8)

        bias_header = QLabel("Fairness Analysis")
        bias_header.setStyleSheet(
            f"""
            color: {COLORS['text_primary']};
            font-weight: 600;
            font-size: 13px;
        """
        )
        bias_layout.addWidget(bias_header)

        self.bias_status = QLabel("✓ No bias indicators detected")
        self.bias_status.setStyleSheet(
            f"""
            background-color: {COLORS['success']}22;
            color: {COLORS['success']};
            padding: 12px;
            border-radius: 6px;
            font-size: 13px;
        """
        )
        bias_layout.addWidget(self.bias_status)

        self.bias_section.hide()
        parent_layout.addWidget(self.bias_section)

        # Override section — enabled only when a result with a persisted
        # match_id is selected.
        self.override_section = QWidget()
        override_layout = QHBoxLayout(self.override_section)
        override_layout.setContentsMargins(0, 0, 0, 0)
        override_layout.setSpacing(8)

        self.override_btn = PrimaryButton("Override Score…")
        self.override_btn.setEnabled(False)
        self.override_btn.clicked.connect(self._on_override_clicked)
        override_layout.addWidget(self.override_btn)

        self.clear_override_btn = SecondaryButton("Clear Override")
        self.clear_override_btn.setVisible(False)
        self.clear_override_btn.clicked.connect(self._on_clear_override_clicked)
        override_layout.addWidget(self.clear_override_btn)

        override_layout.addStretch()
        self.override_section.hide()
        parent_layout.addWidget(self.override_section)

        parent_layout.addStretch()

    def _wrap_layout(self, layout: QHBoxLayout) -> QWidget:
        """Wrap a layout in a widget."""
        widget = QWidget()
        widget.setLayout(layout)
        return widget

    def _import_resumes(self):
        """Import resume files."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Import Resumes",
            "",
            "Documents (*.pdf *.docx *.txt);;All Files (*)",
        )
        if files:
            count = len(files)
            self.resume_count_label.setText(f"{count} resume{'s' if count != 1 else ''} loaded")
            self._resume_files = files

    def _run_matching(self):
        """Run the AI matching process."""
        # Validate inputs
        if not self._resume_files:
            QMessageBox.warning(
                self,
                "No Resumes",
                "Please import resumes before running matching.",
            )
            return

        job_index = self.job_combo.currentIndex()
        if job_index < 0 or job_index >= len(self._jobs):
            QMessageBox.warning(
                self,
                "No Job Selected",
                "Please select a valid job posting.",
            )
            return

        job_data = self._jobs[job_index]

        # Show progress
        self.progress_bar.show()
        self.progress_bar.setValue(0)
        self.status_label.show()
        self.status_label.setText("Initializing...")
        self.run_btn.setEnabled(False)
        self.run_btn.setText("Processing...")

        # Create worker and thread
        self._worker_thread = QThread()
        self._worker = MatchingWorker(job_data, self._resume_files)
        self._worker.moveToThread(self._worker_thread)

        # Connect signals
        self._worker_thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_matching_finished)
        self._worker.error.connect(self._on_matching_error)
        self._worker.finished.connect(self._worker_thread.quit)
        self._worker.error.connect(self._worker_thread.quit)
        self._worker_thread.finished.connect(self._on_thread_finished)

        # Start
        self._worker_thread.start()

    def _on_progress(self, value: int, message: str):
        """Handle progress updates."""
        self.progress_bar.setValue(value)
        self.status_label.setText(message)

    def _on_matching_finished(self, results: list):
        """Handle matching completion."""
        self.progress_bar.hide()
        self.status_label.hide()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶ Run AI Matching")

        self._results = results
        self._display_results(results)

    def _on_thread_finished(self):
        """Release worker and thread references once the thread has stopped."""
        self._worker = None
        self._worker_thread = None

    def _on_matching_error(self, error_message: str):
        """Handle matching error."""
        self.progress_bar.hide()
        self.status_label.hide()
        self.run_btn.setEnabled(True)
        self.run_btn.setText("▶ Run AI Matching")

        QMessageBox.critical(
            self,
            "Matching Error",
            f"An error occurred during matching:\n\n{error_message}",
        )

    def _display_results(self, results: list):
        """Display match results in the table."""
        if not results:
            self.results_count_label.setText("No matches found")
            return

        # Prepare data for table. Override badge (★) prefixed to score
        # gives recruiters a one-glance signal without a separate column.
        table_data = []
        for r in results:
            score_display: str = r.get("score_display", "")
            if r.get("is_override") and "★" not in score_display:
                score_display = f"{score_display} ★"
            table_data.append(
                {
                    "rank": r.get("rank", ""),
                    "candidate": r["candidate"],
                    "score": score_display,
                    "match_level": r["match_level"],
                    "skills_match": r["skills_match"],
                    # Keep full data for selection
                    **r,
                    "score_display": score_display,
                }
            )

        columns_map = {
            "Rank": "rank",
            "Candidate": "candidate",
            "Score": "score",
            "Match Level": "match_level",
            "Skills Match": "skills_match",
        }
        self.results_table.set_data(table_data, columns_map)
        self.results_count_label.setText(f"{len(results)} candidates matched")

    def _on_result_selected(self, result_data: dict):
        """Handle result selection to show details."""
        self._selected_result = result_data
        # Update candidate info
        self.candidate_info.title_label.setText(result_data.get("candidate", "Unknown"))
        self.candidate_info.description_label.setText(
            f"Overall Score: {result_data.get('score_display', result_data.get('score', 'N/A'))} • {result_data.get('match_level', 'Unknown')} Match"
        )

        has_match_id: bool = bool(result_data.get("match_id"))
        is_override: bool = bool(result_data.get("is_override"))
        self.override_section.setVisible(True)
        self.override_btn.setEnabled(has_match_id)
        if is_override:
            self.override_btn.setText("Edit Override…")
            self.clear_override_btn.setVisible(True)
        else:
            self.override_btn.setText("Override Score…")
            self.clear_override_btn.setVisible(False)

        # Show and update score breakdown
        self.score_section.show()
        self.skills_score.set_score(result_data.get("skills_score", 0))
        self.experience_score.set_score(result_data.get("experience_score", 0))
        self.education_score.set_score(result_data.get("education_score", 0))
        self.semantic_score.set_score(result_data.get("semantic_score", 0))

        # Show explanation
        self.explanation_section.show()
        self.explanation_text.setText(result_data.get("explanation", "No explanation available."))

        # Show bias section with appropriate styling
        self.bias_section.show()
        if result_data.get("bias_detected"):
            self.bias_status.setText(
                f"⚠ {result_data.get('bias_info', 'Potential bias indicators detected')}"
            )
            self.bias_status.setStyleSheet(
                f"""
                background-color: {COLORS['warning']}22;
                color: {COLORS['warning']};
                padding: 12px;
                border-radius: 2px;
                font-size: 13px;
            """
            )
        else:
            self.bias_status.setText("✓ No bias indicators detected")
            self.bias_status.setStyleSheet(
                f"""
                background-color: {COLORS['success']}22;
                color: {COLORS['success']};
                padding: 12px;
                border-radius: 2px;
                font-size: 13px;
            """
            )

    def refresh_styles(self) -> None:
        self.job_combo.setStyleSheet(f"""
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
            QComboBox::drop-down {{ border: none; padding-right: 8px; }}
        """)
        self.progress_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {COLORS['surface_overlay']};
                border: none;
                border-radius: 4px;
            }}
            QProgressBar::chunk {{
                background-color: {COLORS['primary']};
                border-radius: 4px;
            }}
        """)
        self.explanation_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['surface_elevated']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border_subtle']};
                border-radius: 4px;
                padding: 12px;
                font-size: 13px;
            }}
        """)
        self.results_table.refresh_styles()

    def refresh(self):
        """Refresh the matching view by reloading jobs."""
        self._load_jobs()

    # Manual override — opens dialog, calls repo, emits audit, refreshes row
    def _on_override_clicked(self):
        if not self._selected_result:
            return
        match_id: Optional[str] = self._selected_result.get("match_id")
        if not match_id:
            QMessageBox.warning(
                self,
                "Override unavailable",
                "This match has not been persisted to the database. Ensure "
                "PostgreSQL is running and rerun the matching step.",
            )
            return

        from src.ui.dialogs.override_score_dialog import OverrideScoreDialog

        job_title: str = ""
        if self._jobs and self.job_combo.currentIndex() >= 0:
            idx = self.job_combo.currentIndex()
            if 0 <= idx < len(self._jobs):
                job_title = self._jobs[idx].get("title", "") or ""

        dialog = OverrideScoreDialog(
            candidate_name=self._selected_result.get("candidate", "Unknown"),
            job_title=job_title or "Unknown Position",
            current_score=float(self._selected_result.get("score", 0.0)),
            ai_score=float(
                self._selected_result.get("ai_score", self._selected_result.get("score", 0.0))
            ),
            existing_reason=None,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        new_score: float = dialog.new_score
        reason: str = dialog.reason

        try:
            from src.data.sql.repositories.match_record_repo import (
                get_match_record_repository,
            )
            from src.services.audit_service import get_audit_service
            import os

            repo = get_match_record_repository()
            overridden_by: str = os.environ.get("USER") or "recruiter"
            original_score: float = float(
                self._selected_result.get("ai_score") or self._selected_result.get("score", 0.0)
            )
            updated = repo.apply_override(
                match_id=match_id,
                new_score=new_score,
                reason=reason,
                overridden_by=overridden_by,
            )
            if updated is None:
                raise RuntimeError("Match row not found")

            # Audit — fail-soft
            try:
                get_audit_service().emit_manual_override(
                    workspace_id=self._selected_result.get("workspace_id") or "",
                    job_id=self._selected_result.get("job_record_id") or "",
                    match_id=match_id,
                    candidate_mongo_id=(
                        updated.candidate_mongo_id if updated.candidate_mongo_id else ""
                    ),
                    candidate_name=self._selected_result.get("candidate", "Unknown"),
                    original_score=original_score,
                    new_score=new_score,
                    reason=reason,
                    actor_name=overridden_by,
                )
            except Exception as exc:
                logger.warning(f"Override audit emit failed: {exc}")

            # Update the in-memory row and refresh UI
            self._selected_result["score"] = new_score
            self._selected_result["score_display"] = f"{new_score:.0%} ★"
            self._selected_result["is_override"] = True
            self._display_results(self._results)
            self._on_result_selected(self._selected_result)

            QMessageBox.information(self, "Override saved", f"Match score set to {new_score:.0%}.")

        except ValueError as exc:
            QMessageBox.warning(self, "Invalid override", str(exc))
        except Exception as exc:
            logger.exception(f"Override failed: {exc}")
            QMessageBox.critical(self, "Override failed", str(exc))

    def _on_clear_override_clicked(self):
        if not self._selected_result:
            return
        match_id: Optional[str] = self._selected_result.get("match_id")
        if not match_id:
            return

        confirm = QMessageBox.question(
            self,
            "Clear override?",
            "Revert this match to the AI-computed score? "
            "The audit record of this override will be preserved.",
        )
        if confirm != QMessageBox.StandardButton.Yes:
            return

        try:
            from src.data.sql.repositories.match_record_repo import (
                get_match_record_repository,
            )

            repo = get_match_record_repository()
            updated = repo.clear_override(match_id)
            if updated is None:
                raise RuntimeError("Match row not found")

            ai_score: float = float(self._selected_result.get("ai_score", updated.overall_score))
            self._selected_result["score"] = ai_score
            self._selected_result["score_display"] = f"{ai_score:.0%}"
            self._selected_result["is_override"] = False
            self._display_results(self._results)
            self._on_result_selected(self._selected_result)
        except Exception as exc:
            logger.exception(f"Clear override failed: {exc}")
            QMessageBox.critical(self, "Clear override failed", str(exc))
