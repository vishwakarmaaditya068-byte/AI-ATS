"""Main screen views for the application."""

from src.ui.views.base_view import BaseView
from src.ui.views.dashboard_view import DashboardView
from src.ui.views.jobs_view import JobsView
from src.ui.views.matching_view import MatchingView
from src.ui.views.settings_view import SettingsView
from src.ui.views.candidates_view import CandidatesView
from src.ui.views.analytics_view import AnalyticsView

__all__ = [
    "BaseView",
    "DashboardView",
    "JobsView",
    "MatchingView",
    "SettingsView",
    "CandidatesView",
    "AnalyticsView",
]
