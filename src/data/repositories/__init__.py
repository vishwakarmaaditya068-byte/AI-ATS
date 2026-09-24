"""
Database repositories for AI-ATS data access.

This module provides repository classes for all database collections,
implementing the repository pattern for clean data access.
"""

# Base repository
from .base import BaseRepository

# Entity repositories
from .candidate_repository import CandidateRepository, get_candidate_repository
from .resume_repository import ResumeRepository, get_resume_repository
from .job_repository import JobRepository, get_job_repository
from .match_repository import MatchRepository, get_match_repository
from .audit_repository import AuditRepository, get_audit_repository

__all__ = [
    # Base
    "BaseRepository",
    # Candidate
    "CandidateRepository",
    "get_candidate_repository",
    # Resume
    "ResumeRepository",
    "get_resume_repository",
    # Job
    "JobRepository",
    "get_job_repository",
    # Match
    "MatchRepository",
    "get_match_repository",
    # Audit
    "AuditRepository",
    "get_audit_repository",
]
