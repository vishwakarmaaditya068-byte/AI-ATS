"""
Reusable UI widgets for AI-ATS application.

This module provides styled, reusable widget components
that maintain consistent appearance across the application.
"""

from .cards import (
    Card,
    StatCard,
    InfoCard,
    ScoreCard,
    CandidateCard,
)

from .buttons import (
    PrimaryButton,
    SecondaryButton,
    DangerButton,
    SuccessButton,
    IconButton,
    TextButton,
    ButtonGroup,
)

from .tables import (
    StyledTable,
    DataTable,
)

__all__ = [
    # Cards
    "Card",
    "StatCard",
    "InfoCard",
    "ScoreCard",
    "CandidateCard",
    # Buttons
    "PrimaryButton",
    "SecondaryButton",
    "DangerButton",
    "SuccessButton",
    "IconButton",
    "TextButton",
    "ButtonGroup",
    # Tables
    "StyledTable",
    "DataTable",
]
