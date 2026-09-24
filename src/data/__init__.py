"""
Data layer for AI-ATS.

Provides database connections, data models, and repository classes
for data access throughout the application.

Submodules:
- database: MongoDB connection management
- models: Pydantic data models/schemas
- repositories: Database operations and queries
- migrations: Database migration scripts
"""

from .database import (
    DatabaseManager,
    get_async_db,
    get_database_manager,
    get_sync_db,
)

__all__ = [
    "DatabaseManager",
    "get_async_db",
    "get_database_manager",
    "get_sync_db",
]
