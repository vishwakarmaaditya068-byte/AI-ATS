"""
AI-ATS Main Entry Point

This module serves as the main entry point for the AI-ATS application.
It initializes all components and launches the GUI.
"""

import sys
from pathlib import Path

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    """
    Main entry point for the AI-ATS application.

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    try:
        # Initialize logging first
        from src.utils.logger import setup_logging, log

        setup_logging()
        log.info("Starting AI-ATS application...")

        # Load configuration
        from src.utils.config import get_settings

        settings = get_settings()
        log.info(f"Environment: {settings.environment}")
        log.info(f"Debug mode: {settings.debug}")

        # Initialize database connection
        log.info("Initializing database connection...")
        from src.data.database import get_database_manager

        db_manager = get_database_manager()
        if db_manager.check_sync_connection():
            log.info("Database connection established")
        else:
            log.warning(
                "Could not connect to MongoDB. "
                "Some features may not work. Run 'ai-ats init-db' to initialize."
            )

        # Initialize ML models (lazy loading - just import to register singletons)
        log.info("Preparing ML components...")
        try:
            from src.ml.nlp import get_resume_parser
            from src.core.matching import get_matching_engine

            # These will lazy-load when first used, but importing ensures the modules are ready
            log.info("ML components ready (will load on first use)")
        except ImportError as e:
            log.warning(f"Some ML components may not be available: {e}")

        # Launch GUI
        log.info("Launching user interface...")
        from src.ui.main_window import run_application

        return run_application()

    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
        return 130
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
