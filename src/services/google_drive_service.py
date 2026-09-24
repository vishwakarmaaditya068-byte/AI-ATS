"""
Google Drive integration service for AI-ATS.

Provides functionality to fetch resume files from Google Drive,
particularly useful for universities using Google Forms for placement registration.

Setup Instructions:
1. Go to Google Cloud Console (https://console.cloud.google.com)
2. Create a new project or select existing
3. Enable Google Drive API
4. Create OAuth 2.0 credentials (Desktop application)
5. Download credentials.json and place in project root
6. Run this service - it will prompt for authorization on first run
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from src.services.ingestion_service import CandidateRepoProtocol

from src.utils.logger import get_logger
from src.utils.config import DATA_DIR

logger = get_logger(__name__)

# Token and credentials paths
CREDENTIALS_FILE = Path("credentials.json")
TOKEN_FILE = DATA_DIR / "google_token.json"


@dataclass
class DriveFile:
    """Represents a file from Google Drive."""
    id: str
    name: str
    mime_type: str
    size: int = 0
    created_time: Optional[str] = None
    modified_time: Optional[str] = None
    web_link: Optional[str] = None


@dataclass
class ImportResult:
    """Result of importing files from Google Drive."""
    total_found: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    files: list[DriveFile] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class GoogleDriveService:
    """
    Service for interacting with Google Drive to fetch resume files.

    Typical workflow:
    1. University collects resumes via Google Forms
    2. PDFs are stored in a Google Drive folder
    3. This service fetches those PDFs for processing

    Usage:
        service = GoogleDriveService()
        if service.authenticate():
            result = service.download_resumes_from_folder(folder_id, output_dir)
    """

    # Supported resume file types
    RESUME_MIME_TYPES = [
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "application/msword",  # .doc
        "text/plain",
    ]

    def __init__(self) -> None:
        """Initialize the Google Drive service."""
        self._service = None
        self._authenticated = False

    def is_available(self) -> bool:
        """Check if Google Drive integration is available (dependencies installed)."""
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            return True
        except ImportError:
            return False

    def has_credentials(self) -> bool:
        """Check if credentials file exists."""
        return CREDENTIALS_FILE.exists()

    def is_authenticated(self) -> bool:
        """Check if already authenticated."""
        return self._authenticated and self._service is not None

    def authenticate(self) -> bool:
        """
        Authenticate with Google Drive API.

        On first run, this will open a browser for OAuth consent.
        Subsequent runs will use the saved token.

        Returns:
            True if authentication successful, False otherwise.
        """
        if not self.is_available():
            logger.error(
                "Google API libraries not installed. "
                "Run: pip install google-auth-oauthlib google-api-python-client"
            )
            return False

        if not self.has_credentials():
            logger.error(
                f"Credentials file not found: {CREDENTIALS_FILE}\n"
                "Please download OAuth credentials from Google Cloud Console."
            )
            return False

        try:
            from google.oauth2.credentials import Credentials
            from google.auth.transport.requests import Request
            from google.auth.exceptions import RefreshError, TransportError
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build

            SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

            creds = None

            # Load existing token if available
            if TOKEN_FILE.exists():
                creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

            # Refresh or get new credentials
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                    except RefreshError:
                        # Refresh token has expired or been revoked (e.g. user
                        # removed app access in Google account settings).  The
                        # stale token file must be deleted so the next branch
                        # runs the full OAuth flow instead of looping forever.
                        logger.warning(
                            "Google OAuth refresh token is invalid or has been "
                            "revoked. Deleting stale token and re-authenticating "
                            "via browser."
                        )
                        TOKEN_FILE.unlink(missing_ok=True)
                        creds = None
                    except TransportError as e:
                        logger.error(
                            f"Network error while refreshing Google token: {e}. "
                            "Check your internet connection and try again."
                        )
                        return False

                if not creds or not creds.valid:
                    # Either never authenticated, or refresh failed above.
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(CREDENTIALS_FILE), SCOPES
                    )
                    creds = flow.run_local_server(port=0)

                # Save token for future use
                TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
                with open(TOKEN_FILE, "w") as token:
                    token.write(creds.to_json())

            # Build the Drive service
            self._service = build("drive", "v3", credentials=creds)
            self._authenticated = True
            logger.info("Google Drive authentication successful")
            return True

        except Exception as e:
            logger.exception(f"Google Drive authentication failed: {e}")
            return False

    def list_folders(self, parent_id: str = "root") -> list[DriveFile]:
        """
        List folders in Google Drive.

        Args:
            parent_id: Parent folder ID (default: root)

        Returns:
            List of folder DriveFile objects.
        """
        if not self.is_authenticated():
            logger.error("Not authenticated. Call authenticate() first.")
            return []

        try:
            query = f"'{parent_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"

            results = self._service.files().list(
                q=query,
                pageSize=100,
                fields="files(id, name, createdTime, modifiedTime, webViewLink)",
            ).execute()

            folders = []
            for item in results.get("files", []):
                folders.append(DriveFile(
                    id=item["id"],
                    name=item["name"],
                    mime_type="application/vnd.google-apps.folder",
                    created_time=item.get("createdTime"),
                    modified_time=item.get("modifiedTime"),
                    web_link=item.get("webViewLink"),
                ))

            return folders

        except Exception as e:
            logger.exception(f"Failed to list folders: {e}")
            return []

    def list_resume_files(self, folder_id: str) -> list[DriveFile]:
        """
        List resume files (PDF, DOCX, DOC) in a folder.

        Args:
            folder_id: Google Drive folder ID

        Returns:
            List of DriveFile objects representing resume files.
        """
        if not self.is_authenticated():
            logger.error("Not authenticated. Call authenticate() first.")
            return []

        try:
            # Build MIME type query
            mime_queries = " or ".join(
                f"mimeType='{mime}'" for mime in self.RESUME_MIME_TYPES
            )
            query = f"'{folder_id}' in parents and ({mime_queries}) and trashed=false"

            files = []
            page_token = None

            while True:
                results = self._service.files().list(
                    q=query,
                    pageSize=100,
                    pageToken=page_token,
                    fields="nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, webViewLink)",
                ).execute()

                for item in results.get("files", []):
                    files.append(DriveFile(
                        id=item["id"],
                        name=item["name"],
                        mime_type=item["mimeType"],
                        size=int(item.get("size", 0)),
                        created_time=item.get("createdTime"),
                        modified_time=item.get("modifiedTime"),
                        web_link=item.get("webViewLink"),
                    ))

                page_token = results.get("nextPageToken")
                if not page_token:
                    break

            logger.info(f"Found {len(files)} resume files in folder")
            return files

        except Exception as e:
            logger.exception(f"Failed to list files: {e}")
            return []

    def download_file(self, file_id: str, output_path: Path) -> bool:
        """
        Download a file from Google Drive.

        Args:
            file_id: Google Drive file ID
            output_path: Local path to save the file

        Returns:
            True if download successful, False otherwise.
        """
        if not self.is_authenticated():
            logger.error("Not authenticated. Call authenticate() first.")
            return False

        try:
            from googleapiclient.http import MediaIoBaseDownload

            request = self._service.files().get_media(fileId=file_id)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "wb") as f:
                downloader = MediaIoBaseDownload(f, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()

            logger.debug(f"Downloaded: {output_path.name}")
            return True

        except Exception as e:
            logger.exception(f"Failed to download file {file_id}: {e}")
            return False

    def download_file_to_bytes(self, file_id: str) -> Optional[bytes]:
        """
        Download a file from Google Drive to memory.

        Args:
            file_id: Google Drive file ID

        Returns:
            File contents as bytes, or None if failed.
        """
        if not self.is_authenticated():
            logger.error("Not authenticated. Call authenticate() first.")
            return None

        try:
            from googleapiclient.http import MediaIoBaseDownload

            request = self._service.files().get_media(fileId=file_id)
            buffer = io.BytesIO()

            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()

            return buffer.getvalue()

        except Exception as e:
            logger.exception(f"Failed to download file {file_id}: {e}")
            return None

    def download_resumes_from_folder(
        self,
        folder_id: str,
        output_dir: Path,
        skip_existing: bool = True,
    ) -> ImportResult:
        """
        Download all resume files from a Google Drive folder.

        Args:
            folder_id: Google Drive folder ID
            output_dir: Local directory to save files
            skip_existing: Skip files that already exist locally

        Returns:
            ImportResult with download statistics.
        """
        result = ImportResult()

        # List files
        files = self.list_resume_files(folder_id)
        result.total_found = len(files)
        result.files = files

        if not files:
            logger.warning("No resume files found in folder")
            return result

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Download each file
        for file in files:
            output_path = output_dir / file.name

            # Handle duplicate filenames
            if output_path.exists():
                if skip_existing:
                    result.skipped += 1
                    continue
                else:
                    # Add suffix to avoid overwriting
                    base = output_path.stem
                    ext = output_path.suffix
                    counter = 1
                    while output_path.exists():
                        output_path = output_dir / f"{base}_{counter}{ext}"
                        counter += 1

            if self.download_file(file.id, output_path):
                result.downloaded += 1
            else:
                result.failed += 1
                result.errors.append(f"Failed to download: {file.name}")

        logger.info(
            f"Download complete: {result.downloaded} downloaded, "
            f"{result.skipped} skipped, {result.failed} failed"
        )
        return result

    def export_sheet_as_csv(self, file_id: str) -> Optional[str]:
        """
        Export a Google Sheet as CSV text.

        Uses the Drive API files.export endpoint, so no Sheets API or
        additional OAuth scope is needed beyond the existing drive.readonly scope.

        Args:
            file_id: The Google Drive file ID of the Sheet (same as the
                     SHEET_ID portion of the spreadsheet URL).

        Returns:
            CSV content as a UTF-8 string, or None if the export failed.
        """
        if not self.is_authenticated():
            logger.error("Not authenticated. Call authenticate() first.")
            return None

        try:
            from googleapiclient.http import MediaIoBaseDownload

            request = self._service.files().export_media(
                fileId=file_id,
                mimeType="text/csv",
            )
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

            return buffer.getvalue().decode("utf-8")

        except Exception as e:
            logger.exception(f"Failed to export sheet {file_id} as CSV: {e}")
            return None

    def get_forms_response_folder(self, form_name: str) -> Optional[str]:
        """
        Find the folder containing responses for a Google Form.

        Google Forms stores uploaded files in a folder named:
        "{Form Name} (File responses)"

        Args:
            form_name: Name of the Google Form

        Returns:
            Folder ID if found, None otherwise.
        """
        if not self.is_authenticated():
            return None

        try:
            # Search for the form responses folder
            folder_name = f"{form_name} (File responses)"
            query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"

            results = self._service.files().list(
                q=query,
                pageSize=10,
                fields="files(id, name)",
            ).execute()

            files = results.get("files", [])
            if files:
                logger.info(f"Found form responses folder: {files[0]['name']}")
                return files[0]["id"]

            logger.warning(f"Form responses folder not found: {folder_name}")
            return None

        except Exception as e:
            logger.exception(f"Failed to find form folder: {e}")
            return None


    def ingest_folder(
        self,
        folder_id: str,
        repo: "Optional[CandidateRepoProtocol]" = None,
    ) -> ImportResult:
        """
        Download every resume in a Drive folder and ingest each one via IngestionService.

        Files are passed as raw bytes — nothing is written to disk.
        Returns an ImportResult with per-file outcome counts.

        Args:
            folder_id: Google Drive folder ID to scan for resume files.
            repo: Optional repository override (used in tests to inject a fake).
        """
        from src.services.ingestion_service import IngestionService, IngestionResult

        svc: IngestionService = IngestionService(repo=repo)
        result: ImportResult = ImportResult()

        files: list[DriveFile] = self.list_resume_files(folder_id)
        result.total_found = len(files)
        result.files = files

        for f in files:
            content: Optional[bytes] = self.download_file_to_bytes(f.id)
            if content is None:
                result.failed += 1
                result.errors.append(f"Download failed: {f.name}")
                continue

            ingestion: IngestionResult = svc.ingest_bytes(content, f.name)

            if ingestion.status == "success":
                result.downloaded += 1
            elif ingestion.status == "duplicate":
                result.skipped += 1
            else:
                result.failed += 1
                result.errors.append(f"{f.name}: {ingestion.error_message}")

        return result


# Singleton instance
_drive_service: Optional[GoogleDriveService] = None


def get_drive_service() -> GoogleDriveService:
    """Get the Google Drive service singleton instance."""
    global _drive_service
    if _drive_service is None:
        _drive_service = GoogleDriveService()
    return _drive_service
