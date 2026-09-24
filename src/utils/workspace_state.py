from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal


class WorkspaceState(QObject):
    """Application-wide active-workspace holder.

    Not a __new__-singleton — QObject's metaclass makes that unsafe.
    Always access via get_workspace_state().
    """

    workspace_changed = pyqtSignal(object)  # emits Workspace | None

    def __init__(self) -> None:
        super().__init__()
        self._workspace: Optional[object] = None

    @property
    def workspace(self) -> Optional[object]:
        return self._workspace

    @property
    def workspace_id_str(self) -> Optional[str]:
        if self._workspace is None:
            return None
        return str(self._workspace.id)  # type: ignore[attr-defined]

    def set_workspace(self, workspace: object) -> None:
        self._workspace = workspace
        self.workspace_changed.emit(workspace)

    def clear(self) -> None:
        self._workspace = None
        self.workspace_changed.emit(None)


_state: Optional[WorkspaceState] = None


def get_workspace_state() -> WorkspaceState:
    global _state
    if _state is None:
        _state = WorkspaceState()
    return _state
