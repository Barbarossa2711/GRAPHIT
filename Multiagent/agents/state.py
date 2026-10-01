from __future__ import annotations

from dataclasses import dataclass

from langgraph.graph import MessagesState
from typing_extensions import NotRequired


class TutorState(MessagesState):
    """
    Graph state shared between supervisor and worker agents: message history plus an optional concept focus.
    """

    current_concept: NotRequired[str | None]


@dataclass
class RuntimeContext:
    """
    Per-request context that identifies the student (the Jupyter user) and cannot be changed by the model.

    ``session_key`` identifies the chat session and stays constant across its turns, so the
    visit counter counts sessions, not messages; ``None`` counts every call. ``slide_log``
    and ``concept_log`` are caller-provided lists that ``get_concept_material`` fills with
    the citations of the loaded slides and the concept id; the server reads them after the
    run, the model never sees them. ``None`` disables the logging.
    """

    student_id: str
    session_key: str | None = None
    slide_log: list[dict] | None = None
    concept_log: list[str] | None = None
