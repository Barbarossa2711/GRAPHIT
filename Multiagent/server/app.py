"""
FastAPI server of GRAPHIT: OpenAI-compatible chat, domain tree, progress, recommendations and quiz.

Start with ``uvicorn Multiagent.server.app:app --port 8077`` and never with more than one
worker: running quizzes are cached in process memory (``_quiz_cache``).
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from Multiagent.agents import build_app
from Multiagent.GraphAccess import DomainTree
from Multiagent.GraphAccess.concept_source import ConceptSource
from Multiagent.LearnerModel import LearnerGraph, ProgressService
from Multiagent.Quiz import QuizService
from Multiagent.Quiz.quiz_service import ITEMS_PER_QUIZ
from Multiagent.Recommender import Recommender
from Multiagent.Recommender.recommender import NEXT_STEPS_LIMIT

from .openai_adapter import (
    completion_response,
    real_streaming,
    run_supervisor,
    stream_completion,
    stream_supervisor,
)

logger = logging.getLogger(__name__)

MODEL_ID = "graphit-tutor"

ANONYMOUS_STUDENT = "anonymous"

_DEFAULT_ORIGINS = "http://localhost:8888,http://127.0.0.1:8888"
CORS_ORIGINS = [
    o.strip() for o in os.getenv("GRAPHIT_CORS_ORIGINS", _DEFAULT_ORIGINS).split(",") if o.strip()
]

_state: dict[str, Any] = {
    "supervisor": None,
    "quiz": None,
    "tree": None,
    "progress": None,
    "recommender": None,
}
_quiz_cache: dict[str, dict] = {}

_CLOSABLE = ("quiz", "tree", "progress", "recommender")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Builds the process-wide services and closes them on shutdown.

    Also ensures the uniqueness constraint on :Student(id) and warns about a legacy
    "anonymous" student node, which belongs to no JupyterHub account and mixes the learning
    state of several people.

    :param app: FastAPI instance (supplied by the runtime).
    :return: Async context wrapping the server's lifetime.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info("Building supervisor app, QuizService, DomainTree, ProgressService, Recommender …")
    _state["supervisor"] = build_app()
    _state["quiz"] = QuizService()
    _state["tree"] = DomainTree()
    _state["progress"] = ProgressService()
    _state["recommender"] = Recommender()
    _state["learner"] = LearnerGraph()
    _state["concepts"] = ConceptSource()
    with LearnerGraph() as _lg:
        _lg.ensure_constraints()
        if _lg.student_exists(ANONYMOUS_STUDENT):
            logger.warning(
                "A :Student node %r exists. It dates from the time when requests without an "
                "id were allowed and belongs to no JupyterHub account; its learning states "
                "mix several people and cannot be separated. Check and remove it.", ANONYMOUS_STUDENT,
            )
    try:
        yield
    finally:
        for key in _CLOSABLE:
            if _state[key] is not None:
                _state[key].close()


app = FastAPI(title="GRAPHIT", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Student-Id"],
)


def _resolve_student(header_id: str | None, body_user: str | None) -> str:
    """
    Determines the student id from header or body, normalised with ``casefold()``.

    One :Student node per JupyterHub account keeps a learning state together, but the
    uniqueness constraint compares exactly; university logins are case-insensitive, so
    ``ABCD001`` and ``abcd001`` must become the same id. Requests without an id are
    rejected instead of mapped to a shared node, which would mix learning states and return
    a plausible but wrong answer. Only user-specific endpoints call this.

    :param header_id: Value of ``X-Student-Id``.
    :param body_user: ``user`` field of the request, used as a fallback.
    :return: The normalised identifier.
    :raises HTTPException: 400 if neither header nor ``user`` field is set.
    """
    raw = (header_id or body_user or "").strip()
    if not raw:
        logger.warning(
            "Request without student id on a user-specific endpoint "
            "(neither X-Student-Id nor 'user'), rejected. The extension sets the header "
            "from the seeded studentId; if it is missing in production, seeding is broken "
            "or someone bypasses the frontend."
        )
        raise HTTPException(
            status_code=400,
            detail="Studenten-Kennung fehlt: X-Student-Id (oder das Feld 'user') ist "
                   "erforderlich. Ohne sie laesst sich kein Lernstand zuordnen.",
        )
    return raw.casefold()


SSE_HEADER = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


class ChatScope(BaseModel):
    """
    Topic selected in the frontend's tree to which the chat is bound; ``type`` is lecture, chapter, topic, subtopic or concept.
    """

    id: str
    name: str | None = None
    type: str | None = None


class ChatCompletionRequest(BaseModel):
    """
    OpenAI chat request extended by ``scope`` and ``session_id``; further OpenAI fields are tolerated.

    ``session_id`` identifies the chat session for the session-based visit counter; without
    it the key is derived from the history (see ``openai_adapter.compute_session_key``).
    """

    model: str = MODEL_ID
    messages: list[dict]
    stream: bool = False
    user: str | None = None
    scope: ChatScope | None = None
    session_id: str | None = None

    model_config = {"extra": "allow"}


@app.get("/v1/models")
def list_models() -> dict:
    """
    Returns the static model list for generic OpenAI clients.

    :return: OpenAI-style list with the single GRAPHIT model.
    """
    return {
        "object": "list",
        "data": [{"id": MODEL_ID, "object": "model", "owned_by": "graphit"}],
    }


@app.post("/v1/chat/completions")
async def chat_completions(
    req: ChatCompletionRequest,
    x_student_id: str | None = Header(default=None, alias="X-Student-Id"),
):
    """
    Chats through the supervisor, OpenAI-compatible, optionally streamed.

    The endpoint is stateless: the client sends the full history. Only metadata is logged,
    never message content. Streaming sets headers that stop proxies from buffering.

    :param req: Request carrying history, optional ``scope`` and ``session_id``.
    :param x_student_id: Student taken from the header.
    :return: A ``chat.completion`` object, or an SSE stream when ``stream`` is set.
    """
    student_id = _resolve_student(x_student_id, req.user)
    logger.info(
        "Chat: student=%s scope=%s session=%s stream=%s messages=%d roles=[%s]",
        student_id,
        req.scope.id if req.scope else None,
        req.session_id or "(derived)",
        req.stream,
        len(req.messages),
        ",".join(str(m.get("role", "?")) for m in req.messages),
    )
    scope = req.scope.model_dump() if req.scope else None

    if req.stream and real_streaming():
        return StreamingResponse(
            stream_supervisor(
                _state["supervisor"],
                req.model,
                student_id,
                req.messages,
                scope,
                req.session_id,
                _state["concepts"],
                _state["learner"],
            ),
            media_type="text/event-stream",
            headers=SSE_HEADER,
        )

    answer, slides = await run_in_threadpool(
        run_supervisor,
        _state["supervisor"],
        student_id,
        req.messages,
        scope,
        req.session_id,
        _state["concepts"],
        _state["learner"],
    )

    if req.stream:
        return StreamingResponse(
            stream_completion(req.model, answer, slides),
            media_type="text/event-stream",
            headers=SSE_HEADER,
        )
    return completion_response(req.model, answer, slides)


@app.get("/domain/tree")
async def domain_tree() -> dict:
    """
    Returns the lecture structure as a nested tree for the selection dialog; progress comes separately from ``/progress``.

    :return: ``{"tree": [...]}``.
    """
    roots = await run_in_threadpool(_state["tree"].fetch_tree)
    return {"tree": roots}


@app.get("/progress")
async def progress(
    student_id: str | None = Query(default=None),
    x_student_id: str | None = Header(default=None, alias="X-Student-Id"),
) -> dict:
    """
    Returns the learning state of all concepts plus roll-ups per hierarchy node.

    :param student_id: Student as a query parameter (fallback).
    :param x_student_id: Student from the header (preferred).
    :return: Learning state of all concepts plus roll-ups per hierarchy node.
    """
    sid = _resolve_student(x_student_id, student_id)
    return await run_in_threadpool(_state["progress"].fetch_progress, sid)


@app.get("/activity")
async def activity(
    student_id: str | None = Query(default=None),
    x_student_id: str | None = Header(default=None, alias="X-Student-Id"),
) -> dict:
    """
    Returns the student's activity days for the heatmap: answered questions and chat sessions, only days with activity.

    :param student_id: Student as a query parameter (fallback).
    :param x_student_id: Student from the header (preferred).
    :return: ``{"student_id": str, "days": [{"date": "YYYY-MM-DD", "count": int}, …]}``.
    """
    sid = _resolve_student(x_student_id, student_id)
    days = await run_in_threadpool(_state["learner"].activity_history, sid)
    return {"student_id": sid, "days": days}


@app.get("/recommend")
async def recommend(
    concept_id: str = Query(..., description="Target concept the learning path is computed for"),
    student_id: str | None = Query(default=None),
    x_student_id: str | None = Header(default=None, alias="X-Student-Id"),
) -> dict:
    """
    Returns the learning path to a target concept, the same deterministic logic the recommender agent uses.

    :param concept_id: Target concept of the learning path.
    :param student_id: Student as a query parameter (fallback).
    :param x_student_id: Student from the header (preferred).
    :return: Learning path with learnable, blocked and due concepts.
    :raises HTTPException: 404 if the concept does not exist.
    """
    sid = _resolve_student(x_student_id, student_id)
    try:
        return await run_in_threadpool(_state["recommender"].recommend, sid, concept_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/next")
async def next_steps(
    limit: int = Query(default=NEXT_STEPS_LIMIT, ge=1, le=50),
    student_id: str | None = Query(default=None),
    x_student_id: str | None = Header(default=None, alias="X-Student-Id"),
) -> dict:
    """
    Returns what to work on next across the corpus, deterministically and without LLM (see :meth:`Recommender.next_steps`).

    :param limit: How many concepts to propose (1–50).
    :param student_id: Student as a query parameter (fallback).
    :param x_student_id: Student from the header (preferred).
    :return: The next concepts plus the counters around them.
    """
    sid = _resolve_student(x_student_id, student_id)
    return await run_in_threadpool(_state["recommender"].next_steps, sid, limit)


class QuizCandidatesRequest(BaseModel):
    """
    Quiz trigger: an explicit concept, a topic, or neither for the fallback via the learning state.
    """

    concept_id: str | None = None
    topic_id: str | None = None
    student_id: str | None = None
    only_unmastered: bool = True


class QuizStartRequest(BaseModel):
    """
    Start of a quiz run; ``items`` is the number of questions, ``ITEMS_PER_QUIZ`` when omitted.
    """

    concept_id: str
    student_id: str | None = None
    items: int | None = None


class QuizSubmitRequest(BaseModel):
    """
    Answers of a quiz run as ``question_id -> answer``.
    """

    quiz_id: str
    answers: dict[str, Any]


@app.post("/quiz/candidates")
async def quiz_candidates(
    req: QuizCandidatesRequest,
    x_student_id: str | None = Header(default=None, alias="X-Student-Id"),
):
    """
    Resolves a quiz request into testable concepts, each with its :CO_OCCURS neighbours for the hint before the quiz.

    :param req: Request carrying ``concept_id`` or ``topic_id``.
    :param x_student_id: Student taken from the header.
    :return: ``{"candidates": [...]}`` per concept, including coupled neighbours.
    """
    student_id = _resolve_student(x_student_id, req.student_id)
    candidates = await run_in_threadpool(
        _state["quiz"].resolve_concepts,
        concept_id=req.concept_id,
        topic_id=req.topic_id,
        student_id=student_id,
        only_unmastered=req.only_unmastered,
    )

    for candidate in candidates:
        candidate["co_occurs"] = await run_in_threadpool(
            _state["quiz"].coupled_concepts, candidate["id"], student_id
        )
    return {"candidates": candidates}


@app.post("/quiz/start")
async def quiz_start(
    req: QuizStartRequest,
    x_student_id: str | None = Header(default=None, alias="X-Student-Id"),
):
    """
    Starts a quiz run and returns the questions without solutions.

    Gating comes before selection, otherwise the filter would shrink the quiz. Only the
    selection is cached, since :meth:`QuizService.submit` counts missing answers as wrong.
    Asked questions are recorded before answering so the rotation moves on even for an
    aborted quiz. The presentation is shuffled before caching, so grading and reloads see
    the same order.

    :param req: Request carrying ``concept_id`` and an optional ``items`` count.
    :param x_student_id: Student taken from the header.
    :return: The ``quiz_id`` and the presented questions.
    :raises HTTPException: 404 if no quiz can be built for the concept.
    """
    student_id = _resolve_student(x_student_id, req.student_id)
    pool = await run_in_threadpool(_state["quiz"].generate, req.concept_id)
    if not pool:
        raise HTTPException(status_code=404, detail=f"Kein Quiz für Concept {req.concept_id!r} erzeugbar.")

    allowed = await run_in_threadpool(_state["quiz"].allowed_items, pool, student_id)
    asked = await run_in_threadpool(
        _state["quiz"].asked_history, student_id, req.concept_id
    )
    quiz = _state["quiz"].select_items(
        allowed, req.items if req.items is not None else ITEMS_PER_QUIZ, asked=asked
    )
    await run_in_threadpool(
        _state["quiz"].record_asked, student_id, req.concept_id,
        [q["question_id"] for q in quiz],
    )
    logger.info(
        "Quiz %s: %d of %d items served (%d allowed, %d distinct stems, "
        "%d questions in the history).",
        req.concept_id, len(quiz), len(pool), len(allowed),
        len({q["stem_id"] for q in quiz}), len(asked),
    )

    quiz = QuizService.shuffle_presentation(quiz)

    quiz_id = uuid.uuid4().hex
    _quiz_cache[quiz_id] = {"student_id": student_id, "concept_id": req.concept_id, "quiz": quiz}
    return {
        "quiz_id": quiz_id,
        "concept_id": req.concept_id,
        "questions": QuizService.strip_solutions(quiz),
    }


@app.post("/quiz/submit")
async def quiz_submit(req: QuizSubmitRequest):
    """
    Grades a run and updates the mastery state.

    :param req: Request carrying ``quiz_id`` and the answers.
    :return: Per-question result and the new learning state.
    :raises HTTPException: 404 if the ``quiz_id`` is unknown.
    """
    entry = _quiz_cache.pop(req.quiz_id, None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Unbekannte oder abgelaufene quiz_id.")

    result = await run_in_threadpool(
        _state["quiz"].submit,
        entry["student_id"],
        entry["concept_id"],
        entry["quiz"],
        req.answers,
    )
    return result
