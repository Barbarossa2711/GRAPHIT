from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import json
import time
import uuid

from langchain_core.messages import AIMessageChunk

logger = logging.getLogger(__name__)

SUPERVISOR_NAME = "supervisor"
LINK_PATTERN = re.compile(r"\[\[([A-Za-z0-9_]{1,64})\|([^\]|]{1,120})\]\]")
BRACKET_PATTERN = re.compile(r"\[\[[^\]]{0,200}\]\]")
LINE_THRESHOLD = 25
LEAD_CHARS = 120
FALLBACK_CHARS = 120
HANDOFF_PREFIX = "Transferring back to"
ERROR_TEXT = "Die Antwort konnte nicht erzeugt werden. Bitte versuche es erneut."
LINE_SPLITTER = re.compile(r"(\n)")


def _content_text(message) -> str:
    """
    Returns the text content of a message, whether LangChain object or dict.

    :param message: LangChain message or dict.
    :return: Plain text; structured blocks are concatenated.
    """
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    if isinstance(content, list):
        parts = [b.get("text", "") if isinstance(b, dict) else str(b) for b in content]
        return "".join(parts)
    return content or ""


def _field(message, name: str):
    """
    Reads a field, no matter whether the message is a LangChain object or a dict.

    :param message: Message to read from.
    :param name: Field name, e.g. ``name`` or ``tool_calls``.
    :return: Field value, or ``None`` if absent.
    """
    value = getattr(message, name, None)
    if value is None and isinstance(message, dict):
        value = message.get(name)
    return value


def _is_ai(message) -> bool:
    """
    Checks whether a message is a reply produced by the model.

    :param message: Message to inspect.
    :return: ``True`` for assistant/AI messages.
    """
    if isinstance(message, dict):
        return message.get("role") in ("assistant", "ai")
    return getattr(message, "type", None) == "ai"


def _is_human(message) -> bool:
    """
    Checks whether a message is input typed by the student.

    :param message: Message to inspect.
    :return: ``True`` for user/human messages.
    """
    if isinstance(message, dict):
        return message.get("role") in ("user", "human")
    return getattr(message, "type", None) == "human"


def _worker_messages(messages: list):
    """
    Yields the agents' messages of the current turn as ``(name, text, calls_tool)``.

    Only the part after the last user question counts, earlier turns would otherwise be
    delivered again. Supervisor messages, empty messages and handoff notices are skipped.

    :param messages: Messages of the supervisor run.
    :return: Generator over the agents' messages of the current turn, in order.
    """
    start = 0
    for i, message in enumerate(messages):
        if _is_human(message):
            start = i

    for message in messages[start:]:
        if not _is_ai(message):
            continue
        name = _field(message, "name")
        if not name or name == SUPERVISOR_NAME:
            continue
        text = _content_text(message).strip()
        if not text or text.startswith(HANDOFF_PREFIX):
            continue
        yield name, text, bool(_field(message, "tool_calls"))


def worker_answers(messages: list) -> list[str]:
    """
    Returns the substantive agent answers of the current turn, in order.

    Messages that call a tool carry at most an announcement and are dropped (see
    :func:`discarded_answers`). Only the first answer per agent counts: the supervisor
    sometimes delegates the same question twice, and the student must not get the same
    explanation twice; the streaming path stops after the first answer anyway.

    :param messages: Messages of the supervisor run.
    :return: The agents' substantive answers of the current turn, in order.
    """
    answers: list[str] = []
    seen_agents: set[str] = set()
    for name, text, calls_tool in _worker_messages(messages):
        if calls_tool or name in seen_agents:
            continue
        seen_agents.add(name)
        answers.append(text)
    return answers


def discarded_answers(messages: list) -> list[str]:
    """
    Returns agent texts that were dropped only because the same message also called a tool.

    Covers the model writing the finished explanation and a tool call into one message.
    Texts shorter than ``FALLBACK_CHARS`` are announcements like "Einen Moment, ich
    schaue nach." and stay out.

    :param messages: Messages of the supervisor run.
    :return: The discarded texts long enough to be an answer, in order.
    """
    return [
        text
        for _name, text, calls_tool in _worker_messages(messages)
        if calls_tool and len(text) >= FALLBACK_CHARS
    ]


def has_delegated(messages: list) -> bool:
    """
    Checks whether the supervisor handed off to an agent in the current turn.

    The supervisor only has handoff tools, so any tool call of it is a delegation. Without
    delegation the supervisor's text is the answer (greeting, follow-up question); with it,
    anything the supervisor writes afterwards is commentary.

    :param messages: Messages of the supervisor run.
    :return: ``True`` if a handoff happened in this turn.
    """
    start = 0
    for i, message in enumerate(messages):
        if _is_human(message):
            start = i
    for message in messages[start:]:
        if not _is_ai(message):
            continue
        name = _field(message, "name")
        if name and name != SUPERVISOR_NAME:
            continue
        if _field(message, "tool_calls"):
            return True
    return False


def extract_final_text(result: dict) -> str:
    """
    Extracts the answer for the student from the supervisor result.

    Not simply ``messages[-1]``: after the worker returns, the supervisor answers again with
    a retelling that sometimes shrinks to "see above" and loses the explanation. Fallback
    order: the agents' answers, then an agent text dropped only because of an attached tool
    call, then the supervisor itself but only without delegation; after a delegation its
    final message is commentary, so the error text is returned instead.

    :param result: Result of ``app.invoke``.
    :return: The answer shown to the student.
    """
    messages = result.get("messages") or []
    if not messages:
        return ""
    answers = worker_answers(messages)
    if answers:
        return "\n\n".join(answers)
    discarded = discarded_answers(messages)
    if discarded:
        logger.info("Answer recovered from a discarded tool-call message")
        return discarded[-1]
    if has_delegated(messages):
        logger.warning("No agent text in the run: supervisor commentary suppressed")
        return ERROR_TEXT
    return _content_text(messages[-1])


def scope_system_message(scope: dict | None) -> dict | None:
    """
    Builds a system message that binds the chat to the topic selected in the frontend's tree.

    Without it the tutor would have to guess the concept from the question, which fails for
    follow-ups like "and how was that again?". It is prepended as system message so the
    student's history stays unchanged. For a concept the id is used directly instead of
    ``find_concept``.

    :param scope: Frontend selection ``{id, name, type}``, or ``None``.
    :return: System message binding the chat to that topic; ``None`` without a selection.
    """
    if not scope or not scope.get("id"):
        return None

    scope_type = (scope.get("type") or "concept").lower()
    label = {
        "lecture": "die Vorlesung",
        "chapter": "das Kapitel",
        "topic": "das Thema",
        "subtopic": "das Unterthema",
        "concept": "das Konzept",
    }.get(scope_type, "den Bereich")
    name = scope.get("name") or scope["id"]

    text = (
        f"Der Student hat im Auswahlbaum {label} '{name}' (ID: {scope['id']}) gewählt. "
        "Beziehe deine Antworten auf diesen Bereich der Vorlesung. "
    )
    if scope_type == "concept":
        text += (
            f"Rufe get_concept_material direkt mit der ID '{scope['id']}' auf, statt das "
            "Konzept über find_concept zu suchen. "
        )
    else:
        text += (
            "Bestimme das passende Konzept innerhalb dieses Bereichs mit find_concept. "
        )
    text += (
        "HARTE GRENZE: Beantworte ausschließlich Fragen zu diesem Bereich. Liegt die "
        f"Frage außerhalb von '{name}' — auch wenn sie zu einem anderen Teil dieser "
        "Vorlesung gehört —, erkläre den Inhalt NICHT. Gib auch keine Kurzfassung und "
        "keinen Anriss — ein 'trotzdem kurz erklärt' ist genau das, was hier nicht "
        "passieren soll.\n"
        "Verweise stattdessen an die richtige Stelle, und zwar belegt: Rufe find_concept "
        "mit dem erfragten Begriff auf und gib vom besten Treffer ZWEI Felder "
        "unverändert aus: erst das Feld 'link' — kopiere es Zeichen für Zeichen "
        "mitsamt der doppelten eckigen Klammern, es wird im Chat zu einem Klick auf "
        "genau dieses Konzept — und dann das Feld 'path' als Pfad im Auswahlbaum. "
        "Baue beide nicht selbst zusammen und kürze sie nicht.\n"
        "Rate den Ort niemals aus eigenem Wissen. Findet find_concept nichts, ist der "
        "Begriff nicht im Vorlesungsmodell — dann gilt die nächste Regel."
    )
    return {"role": "system", "content": text}


def compute_session_key(
    student_id: str,
    messages: list[dict],
    scope: dict | None = None,
    explicit: str | None = None,
) -> str | None:
    """
    Computes the key of the current chat session, the basis of the session-based visit counter.

    A ``session_id`` sent by the client wins, since only the client knows when a chat was
    reopened. Otherwise the key is derived from student, scope and the first user question
    of the history, which stays constant across the turns of a conversation.

    :param student_id: Student the key belongs to.
    :param messages: Chat history; its first user question serves as the anchor.
    :param scope: Selected topic, part of the key.
    :param explicit: ``session_id`` supplied by the client; takes precedence.
    :return: Session key, or ``None`` if the history holds no user question.
    """
    if explicit:
        return f"c:{explicit}"

    first_question = None
    for message in messages:
        if str(message.get("role", "")) in ("user", "human"):
            first_question = str(message.get("content") or "")
            break
    if first_question is None:
        return None

    raw = "|".join([student_id, str((scope or {}).get("id") or ""), first_question])
    return "h:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def last_question(messages: list[dict]) -> str:
    """
    Returns the text of the most recent user message.

    :param messages: Chat history as sent by the client.
    :return: Text of the last user message; empty if there is none.
    """
    for message in reversed(messages):
        if str(message.get("role", "")) in ("user", "human"):
            return str(message.get("content") or "")
    return ""


def append_pointer(answer: str, question: str, scope: dict | None, source) -> str:
    """
    Appends a pointer to the concept the question is about, if the answer lacks one.

    The scope prompt tells the tutor to output the ``link`` of ``find_concept`` for questions
    outside the selected topic, but it did so in only two of four measured runs, so the
    pointer is added here. Only if a topic is selected, the answer has no link yet, the
    question names a concept (``find_mentioned``) and that concept is not the selected one.

    :param answer: Answer produced by the supervisor run.
    :param question: The student's last question, verbatim.
    :param scope: Selected topic, or ``None``.
    :param source: A ``ConceptSource`` providing ``find_mentioned``.
    :return: The answer, possibly with one appended pointer line.
    """
    if not scope or not scope.get("id") or not question:
        return answer

    answer = clean_brackets(answer)
    pointer = pointer_line(answer, question, scope, source)
    return answer + pointer if pointer else answer


def clean_brackets(answer: str) -> str:
    """
    Removes broken ``[[...]]`` leftovers line by line.

    The model sometimes outputs field names instead of values ("[[link]] Pfad: [[path]]").
    A touched line is dropped entirely if its cleaned rest is shorter than
    ``LINE_THRESHOLD`` characters; longer lines carry content of their own. This rule is
    what makes the cleaning streamable (see :class:`ReplyFilter`).

    :param answer: Answer text as produced by the model.
    :return: The answer without broken bracket leftovers.
    """
    lines = []
    for line in answer.split("\n"):
        cleaned = strip_brackets(line)
        if cleaned != line and len(cleaned.strip()) < LINE_THRESHOLD:
            continue
        lines.append(cleaned)
    return "\n".join(lines).strip()


def strip_brackets(text: str) -> str:
    """
    Removes all ``[[...]]`` groups except valid links, whose pattern must match the frontend's ``CONCEPT_LINK``.

    :param text: One line, or a fragment of one.
    :return: The same text without broken bracket groups.
    """
    return BRACKET_PATTERN.sub(
        lambda m: m.group(0) if LINK_PATTERN.fullmatch(m.group(0)) else "", text
    )


def pointer_line(answer: str, question: str, scope: dict | None, source) -> str | None:
    """
    Computes the pointer line to append; split from :func:`append_pointer` because the streaming path needs it at the end.

    Errors in the lookup must not cost the answer.

    :param answer: Answer produced so far, already cleaned.
    :param question: The student's last question, verbatim.
    :param scope: Selected topic, or ``None``.
    :param source: A ``ConceptSource`` providing ``find_mentioned``, or ``None``.
    :return: The text to append, or ``None``.
    """
    if source is None or not scope or not scope.get("id") or not question:
        return None
    if LINK_PATTERN.search(answer):
        return None
    try:
        hits = source.find_mentioned(question)
    except Exception:
        return None
    for t in hits:
        if t["id"] == scope["id"]:
            continue
        path_suffix = " — Pfad: " + t["path"] if t.get("path") else ""
        return "\n" * 2 + "Du findest es hier: " + t["link"] + path_suffix
    return None


def sort_citations(citations: list[dict]) -> list[dict]:
    """
    Deduplicates the slide citations of a run by source and page and sorts them in lecture order.

    The same slides can be loaded several times in one run. Chapters without number go last.

    :param citations: Citations collected during one run, in call order.
    :return: The same citations, deduplicated and in lecture order.
    """
    by_slide: dict[tuple[str, int], dict] = {}
    for citation in citations:
        message_key = (str(citation.get("source") or ""), int(citation.get("page") or 0))
        by_slide.setdefault(message_key, citation)
    return sorted(
        by_slide.values(),
        key=lambda b: (
            b.get("chapter_num") if isinstance(b.get("chapter_num"), int) else 10**6,
            str(b.get("source") or ""),
            int(b.get("page") or 0),
        ),
    )


def record_visits(
    learner,
    student_id: str,
    session_key: str | None,
    scope: dict | None,
    concept_ids: list[str],
) -> None:
    """
    Records a visit for the concepts covered in the run.

    Done by the server instead of a ``mark_visited`` tool, which cost an extra model turn
    and was called unreliably. The selected topic counts if there is one, otherwise the
    concepts whose slides were loaded. Write errors must not cost the answer.

    :param learner: A ``LearnerGraph``, or ``None`` to skip recording.
    :param student_id: Student whose visit is recorded.
    :param session_key: Chat session; the same key does not count twice.
    :param scope: Selected topic, or ``None``.
    :param concept_ids: Concepts whose material was loaded during the run.
    """
    if learner is None:
        return
    targets = [scope["id"]] if scope and scope.get("id") else list(concept_ids)
    for concept_id in targets:
        try:
            learner.mark_visited(student_id, concept_id, session_key)
        except Exception:
            logger.warning("Visit of %s not recorded.", concept_id, exc_info=True)


def run_supervisor(
    app,
    student_id: str,
    messages: list[dict],
    scope: dict | None = None,
    session_id: str | None = None,
    concepts=None,
    learner=None,
) -> tuple[str, list[dict]]:
    """
    Runs the supervisor app on the chat history and returns the final answer with its slide citations.

    The citation list goes into the run empty and comes back filled by
    ``get_concept_material``. A visit is only recorded for a real answer.

    :param app: Compiled supervisor app.
    :param student_id: Student, passed through as runtime context.
    :param messages: Full chat history sent by the client.
    :param scope: Selected topic, prepended as a system message.
    :param session_id: Chat session identifier for the visit counter.
    :param concepts: Optional ``ConceptSource`` for the pointer fallback (s.
        :func:`append_pointer`); without it the answer is returned unchanged.
    :param learner: Optional ``LearnerGraph`` for the visit counter (s.
        :func:`record_visits`); without it no visit is recorded.
    :return: Answer text for the student and the citations of the slides it drew on.
    """
    session_key = compute_session_key(student_id, messages, scope, session_id)
    question = last_question(messages)

    system = scope_system_message(scope)
    if system is not None:
        messages = [system, *messages]

    citations: list[dict] = []
    visited: list[str] = []
    result = app.invoke(
        {"messages": messages},
        context={
            "student_id": student_id,
            "session_key": session_key,
            "slide_log": citations,
            "concept_log": visited,
        },
    )
    answer = extract_final_text(result)
    if answer and answer != ERROR_TEXT:
        record_visits(learner, student_id, session_key, scope, visited)
    if concepts is not None:
        answer = append_pointer(answer, question, scope, concepts)
    return answer, sort_citations(citations)


def completion_response(model: str, content: str, slides: list[dict] | None = None) -> dict:
    """
    Builds a non-streamed OpenAI ``chat.completion`` object.

    ``graphit_slides`` is an extension field, ignored by foreign clients. Citations stay
    next to the answer text and not in it, otherwise the stateless client would send them
    back to the model with the next turn.

    :param model: Model name echoed back.
    :param content: Finished answer text.
    :param slides: Citations of the slides the answer drew on, if any.
    :return: OpenAI-compatible ``chat.completion`` object.
    """
    payload = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }
    if slides:
        payload["graphit_slides"] = slides
    return payload


def stream_completion(model: str, content: str, slides: list[dict] | None = None):
    """
    Streams a pre-computed answer word by word as OpenAI SSE, the simulated streaming variant.

    :param model: Model name carried in every chunk.
    :param content: Pre-computed answer that gets split up.
    :param slides: Citations of the slides the answer drew on, sent with the closing chunk.
    :return: Generator over SSE lines including the terminator and ``[DONE]``.
    """
    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    yield sse_chunk(cid, created, model, {"role": "assistant"})
    for token in _tokenize(content):
        yield sse_chunk(cid, created, model, {"content": token})
    yield sse_chunk(cid, created, model, {}, finish_reason="stop", slides=slides)
    yield "data: [DONE]\n\n"


def sse_chunk(
    cid: str,
    created: int,
    model: str,
    delta: dict,
    finish_reason=None,
    slides: list[dict] | None = None,
) -> str:
    """
    Builds a single ``chat.completion.chunk`` SSE line; citations belong to the closing chunk, not to a delta.

    :param cid: Identifier shared by every chunk of one reply.
    :param created: Unix timestamp shared by every chunk of one reply.
    :param model: Model name carried in every chunk.
    :param delta: The delta payload, e.g. ``{"content": "..."}``.
    :param finish_reason: Reason string on the closing chunk, ``None`` otherwise.
    :param slides: Citations to attach as ``graphit_slides``, if any.
    :return: One ``data:`` line ready to be sent.
    """
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    if slides:
        payload["graphit_slides"] = slides
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _tokenize(text: str) -> list[str]:
    """
    Splits text into words including their trailing space, for streaming.

    :param text: Answer text to split.
    :return: Word-sized pieces with their trailing spaces preserved.
    """
    out: list[str] = []
    current = ""
    for ch in text:
        current += ch
        if ch == " ":
            out.append(current)
            current = ""
    if current:
        out.append(current)
    return out


class ReplyFilter:
    """
    Filters the tokens of a run down to exactly the text the student may see, while it is still being written.

    Streaming counterpart of :func:`worker_answers` and :func:`clean_brackets`, which
    works because both rules only drop short text:

    - A line is released only once its cleaned rest exceeds ``LINE_THRESHOLD`` characters,
      so nothing released can be dropped later.
    - An open ``[[`` is held back until it closes and is known to be a valid link or junk.
    - A message that ends in a tool call is no content, but that is only known at the end
      of the model turn. The first model turn of an agent is therefore buffered completely;
      later turns are held back for ``LEAD_CHARS`` characters. Announcements that
      leak anyway are counted in ``leaked_preambles``. Longer text of a tool-calling message
      is kept in ``discarded`` as fallback instead of being lost.

    Suppressing a second answer of the same agent cannot be streamed: what is sent is sent.
    ``emitted`` holds everything that has left the server.
    """

    def __init__(self) -> None:
        """
        Creates an empty filter for one run.
        """
        self.emitted = ""
        self.leaked_preambles = 0
        self.discarded: list[str] = []
        self._message_key = None
        self._runs: set[str] = set()
        self._emitted_any = False
        self._reset_message()

    def _reset_message(self) -> None:
        """
        Resets the per-message state: raw text, current line and its released part, pending output and whitespace.
        """
        self._raw = ""
        self._line = ""
        self._line_released = ""
        self._pending = ""
        self._whitespace = ""
        self._open = True
        self._message_started = False
        self._first_turn = False

    def begin_message(self, message_key, run_id: str = "") -> str:
        """
        Signals that the following chunks belong to another message.

        :param message_key: Identifier of the message, e.g. ``(node, chunk.id)``.
        :param run_id: Identifier of the agent run the message belongs to. Its first
            message is buffered completely.
        :return: Remaining text of the previous message, ready to be sent.
        """
        if message_key == self._message_key:
            return ""
        rest = self.finish()
        self._message_key = message_key
        self._reset_message()
        self._first_turn = run_id not in self._runs
        self._runs.add(run_id)
        return rest

    def text(self, piece: str) -> str:
        """
        Takes the text part of a chunk.

        :param piece: Content delta of one chunk.
        :return: The part of it that may be sent right now.
        """
        if not self._open or not piece:
            return ""
        self._raw += piece
        released = []
        for part in LINE_SPLITTER.split(piece):
            if part == "\n":
                released.append(self._end_line())
            elif part:
                released.append(self._continue_line(part))
        return "".join(released)

    def mark_tool_call(self) -> None:
        """
        Signals that the current message calls a tool, which drops it; long enough text is kept for :meth:`fallback`.
        """
        if not self._open:
            return
        if self._message_started:
            self.leaked_preambles += 1
            logger.info(
                "Streaming: preamble before a tool call was delivered (%d characters)",
                len(self._raw),
            )
        elif len(self._raw.strip()) >= FALLBACK_CHARS:
            self.discarded.append(self._raw.strip())
            logger.info(
                "Streaming: tool-call message carried %d characters of text; kept as "
                "fallback",
                len(self._raw.strip()),
            )
        self._open = False
        self._pending = ""

    def fallback(self) -> str:
        """
        Returns the longest text that an attached tool call cost.

        :return: The discarded text, or ``""`` when nothing was discarded.
        """
        return max(self.discarded, key=len, default="")

    def finish(self) -> str:
        """
        Finishes the current message and releases its remainder.

        :return: Remaining text of the current message.
        """
        if not self._open:
            return ""
        if self._line:
            cleaned = strip_brackets(self._line)
            dropped = (
                cleaned != self._line
                and len(cleaned.strip()) < LINE_THRESHOLD
            )
            if not dropped:
                self._pending += cleaned[len(self._line_released):]
            self._line = ""
            self._line_released = ""
        return self._pump(final=True)

    def _continue_line(self, part: str) -> str:
        """
        Processes text within a line that is still open; nothing is released while the whole line could still be dropped.

        :param part: Piece of the current line.
        :return: Text released by this piece.
        """
        self._line += part
        cleaned = strip_brackets(_stable_part(self._line))
        if len(cleaned.strip()) < LINE_THRESHOLD:
            return ""
        new_text = cleaned[len(self._line_released):]
        self._line_released = cleaned
        self._pending += new_text
        return self._pump()

    def _end_line(self) -> str:
        """
        Finishes a line and applies the rule of :func:`clean_brackets`.

        :return: Text released by the finished line.
        """
        cleaned = strip_brackets(self._line)
        dropped = cleaned != self._line and len(cleaned.strip()) < LINE_THRESHOLD
        self._line = ""
        if dropped:
            self._line_released = ""
            return ""
        self._pending += cleaned[len(self._line_released):] + "\n"
        self._line_released = ""
        return self._pump()

    def _pump(self, final: bool = False) -> str:
        """
        Releases pending text once the buffers allow it.

        Nothing leaves during an agent's first turn or before ``LEAD_CHARS`` characters.
        Handoff notices are dropped, leading whitespace is stripped, several agents are
        separated by a blank line, and whitespace is only sent together with following text.

        :param final: ``True`` at the end of a message; releases regardless.
        :return: Text that may leave the server now.
        """
        if not self._pending:
            return ""
        if not final and self._first_turn:
            return ""
        if not final and not self._message_started and len(self._raw) <= LEAD_CHARS:
            return ""

        text = self._pending
        self._pending = ""

        if not self._message_started:
            if self._raw.lstrip().startswith(HANDOFF_PREFIX):
                self._open = False
                return ""
            text = text.lstrip()
            if not text:
                return ""
            self._message_started = True
            if self._emitted_any:
                text = "\n\n" + text
        else:
            if not text.strip():
                self._whitespace += text
                return ""
            text = self._whitespace + text
        self._whitespace = ""
        self._emitted_any = True
        self.emitted += text
        return text


def _stable_part(line: str) -> str:
    """
    Cuts off an unclosed ``[[`` (or a trailing ``[``) at the end of the line, whose cleaning is not settled yet.

    :param line: The current line as written so far.
    :return: The part of it whose cleaning is already settled.
    """
    i = line.rfind("[[")
    if i != -1 and "]]" not in line[i:]:
        return line[:i]
    if line.endswith("["):
        return line[:-1]
    return line


def real_streaming() -> bool:
    """
    Checks whether answers are streamed token by token from the model.

    ``CHAT_STREAM_SIMULIERT=true`` switches back to computing the full answer first and
    streaming it word by word, kept to compare the latency to the first character.

    :return: ``True`` for token streaming, ``False`` for the simulated variant.
    """
    return (os.getenv("CHAT_STREAM_SIMULIERT") or "").strip().lower() not in {
        "true",
        "1",
        "yes",
    }


def _choose_fallback(
    reply_filter: "ReplyFilter",
    supervisor_text: dict[str, str],
    last_supervisor_id: str | None,
    delegated: bool,
) -> str:
    """
    Chooses what to deliver when no agent text passed the filter, in the same order as :func:`extract_final_text`.

    First an agent text dropped because of an attached tool call, then the supervisor if
    it did not delegate, otherwise the error text.

    :param reply_filter: The filter of the finished run.
    :param supervisor_text: Text of each supervisor message, by message id.
    :param last_supervisor_id: Id of the last supervisor message carrying text.
    :param delegated: Whether the supervisor handed off to an agent in this run.
    :return: The text to deliver.
    """
    recovered = reply_filter.fallback()
    if recovered:
        logger.info("Streaming: answer recovered from a discarded tool-call message")
        return recovered
    if not delegated and last_supervisor_id:
        return supervisor_text.get(last_supervisor_id, "")
    logger.warning("Streaming: no agent text in the run, supervisor commentary suppressed")
    return ERROR_TEXT


async def stream_supervisor(
    app,
    model: str,
    student_id: str,
    messages: list[dict],
    scope: dict | None = None,
    session_id: str | None = None,
    concepts=None,
    learner=None,
):
    """
    Streams the supervisor run token by token as OpenAI SSE, the counterpart of :func:`run_supervisor`.

    Only worker nodes go out live (the first namespace element names the top-level node);
    the supervisor's text is recorded as fallback. Finished messages written to the state
    are skipped, otherwise every answer would appear twice. Link cleaning runs along
    (:class:`ReplyFilter`), while the pointer line and the slide citations are only known
    at the end. As soon as a worker has answered and control returns to the supervisor,
    the loop stops: the supervisor's second model call takes 2.4 to 4.5 s and only
    produces text that would be dropped. The visit is recorded after the answer, so the
    Neo4j write does not delay the output.

    :param app: Compiled supervisor app.
    :param model: Model name echoed back in every chunk.
    :param student_id: Student, passed through as runtime context.
    :param messages: Full chat history sent by the client.
    :param scope: Selected topic, prepended as a system message.
    :param session_id: Chat session identifier for the visit counter.
    :param concepts: Optional ``ConceptSource`` for the pointer fallback.
    :param learner: Optional ``LearnerGraph`` for the visit counter (s.
        :func:`record_visits`); without it no visit is recorded.
    :return: Async generator over SSE lines including the terminator and ``[DONE]``.
    """
    session_key = compute_session_key(student_id, messages, scope, session_id)
    question = last_question(messages)

    system = scope_system_message(scope)
    if system is not None:
        messages = [system, *messages]

    cid = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    yield sse_chunk(cid, created, model, {"role": "assistant"})

    reply_filter = ReplyFilter()
    supervisor_text: dict[str, str] = {}
    supervisor_tool_calls: set[str] = set()
    last_supervisor_id: str | None = None
    citations: list[dict] = []
    visited: list[str] = []

    try:
        async for namespace, (chunk, _meta) in app.astream(
            {"messages": messages},
            context={
                "student_id": student_id,
                "session_key": session_key,
                "slide_log": citations,
                "concept_log": visited,
            },
            stream_mode="messages",
            subgraphs=True,
        ):
            if not isinstance(chunk, AIMessageChunk):
                continue
            node = str(namespace[0]).split(":")[0] if namespace else ""
            piece = _content_text(chunk)

            if node in ("", SUPERVISOR_NAME):
                rest = reply_filter.finish()
                if rest:
                    yield sse_chunk(cid, created, model, {"content": rest})

                if reply_filter.emitted.strip():
                    break

                message_id = chunk.id or SUPERVISOR_NAME
                if chunk.tool_call_chunks:
                    supervisor_tool_calls.add(message_id)
                if piece:
                    supervisor_text[message_id] = supervisor_text.get(message_id, "") + piece
                    last_supervisor_id = message_id
                continue

            released = reply_filter.begin_message((node, chunk.id), str(namespace[0]))
            if released:
                yield sse_chunk(cid, created, model, {"content": released})
            if chunk.tool_call_chunks:
                reply_filter.mark_tool_call()
                continue
            released = reply_filter.text(piece)
            if released:
                yield sse_chunk(cid, created, model, {"content": released})

        released = reply_filter.finish()
        if released:
            yield sse_chunk(cid, created, model, {"content": released})
    except Exception:
        logger.exception("Streaming aborted (student=%s)", student_id)
        if not reply_filter.emitted.strip():
            yield sse_chunk(
                cid,
                created,
                model,
                {"content": ERROR_TEXT},
            )
        yield sse_chunk(cid, created, model, {}, finish_reason="stop")
        yield "data: [DONE]\n\n"
        return

    answer = reply_filter.emitted
    if not answer.strip():
        answer = clean_brackets(
            _choose_fallback(reply_filter, supervisor_text, last_supervisor_id, bool(supervisor_tool_calls))
        )
        for token in _tokenize(answer):
            yield sse_chunk(cid, created, model, {"content": token})

    if answer.strip() and answer != ERROR_TEXT:
        await asyncio.to_thread(
            record_visits, learner, student_id, session_key, scope, visited
        )

    pointer = await asyncio.to_thread(pointer_line, answer, question, scope, concepts)
    if pointer:
        yield sse_chunk(cid, created, model, {"content": pointer})

    yield sse_chunk(
        cid, created, model, {}, finish_reason="stop", slides=sort_citations(citations)
    )
    yield "data: [DONE]\n\n"
