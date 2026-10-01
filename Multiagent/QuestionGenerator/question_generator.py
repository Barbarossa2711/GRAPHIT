from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

from typing import Callable

import dotenv
import httpx
from langchain_openai import ChatOpenAI

from Multiagent.Assessment.mastery import s_min

from .checks import quality_warnings, semantic_errors

from .models import (
    Concept,
    GeneratedStem,
    QualityWarning,
    Question,
    QuestionSet,
    StemPlan,
    StemPlanList,
    StemSource,
    TYPE_TO_MODEL,
)
from .prompts import (
    LECTURE_SYSTEM,
    LECTURE_USER,
    PLAN_SYSTEM,
    VARIANT_SYSTEM,
    build_plan_user,
    build_variant_user,
)
from Multiagent.llm_endpoint import tls_verify

from .validation import validate_question

logger = logging.getLogger(__name__)

_DEFAULT_ENV_PATH = Path(__file__).parents[1] / ".env"
_DEFAULT_MODEL = "gpt-4o"
_DEFAULT_STRUCTURED_METHOD = "function_calling"

_tls_verify = tls_verify

FAILURE_BUDGET = 2
VALIDATION_RETRIES = 3
NETWORK_RETRIES = 3
NETWORK_BACKOFF = 2.0


def _is_transient(exc: BaseException) -> bool:
    """
    Decides whether an LLM error is transient (timeout, connection, rate limit, 5xx) and worth a retry.

    Permanent errors such as 400 or 401 stay the same on every attempt and must stay visible.

    :param exc: Exception raised by the LLM call.
    :return: ``True`` if retrying is worthwhile, ``False`` for permanent failures.
    """
    try:
        import openai
    except ImportError:  # pragma: no cover
        return isinstance(exc, (TimeoutError, ConnectionError))

    transient = (
        openai.APIConnectionError,
        openai.RateLimitError,
        openai.InternalServerError,
    )
    if isinstance(exc, transient):
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and status >= 500


def _invoke_robust(runnable, messages):
    """
    Calls the LLM and retries transient errors up to ``NETWORK_RETRIES`` times with doubling backoff.

    Separate from ``VALIDATION_RETRIES``: here the model did not answer at all, so the same
    call may succeed next time; without retries a single outage silently costs an item.

    :param runnable: The prepared LLM chain to call.
    :param messages: Messages passed to the model.
    :return: The reply of the model.
    :raises Exception: The last error once the retry budget is spent or the error is permanent.
    """
    backoff = NETWORK_BACKOFF
    for versuch in range(1, NETWORK_RETRIES + 1):
        try:
            return runnable.invoke(messages)
        except Exception as exc:
            if versuch == NETWORK_RETRIES or not _is_transient(exc):
                raise
            logger.warning(
                "LLM call failed (%s), retry %d/%d in %.0fs: %s",
                type(exc).__name__, versuch + 1, NETWORK_RETRIES, backoff, str(exc)[:160],
            )
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError("unerreichbar")  # pragma: no cover


ProgressHook = Callable[[str, int], None]


def min_items(f: int = FAILURE_BUDGET) -> int:
    """
    Returns the minimum number of distinct items a concept needs, the target of the item budget.

    Taken from :func:`Multiagent.Assessment.mastery.s_min`, since it is a property of the
    learner model: a concept with fewer items cannot reach the mastery threshold without
    repeating a question. The default tolerates two wrong answers, matching the simulated
    error rate of 20 %.

    :param f: Wrong answers the budget should tolerate.
    :return: Distinct items a concept needs at minimum.
    """
    return s_min(f)


class QuestionGenerator:
    """
    Generates validated question sets from a concept and its slides via any OpenAI-compatible endpoint.

    Phase 1 plans the stems (testable facts and suitable question types) within a span
    derived from the slide count; phase 2 generates one validated variant per stem and type;
    phase 3 fills up the item budget with isomorphic items.

    A ``progress`` hook receives ``("planned", n)`` once after planning (a lower bound),
    ``("item", 1)`` per variant from phase 2, ``("budget", 1)`` per item added in phase 3
    and ``("warning", n)`` per item with ``n`` quality findings, which counts no items.
    """

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.4,
        env_path: str | Path | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        structured_method: str | None = None,
    ) -> None:
        """
        Sets up the LLM client; each endpoint setting is resolved as argument, then .env, then default.

        Settings come from ``LLM_BASE_URL``, ``LLM_API_KEY``, ``LLM_MODEL`` and
        ``LLM_STRUCTURED_METHOD``. ``LLM_API_KEY`` is only used for a custom endpoint, so
        that the OpenAI key is never sent to a foreign endpoint. Custom httpx clients are
        only created when the endpoint needs a different TLS verification.

        :param model: Model name.
        :param temperature: Sampling temperature.
        :param env_path: Path of the ``.env`` to read.
        :param base_url: Base URL of an OpenAI-compatible endpoint; api.openai.com when empty.
        :param api_key: API key for that endpoint.
        :param structured_method: How the output structure is enforced: ``function_calling``
            (default; strict ``json_schema`` rejects the open map of cloze solutions),
            ``json_schema`` or ``json_mode``.
        """
        dotenv.load_dotenv(env_path or _DEFAULT_ENV_PATH)
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or None
        self.model = model or os.getenv("LLM_MODEL") or _DEFAULT_MODEL
        self.temperature = temperature
        self.structured_method = (
            structured_method or os.getenv("LLM_STRUCTURED_METHOD") or _DEFAULT_STRUCTURED_METHOD
        )
        key = api_key or (os.getenv("LLM_API_KEY") if self.base_url else None)

        verify = _tls_verify() if self.base_url else True
        tls: dict = {}
        if verify is not True:
            tls = {
                "http_client": httpx.Client(verify=verify),
                "http_async_client": httpx.AsyncClient(verify=verify),
            }

        self._llm = ChatOpenAI(
            model=self.model,
            temperature=temperature,
            **({"base_url": self.base_url} if self.base_url else {}),
            **({"api_key": key} if key else {}),
            **tls,
        )
        logger.info(
            "QuestionGenerator: model '%s' via %s (structured_output=%s)",
            self.model,
            self.base_url or "api.openai.com",
            self.structured_method,
        )

    def _structured(self, model_cls):
        """
        Wraps the LLM so that it has to produce the given structure; used by all phases.

        :param model_cls: Pydantic model describing the expected output.
        :return: An LLM runnable forced to produce that structure.
        """
        return self._llm.with_structured_output(model_cls, method=self.structured_method)

    @staticmethod
    def slides_to_stem_range(n_slides: int) -> tuple[int, int]:
        """
        Derives a span of stems from the slide count, within which the LLM picks the number of distinct facts.

        The slide count only measures size, not density, and about 85 % of the concepts have
        one or two slides, so a span calibrated to that distribution is used. It only
        controls the content structure; the item budget is ensured separately by
        :meth:`fill_item_budget`.

        :param n_slides: Number of slides available for the concept.
        :return: The ``(min, max)`` span of stems to plan.
        """
        if n_slides <= 1:
            return 1, 2
        if n_slides <= 3:
            return 2, 3
        if n_slides <= 5:
            return 3, 4
        return 4, 5

    def plan_stems(
        self,
        concept: Concept,
        slides: list[str],
        min_stems: int,
        max_stems: int,
        avoid: list[str] | None = None,
        foreign_concepts: list[str] | None = None,
    ) -> list[StemPlan]:
        """
        Plans new stems from the slides (phase 1), at most ``max_stems``.

        :param concept: Concept the stems are planned for.
        :param slides: Slide texts serving as the only source.
        :param min_stems: Lower bound of the stem span.
        :param max_stems: Upper bound of the stem span.
        :param avoid: Prompt texts that must not be repeated.
        :param foreign_concepts: Neighbouring concepts the plan must stay clear of.
        :return: The planned stems.
        """
        objective_line = f"\nÜbergeordnetes Lernziel: {concept.objective}" if concept.objective else ""
        user = build_plan_user(
            concept_name=concept.name,
            objective_line=objective_line,
            n_slides=len(slides),
            slides=_format_slides(slides),
            min_stems=min_stems,
            max_stems=max_stems,
            avoid=avoid,
            foreign_concepts=foreign_concepts,
        )
        planner = self._structured(StemPlanList)
        result: StemPlanList = _invoke_robust(
            planner, [("system", PLAN_SYSTEM), ("human", user)]
        )
        return result.stems[:max_stems]

    def plan_lecture_stems(
        self, concept: Concept, existing_questions: list[str]
    ) -> list[StemPlan]:
        """
        Derives one stem from each existing lecture question (phase 1b).

        :param concept: Concept the stems belong to.
        :param existing_questions: Original lecture questions to derive stems from.
        :return: One stem per usable lecture question.
        """
        user = LECTURE_USER.format(
            concept_name=concept.name,
            questions="\n".join(f"{i}. {q}" for i, q in enumerate(existing_questions, start=1)),
        )
        planner = self._structured(StemPlanList)
        result: StemPlanList = _invoke_robust(
            planner, [("system", LECTURE_SYSTEM), ("human", user)]
        )
        return result.stems

    def generate_variant(
        self,
        concept: Concept,
        objective: str,
        qtype: str,
        slides: list[str],
        *,
        avoid: list[str] | None = None,
    ) -> dict | None:
        """
        Generates a single question variant that passes the JSON schema and the semantic checks (phase 2).

        On errors it retries up to ``VALIDATION_RETRIES`` times, passing only the errors of
        the last attempt as correction hint so the prompt does not accumulate contradicting
        hints. A non-empty ``avoid`` produces an isomorphic item.

        :param concept: Concept the item belongs to.
        :param objective: Learning objective the item should test.
        :param qtype: Question type to produce.
        :param slides: Slide texts serving as the only source.
        :param avoid: Prompt texts that must not be repeated.
        :return: A schema-valid payload, or ``None`` if no sound item could be produced.
        """
        model_cls = TYPE_TO_MODEL[qtype]
        structured = self._structured(model_cls)
        user = build_variant_user(
            concept_name=concept.name,
            objective=objective,
            slides=_format_slides(slides),
            qtype=qtype,
            avoid=avoid,
        )
        messages = [("system", VARIANT_SYSTEM), ("human", user)]

        for attempt in range(1, VALIDATION_RETRIES + 1):
            try:
                payload = _invoke_robust(structured, messages).model_dump(exclude_none=True)
            except Exception as exc:
                logger.warning("Variant '%s' could not be generated: %s: %s",
                               qtype, type(exc).__name__, exc)
                return None

            errors = validate_question(payload) + _semantic_errors(qtype, payload)
            if not errors:
                if attempt > 1:
                    logger.info("Variant '%s' valid on attempt %d.", qtype, attempt)
                return payload

            logger.info(
                "Validation errors for '%s' (attempt %d/%d): %s",
                qtype, attempt, VALIDATION_RETRIES, "; ".join(errors),
            )
            messages = messages[:2] + [
                (
                    "human",
                    "Die vorige Ausgabe war fehlerhaft:\n- "
                    + "\n- ".join(errors)
                    + "\nKorrigiere die Frage entsprechend.",
                )
            ]

        logger.warning("Variant '%s' skipped (still invalid after %d attempts).",
                       qtype, VALIDATION_RETRIES)
        return None

    def generate(
        self,
        concept: Concept,
        slides: list[str],
        *,
        existing_questions: list[str] | None = None,
        n_stems: int | None = None,
        foreign_concepts: list[str] | None = None,
        progress: ProgressHook | None = None,
    ) -> QuestionSet:
        """
        Generates the complete, validated question set for a concept.

        Each existing lecture question becomes a stem with ``source="lecture"``; newly
        planned stems avoid those facts and the stem span is reduced by their number.
        Quality findings are logged per item at INFO and summarised per concept at WARNING.

        :param concept: Concept to generate for.
        :param slides: Slide texts serving as the only source.
        :param existing_questions: Original lecture questions, each turned into a stem.
        :param n_stems: Fixed total number of stems; derived from the slide count when omitted.
        :param foreign_concepts: Names of the :CO_OCCURS neighbour concepts, excluded in the
            planning prompt so that shared slides do not produce questions about them.
        :param progress: Optional progress hook, see :class:`QuestionGenerator`.
        :return: The complete, validated question set.
        :raises ValueError: If no slide text is given.
        """
        def report(event: str, count: int) -> None:
            """
            Passes a progress event on to the hook supplied by the caller.

            :param event: Event name, e.g. ``geplant``, ``item`` or ``budget``.
            :param count: How many units the event covers.
            """
            if progress is not None:
                progress(event, count)

        if not slides:
            raise ValueError("At least one slide text must be passed.")

        if n_stems is not None:
            min_stems, max_stems = n_stems, n_stems
        else:
            min_stems, max_stems = self.slides_to_stem_range(len(slides))

        lecture_plans = (
            self.plan_lecture_stems(concept, existing_questions) if existing_questions else []
        )
        n_lecture = len(lecture_plans)

        remaining_min = max(0, min_stems - n_lecture)
        remaining_max = max(0, max_stems - n_lecture)
        covered = [p.objective for p in lecture_plans]
        new_plans = (
            self.plan_stems(
                concept, slides, remaining_min, remaining_max,
                avoid=covered, foreign_concepts=foreign_concepts,
            )
            if remaining_max > 0
            else []
        )

        planned: list[tuple[StemPlan, StemSource]] = (
            [(p, "lecture") for p in lecture_plans] + [(p, "generated") for p in new_plans]
        )

        planned_types = sum(
            len([t for t in _normalize_question_types(p.question_types) if t in TYPE_TO_MODEL])
            for p, _ in planned
        )
        report("planned", max(planned_types, min_items()))

        warnings: list[QualityWarning] = []
        stems: list[GeneratedStem] = []
        for idx, (plan, source) in enumerate(planned, start=1):
            stem_id = f"{concept.id}_ST{idx:02d}"
            questions: list[Question] = []
            for qtype in _normalize_question_types(plan.question_types):
                if qtype not in TYPE_TO_MODEL:
                    continue
                payload = self.generate_variant(concept, plan.objective, qtype, slides)
                if payload is None:
                    continue
                question = _build_question(stem_id, qtype, payload, questions)
                questions.append(question)
                report("item", 1)
                _collect_warnings(question, payload, foreign_concepts, warnings, progress)
            if questions:
                stems.append(
                    GeneratedStem(id=stem_id, objective=plan.objective, source=source, questions=questions)
                )

        self.fill_item_budget(
            concept, slides, stems,
            progress=progress, foreign_concepts=foreign_concepts, warnings=warnings,
        )

        if warnings:
            logger.warning(
                "Concept '%s': %d quality warnings on %d of %d items.",
                concept.id,
                len(warnings),
                len({w.question_id for w in warnings}),
                sum(len(st.questions) for st in stems),
            )

        return QuestionSet(
            concept_id=concept.id,
            concept_name=concept.name,
            n_slides=len(slides),
            stems=stems,
            warnings=warnings,
        )

    def fill_item_budget(
        self,
        concept: Concept,
        slides: list[str],
        stems: list[GeneratedStem],
        *,
        target: int | None = None,
        progress: ProgressHook | None = None,
        foreign_concepts: list[str] | None = None,
        warnings: list[QualityWarning] | None = None,
    ) -> int:
        """
        Fills ``stems`` in place with isomorphic items until ``target`` items exist (phase 3).

        Fills via additional variants of existing stems (same fact, different wording and
        distractors), not via additional stems, so a concept with one testable fact gets no
        forced second one. The stem with the fewest items is filled first. Every failure
        costs an attempt (at most twice the gap), so a failing LLM cannot loop forever.
        Added items are quality-checked too.

        :param concept: Concept being filled up.
        :param slides: Slide texts serving as the only source.
        :param stems: Stems to extend in place.
        :param target: Number of items to reach; :func:`min_items` when omitted.
        :param progress: Optional progress hook.
        :param foreign_concepts: Neighbouring concepts, for the delimitation check.
        :param warnings: List the quality findings of the added items are appended to.
        :return: Number of items added; 0 if the budget was already met or there are no stems.
        """
        target = min_items() if target is None else target
        if not stems:
            logger.warning(
                "Concept '%s': no stems generated, item budget cannot be met.", concept.id
            )
            return 0

        available = sum(len(s.questions) for s in stems)
        added = 0
        attempts = 2 * max(0, target - available)

        while available < target and attempts > 0:
            attempts -= 1
            stem = min(stems, key=lambda s: (len(s.questions), s.id))
            qtype = _isomorphic_type(stem)
            payload = self.generate_variant(
                concept,
                stem.objective,
                qtype,
                slides,
                avoid=_existing_prompts(stem),
            )
            if payload is None:
                continue
            question = _build_question(stem.id, qtype, payload, stem.questions)
            stem.questions.append(question)
            available += 1
            added += 1
            if progress is not None:
                progress("budget", 1)
            _collect_warnings(question, payload, foreign_concepts, warnings, progress)

        if available < target:
            logger.warning(
                "Concept '%s': only %d of %d required items generated; the concept cannot reach "
                "the mastery threshold without repeating items.",
                concept.id,
                available,
                target,
            )
        elif added:
            logger.info(
                "Concept '%s': %d isomorphic items added (item budget %d reached).",
                concept.id,
                added,
                target,
            )
        return added


def _collect_warnings(
    question: Question,
    payload: dict,
    foreign_concepts: list[str] | None,
    sink: list[QualityWarning] | None,
    progress: ProgressHook | None = None,
) -> None:
    """
    Checks an accepted item for construction weaknesses and records the findings.

    Runs only on the accepted payload, not inside the retry loop; findings never abort.

    :param question: The finished item the findings belong to.
    :param payload: Its payload, already validated.
    :param foreign_concepts: Neighbouring concepts, for the delimitation check.
    :param sink: List the findings are appended to; ``None`` logs only.
    :param progress: Optional progress hook, notified once per item with findings.
    """
    findings = _quality_warnings(question.type, payload, foreign_concepts)
    if not findings:
        return
    for text in findings:
        logger.info("Quality warning for %s: %s", question.id, text)
        if sink is not None:
            sink.append(
                QualityWarning(question_id=question.id, type=question.type, message=text)
            )
    if progress is not None:
        progress("warning", len(findings))


def _format_slides(slides: list[str]) -> str:
    """
    Numbers the slide texts for use inside a prompt.

    :param slides: Slide texts of the concept.
    :return: The slides numbered for use inside a prompt.
    """
    return "\n".join(f"[Folie {i}] {text}" for i, text in enumerate(slides, start=1))


def _build_question(
    stem_id: str, qtype: str, payload: dict, existing: list[Question]
) -> Question:
    """
    Builds a :class:`Question` with a unique id, numbered per stem and type because a stem can hold several items of one type.

    :param stem_id: Stem the variant belongs to.
    :param qtype: Question type of the variant.
    :param payload: Validated payload of the item.
    :param existing: Items already built, used to keep the id unique.
    :return: The assembled :class:`Question`.
    """
    n = sum(1 for q in existing if q.type == qtype) + 1
    return Question(
        id=f"{stem_id}_{qtype.upper()}_{n:02d}",
        type=qtype,
        payload=json.dumps(payload, ensure_ascii=False),
    )


_ISOMORPH_SUITABILITY = {"single": 0, "multiple": 0, "cloze": 1, "match": 2, "order": 2}


def _isomorphic_type(stem: GeneratedStem) -> str:
    """
    Chooses the question type for the next isomorphic item of a stem.

    Only types already planned for the stem are considered. Choice and cloze items can be
    varied freely, while ``match`` and ``order`` depend on the structure of the fact and
    are only used if nothing else was planned. Among suitable types the one with the fewest
    items wins, so formats alternate.

    :param stem: Stem whose next variant is planned.
    :return: The question type the next item should use.
    """
    count: dict[str, int] = {}
    for q in stem.questions:
        count[q.type] = count.get(q.type, 0) + 1
    suitable = {t: n for t, n in count.items() if _ISOMORPH_SUITABILITY.get(t, 3) <= 1}
    candidates = suitable or count
    return min(candidates, key=lambda t: (candidates[t], _ISOMORPH_SUITABILITY.get(t, 3), t))


def _existing_prompts(stem: GeneratedStem) -> list[str]:
    """
    Collects the question texts of all previous items of a stem, across all types, as a template to differ from.

    :param stem: Stem whose previous items are collected.
    :return: Prompt texts of all existing items, used as a delimitation template.
    """
    prompts: list[str] = []
    for q in stem.questions:
        try:
            payload = json.loads(q.payload)
        except json.JSONDecodeError:
            continue
        text = payload.get("text") if q.type == "cloze" else payload.get("prompt")
        if text:
            prompts.append(text)
    return prompts


_CHOICE_FAMILY = {"single", "multiple"}


def _normalize_question_types(qtypes: list[str]) -> list[str]:
    """
    Deduplicates question types keeping their order and allows at most one choice type.

    A multiple choice item with one correct answer is effectively a single choice item, so
    if the planner returns both, only the first one is kept.

    :param qtypes: Requested question types, possibly with duplicates.
    :return: Deduplicated list keeping order, with at most one choice type.
    """
    result: list[str] = []
    choice_used = False
    for qtype in dict.fromkeys(qtypes):
        if qtype in _CHOICE_FAMILY:
            if choice_used:
                continue
            choice_used = True
        result.append(qtype)
    return result


_semantic_errors = semantic_errors
_quality_warnings = quality_warnings
