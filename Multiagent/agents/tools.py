from __future__ import annotations

import logging

from langchain.tools import tool
from langgraph.runtime import get_runtime

from Multiagent.GraphAccess import ConceptSource
from Multiagent.LearnerModel import LearnerGraph
from Multiagent.Recommender import Recommender

from .state import RuntimeContext

logger = logging.getLogger(__name__)

_concept_source: ConceptSource | None = None
_learner_graph: LearnerGraph | None = None
_recommender: Recommender | None = None


def _cs() -> ConceptSource:
    """
    Returns the lazily created module-wide ``ConceptSource`` so that tool calls share one Neo4j driver.

    :return: The shared concept source.
    """
    global _concept_source
    if _concept_source is None:
        _concept_source = ConceptSource()
    return _concept_source


def _lg() -> LearnerGraph:
    """
    Returns the lazily created module-wide ``LearnerGraph``.

    :return: The shared learner graph.
    """
    global _learner_graph
    if _learner_graph is None:
        _learner_graph = LearnerGraph()
    return _learner_graph


def _rec() -> Recommender:
    """
    Returns the lazily created module-wide ``Recommender``.

    :return: The shared recommender.
    """
    global _recommender
    if _recommender is None:
        _recommender = Recommender()
    return _recommender


@tool
def find_concept(query: str) -> list[dict]:
    """
    Findet Vorlesungskonzepte anhand ihres Namens (Teilstring, case-insensitive).

    Nutze dies zuerst, um eine natürlichsprachliche Nennung (z. B. 'CAP-Theorem') in eine
    Konzept-ID aufzulösen. Rückgabe je Treffer: {"id", "name", "objective", "path"}.

    "path" ist die Position im Auswahlbaum ("Kapitel > Thema > Unterthema"). Nenne sie
    wörtlich, wenn du einen Studenten auf eine andere Stelle verweist — dann muss er
    nicht suchen. Rate den Ort nie selbst.

    :param query: Natural-language mention, e.g. "CAP theorem".
    :return: Matches as ``{id, name, objective, path}``.
    """
    return _cs().find_concept(query)


def _log_concept(concept_id: str) -> None:
    """
    Records in the runtime context that material for this concept was loaded.

    Replaces the former ``mark_visited`` tool, which the model called unreliably; the server
    writes the visit after the run (:func:`Multiagent.server.openai_adapter.record_visits`).
    Has no effect on the agent, and a missing context must not cost the explanation.

    :param concept_id: Concept whose material was loaded.
    """
    try:
        log_list = get_runtime(RuntimeContext).context.concept_log
    except Exception:
        return
    if log_list is None:
        return
    if concept_id not in log_list:
        log_list.append(concept_id)


def _log_slides(slide_source: ConceptSource, pairs: list[tuple]) -> None:
    """
    Writes the citations of the loaded slides into the runtime context for the server to read after the run.

    This is the only place where the slide origin is known, since the model only receives
    plain texts. Has no effect on the agent; a missing context or a failing citation must
    not cost the explanation.

    :param slide_source: The ``ConceptSource`` the slides were loaded from.
    :param pairs: Pairs of slide reference and text, as returned by ``fetch_slides``.
    """
    try:
        log_list = get_runtime(RuntimeContext).context.slide_log
    except Exception:
        return
    if log_list is None:
        return
    for ref, _ in pairs:
        try:
            log_list.append(slide_source.slide_citation(ref))
        except Exception:
            logger.warning("Slide citation for %s p.%s skipped.", ref.source, ref.page)


@tool
def get_concept_material(concept_id: str) -> list[str]:
    """
    Lädt die Vorlesungsfolien-Texte zu einer Konzept-ID.

    Dies ist die EINZIGE erlaubte Wissensquelle für die Erklärung. Erwartet eine Konzept-ID
    aus find_concept und liefert die Folientexte als Liste von Strings.

    :param concept_id: Concept whose slides are loaded.
    :return: Slide texts as a list.
    """
    slide_source = _cs()
    pairs = slide_source.fetch_slides(concept_id)
    _log_slides(slide_source, pairs)
    _log_concept(concept_id)
    return [text for _, text in pairs]


@tool
def next_steps(limit: int = 5) -> dict:
    """
    Nennt konkrete Konzepte, mit denen der Student jetzt weitermachen kann.

    Das Werkzeug für die offene Frage ("Womit weiter?", "Was soll ich als Nächstes
    lernen?"), die KEIN Ziel-Konzept nennt. Es braucht deshalb auch keines.

    Geliefert wird die lernbare Front des gesamten Korpus: Konzepte, die noch nicht
    beherrscht sind und deren sämtliche Voraussetzungen erfüllt sind — einschließlich
    derer ganz ohne Voraussetzungen. Sortiert in der Reihenfolge der Vorlesung, das
    oberste ist also das, was als Nächstes drankommt.

    Rückgabe:
    - ``next``: die Vorschläge als ``{id, name, path, status, mastery, visited_count}``.
      ``path`` ist die Position im Auswahlbaum ("Kapitel > Thema > Unterthema") und sagt
      dir, welche Vorschläge beim zuletzt besprochenen Konzept liegen.
    - ``next_total``: wie groß die Front insgesamt ist (vor dem Kürzen auf ``limit``).
    - ``due_count``: fällige Wiederholungen. Blockieren nichts, eigene Ansicht.
    - ``mastered`` / ``total``: beherrschte Konzepte und Gesamtzahl.

    Ist ``next`` leer und ``mastered == total``, ist die Vorlesung durch.

    :param limit: How many concepts to propose; 5 is a good default.
    :return: The learnable front in curriculum order plus the surrounding counters.
    """
    runtime = get_runtime(RuntimeContext)
    return _rec().next_steps(runtime.context.student_id, limit)


@tool
def recommend_next(concept_id: str) -> dict:
    """
    Ermittelt für ein Ziel-Konzept, was der Student als Nächstes lernen sollte.

    Wertet die Voraussetzungen (transitiv) und den Lernstand aus. Rückgabe:
    - ``target``: das Ziel-Konzept mit Status.
    - ``ready_to_learn``: die jetzt lernbaren Konzepte (Voraussetzungen erfüllt) — die eigentliche
      Empfehlung. Sind es mehrere, soll der Student wählen.
    - ``blocked``: noch nicht lernbare Konzepte samt fehlender Voraussetzungen.
    - ``facilitators``: OPTIONALE, hilfreiche Zusatzkonzepte (keine Pflicht).

    Erwartet eine Konzept-ID (zuvor per find_concept auflösen).

    :param concept_id: The student's target concept.
    :return: Learning path with ``target``, ``ready_to_learn``, ``blocked`` and ``facilitators``.
    """
    runtime = get_runtime(RuntimeContext)
    return _rec().recommend(runtime.context.student_id, concept_id)
