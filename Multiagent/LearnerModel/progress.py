from __future__ import annotations

from pathlib import Path

from neo4j import GraphDatabase

from Multiagent.Assessment.mastery import (
    GATE_THRESHOLD,
    MASTERY_THRESHOLD,
    MasteryParams,
    compute_mastery,
    days_until_due,
    today_daynum,
)
from Multiagent.GraphAccess.config import GraphSchema, Neo4jConfig, load_config
from Multiagent.GraphAccess.domain_tree import MAX_DEPTH

from .learner_graph import LearnerSchema

__all__ = [
    "ProgressService",
    "STATUSES",
    "concept_status",
    "effective_mastery",
    "gate_passed",
    "peak_mastery",
]

STATUSES = ("new", "visited", "in_progress", "due_review", "mastered")


def effective_mastery(node: dict, t_now: int, params: MasteryParams = MasteryParams()) -> float | None:
    """
    Computes mastery from ``(s, f, t_last)`` on day ``t_now`` (compute-on-read).

    The stored ``mastery`` snapshot is deliberately ignored; it is from the last quiz and
    would make the time decay ineffective.

    :param node: Concept entry carrying ``s``, ``f`` and ``t_last``.
    :param t_now: Day the value is evaluated for.
    :param params: Model parameters used for the computation.
    :return: Current mastery, or ``None`` if the concept was never tested.
    """
    if node.get("t_last") is None:
        return None
    return compute_mastery(
        int(node.get("s") or 0), int(node.get("f") or 0), node["t_last"], t_now, params
    )


def peak_mastery(node: dict) -> float:
    """
    Returns the highest mastery ever reached, which drives the PREREQUISITE gate.

    Falls back to the ``mastery`` snapshot for edges written before the peak existed.

    :param node: Concept entry carrying ``mastery_peak``.
    :return: Highest mastery ever reached; ``0.0`` if never tested.
    """
    peak = node.get("mastery_peak")
    if peak is None:
        peak = node.get("mastery")
    return float(peak) if peak is not None else 0.0


def gate_passed(node: dict) -> bool:
    """
    Checks whether a concept passes the PREREQUISITE gate, decided solely by its peak mastery.

    :param node: Concept entry carrying ``mastery_peak``.
    :return: ``True`` if the concept satisfies the PREREQUISITE gate.
    """
    return peak_mastery(node) >= GATE_THRESHOLD


def concept_status(has_edge: bool, mastery: float | None, peak: float) -> str:
    """
    Classifies a concept by gate and due state; the single definition shared with the recommender.

    ==============  =========================================================
    ``new``         no :LEARNS edge, never visited
    ``visited``     edge exists, never tested
    ``in_progress`` tested, gate not passed -> blocks follow-up concepts
    ``due_review``  gate passed, current mastery dropped -> review due, does not block
    ``mastered``    gate passed and currently above the threshold
    ==============  =========================================================

    :param has_edge: Whether a :LEARNS edge exists at all.
    :param mastery: Current, decaying mastery value.
    :param peak: Highest mastery ever reached.
    :return: One of ``new``, ``visited``, ``in_progress``, ``due_review``, ``mastered``.
    """
    if not has_edge:
        return "new"
    if mastery is None:
        return "visited"
    if peak < GATE_THRESHOLD:
        return "in_progress"
    return "mastered" if mastery >= MASTERY_THRESHOLD else "due_review"


class ProgressService:
    """
    Computes a student's progress over all concepts for the frontend; usable as a context manager.

    Mastery has to be computed server side, since the frontend knows neither ``s``/``f``/``t_last``
    nor the model parameters.
    """

    def __init__(
        self,
        config: Neo4jConfig | None = None,
        schema: GraphSchema | None = None,
        learner: LearnerSchema | None = None,
        env_path: str | Path | None = None,
        params: MasteryParams = MasteryParams(),
        prerequisite_rel: str = "PREREQUISITE",
    ) -> None:
        """
        Opens the Neo4j driver.

        :param config: Connection settings; loaded from the .env when omitted.
        :param schema: Label/property names of the domain graph.
        :param learner: Label/property names of the learner subgraph.
        :param env_path: Path of the ``.env`` used when ``config`` is omitted.
        :param params: Mastery model parameters.
        :param prerequisite_rel: Prerequisite relationship; ``(A)-[:PREREQUISITE]->(B)`` means
            B is a prerequisite of A.
        """
        self.config = config or load_config(env_path)
        self.schema = schema or GraphSchema()
        self.learner = learner or LearnerSchema()
        self.params = params
        self.prerequisite_rel = prerequisite_rel
        self._driver = GraphDatabase.driver(self.config.uri, auth=self.config.auth)

    def close(self) -> None:
        """
        Closes the Neo4j driver.
        """
        self._driver.close()

    def __enter__(self) -> "ProgressService":
        """
        Enters the context manager.

        :return: This instance.
        """
        return self

    def __exit__(self, *exc) -> None:
        """
        Closes the driver when leaving the context manager.

        :param exc: Exception information, ignored.
        """
        self.close()

    def _fetch_rows(self, student_id: str) -> list[dict]:
        """
        Reads all concepts with their possibly empty learning state and their position in the hierarchy.

        Concepts without :LEARNS edge are included as ``new``. Ancestors are collected over
        the structure relationships with variable depth, including the lecture, because a
        fixed chain would leave roll-ups of nested nodes empty. Ancestors are aggregated
        before the prerequisites are attached as a COLLECT subquery, which avoids a cross
        product. ``prereq_ids`` are only the direct prerequisites.

        :param student_id: Student whose learning state is read.
        :return: One row per concept, with learning state, ancestor ids and
            direct prerequisite ids.
        """
        s, ls = self.schema, self.learner
        structure = "|".join(
            (s.has_chapter_rel, s.has_topic_rel, s.has_subtopic_rel, s.has_concept_rel)
        )
        query = (
            f"MATCH (c:{s.concept_label}) "
            f"OPTIONAL MATCH (st:{ls.student_label} {{{ls.student_id_prop}: $sid}})"
            f"-[r:{ls.learns_rel}]->(c) "
            f"OPTIONAL MATCH (c)<-[:{structure}*1..{MAX_DEPTH}]-(a) "
            f"WITH c, r, collect(DISTINCT a.{s.concept_id_prop}) AS ancestor_ids "
            f"RETURN c.{s.concept_id_prop} AS id, c.{s.concept_name_prop} AS name, "
            f"r.s AS s, r.f AS f, r.t_last AS t_last, r.mastery AS mastery, "
            f"r.mastery_peak AS mastery_peak, r.visited_count AS visited_count, "
            f"(r IS NOT NULL) AS has_edge, ancestor_ids, "
            f"COLLECT {{ MATCH (c)-[:{self.prerequisite_rel}]->(p:{s.concept_label}) "
            f"          RETURN p.{s.concept_id_prop} }} AS prereq_ids "
            f"ORDER BY id"
        )
        with self._driver.session(database=self.config.database) as session:
            return session.run(query, sid=student_id).data()

    def fetch_progress(self, student_id: str) -> dict:
        """
        Assembles the learning state over the whole corpus for the progress view.

        Structure::

            {
              "student_id": str,
              "concepts": [ {id, name, status, mastery, mastery_peak, mastered,
                             gate_passed, prereqs_met, prereqs_missing,
                             days_until_due, days_since_quiz, visited_count, s, f} ],
              "rollup":   { node_id: {total, new, visited, in_progress, due_review,
                                      mastered, gate_passed, mastery_avg} },
              "summary":  { same counters over the whole corpus },
            }

        ``rollup`` is keyed by the ids of the :class:`~Multiagent.GraphAccess.DomainTree`
        nodes. ``gate_passed`` says whether this concept unlocks its follow-ups,
        ``prereqs_met`` whether every direct prerequisite has passed its gate, i.e. the
        learnable front of the recommender; prerequisites pointing to unknown ids are
        ignored. ``days_until_due`` and ``days_since_quiz`` are ``None`` only if the concept
        was never answered.

        :param student_id: Student whose progress is assembled.
        :return: Per-concept state plus roll-ups per hierarchy node.
        """
        t_now = today_daynum()
        rows = self._fetch_rows(student_id)

        concepts: list[dict] = []
        mapping: dict[str, list[dict]] = {}
        prerequisites: dict[str, list[str]] = {}

        for row in rows:
            if row["id"] is None:
                continue
            prerequisites[row["id"]] = [p for p in (row.get("prereq_ids") or []) if p]
            mastery = effective_mastery(row, t_now, self.params)
            peak = peak_mastery(row)
            entry = {
                "id": row["id"],
                "name": row["name"] or row["id"],
                "status": concept_status(row["has_edge"], mastery, peak),
                "mastery": mastery,
                "mastery_peak": peak,
                "mastered": mastery is not None and mastery >= MASTERY_THRESHOLD,
                "gate_passed": peak >= GATE_THRESHOLD,
                "days_until_due": (
                    None
                    if row["t_last"] is None
                    else days_until_due(
                        int(row["s"] or 0), int(row["f"] or 0), row["t_last"], t_now, self.params
                    )
                ),
                "visited_count": int(row["visited_count"] or 0),
                "s": int(row["s"] or 0),
                "f": int(row["f"] or 0),
                "days_since_quiz": (
                    None if row["t_last"] is None else max(0, t_now - int(row["t_last"]))
                ),
            }
            concepts.append(entry)

            for node_id in row["ancestor_ids"] or []:
                if node_id is not None:
                    mapping.setdefault(node_id, []).append(entry)

        by_id = {e["id"]: e for e in concepts}
        for entry in concepts:
            missing = [
                by_id[p]["name"]
                for p in prerequisites.get(entry["id"], [])
                if p in by_id and not by_id[p]["gate_passed"]
            ]
            entry["prereqs_met"] = not missing
            entry["prereqs_missing"] = missing

        return {
            "student_id": student_id,
            "concepts": concepts,
            "rollup": {nid: _aggregate(entries) for nid, entries in mapping.items()},
            "summary": _aggregate(concepts),
        }


def _aggregate(entries: list[dict]) -> dict:
    """
    Counts statuses and mean mastery over a group of concept entries.

    ``mastery_avg`` averages over all concepts of the group and counts untested ones as 0,
    so that testing a single concept does not fake chapter progress.

    :param entries: Concept entries belonging to one hierarchy node.
    :return: Status counts and mean mastery for that node.
    """
    counter = {status: 0 for status in STATUSES}
    for e in entries:
        counter[e["status"]] += 1
    total = len(entries)
    return {
        "total": total,
        **counter,
        "gate_passed": sum(1 for e in entries if e["gate_passed"]),
        "mastery_avg": (sum(e["mastery"] or 0.0 for e in entries) / total) if total else 0.0,
    }
