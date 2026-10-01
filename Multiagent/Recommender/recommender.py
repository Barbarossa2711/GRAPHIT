from __future__ import annotations

from pathlib import Path

from neo4j import GraphDatabase

from Multiagent.Assessment.mastery import (
    GATE_THRESHOLD,
    MASTERY_THRESHOLD,
    days_until_due,
    today_daynum,
)
from Multiagent.GraphAccess.config import GraphSchema, Neo4jConfig, load_config
from Multiagent.GraphAccess.domain_tree import MAX_DEPTH
from Multiagent.LearnerModel.learner_graph import LearnerSchema
from Multiagent.LearnerModel.progress import (
    concept_status as _status,
    effective_mastery as _effective_mastery,
    gate_passed as _gate_passed,
    peak_mastery as _peak,
)

__all__ = ["Recommender", "MASTERY_THRESHOLD", "GATE_THRESHOLD", "NEXT_STEPS_LIMIT"]

NEXT_STEPS_LIMIT = 5


class Recommender:
    """
    Recommends the next learnable concepts from prerequisites and learning state, deterministically and without an LLM; usable as a context manager.

    The PREREQUISITE gate uses ``mastery_peak`` (storage strength), which never decays; the
    review due state uses the decaying mastery (retrieval strength). A concept whose mastery
    dropped below the threshold therefore shows up as due for review but no longer blocks
    its follow-up concepts. ``(A)-[:PREREQUISITE]->(B)`` means B is a prerequisite of A,
    ``(A)-[:FACILITATOR]->(B)`` means B is helpful for A.
    """

    def __init__(
        self,
        config: Neo4jConfig | None = None,
        schema: GraphSchema | None = None,
        learner: LearnerSchema | None = None,
        env_path: str | Path | None = None,
        prerequisite_rel: str = "PREREQUISITE",
        facilitator_rel: str = "FACILITATOR",
    ) -> None:
        """
        Opens the Neo4j driver.

        :param config: Connection settings; loaded from the .env when omitted.
        :param schema: Label/property names of the domain graph.
        :param learner: Label/property names of the learner subgraph.
        :param env_path: Path of the ``.env`` used when ``config`` is omitted.
        :param prerequisite_rel: Relationship type of mandatory prerequisites.
        :param facilitator_rel: Relationship type of optional, helpful concepts.
        """
        self.config = config or load_config(env_path)
        self.schema = schema or GraphSchema()
        self.learner = learner or LearnerSchema()
        self.prerequisite_rel = prerequisite_rel
        self.facilitator_rel = facilitator_rel
        self._driver = GraphDatabase.driver(self.config.uri, auth=self.config.auth)

    def close(self) -> None:
        """
        Closes the Neo4j driver.
        """
        self._driver.close()

    def __enter__(self) -> "Recommender":
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

    def _fetch_prereq_graph(self, student_id: str, concept_id: str) -> dict[str, dict]:
        """
        Reads the target and its transitive prerequisites with mastery state and direct prerequisites.

        The stored ``mastery`` snapshot only serves as fallback for ``mastery_peak``; current
        mastery is computed on read from ``s``/``f``/``t_last``.

        :param student_id: Student whose mastery state is joined in.
        :param concept_id: Target concept whose prerequisites are collected.
        :return: Target and transitive prerequisites, keyed by concept id.
        """
        s, ls = self.schema, self.learner
        query = (
            f"MATCH (target:{s.concept_label} {{{s.concept_id_prop}: $cid}}) "
            f"OPTIONAL MATCH (target)-[:{self.prerequisite_rel}*0..]->(c:{s.concept_label}) "
            f"WITH DISTINCT c "
            f"OPTIONAL MATCH (st:{ls.student_label} {{{ls.student_id_prop}: $sid}})"
            f"-[r:{ls.learns_rel}]->(c) "
            f"OPTIONAL MATCH (c)-[:{self.prerequisite_rel}]->(dp:{s.concept_label}) "
            f"RETURN c.{s.concept_id_prop} AS id, c.{s.concept_name_prop} AS name, "
            f"r.s AS s, r.f AS f, r.t_last AS t_last, r.mastery AS mastery, "
            f"r.mastery_peak AS mastery_peak, (r IS NOT NULL) AS has_edge, "
            f"collect(DISTINCT dp.{s.concept_id_prop}) AS direct_prereqs"
        )
        with self._driver.session(database=self.config.database) as session:
            rows = session.run(query, cid=concept_id, sid=student_id).data()

        graph: dict[str, dict] = {}
        for row in rows:
            if row["id"] is None:
                continue
            graph[row["id"]] = {
                "name": row["name"],
                "s": row["s"],
                "f": row["f"],
                "t_last": row["t_last"],
                "mastery": row["mastery"],
                "mastery_peak": row["mastery_peak"],
                "has_edge": row["has_edge"],
                "direct_prereqs": [p for p in row["direct_prereqs"] if p is not None],
            }
        return graph

    def _fetch_facilitators(self, student_id: str, concept_id: str, t_now: int) -> list[dict]:
        """
        Reads the optional facilitator concepts of the target.

        :param student_id: Student whose mastery state is joined in.
        :param concept_id: Target concept whose facilitators are looked up.
        :param t_now: Day used to evaluate the decaying mastery.
        :return: Facilitators with their current state; never blocking.
        """
        s, ls = self.schema, self.learner
        query = (
            f"MATCH (target:{s.concept_label} {{{s.concept_id_prop}: $cid}})"
            f"-[:{self.facilitator_rel}]->(f:{s.concept_label}) "
            f"OPTIONAL MATCH (st:{ls.student_label} {{{ls.student_id_prop}: $sid}})"
            f"-[r:{ls.learns_rel}]->(f) "
            f"RETURN f.{s.concept_id_prop} AS id, f.{s.concept_name_prop} AS name, "
            f"r.s AS s, r.f AS f, r.t_last AS t_last, r.mastery AS mastery, "
            f"r.mastery_peak AS mastery_peak, (r IS NOT NULL) AS has_edge"
        )
        with self._driver.session(database=self.config.database) as session:
            rows = session.run(query, cid=concept_id, sid=student_id).data()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "status": _status(r["has_edge"], _effective_mastery(r, t_now), _peak(r)),
            }
            for r in rows
        ]

    def recommend(self, student_id: str, concept_id: str) -> dict:
        """
        Recommends what to learn next on the way to a target concept.

        Structure::

            {
              "target": {"id","name","status"},
              "ready_to_learn": [ {"id","name","status"} ],
              "blocked": [ {"id","name","status","missing_prereqs":[names]} ],
              "due_for_review": [ {"id","name","status","days_until_due"} ],
              "facilitators": [ {"id","name","status"} ],
            }

        ``ready_to_learn`` is the learnable front: concepts that have not passed the gate
        but whose direct prerequisites all have (empty if the target is mastered).
        ``blocked`` lists concepts with prerequisites that never passed the gate;
        ``due_for_review`` lists concepts that passed the gate but whose mastery decayed,
        most overdue first; they never block. Facilitators are optional hints.

        :param student_id: Student the recommendation is made for.
        :param concept_id: Target concept the student wants to reach.
        :return: ``target``, ``ready_to_learn``, ``blocked``, ``due_for_review`` and ``facilitators``.
        :raises ValueError: If the target concept does not exist.
        """
        graph = self._fetch_prereq_graph(student_id, concept_id)
        if concept_id not in graph:
            raise ValueError(
                f"Kein :{self.schema.concept_label} mit "
                f"{self.schema.concept_id_prop}={concept_id!r} gefunden."
            )

        t_now = today_daynum()
        eff: dict[str, float | None] = {cid: _effective_mastery(node, t_now) for cid, node in graph.items()}
        status: dict[str, str] = {
            cid: _status(node["has_edge"], eff[cid], _peak(node)) for cid, node in graph.items()
        }

        ready: list[dict] = []
        blocked: list[dict] = []
        due: list[dict] = []
        for cid, node in graph.items():
            entry = {"id": cid, "name": node["name"], "status": status[cid]}

            if status[cid] == "due_review":
                entry["days_until_due"] = days_until_due(
                    int(node["s"] or 0), int(node["f"] or 0), node["t_last"], t_now
                )
                due.append(entry)
                continue
            if _gate_passed(node):
                continue

            missing = [p for p in node["direct_prereqs"] if not _gate_passed(graph.get(p, {}))]
            if missing:
                entry["missing_prereqs"] = [graph.get(p, {}).get("name", p) for p in missing]
                blocked.append(entry)
            else:
                ready.append(entry)

        def _by_name(e: dict) -> str:
            """
            Returns the sort key that orders entries by their display name.

            :param e: Entry carrying a ``name``.
            :return: The name, or the id if the name is missing.
            """
            return e["name"] or e["id"]

        return {
            "target": {
                "id": concept_id,
                "name": graph[concept_id]["name"],
                "status": status[concept_id],
            },
            "ready_to_learn": sorted(ready, key=_by_name),
            "blocked": sorted(blocked, key=_by_name),
            "due_for_review": sorted(due, key=lambda e: e["days_until_due"]),
            "facilitators": self._fetch_facilitators(student_id, concept_id, t_now),
        }

    def _fetch_all_concepts(self, student_id: str) -> dict[str, dict]:
        """
        Reads all concepts of the corpus with learning state and direct prerequisites in one query.

        The learnable front only depends on direct prerequisites, so no transitive closure
        is needed.

        :param student_id: Student whose mastery state is joined in.
        :return: Every concept keyed by id, with learning state and direct prerequisites.
        """
        s, ls = self.schema, self.learner
        query = (
            f"MATCH (c:{s.concept_label}) "
            f"OPTIONAL MATCH (st:{ls.student_label} {{{ls.student_id_prop}: $sid}})"
            f"-[r:{ls.learns_rel}]->(c) "
            f"OPTIONAL MATCH (c)-[:{self.prerequisite_rel}]->(dp:{s.concept_label}) "
            f"RETURN c.{s.concept_id_prop} AS id, c.{s.concept_name_prop} AS name, "
            f"r.s AS s, r.f AS f, r.t_last AS t_last, r.mastery AS mastery, "
            f"r.mastery_peak AS mastery_peak, r.visited_count AS visited_count, "
            f"(r IS NOT NULL) AS has_edge, "
            f"collect(DISTINCT dp.{s.concept_id_prop}) AS direct_prereqs"
        )
        with self._driver.session(database=self.config.database) as session:
            rows = session.run(query, sid=student_id).data()

        graph: dict[str, dict] = {}
        for row in rows:
            if row["id"] is None:
                continue
            graph[row["id"]] = {
                "name": row["name"],
                "s": row["s"],
                "f": row["f"],
                "t_last": row["t_last"],
                "mastery": row["mastery"],
                "mastery_peak": row["mastery_peak"],
                "visited_count": int(row["visited_count"] or 0),
                "has_edge": row["has_edge"],
                "direct_prereqs": [p for p in row["direct_prereqs"] if p is not None],
            }
        return graph

    def _fetch_paths(self, concept_ids: list[str]) -> dict[str, str]:
        """
        Looks up the position in the selection tree as ``"Chapter > Topic > Subtopic"`` for the given concepts only.

        Variable depth, without lecture and concept name.

        :param concept_ids: The concepts to locate.
        :return: ``{concept_id: "Kapitel > Thema"}``; concepts without a path map to ``""``.
        """
        if not concept_ids:
            return {}
        s = self.schema
        structure = "|".join(
            (s.has_chapter_rel, s.has_topic_rel, s.has_subtopic_rel, s.has_concept_rel)
        )
        query = (
            f"UNWIND $ids AS cid "
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: cid}}) "
            f"OPTIONAL MATCH tree_path = shortestPath("
            f"  (l:{s.lecture_label})-[:{structure}*1..{MAX_DEPTH}]->(c)) "
            f"RETURN cid AS id, [n IN nodes(tree_path) | n.{s.concept_name_prop}] AS path"
        )
        with self._driver.session(database=self.config.database) as session:
            rows = session.run(query, ids=list(concept_ids)).data()
        return {
            r["id"]: " > ".join([n for n in (r["path"] or []) if n][1:-1]) for r in rows
        }

    def next_steps(self, student_id: str, limit: int = NEXT_STEPS_LIMIT) -> dict:
        """
        Recommends what to work on next across the whole corpus, without a target concept.

        Returns the learnable front: concepts that have not passed the gate but whose direct
        prerequisites all have, the same definition as :meth:`recommend`. Sorted by concept
        id, which reproduces the curriculum order because ids are hierarchical
        (``BDT_CH01_T03_S02_C02``); sorting by learning state was rejected because it breaks
        that order. Due reviews have their own view and are only counted. The default limit
        of 5 keeps the answer a recommendation rather than a second table of contents.

        Structure::

            {
              "student_id": str,
              "next": [ {"id","name","path","status","mastery","visited_count"} ],
              "next_total": int,
              "due_count": int,
              "mastered": int, "total": int,
            }

        An empty ``next`` with ``mastered == total`` means the lecture is complete.

        :param student_id: Student the recommendation is made for.
        :param limit: How many concepts to propose; ``<= 0`` returns the whole front.
        :return: The next concepts in curriculum order plus the surrounding counters.
        """
        graph = self._fetch_all_concepts(student_id)
        t_now = today_daynum()

        front: list[dict] = []
        due_count = 0
        mastered = 0
        for cid, node in graph.items():
            eff = _effective_mastery(node, t_now)
            status = _status(node["has_edge"], eff, _peak(node))
            if status == "due_review":
                due_count += 1
                continue
            if _gate_passed(node):
                mastered += 1
                continue
            if any(not _gate_passed(graph.get(p, {})) for p in node["direct_prereqs"]):
                continue
            front.append(
                {
                    "id": cid,
                    "name": node["name"] or cid,
                    "status": status,
                    "mastery": eff,
                    "visited_count": node["visited_count"],
                }
            )

        front.sort(key=lambda e: e["id"])
        suggestion = front[:limit] if limit and limit > 0 else front

        paths = self._fetch_paths([e["id"] for e in suggestion])
        for suggestion_entry in suggestion:
            suggestion_entry["path"] = paths.get(suggestion_entry["id"], "")

        return {
            "student_id": student_id,
            "next": suggestion,
            "next_total": len(front),
            "due_count": due_count,
            "mastered": mastered,
            "total": len(graph),
        }
