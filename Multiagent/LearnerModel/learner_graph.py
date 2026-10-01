from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from neo4j import GraphDatabase

from Multiagent.Assessment.mastery import today_daynum
from Multiagent.GraphAccess.config import GraphSchema, Neo4jConfig, load_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LearnerSchema:
    """
    Label, relationship and property names of the learner subgraph ``(:Student)-[:LEARNS]->(:Concept)``.
    """

    student_label: str = "Student"
    student_id_prop: str = "id"
    learns_rel: str = "LEARNS"


class LearnerGraph:
    """
    Reads and writes a student's learning state in Neo4j; usable as a context manager.

    The :Student node is created lazily via MERGE; a single :LEARNS edge to the :Concept holds
    visit counter and mastery state. The chat server only calls :meth:`mark_visited`, the
    quiz service only :meth:`write_mastery`.
    """

    def __init__(
        self,
        config: Neo4jConfig | None = None,
        schema: GraphSchema | None = None,
        learner: LearnerSchema | None = None,
        env_path: str | Path | None = None,
    ) -> None:
        """
        Opens the Neo4j driver.

        :param config: Connection settings; loaded from the .env when omitted.
        :param schema: Label/property names of the domain graph.
        :param learner: Label/property names of the learner subgraph.
        :param env_path: Path of the ``.env`` used when ``config`` is omitted.
        """
        self.config = config or load_config(env_path)
        self.schema = schema or GraphSchema()
        self.learner = learner or LearnerSchema()
        self._driver = GraphDatabase.driver(self.config.uri, auth=self.config.auth)

    def close(self) -> None:
        """
        Closes the Neo4j driver.
        """
        self._driver.close()

    def __enter__(self) -> "LearnerGraph":
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

    def ensure_constraints(self) -> None:
        """
        Ensures uniqueness of ``:Student(id)`` (idempotent).

        Without the constraint ``MERGE (st:Student {id: $sid})`` is not atomic: two concurrent
        requests of the same student create two nodes, and every later ``.single()`` fails.
        """
        ls = self.learner
        query = (
            f"CREATE CONSTRAINT student_id IF NOT EXISTS "
            f"FOR (s:{ls.student_label}) REQUIRE s.{ls.student_id_prop} IS UNIQUE"
        )
        with self._driver.session(database=self.config.database) as session:
            session.run(query)

    def student_exists(self, student_id: str) -> bool:
        """
        Checks whether a :Student node exists for an identifier.

        :param student_id: Identifier to look up.
        :return: ``True`` if a node with this identifier exists.
        """
        ls = self.learner
        query = (
            f"MATCH (s:{ls.student_label} {{{ls.student_id_prop}: $sid}}) "
            f"RETURN count(s) > 0 AS present"
        )
        with self._driver.session(database=self.config.database) as session:
            rec = session.run(query, sid=student_id).single()
        return bool(rec["present"]) if rec else False

    def merge_student(self, student_id: str) -> None:
        """
        Creates the :Student node if it does not exist yet (idempotent).

        :param student_id: Student identifier (the hub user name).
        """
        ls = self.learner
        query = f"MERGE (s:{ls.student_label} {{{ls.student_id_prop}: $sid}})"
        with self._driver.session(database=self.config.database) as session:
            session.run(query, sid=student_id)

    def mark_visited(
        self, student_id: str, concept_id: str, session_key: str | None = None
    ) -> int:
        """
        Increments the visit counter of the :LEARNS edge, creating student and edge lazily.

        Only ``visited_count`` is touched. The counter is session based: if ``session_key``
        equals the key stored on the edge, it stays unchanged. The "new session?" decision is
        made in the same Cypher statement before the SET, so concurrent turns cannot both
        count. ``session_key=None`` counts every call. A new session is also recorded as one
        unit of activity.

        :param student_id: Student whose visit is recorded.
        :param concept_id: Concept that was visited.
        :param session_key: Chat session identifier; the same key does not count twice.
        :return: New value of ``visited_count``.
        :raises ValueError: If the concept does not exist.
        """
        ls, s = self.learner, self.schema
        query = (
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $cid}}) "
            f"MERGE (st:{ls.student_label} {{{ls.student_id_prop}: $sid}}) "
            f"MERGE (st)-[r:{ls.learns_rel}]->(c) "
            f"WITH r, CASE WHEN $session IS NOT NULL AND r.last_chat_session = $session "
            f"             THEN false ELSE true END AS neue_sitzung "
            f"SET r.visited_count = coalesce(r.visited_count, 0) + "
            f"  CASE WHEN neue_sitzung THEN 1 ELSE 0 END, "
            f"    r.last_chat_session = coalesce($session, r.last_chat_session) "
            f"RETURN r.visited_count AS visited_count, neue_sitzung AS neue_sitzung"
        )
        with self._driver.session(database=self.config.database) as session:
            record = session.run(
                query, sid=student_id, cid=concept_id, session=session_key
            ).single()

        if record is None:
            raise ValueError(
                f"No :{s.concept_label} with {s.concept_id_prop}={concept_id!r} found in database "
                f"{self.config.database!r}; visit not recorded."
            )
        if record["neue_sitzung"]:
            self.record_activity(student_id, 1)
        return int(record["visited_count"])

    ACTIVITY_HISTORY_DAYS = 400

    def record_activity(self, student_id: str, count: int = 1) -> None:
        """
        Records today's learning activity for the activity heatmap.

        Stored as two parallel lists on the :Student node, ``activity_days`` (ascending day
        numbers) and ``activity_counts``, since Neo4j has no map properties. If today is
        already the last entry, only its count grows; otherwise an entry is appended and the
        history is cut to ``ACTIVITY_HISTORY_DAYS`` (the heatmap shows 53 weeks). Done in
        Cypher so that concurrent quizzes cannot overwrite each other.

        :param student_id: Student whose activity is recorded.
        :param count: How much happened — answered questions, or 1 for a chat session.
        """
        if count <= 0:
            return
        ls = self.learner
        query = (
            f"MERGE (st:{ls.student_label} {{{ls.student_id_prop}: $sid}}) "
            "WITH st, coalesce(st.activity_days, []) AS tage, "
            "     coalesce(st.activity_counts, []) AS zahlen "
            "WITH st, tage, zahlen, size(tage) AS n "
            "WITH st, "
            "  CASE WHEN n > 0 AND tage[n-1] = $today THEN tage ELSE tage + [$today] END "
            "    AS neueTage, "
            "  CASE WHEN n > 0 AND tage[n-1] = $today "
            "       THEN zahlen[0..n-1] + [zahlen[n-1] + $count] "
            "       ELSE zahlen + [$count] END AS neueZahlen "
            "SET st.activity_days = CASE WHEN size(neueTage) > $keep "
            "                            THEN neueTage[size(neueTage) - $keep..] "
            "                            ELSE neueTage END, "
            "    st.activity_counts = CASE WHEN size(neueZahlen) > $keep "
            "                              THEN neueZahlen[size(neueZahlen) - $keep..] "
            "                              ELSE neueZahlen END"
        )
        with self._driver.session(database=self.config.database) as session:
            session.run(
                query, sid=student_id, today=today_daynum(),
                count=int(count), keep=self.ACTIVITY_HISTORY_DAYS,
            )

    def activity_history(self, student_id: str) -> list[dict]:
        """
        Returns the activity history as ``[{"date": "YYYY-MM-DD", "count": int}, …]``.

        Only days with activity are returned; the frontend fills the gaps.

        :param student_id: Student whose history is read.
        :return: Days with activity, oldest first.
        """
        ls = self.learner
        query = (
            f"MATCH (st:{ls.student_label} {{{ls.student_id_prop}: $sid}}) "
            "RETURN coalesce(st.activity_days, []) AS tage, "
            "       coalesce(st.activity_counts, []) AS zahlen"
        )
        with self._driver.session(database=self.config.database) as session:
            record = session.run(query, sid=student_id).single()
        if record is None:
            return []
        entries = []
        for day, count_value in zip(record["tage"], record["zahlen"]):
            entries.append(
                {"date": date.fromordinal(int(day)).isoformat(), "count": int(count_value)}
            )
        return entries

    ASKED_HISTORY_LIMIT = 60

    def record_asked(
        self, student_id: str, concept_id: str, question_ids: list[str]
    ) -> None:
        """
        Records which questions were just presented, as the basis of the round-robin in :meth:`QuizService.select_items`.

        The list is chronological (newest last) and capped at ``ASKED_HISTORY_LIMIT`` in
        Cypher, so concurrent quiz starts cannot overwrite each other. Correctness is not
        stored here but in ``s``/``f``.

        :param student_id: Student the questions were presented to.
        :param concept_id: Concept of this run.
        :param question_ids: Ids of the questions just presented, in presentation order.
        """
        if not question_ids:
            return
        ls, sc = self.learner, self.schema
        query = (
            f"MATCH (c:{sc.concept_label} {{{sc.concept_id_prop}: $cid}}) "
            f"MERGE (st:{ls.student_label} {{{ls.student_id_prop}: $sid}}) "
            f"MERGE (st)-[r:{ls.learns_rel}]->(c) "
            f"WITH r, coalesce(r.asked_ids, []) + $qids AS merged "
            f"SET r.asked_ids = CASE WHEN size(merged) > $keep "
            f"                       THEN merged[size(merged) - $keep..] "
            f"                       ELSE merged END"
        )
        with self._driver.session(database=self.config.database) as session:
            session.run(
                query, sid=student_id, cid=concept_id,
                qids=list(question_ids), keep=self.ASKED_HISTORY_LIMIT,
            )

    def asked_history(self, student_id: str, concept_id: str) -> list[str]:
        """
        Returns the most recently presented question ids.

        :param student_id: Student whose history is read.
        :param concept_id: Concept the history refers to.
        :return: Question ids, oldest first; empty if nothing was asked yet.
        """
        ls, sc = self.learner, self.schema
        query = (
            f"MATCH (st:{ls.student_label} {{{ls.student_id_prop}: $sid}})"
            f"-[r:{ls.learns_rel}]->(c:{sc.concept_label} {{{sc.concept_id_prop}: $cid}}) "
            f"RETURN coalesce(r.asked_ids, []) AS asked"
        )
        with self._driver.session(database=self.config.database) as session:
            rec = session.run(query, sid=student_id, cid=concept_id).single()
        return list(rec["asked"]) if rec else []

    def write_mastery(
        self,
        student_id: str,
        concept_id: str,
        s: int,
        f: int,
        t_last: int,
        mastery: float,
    ) -> None:
        """
        Writes the mastery state as absolute values to the :LEARNS edge.

        ``s``, ``f`` and ``mastery`` are computed by the caller (QuizService), so this class
        stays a pure graph writer; ``visited_count`` is untouched. ``mastery_peak``, which
        drives the PREREQUISITE gate, is updated monotonically inside Cypher so that it cannot
        be lost with concurrent writers.

        :param student_id: Student whose state is written.
        :param concept_id: Concept being tested.
        :param s: Accumulated correct answers.
        :param f: Accumulated wrong answers.
        :param t_last: Day number of the attempt.
        :param mastery: Pre-computed mastery snapshot.
        :raises ValueError: If the concept does not exist.
        """
        ls, sc = self.learner, self.schema
        query = (
            f"MATCH (c:{sc.concept_label} {{{sc.concept_id_prop}: $cid}}) "
            f"MERGE (st:{ls.student_label} {{{ls.student_id_prop}: $sid}}) "
            f"MERGE (st)-[r:{ls.learns_rel}]->(c) "
            f"SET r.s = $s, r.f = $f, r.t_last = $tl, r.mastery = $m, "
            f"    r.mastery_peak = CASE WHEN coalesce(r.mastery_peak, 0.0) < $m "
            f"                          THEN $m ELSE coalesce(r.mastery_peak, 0.0) END"
        )
        with self._driver.session(database=self.config.database) as session:
            summary = session.run(
                query, sid=student_id, cid=concept_id, s=s, f=f, tl=t_last, m=mastery
            ).consume()
        if summary.counters.relationships_created == 0 and summary.counters.properties_set == 0:
            raise ValueError(
                f"No :{sc.concept_label} with {sc.concept_id_prop}={concept_id!r} found; "
                f"mastery not written."
            )

    def get_state(self, student_id: str, concept_id: str) -> dict | None:
        """
        Returns a student's learning state for a concept.

        Mastery fields are ``None`` until a quiz was taken. ``mastery_peak`` falls back to
        ``mastery`` for edges written before the peak existed.

        :param student_id: Student whose state is read.
        :param concept_id: Concept the state refers to.
        :return: ``{visited_count, mastery, mastery_peak, s, f, t_last}``, or ``None`` if there is no edge.
        """
        ls, s = self.learner, self.schema
        query = (
            f"MATCH (st:{ls.student_label} {{{ls.student_id_prop}: $sid}})"
            f"-[r:{ls.learns_rel}]->(c:{s.concept_label} {{{s.concept_id_prop}: $cid}}) "
            f"RETURN r.visited_count AS visited_count, r.mastery AS mastery, "
            f"coalesce(r.mastery_peak, r.mastery) AS mastery_peak, "
            f"r.s AS s, r.f AS f, r.t_last AS t_last LIMIT 1"
        )
        with self._driver.session(database=self.config.database) as session:
            record = session.run(query, sid=student_id, cid=concept_id).single()
        return dict(record) if record is not None else None
