from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

from neo4j import Driver, GraphDatabase
from neo4j.exceptions import Neo4jError

from Multiagent.QuestionGenerator.models import GeneratedStem, Question, QuestionSet

from ..GraphAccess.config import GraphSchema, Neo4jConfig, load_config

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 2

_WITHOUT_INDEX = 2147483647


class QuestionStore:
    """
    Persists generated question sets in the Neo4j graph; usable as a context manager.

    Model: ``(:Concept)<-[:TESTS]-(:QuestionStem)-[:HAS_VARIANT]->(:GeneratedQuestion)``.
    A stem is one knowledge component and all its variants update the same mastery
    estimate. ``payload`` is stored as a JSON string because its structure differs per
    question type and Neo4j has no nested property type. The curated lecture questions
    (label :Question) are never touched.
    """

    def __init__(
        self,
        config: Neo4jConfig | None = None,
        schema: GraphSchema | None = None,
        env_path: str | Path | None = None,
        *,
        driver: Driver | None = None,
    ) -> None:
        """
        Opens a Neo4j driver or shares the one of the caller.

        :param config: Connection settings; loaded from the .env when omitted.
        :param schema: Label/property names of the graph.
        :param env_path: Path of the ``.env`` used when ``config`` is omitted.
        :param driver: Driver of the caller (e.g. ``QuizService``); it is not closed by :meth:`close`.
        """
        self.config = config or load_config(env_path)
        self.schema = schema or GraphSchema()
        self._owns_driver = driver is None
        self._driver = driver or GraphDatabase.driver(self.config.uri, auth=self.config.auth)

    def close(self) -> None:
        """
        Closes the driver if this store created it.
        """
        if self._owns_driver:
            self._driver.close()

    def __enter__(self) -> "QuestionStore":
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

    def ping(self) -> bool:
        """
        Checks the connection without raising, so that the caller can degrade gracefully.

        :return: ``True`` if Neo4j is reachable.
        """
        try:
            self._driver.verify_connectivity()
            return True
        except Neo4jError as exc:
            logger.warning("Neo4j not reachable (%s): %s", self.config.uri, exc)
            return False

    def ensure_constraints(self) -> None:
        """
        Creates the unique constraints on the question labels (idempotent).

        Deliberately not done in the constructor, since it changes the schema.
        """
        s = self.schema
        statements = [
            f"CREATE CONSTRAINT stem_id IF NOT EXISTS "
            f"FOR (n:{s.stem_label}) REQUIRE n.id IS UNIQUE",
            f"CREATE CONSTRAINT generated_question_id IF NOT EXISTS "
            f"FOR (n:{s.generated_question_label}) REQUIRE n.id IS UNIQUE",
        ]
        with self._driver.session(database=self.config.database) as session:
            for statement in statements:
                session.run(statement)
        logger.info("Constraints on :%s and :%s ensured.",
                    s.stem_label, s.generated_question_label)

    def save(self, qset: QuestionSet, *, model: str | None = None) -> int:
        """
        Stores the question set at its concept, replacing the concept's previous set.

        Stems and variants missing from the new set are removed; ``created_at`` survives for
        unchanged ids. Everything runs in one transaction so no half-replaced set remains.

        :param qset: Question set to store.
        :param model: Name of the model that produced it, kept for provenance.
        :return: Number of records written.
        """
        stems = [
            {
                "id": stem.id,
                "objective": stem.objective,
                "source": stem.source,
                "questions": [
                    {"id": q.id, "type": q.type, "payload": q.payload} for q in stem.questions
                ],
            }
            for stem in qset.stems
        ]
        parameter = {
            "cid": qset.concept_id,
            "stems": stems,
            "stem_ids": [st["id"] for st in stems],
            "question_ids": [q["id"] for st in stems for q in st["questions"]],
            "n_slides": qset.n_slides,
            "model": model,
            "schema_version": SCHEMA_VERSION,
            "now": datetime.now(timezone.utc),
        }

        with self._driver.session(database=self.config.database) as session:
            written_count = session.execute_write(self._save_tx, self.schema, parameter)

        logger.info(
            "Question set %s stored: %d stems, %d items.",
            qset.concept_id, len(stems), written_count,
        )
        return written_count

    @staticmethod
    def _save_tx(tx, s: GraphSchema, p: dict) -> int:
        """
        Writes the set and removes stale variants and stems in one transaction.

        ``model`` is stored on every variant as well, so model comparisons need no join.

        :param tx: Open write transaction.
        :param s: Label/property names of the graph.
        :param p: Query parameters assembled by the caller.
        :return: Number of records written.
        :raises ValueError: If the concept does not exist, since MATCH would silently write nothing.
        """
        if tx.run(
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $cid}}) RETURN c LIMIT 1",
            cid=p["cid"],
        ).single() is None:
            raise ValueError(f"No :{s.concept_label} with id={p['cid']!r} in the graph.")

        tx.run(
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $cid}}) "
            "UNWIND $stems AS st "
            f"  MERGE (stem:{s.stem_label} {{id: st.id}}) "
            "    ON CREATE SET stem.created_at = $now "
            "    SET stem.objective = st.objective, stem.source = st.source, "
            "        stem.concept_id = $cid, stem.n_slides = $n_slides, stem.model = $model, "
            "        stem.schema_version = $schema_version, stem.updated_at = $now "
            f"  MERGE (stem)-[:{s.tests_rel}]->(c) "
            "  WITH stem, st "
            "  UNWIND st.questions AS q "
            f"    MERGE (v:{s.generated_question_label} {{id: q.id}}) "
            "      ON CREATE SET v.created_at = $now "
            "      SET v.type = q.type, v.payload = q.payload, v.stem_id = st.id, "
            "          v.model = $model, v.updated_at = $now "
            f"    MERGE (stem)-[:{s.has_variant_rel}]->(v)",
            **p,
        )

        tx.run(
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $cid}})"
            f"<-[:{s.tests_rel}]-(:{s.stem_label})"
            f"-[:{s.has_variant_rel}]->(v:{s.generated_question_label}) "
            "WHERE NOT v.id IN $question_ids "
            "DETACH DELETE v",
            cid=p["cid"], question_ids=p["question_ids"],
        )
        tx.run(
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $cid}})"
            f"<-[:{s.tests_rel}]-(stem:{s.stem_label}) "
            "WHERE NOT stem.id IN $stem_ids "
            "DETACH DELETE stem",
            cid=p["cid"], stem_ids=p["stem_ids"],
        )
        return len(p["question_ids"])

    def load(self, concept_id: str) -> QuestionSet | None:
        """
        Reads the question set of a concept.

        ``concept_name`` is taken from the :Concept node, so renaming the concept takes
        effect immediately.

        :param concept_id: Concept whose set is read.
        :return: The stored set, or ``None`` if the concept has none.
        """
        s = self.schema
        query = (
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $cid}})"
            f"<-[:{s.tests_rel}]-(stem:{s.stem_label}) "
            f"OPTIONAL MATCH (stem)-[:{s.has_variant_rel}]->(v:{s.generated_question_label}) "
            "WITH c, stem, v ORDER BY stem.id, v.id "
            "WITH c, stem, collect(v) AS vs "
            f"RETURN c.{s.concept_name_prop} AS concept_name, stem.id AS id, "
            "       stem.objective AS objective, stem.source AS source, "
            "       stem.n_slides AS n_slides, "
            "       [v IN vs | {id: v.id, type: v.type, payload: v.payload}] AS questions "
            "ORDER BY stem.id"
        )
        with self._driver.session(database=self.config.database) as session:
            lines = session.run(query, cid=concept_id).data()

        if not lines:
            return None
        return QuestionSet(
            concept_id=concept_id,
            concept_name=lines[0]["concept_name"] or concept_id,
            n_slides=lines[0]["n_slides"] or 0,
            stems=[
                GeneratedStem(
                    id=z["id"],
                    objective=z["objective"],
                    source=z["source"] or "generated",
                    questions=[
                        Question(id=q["id"], type=q["type"], payload=q["payload"])
                        for q in z["questions"]
                    ],
                )
                for z in lines
            ],
        )

    def delete(self, concept_id: str) -> int:
        """
        Removes the stems and variants of a concept; curated lecture questions are untouched.

        :param concept_id: Concept whose stems and variants are removed.
        :return: Number of deleted items.
        """
        s = self.schema
        query = (
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $cid}})"
            f"<-[:{s.tests_rel}]-(stem:{s.stem_label}) "
            f"OPTIONAL MATCH (stem)-[:{s.has_variant_rel}]->(v:{s.generated_question_label}) "
            "WITH collect(DISTINCT stem) AS stems, collect(DISTINCT v) AS vs "
            "WITH stems, [x IN vs WHERE x IS NOT NULL] AS vs "
            "WITH stems, vs, size(stems) AS n_stems, size(vs) AS n_questions "
            "FOREACH (x IN vs | DETACH DELETE x) "
            "FOREACH (x IN stems | DETACH DELETE x) "
            "RETURN n_stems, n_questions"
        )
        with self._driver.session(database=self.config.database) as session:
            record = session.run(query, cid=concept_id).single()

        if record is None or record["n_stems"] == 0:
            return 0
        logger.info(
            "Questions of %s deleted: %d stems, %d items.",
            concept_id, record["n_stems"], record["n_questions"],
        )
        return record["n_questions"]

    def stats(self) -> list[dict]:
        """
        Counts the stored items per chapter.

        The chapter-to-concept path has variable length (depth 2 to 5), a fixed path would
        miss about 40 % of the concepts. A concept under several chapters counts for the one
        with the smallest ``index``; concepts without chapter appear with ``chapter_id = None``.

        :return: Rows ``{chapter_id, chapter_name, chapter_index, n_concepts, n_stems, n_questions}``.
        """
        s = self.schema
        query = (
            f"MATCH (stem:{s.stem_label})-[:{s.tests_rel}]->(c:{s.concept_label}) "
            f"OPTIONAL MATCH (stem)-[:{s.has_variant_rel}]->(v:{s.generated_question_label}) "
            "WITH c, stem, count(v) AS nq "
            f"OPTIONAL MATCH (ch:{s.chapter_label})-[:{s.has_topic_rel}|{s.has_subtopic_rel}"
            f"|{s.has_concept_rel}*1..5]->(c) "
            f"WITH c, stem, nq, ch ORDER BY coalesce(ch.index, {_WITHOUT_INDEX}), ch.id "
            "WITH c, stem, nq, head(collect(ch)) AS ch "
            "WITH ch, count(DISTINCT c) AS n_concepts, count(DISTINCT stem) AS n_stems, "
            "     sum(nq) AS n_questions "
            f"RETURN ch.{s.concept_id_prop} AS chapter_id, ch.{s.concept_name_prop} AS chapter_name, "
            "       ch.index AS chapter_index, n_concepts, n_stems, n_questions "
            f"ORDER BY coalesce(chapter_index, {_WITHOUT_INDEX}), chapter_id"
        )
        with self._driver.session(database=self.config.database) as session:
            return session.run(query).data()

    def concepts_below_budget(self, min_n: int, limit: int = 100) -> list[dict]:
        """
        Lists concepts whose item count is below ``min_n``, the work list of the generator.

        Only when it is empty can every concept reach the mastery threshold; ``min_n`` comes
        from ``QuestionGenerator.min_items()``.

        :param min_n: Item count a concept must reach to be considered complete.
        :param limit: Maximum number of concepts to return.
        :return: Concepts whose item count is below ``min_n``, zero included.
        """
        s = self.schema
        query = (
            f"MATCH (c:{s.concept_label}) "
            f"OPTIONAL MATCH (c)<-[:{s.tests_rel}]-(:{s.stem_label})"
            f"-[:{s.has_variant_rel}]->(v:{s.generated_question_label}) "
            "WITH c, count(v) AS n "
            "WHERE n < $min_n "
            f"RETURN c.{s.concept_id_prop} AS concept_id, c.{s.concept_name_prop} AS concept_name, "
            "       n AS n_questions "
            "ORDER BY n DESC, concept_id LIMIT $limit"
        )
        with self._driver.session(database=self.config.database) as session:
            return session.run(query, min_n=min_n, limit=limit).data()
