from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError

from Multiagent.Assessment.mastery import (
    GATE_THRESHOLD,
    MASTERY_THRESHOLD,
    MasteryParams,
    compute_mastery,
    days_until_due,
    today_daynum,
)
from Multiagent.Assessment.grading import grade_answer
from Multiagent.GraphAccess import ConceptSource
from Multiagent.GraphAccess.config import GraphSchema, Neo4jConfig, load_config
from Multiagent.LearnerModel import LearnerGraph
from Multiagent.LearnerModel.learner_graph import LearnerSchema
from Multiagent.QuestionGenerator import QuestionGenerator, min_items
from Multiagent.QuestionGenerator.question_generator import ProgressHook
from Multiagent.QuestionGenerator.models import QuestionSet
from Multiagent.QuestionStore import QuestionStore
from Multiagent.QuestionStore.question_scope import annotate_concept

logger = logging.getLogger(__name__)

ITEMS_PER_QUIZ = 1

MIN_EXCLUSIVE_QUESTIONS = 3
MIN_EXCLUSIVE_STEMS = 2

SHUFFLE_TRIES = 8


class QuizService:
    """
    Builds quizzes, grades answers deterministically and updates mastery; usable as a context manager.

    Request/response only: the caller keeps the quiz with solutions between :meth:`generate`
    and :meth:`submit`. Generated questions are read from the graph first and the LLM only
    runs if none exist or the stored set misses the item budget.
    """

    def __init__(
        self,
        config: Neo4jConfig | None = None,
        schema: GraphSchema | None = None,
        learner: LearnerSchema | None = None,
        env_path: str | Path | None = None,
        model: str | None = None,
        params: MasteryParams = MasteryParams(),
        question_label: str = "Question",
        question_text_prop: str = "text",
        question_index_prop: str = "index",
        tests_rel: str = "TESTS",
        topic_label: str = "Topic",
        subtopic_label: str = "Subtopic",
        has_subtopic_rel: str = "HAS_SUBTOPIC",
        has_concept_rel: str = "HAS_CONCEPT",
        store: QuestionStore | None | bool = None,
    ) -> None:
        """
        Opens the Neo4j driver and sets up concept source, learner graph, generator and store.

        :param config: Connection settings; loaded from the .env when omitted.
        :param schema: Label/property names of the domain graph.
        :param learner: Label/property names of the learner subgraph.
        :param env_path: Path of the ``.env`` used when ``config`` is omitted.
        :param model: LLM model; ``LLM_MODEL``/``LLM_BASE_URL`` from the .env when omitted.
        :param params: Mastery model parameters.
        :param question_label: Label of the curated lecture questions.
        :param question_text_prop: Text property of the lecture questions.
        :param question_index_prop: Sort property of the lecture questions.
        :param tests_rel: Relationship from a lecture question to its concept.
        :param topic_label: Label of topic nodes.
        :param subtopic_label: Label of subtopic nodes.
        :param has_subtopic_rel: Relationship from topic to subtopic.
        :param has_concept_rel: Relationship to a concept.
        :param store: Question store; ``None`` creates one, ``False`` disables storage.
        """
        self.config = config or load_config(env_path)
        self.schema = schema or GraphSchema()
        self.learner_schema = learner or LearnerSchema()
        self.params = params
        self.question_label = question_label
        self.question_text_prop = question_text_prop
        self.question_index_prop = question_index_prop
        self.tests_rel = tests_rel
        self.topic_label = topic_label
        self.subtopic_label = subtopic_label
        self.has_subtopic_rel = has_subtopic_rel
        self.has_concept_rel = has_concept_rel

        self._driver = GraphDatabase.driver(self.config.uri, auth=self.config.auth)
        self._concepts = ConceptSource(config=self.config, schema=self.schema)
        self._learner = LearnerGraph(config=self.config, schema=self.schema, learner=self.learner_schema)
        self._generator = QuestionGenerator(model=model, env_path=env_path)
        self.model = self._generator.model
        self._store = self._setup_store(store, self.config, self.schema, self._driver)

    @staticmethod
    def _setup_store(
        store: QuestionStore | None | bool,
        config: Neo4jConfig,
        schema: GraphSchema,
        driver,
    ) -> QuestionStore | None:
        """
        Sets up the question store unless disabled with ``store=False``; a created store shares this service's driver.

        :param store: Ready-made store; ``None`` creates one, ``False`` disables storage.
        :param config: Neo4j connection used for the store created here.
        :param schema: Label/property names of the question storage.
        :param driver: Already open driver that is shared.
        :return: Store instance, or ``None`` if storage is disabled.
        """
        if store is False:
            return None
        if isinstance(store, QuestionStore):
            return store
        return QuestionStore(config=config, schema=schema, driver=driver)

    def close(self) -> None:
        """
        Closes all drivers; the store shares this service's driver and does not close it itself.
        """
        self._driver.close()
        self._concepts.close()
        self._learner.close()

    def __enter__(self) -> "QuizService":
        """
        Enters the context manager.

        :return: This instance.
        """
        return self

    def __exit__(self, *exc) -> None:
        """
        Closes the drivers when leaving the context manager.

        :param exc: Exception information, ignored.
        """
        self.close()

    def _fetch_lecture_questions(self, concept_id: str) -> list[str]:
        """
        Reads the text of the curated lecture questions of a concept.

        :param concept_id: Concept whose lecture questions are looked up.
        :return: Texts of the :Question nodes, used as a template for the generator.
        """
        sc = self.schema
        query = (
            f"MATCH (q:{self.question_label})-[:{self.tests_rel}]->"
            f"(c:{sc.concept_label} {{{sc.concept_id_prop}: $cid}}) "
            f"WHERE q.{self.question_text_prop} IS NOT NULL "
            f"RETURN q.{self.question_text_prop} AS text "
            f"ORDER BY q.{self.question_index_prop}"
        )
        with self._driver.session(database=self.config.database) as session:
            return [r["text"] for r in session.run(query, cid=concept_id).data()]

    def _load_stored(self, concept_id: str) -> QuestionSet | None:
        """
        Loads the stored question set if it exists and meets the item budget.

        A set below the budget is discarded, since it could never lift the concept over the
        threshold. A set that cannot test the concept on its own (see
        :meth:`exclusive_coverage`) is only reported, not regenerated, so students never wait
        for the LLM before their first quiz.

        :param concept_id: Concept whose stored set is loaded.
        :return: Stored set, or ``None`` if absent or below the item budget.
        """
        if self._store is None:
            return None
        qset = self._store.load(concept_id)
        if qset is None:
            return None
        if qset.n_questions < min_items():
            logger.info(
                "Stored set of %s discarded: %d items below the budget of %d, regenerating.",
                concept_id, qset.n_questions, min_items(),
            )
            return None
        status = self.exclusive_coverage(concept_id)
        if not status["satisfied"]:
            logger.warning(
                "Concept %s cannot be tested in isolation: only %d exclusive questions from "
                "%d stems (required: %d/%d). Mastery can only be reached with questions "
                "that also test a neighbouring concept.",
                concept_id, status["questions"], status["stems"],
                MIN_EXCLUSIVE_QUESTIONS, MIN_EXCLUSIVE_STEMS,
            )
        logger.info("Questions for %s from the store (%d items).", concept_id, qset.n_questions)
        return qset

    def generate(
        self, concept_id: str, *, refresh: bool = False, progress: ProgressHook | None = None
    ) -> list[dict]:
        """
        Returns all items of a concept including solutions, generating and storing them if needed.

        Generation passes the :CO_OCCURS neighbour concepts to the generator so that it does
        not test them, then stores the set and annotates its :REQUIRES edges.

        :param concept_id: Concept the quiz is built for.
        :param refresh: ``True`` forces regeneration and replaces the stored set.
        :param progress: Optional progress hook; only used on actual generation.
        :return: All items of the concept including solutions —
            ``{question_id, stem_id, source, type, payload}``.
        """
        qset = None if refresh else self._load_stored(concept_id)

        if qset is None:
            concept, slides = self._concepts.load(concept_id)
            existing = self._fetch_lecture_questions(concept_id)
            foreign = self._concepts.fetch_co_occurring(concept_id)
            logger.info(
                "Quiz for %s: %d slides, %d lecture questions, %d neighbouring concepts to delimit.",
                concept_id, len(slides), len(existing), len(foreign),
            )
            qset = self._generator.generate(
                concept, slides, existing_questions=existing or None,
                foreign_concepts=foreign or None, progress=progress,
            )
            self._persist(qset)
            self._annotate_scope(concept_id)

        quiz: list[dict] = []
        for rec in qset.iter_questions():
            quiz.append(
                {
                    "question_id": rec.question_id,
                    "stem_id": rec.stem_id,
                    "source": rec.source,
                    "type": rec.type,
                    "payload": json.loads(rec.payload),
                }
            )
        return quiz

    def _persist(self, qset: QuestionSet) -> None:
        """
        Stores a freshly generated set; write errors are logged and never abort the quiz.

        :param qset: Freshly generated set to store.
        """
        if self._store is None:
            return
        try:
            self._store.save(qset, model=self.model)
        except Neo4jError as exc:
            logger.warning("Question set %s could not be stored: %s",
                           qset.concept_id, exc)

    _PRESENTATION_LISTS = {
        "cloze": ("bank",),
        "single": ("options",),
        "multiple": ("options",),
        "match": ("left", "right"),
        "order": ("items",),
    }

    @staticmethod
    def _reveals_solution(qtype_: str, payload: dict) -> bool:
        """
        Checks whether the presented order gives the answer away: the correct entries come first, in solution order.

        An unexpectedly shaped payload counts as not revealing.

        :param qtype_: Question type.
        :param payload: Payload including its ``solution``.
        :return: ``True`` if the presented order gives the answer away.
        """
        sol = payload.get("solution")
        try:
            if qtype_ == "cloze":
                expected = [(sol or {}).get(b) for b in payload.get("blanks") or []]
                bank = [b["id"] for b in payload.get("bank") or []]
                return bool(expected) and all(expected) and expected == bank[: len(expected)]
            if qtype_ in ("single", "multiple"):
                opts = [o["id"] for o in payload.get("options") or []]
                correct_ids_ = sol.get("correct") if isinstance(sol, dict) else sol
                correct_ids_ = correct_ids_ if isinstance(correct_ids_, list) else [correct_ids_]
                if not opts or not all(r in opts for r in correct_ids_):
                    return False
                slots = sorted(opts.index(r) for r in correct_ids_)
                return slots == list(range(len(slots)))
            if qtype_ == "match":
                right_ids = [x["id"] for x in payload.get("right") or []]
                expected = [p.get("right") for p in (sol or []) if isinstance(p, dict)]
                return bool(expected) and all(expected) and expected == right_ids[: len(expected)]
            if qtype_ == "order":
                items = [x["id"] if isinstance(x, dict) else x for x in payload.get("items") or []]
                expected = sol.get("order") if isinstance(sol, dict) else sol
                return bool(expected) and bool(items) and list(expected) == items[: len(expected)]
        except (AttributeError, TypeError, KeyError):
            return False
        return False

    @classmethod
    def shuffle_presentation(
        cls, quiz: list[dict], rng: random.Random | None = None
    ) -> list[dict]:
        """
        Shuffles the presented lists of each item without changing the stored question.

        The generator stores options, bank entries and match columns in solution order, so
        the presentation gave the answer away for up to 91 % of the items. Only the display
        order changes: ids and ``solution`` stay, and grading compares ids. Must be called
        before the run is cached, so the order stays stable for the whole run. Up to
        ``SHUFFLE_TRIES`` attempts avoid a revealing order; with two entries no order is
        safe, so the last attempt is accepted.

        :param quiz: Items including solutions.
        :param rng: Random source; a seeded instance makes the result reproducible.
        :return: A copy with shuffled presentation lists; input untouched.
        """
        rng = rng or random.Random()
        shuffled: list[dict] = []
        for item in quiz:
            copied = dict(item)
            payload = dict(item.get("payload") or {})
            field_names = cls._PRESENTATION_LISTS.get(item.get("type"), ())
            for field_name in field_names:
                field_values = payload.get(field_name)
                if not isinstance(field_values, list) or len(field_values) < 2:
                    continue
                for _ in range(SHUFFLE_TRIES):
                    new_list = list(field_values)
                    rng.shuffle(new_list)
                    payload[field_name] = new_list
                    if not cls._reveals_solution(item.get("type"), payload):
                        break
            copied["payload"] = payload
            shuffled.append(copied)
        return shuffled

    @staticmethod
    def strip_solutions(quiz: list[dict]) -> list[dict]:
        """
        Removes ``solution`` and ``explanation`` from every payload for the client.

        :param quiz: Items including solutions.
        :return: The same items without ``solution``/``explanation`` — the client-facing view.
        """
        client: list[dict] = []
        for q in quiz:
            payload = {k: v for k, v in q["payload"].items() if k not in ("solution", "explanation")}
            client.append({"question_id": q["question_id"], "type": q["type"], "payload": payload})
        return client

    def _annotate_scope(self, concept_id: str) -> None:
        """
        Writes the :REQUIRES edges for a freshly generated set, so that :meth:`allowed_items` can gate it.

        Failures are logged and never abort the quiz.

        :param concept_id: Concept whose freshly generated questions are annotated.
        """
        try:
            written = annotate_concept(self._driver, self.config, self.schema, concept_id)
            logger.info("Scope annotation for %s: %d :REQUIRES edges.", concept_id, written)
        except Neo4jError as exc:
            logger.warning("Scope annotation for %s failed: %s", concept_id, exc)

    def asked_history(self, student_id: str, concept_id: str) -> list[str]:
        """
        Returns the recently asked question ids, the input of the round-robin in :meth:`select_items`.

        :param student_id: Student whose history is read.
        :param concept_id: Concept the history refers to.
        :return: Previously asked question ids, oldest first.
        """
        return self._learner.asked_history(student_id, concept_id)

    def record_asked(
        self, student_id: str, concept_id: str, question_ids: list[str]
    ) -> None:
        """
        Records the questions just presented (see :meth:`LearnerGraph.record_asked`).

        :param student_id: Student the questions were presented to.
        :param concept_id: Concept of this run.
        :param question_ids: Ids of the questions just presented.
        """
        self._learner.record_asked(student_id, concept_id, question_ids)

    def exclusive_coverage(self, concept_id: str) -> dict:
        """
        Measures how much of the stored set tests this concept alone, i.e. without :REQUIRES edge.

        Satisfied means at least ``MIN_EXCLUSIVE_QUESTIONS`` (3, the correct answers needed
        for mastery) such questions from at least ``MIN_EXCLUSIVE_STEMS`` (2) stems, so the
        three proofs are not the same fact; 66 % of the concepts meet this. Only informs,
        the caller decides whether to regenerate.

        :param concept_id: Concept whose stored set is assessed.
        :return: ``{"questions", "stems", "satisfied"}`` — how many items carry no :REQUIRES edge
            and whether that suffices to reach mastery on this concept alone.
        """
        s, ls = self.schema, self.learner_schema  # noqa: F841
        query = (
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $cid}})"
            f"<-[:{s.tests_rel}]-(st:{s.stem_label})"
            f"-[:{s.has_variant_rel}]->(q:{s.generated_question_label}) "
            f"WHERE NOT EXISTS {{ (q)-[:REQUIRES]->(:{s.concept_label}) }} "
            f"RETURN count(q) AS questions, count(DISTINCT st) AS stems"
        )
        with self._driver.session(database=self.config.database) as session:
            rec = session.run(query, cid=concept_id).single()
        questions = int(rec["questions"] or 0) if rec else 0
        stems = int(rec["stems"] or 0) if rec else 0
        return {
            "questions": questions,
            "stems": stems,
            "satisfied": questions >= MIN_EXCLUSIVE_QUESTIONS and stems >= MIN_EXCLUSIVE_STEMS,
        }

    def allowed_items(self, quiz: list[dict], student_id: str) -> list[dict]:
        """
        Withholds items whose co-tested neighbour concepts (:REQUIRES) the student does not know yet.

        A neighbour counts as known if it was visited in the chat (``visited_count > 0``) or
        its ``mastery_peak`` passed the gate; both never expire. If nothing would be left,
        the unfiltered pool is returned, since some concepts have no question without a
        neighbour and would otherwise be untestable.

        :param quiz: Items of the concept to choose from.
        :param student_id: Student whose visit state decides what is released.
        :return: The permitted items; the untouched pool if everything would be withheld.
        """
        if not quiz:
            return quiz
        query = (
            f"UNWIND $qids AS qid "
            f"MATCH (q:{self.schema.generated_question_label} "
            f"        {{{self.schema.concept_id_prop}: qid}}) "
            f"OPTIONAL MATCH (q)-[:REQUIRES]->(c:{self.schema.concept_label}) "
            f"OPTIONAL MATCH (st:{self.learner_schema.student_label} "
            f"        {{{self.learner_schema.student_id_prop}: $sid}})"
            f"-[l:{self.learner_schema.learns_rel}]->(c) "
            f"WITH qid, c, coalesce(l.visited_count, 0) AS visited, "
            f"     coalesce(l.mastery_peak, 0.0) AS peak "
            f"WITH qid, collect(CASE WHEN c IS NOT NULL AND visited = 0 "
            f"                        AND peak < {GATE_THRESHOLD} "
            f"                       THEN c.{self.schema.concept_name_prop} END) AS roh "
            f"RETURN qid, [x IN roh WHERE x IS NOT NULL] AS missing"
        )
        qids = [item["question_id"] for item in quiz]
        with self._driver.session(database=self.config.database) as session:
            missing = {r["qid"]: r["missing"] for r in session.run(query, qids=qids, sid=student_id)}

        allowed = [item for item in quiz if not missing.get(item["question_id"])]
        if not allowed:
            logger.info(
                "All %d items require unvisited neighbouring concepts; the set is served "
                "unfiltered.", len(quiz),
            )
            return quiz
        if len(allowed) < len(quiz):
            logger.info("%d of %d items held back because of unvisited neighbouring concepts.",
                        len(quiz) - len(allowed), len(quiz))
        return allowed

    @staticmethod
    def select_items(
        quiz: list[dict],
        n: int = ITEMS_PER_QUIZ,
        *,
        asked: list[str] | None = None,
        rng: random.Random | None = None,
    ) -> list[dict]:
        """
        Selects the items of one run from the concept's pool.

        First, distinct stems: a stem is one fact and its variants are the same question in
        another format, so a second variant only comes once every stem is represented.
        Second, round-robin by ``asked``: never asked questions first, then the ones asked
        longest ago; ties are broken randomly, so shuffling happens before the stable sort.
        One item per run by default (``ITEMS_PER_QUIZ``), so mastery needs at least three runs.

        :param quiz: Already filtered pool of the concept.
        :param n: Number of items this run presents.
        :param asked: History of recently asked question ids, driving the round-robin.
        :param rng: Random source; injectable so tests stay deterministic.
        :return: The selected items, at most ``n``.
        """
        if n <= 0 or len(quiz) <= n:
            return list(quiz)

        r = rng or random.Random()
        last_seen: dict[str, int] = {}
        for position, qid in enumerate(asked or []):
            last_seen[qid] = position

        def recency(item: dict) -> int:
            """
            Returns the rank of a question in the history; lower means longer ago.

            :param item: A quiz item carrying ``question_id``.
            :return: Position of the last presentation; ``-1`` if never asked.
            """
            return last_seen.get(item["question_id"], -1)

        by_stem: dict[str, list[dict]] = {}
        for item in quiz:
            by_stem.setdefault(item["stem_id"], []).append(item)

        groups = list(by_stem.values())
        for group in groups:
            r.shuffle(group)
            group.sort(key=recency)
        r.shuffle(groups)
        groups.sort(key=lambda g: recency(g[0]))

        selected: list[dict] = []
        round_no = 0
        while len(selected) < n:
            next_up = [g[round_no] for g in groups if len(g) > round_no]
            if not next_up:
                break
            for item in next_up:
                selected.append(item)
                if len(selected) == n:
                    break
            round_no += 1
        return selected

    def submit(
        self, student_id: str, concept_id: str, quiz: list[dict], answers: dict
    ) -> dict:
        """
        Grades the answers deterministically and updates the concept's mastery.

        Every question counts equally: ``s += correct``, ``f += wrong``, ``t_last = today``;
        the snapshot is computed with ``dt = 0``. ``mastery_peak`` is updated monotonically
        in :meth:`LearnerGraph.write_mastery` and returned with the same value. Every
        answered question counts as activity, right or wrong. ``next_due_in_days`` uses
        :func:`days_until_due` with ``t_last = t_now``, so it matches the progress view.

        :param student_id: Student whose learning state is updated.
        :param concept_id: Concept being tested.
        :param quiz: The presented items including solutions (from the quiz cache).
        :param answers: ``question_id -> answer``; missing entries count as wrong.
        :return: Per-question result plus the new mastery state.
        """
        results: list[dict] = []
        n_correct = 0
        for q in quiz:
            qid = q["question_id"]
            ans = answers.get(qid)
            correct = bool(grade_answer(q["payload"], ans)) if ans is not None else False
            n_correct += int(correct)
            results.append(
                {
                    "question_id": qid,
                    "type": q["type"],
                    "correct": correct,
                    "your_answer": ans,
                    "solution": q["payload"].get("solution"),
                    "explanation": q["payload"].get("explanation"),
                }
            )

        n_total = len(quiz)
        n_wrong = n_total - n_correct

        prev = self._learner.get_state(student_id, concept_id) or {}
        new_s = int(prev.get("s") or 0) + n_correct
        new_f = int(prev.get("f") or 0) + n_wrong
        t_now = today_daynum()
        mastery = compute_mastery(new_s, new_f, t_now, t_now, self.params)
        self._learner.write_mastery(student_id, concept_id, new_s, new_f, t_now, mastery)
        self._learner.record_activity(student_id, n_total)

        peak = max(float(prev.get("mastery_peak") or 0.0), mastery)

        return {
            "concept_id": concept_id,
            "n_correct": n_correct,
            "n_total": n_total,
            "mastery": mastery,
            "mastered": mastery >= MASTERY_THRESHOLD,
            "mastery_peak": peak,
            "gate_passed": peak >= GATE_THRESHOLD,
            "next_due_in_days": days_until_due(new_s, new_f, t_now, t_now, self.params),
            "s": new_s,
            "f": new_f,
            "results": results,
        }

    def coupled_concepts(self, concept_id: str, student_id: str | None = None) -> list[dict]:
        """
        Lists the concepts treated on the same slides (:CO_OCCURS) with their visit state, for the hint before a quiz.

        :param concept_id: Concept whose :CO_OCCURS neighbours are looked up.
        :param student_id: Optional; adds whether each neighbour has been visited.
        :return: ``{id, name, shared_slides, visited}`` per coupled concept.
        """
        sc = self._concepts.schema
        query = (
            f"MATCH (c:{sc.concept_label} {{{sc.concept_id_prop}: $cid}})"
            f"-[r:CO_OCCURS]-(o:{sc.concept_label}) "
            "OPTIONAL MATCH (s:Student {id: $sid})-[l:LEARNS]->(o) "
            f"RETURN o.{sc.concept_id_prop} AS id, o.{sc.concept_name_prop} AS name, "
            "       r.slides AS shared_slides, "
            "       coalesce(l.visited_count, 0) AS visited_count "
            "ORDER BY shared_slides DESC, name"
        )
        with self._concepts._driver.session(database=self._concepts.config.database) as session:
            lines = session.run(query, cid=concept_id, sid=student_id or "").data()
        return [
            {
                "id": z["id"],
                "name": z["name"],
                "shared_slides": z["shared_slides"],
                "visited": (z["visited_count"] or 0) > 0,
            }
            for z in lines
        ]

    def resolve_concepts(
        self,
        *,
        concept_id: str | None = None,
        topic_id: str | None = None,
        student_id: str | None = None,
        only_unmastered: bool = True,
    ) -> list[dict]:
        """
        Resolves a quiz trigger into concrete concepts.

        Priority: explicit ``concept_id``, then ``topic_id`` via the hierarchy (variable path
        length), then the concepts the student has touched. ``only_unmastered`` filters by
        the decaying mastery, i.e. review selection, not the PREREQUISITE gate; untested
        concepts always count as unmastered.

        :param concept_id: Explicitly chosen concept (takes precedence).
        :param topic_id: Hierarchy node whose concepts are collected.
        :param student_id: Student whose learning state is used for filtering.
        :param only_unmastered: ``True`` drops concepts that are already mastered.
        :return: Testable concepts as ``{id, name}``.
        """
        sc, ls = self.schema, self.learner_schema
        if concept_id:
            rows = [{"id": concept_id, "name": None, "s": None, "f": None, "t_last": None}]
        elif topic_id:
            query = (
                f"MATCH (t:{self.topic_label} {{{sc.concept_id_prop}: $tid}})"
                f"-[:{self.has_subtopic_rel}|{self.has_concept_rel}*1..4]->"
                f"(c:{sc.concept_label}) "
                f"OPTIONAL MATCH (st:{ls.student_label} {{{ls.student_id_prop}: $sid}})"
                f"-[r:{ls.learns_rel}]->(c) "
                f"RETURN c.{sc.concept_id_prop} AS id, c.{sc.concept_name_prop} AS name, "
                f"r.s AS s, r.f AS f, r.t_last AS t_last"
            )
            with self._driver.session(database=self.config.database) as session:
                rows = session.run(query, tid=topic_id, sid=student_id).data()
        else:
            query = (
                f"MATCH (st:{ls.student_label} {{{ls.student_id_prop}: $sid}})"
                f"-[r:{ls.learns_rel}]->(c:{sc.concept_label}) "
                f"RETURN c.{sc.concept_id_prop} AS id, c.{sc.concept_name_prop} AS name, "
                f"r.s AS s, r.f AS f, r.t_last AS t_last"
            )
            with self._driver.session(database=self.config.database) as session:
                rows = session.run(query, sid=student_id).data()

        if not only_unmastered:
            return [{"id": r["id"], "name": r["name"]} for r in rows]

        t_now = today_daynum()
        out: list[dict] = []
        for r in rows:
            if r["t_last"] is None:
                out.append({"id": r["id"], "name": r["name"]})
                continue
            m = compute_mastery(int(r["s"] or 0), int(r["f"] or 0), r["t_last"], t_now, self.params)
            if m < MASTERY_THRESHOLD:
                out.append({"id": r["id"], "name": r["name"]})
        return out
