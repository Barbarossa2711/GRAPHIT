from __future__ import annotations

import logging
from pathlib import Path

from neo4j import GraphDatabase
from pydantic import BaseModel

from Multiagent.QuestionGenerator import Concept

from .config import GraphSchema, Neo4jConfig, load_config
from .domain_tree import MAX_DEPTH
from .slide_index import SlideIndex

logger = logging.getLogger(__name__)


class SlideRef(BaseModel):
    """
    Reference to a slide by source file name and page number; the content itself is not stored in the graph.
    """

    source: str
    page: int


class ConceptSource:
    """
    Loads concepts and their slide texts from Neo4j; usable as a context manager.

    Slide texts come from the merged chunks JSON (:class:`SlideIndex`), joined via the
    :Slide reference ``(source, pageNumber)``.
    """

    def __init__(
        self,
        config: Neo4jConfig | None = None,
        schema: GraphSchema | None = None,
        env_path: str | Path | None = None,
    ) -> None:
        """
        Opens the Neo4j driver; the slide index is loaded lazily.

        :param config: Connection settings; loaded from the .env when omitted.
        :param schema: Label/property names of the graph.
        :param env_path: Path of the ``.env`` used when ``config`` is omitted.
        """
        self.config = config or load_config(env_path)
        self.schema = schema or GraphSchema()
        self._driver = GraphDatabase.driver(self.config.uri, auth=self.config.auth)
        self._slide_index: SlideIndex | None = None

    @property
    def slide_index(self) -> SlideIndex:
        """
        Loads the slide content JSON on first access.

        :return: The slide index.
        """
        if self._slide_index is None:
            self._slide_index = SlideIndex(self.config.slide_content_json)
        return self._slide_index

    def close(self) -> None:
        """
        Closes the Neo4j driver.
        """
        self._driver.close()

    def __enter__(self) -> "ConceptSource":
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

    def fetch_concept(self, concept_id: str) -> Concept:
        """
        Reads the :Concept node and builds a ``Concept`` from it.

        :param concept_id: Concept to read.
        :return: The concept as a ``Concept`` object.
        :raises ValueError: If no such concept exists.
        """
        s = self.schema
        query = (
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $id}}) "
            "RETURN properties(c) AS props LIMIT 1"
        )
        with self._driver.session(database=self.config.database) as session:
            record = session.run(query, id=concept_id).single()

        if record is None:
            raise ValueError(
                f"No :{s.concept_label} with {s.concept_id_prop}={concept_id!r} found in database "
                f"{self.config.database!r}."
            )

        props = record["props"]
        return Concept(
            id=str(props.get(s.concept_id_prop, concept_id)),
            name=str(props.get(s.concept_name_prop) or concept_id),
            objective=props.get(s.concept_objective_prop),
        )

    def fetch_slide_refs(self, concept_id: str, include_same_as: bool = True) -> list[SlideRef]:
        """
        Reads the slide references of a concept, sorted by source and page.

        With ``include_same_as`` the slides of concepts linked via :SAME_AS (the same concept
        in another chapter) are included as well, undirected and deduplicated.

        :param concept_id: Concept whose slides are looked up.
        :param include_same_as: Whether slides reached via :SAME_AS count as well.
        :return: Slide references sorted by source and page.
        """
        s = self.schema
        if include_same_as:
            query = (
                f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $id}}) "
                f"OPTIONAL MATCH (c)-[:{s.same_as_rel}]-(alias:{s.concept_label}) "
                f"WITH collect(DISTINCT c) + collect(DISTINCT alias) AS concepts "
                f"UNWIND concepts AS cc "
                f"MATCH (sl:{s.slide_label})-[:{s.covers_rel}]->(cc) "
                f"RETURN DISTINCT sl.{s.slide_source_prop} AS source, sl.{s.slide_page_prop} AS page "
                f"ORDER BY source, page"
            )
        else:
            query = (
                f"MATCH (sl:{s.slide_label})-[:{s.covers_rel}]->(c:{s.concept_label} "
                f"{{{s.concept_id_prop}: $id}}) "
                f"RETURN sl.{s.slide_source_prop} AS source, sl.{s.slide_page_prop} AS page "
                f"ORDER BY source, page"
            )
        with self._driver.session(database=self.config.database) as session:
            rows = session.run(query, id=concept_id).data()

        refs: list[SlideRef] = []
        for row in rows:
            if row.get("source") is None or row.get("page") is None:
                logger.warning("Slide with missing source/page skipped: %s", row)
                continue
            refs.append(SlideRef(source=str(row["source"]), page=int(row["page"])))
        return refs

    def fetch_slides(self, concept_id: str) -> list[tuple[SlideRef, str]]:
        """
        Returns the slides of a concept that have content, as ``(reference, text)`` pairs.

        The reference is kept for the citation below the chat answer
        (:meth:`slide_citation`); slides without content are dropped, so only what the tutor
        actually received is cited.

        :param concept_id: Concept whose slides are assembled.
        :return: Pairs of slide reference and slide text, page-sorted.
        """
        pairs: list[tuple[SlideRef, str]] = []
        for ref in self.fetch_slide_refs(concept_id):
            text = self.slide_index.get_text(ref.source, ref.page)
            if text:
                pairs.append((ref, text))
            else:
                logger.warning(
                    "No content in the JSON for %s p.%s (join missed?).", ref.source, ref.page
                )
        return pairs

    def fetch_slides_text(self, concept_id: str) -> list[str]:
        """
        Returns only the slide texts of a concept.

        :param concept_id: Concept whose slide texts are assembled.
        :return: The slide texts, joined via (source, pageNumber).
        """
        return [text for _, text in self.fetch_slides(concept_id)]

    def slide_citation(self, ref: SlideRef) -> dict:
        """
        Builds the content-free citation of a slide: page, chapter and slide set version, never title or text.

        The source is taken from the graph reference, since the JSON may use another file
        extension for the same slide set.

        :param ref: The slide reference to describe.
        :return: ``{page, chapter, chapter_num, source, stand}`` for that slide.
        """
        meta = self.slide_index.get_meta(ref.source, ref.page)
        return {
            "page": ref.page,
            "chapter": meta.get("chapter") or "",
            "chapter_num": meta.get("chapter_num"),
            "source": ref.source,
            "stand": self.config.slide_version,
        }

    def fetch_co_occurring(self, concept_id: str, min_slides: int = 1) -> list[str]:
        """
        Returns the names of concepts that share slides with this one according to :CO_OCCURS.

        The generator excludes them in the planning prompt, otherwise it inevitably produces
        questions about the neighbour. Queried undirected; without the edges the list is empty.

        :param concept_id: Concept whose :CO_OCCURS neighbours are looked up.
        :param min_slides: Minimum number of shared slides a neighbour must have.
        :return: Names of the neighbouring concepts.
        """
        s = self.schema
        query = (
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $cid}})"
            f"-[r:CO_OCCURS]-(o:{s.concept_label}) "
            "WHERE r.slides >= $min_slides "
            f"RETURN o.{s.concept_name_prop} AS name ORDER BY r.slides DESC, name"
        )
        with self._driver.session(database=self.config.database) as session:
            return [r["name"] for r in session.run(query, cid=concept_id, min_slides=min_slides).data() if r["name"]]

    def load(self, concept_id: str) -> tuple[Concept, list[str]]:
        """
        Loads a concept together with its slide texts.

        :param concept_id: Concept to load.
        :return: The pair ``(concept, slide texts)`` used by ``QuestionGenerator.generate``.
        """
        return self.fetch_concept(concept_id), self.fetch_slides_text(concept_id)

    def find_mentioned(self, text: str, min_name_len: int = 4, limit: int = 5) -> list[dict]:
        """
        Finds concepts whose name occurs in the text, the reverse of :meth:`find_concept`.

        Compares against the name without its parenthesised suffix ("Apache Kafka" matches
        "Apache Kafka (Publish & Subscribe)"). Longest name first, as the more specific hit;
        ``min_name_len`` keeps out short tokens like "DB".

        :param text: Free text, typically the student's question.
        :param min_name_len: Shortest concept name considered; guards against false hits.
        :param limit: Maximum number of matches.
        :return: Matches as ``{id, name, path, link}``, most specific first.
        """
        s = self.schema
        structure = "|".join(
            (s.has_chapter_rel, s.has_topic_rel, s.has_subtopic_rel, s.has_concept_rel)
        )
        query_str = (
            f"MATCH (c:{s.concept_label}) "
            f"WITH c, CASE WHEN c.{s.concept_name_prop} CONTAINS ' (' "
            f"            THEN split(c.{s.concept_name_prop}, ' (')[0] "
            f"            ELSE c.{s.concept_name_prop} END AS kurz "
            f"WHERE size(kurz) >= $minlen AND toLower($text) CONTAINS toLower(kurz) "
            f"WITH c, kurz ORDER BY size(kurz) DESC LIMIT $limit "
            f"OPTIONAL MATCH tree_path = shortestPath("
            f"  (l:{s.lecture_label})-[:{structure}*1..{MAX_DEPTH}]->(c)) "
            f"RETURN c.{s.concept_id_prop} AS id, c.{s.concept_name_prop} AS name, "
            f"[n IN nodes(tree_path) | n.{s.concept_name_prop}] AS path"
        )
        with self._driver.session(database=self.config.database) as session:
            rows = session.run(
                query_str, text=text, minlen=min_name_len, limit=limit
            ).data()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "path": " > ".join([n for n in (r["path"] or []) if n][1:-1]),
                "link": f"[[{r['id']}|{r['name']}]]",
            }
            for r in rows
        ]

    def find_concept(self, query: str, limit: int = 10) -> list[dict]:
        """
        Finds concepts by case-insensitive substring of their name, shortest name first.

        ``path`` is the position in the selection tree as ``"Chapter > Topic > Subtopic"``
        (without lecture and concept), found as the shortest path of variable depth, so the
        tutor can refer to the right place. ``link`` is the ready-made chat link
        ``[[id|name]]``: models copy a given string far more reliably than assembling it.

        :param query: Substring searched in the concept name, case-insensitive.
        :param limit: Maximum number of matches.
        :return: Matches as ``{id, name, objective, path, link}``.
        """
        s = self.schema
        structure = "|".join(
            (s.has_chapter_rel, s.has_topic_rel, s.has_subtopic_rel, s.has_concept_rel)
        )
        query_str = (
            f"MATCH (c:{s.concept_label}) "
            f"WHERE toLower(c.{s.concept_name_prop}) CONTAINS toLower($q) "
            f"WITH c ORDER BY size(c.{s.concept_name_prop}) ASC LIMIT $limit "
            f"OPTIONAL MATCH tree_path = shortestPath("
            f"  (l:{s.lecture_label})-[:{structure}*1..{MAX_DEPTH}]->(c)) "
            f"RETURN c.{s.concept_id_prop} AS id, c.{s.concept_name_prop} AS name, "
            f"properties(c) AS props, "
            f"[n IN nodes(tree_path) | n.{s.concept_name_prop}] AS path"
        )
        with self._driver.session(database=self.config.database) as session:
            rows = session.run(query_str, q=query, limit=limit).data()

        entries = []
        for r in rows:
            path = [n for n in (r["path"] or []) if n][1:-1]
            entries.append(
                {
                    "id": r["id"],
                    "name": r["name"],
                    "objective": r["props"].get(s.concept_objective_prop),
                    "path": " > ".join(path),
                    "link": f"[[{r['id']}|{r['name']}]]",
                }
            )
        return entries

    def describe(self, concept_id: str) -> None:
        """
        Prints the concept node and its slide nodes with all raw properties, to verify the actual field names in the graph.

        :param concept_id: Concept to dump.
        """
        s = self.schema
        with self._driver.session(database=self.config.database) as session:
            crec = session.run(
                f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $id}}) "
                "RETURN properties(c) AS props LIMIT 1",
                id=concept_id,
            ).single()
            slides = session.run(
                f"MATCH (sl:{s.slide_label})-[:{s.covers_rel}]->(c:{s.concept_label} "
                f"{{{s.concept_id_prop}: $id}}) RETURN properties(sl) AS props",
                id=concept_id,
            ).data()

        print(f"=== Concept (:{s.concept_label} {s.concept_id_prop}={concept_id!r}) ===")
        if crec is None:
            print("  (not found)")
        else:
            for k, v in crec["props"].items():
                print(f"  {k}: {v!r}")

        print(f"\n=== Slides (:{s.slide_label})-[:{s.covers_rel}]->Concept ===  count: {len(slides)}")
        for i, row in enumerate(slides, start=1):
            print(f"  [{i}] " + ", ".join(f"{k}={v!r}" for k, v in row["props"].items()))
