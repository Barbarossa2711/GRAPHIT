from __future__ import annotations

import logging
from pathlib import Path

from neo4j import GraphDatabase

from .config import GraphSchema, Neo4jConfig, load_config

logger = logging.getLogger(__name__)

_TYPE_FIELDS = [
    ("lecture_label", "lecture"),
    ("chapter_label", "chapter"),
    ("topic_label", "topic"),
    ("subtopic_label", "subtopic"),
    ("concept_label", "concept"),
]

MAX_DEPTH = 10


def nest_rows(rows: list[dict]) -> list[dict]:
    """
    Nests the path rows of the tree query into a tree, without database access so it can be tested standalone.

    Nodes are indexed by their full path, not by id, so a concept that appears in two
    chapters stays two nodes. Shorter paths are prefixes of longer ones and create no
    duplicates. A node without id ends its branch.

    :param rows: Path rows, each holding ``path`` as a root-to-leaf list of ``{id, name, type}``.
    :return: The paths nested into a tree, children sorted by id.
    """
    root_nodes: list[dict] = []
    index: dict[tuple, dict] = {}

    for row in rows:
        parent_ids: dict | None = None
        path_ids: tuple = ()
        for entry in row.get("path") or []:
            child_id = entry.get("id")
            if child_id is None:
                break
            path_ids = path_ids + (child_id,)
            node = index.get(path_ids)
            if node is None:
                node = {
                    "id": child_id,
                    "name": entry.get("name") or child_id,
                    "type": entry.get("type") or "concept",
                    "children": [],
                }
                index[path_ids] = node
                if parent_ids is None:
                    root_nodes.append(node)
                else:
                    parent_ids["children"].append(node)
            parent_ids = node

    _sort_tree(root_nodes)
    return root_nodes


def _sort_tree(node: list[dict]) -> None:
    """
    Sorts siblings recursively by ``id`` in place.

    The corpus ids are hierarchical (``BDT_CH01_T03_S02_C02``), so their lexicographic order
    is the curriculum order.

    :param node: Sibling list to sort; its subtrees are sorted as well.
    """
    node.sort(key=lambda n: n["id"])
    for n in node:
        _sort_tree(n["children"])


class DomainTree:
    """
    Reads the lecture hierarchy from Neo4j as a nested tree for the frontend's selection tree; usable as a context manager.

    The structure contains no learning state: it is static and cacheable, while progress is
    joined in the frontend via ``id``.
    """

    def __init__(
        self,
        config: Neo4jConfig | None = None,
        schema: GraphSchema | None = None,
        env_path: str | Path | None = None,
    ) -> None:
        """
        Opens the Neo4j driver.

        :param config: Connection settings; loaded from the .env when omitted.
        :param schema: Label/property names of the graph.
        :param env_path: Path of the ``.env`` used when ``config`` is omitted.
        """
        self.config = config or load_config(env_path)
        self.schema = schema or GraphSchema()
        self._driver = GraphDatabase.driver(self.config.uri, auth=self.config.auth)

    def close(self) -> None:
        """
        Closes the Neo4j driver.
        """
        self._driver.close()

    def __enter__(self) -> "DomainTree":
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

    def _structure_rels(self) -> str:
        """
        Joins the four structure relationships as a Cypher alternative (``A|B|C|D``).

        :return: Relationship types joined for a variable-length pattern.
        """
        s = self.schema
        return "|".join(
            (s.has_chapter_rel, s.has_topic_rel, s.has_subtopic_rel, s.has_concept_rel)
        )

    def _build_query(self) -> str:
        """
        Builds the path query over the structure relationships with variable depth.

        A fixed four-level chain misses 40 % of the concepts, because concepts may hang
        directly below a topic and subtopics may be nested. Depth starts at 0 so that empty
        branches still appear as empty nodes. Each hit returns its full path from the
        lecture; ``type`` comes from the node label, not from its depth.

        :return: The Cypher query producing one ``path`` list per hierarchy node.
        """
        s = self.schema
        type_cases = " ".join(
            f'WHEN "{getattr(s, field_name)}" IN labels(x) THEN "{node_type}"'
            for field_name, node_type in _TYPE_FIELDS
        )
        return (
            f"MATCH pfad = (l:{s.lecture_label})"
            f"-[:{self._structure_rels()}*0..{MAX_DEPTH}]->(n) "
            "RETURN [x IN nodes(pfad) | {"
            f"  id: x.{s.concept_id_prop}, name: x.{s.concept_name_prop}, "
            f"  type: CASE {type_cases} ELSE head(labels(x)) END"
            "}] AS path"
        )

    def fetch_tree(self) -> list[dict]:
        """
        Returns the hierarchy as a list of root nodes, usually exactly one lecture.

        Node format: ``{"id", "name", "type", "children"}`` with ``type`` one of
        ``lecture|chapter|topic|subtopic|concept``; a missing name falls back to the id.

        :return: Root nodes of the lecture hierarchy.
        """
        with self._driver.session(database=self.config.database) as session:
            rows = session.run(self._build_query()).data()
        return nest_rows(rows)

    def count_concepts(self, node_id: str) -> int:
        """
        Counts the concepts below any hierarchy node.

        :param node_id: Any hierarchy node.
        :return: Number of concepts below that node.
        """
        s = self.schema
        query = (
            f"MATCH (n {{{s.concept_id_prop}: $nid}}) "
            f"OPTIONAL MATCH (n)-[:{self._structure_rels()}*0..{MAX_DEPTH}]->"
            f"(c:{s.concept_label}) "
            f"RETURN count(DISTINCT c) AS n"
        )
        with self._driver.session(database=self.config.database) as session:
            record = session.run(query, nid=node_id).single()
        return int(record["n"]) if record else 0
