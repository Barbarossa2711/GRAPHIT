from __future__ import annotations

import argparse
import logging

from neo4j import GraphDatabase

from .config import GraphSchema, Neo4jConfig, load_config

logger = logging.getLogger(__name__)

CO_OCCURS_REL = "CO_OCCURS"

_COUNT = """
MATCH (s:{slide})-[:{covers}]->(a:{concept}), (s)-[:{covers}]->(b:{concept})
WHERE a.{idp} < b.{idp}
WITH a, b, count(DISTINCT s) AS shared
RETURN a.{idp} AS a, b.{idp} AS b, shared
ORDER BY shared DESC, a, b
"""

_WRITE = """
UNWIND $pairs AS p
MATCH (a:{concept} {{{idp}: p.a}}), (b:{concept} {{{idp}: p.b}})
MERGE (a)-[r:{rel}]->(b)
SET r.slides = p.shared
RETURN count(r) AS written
"""


def concept_pairs(config: Neo4jConfig, schema: GraphSchema | None = None) -> list[dict]:
    """
    Reads all concept pairs that share at least one slide.

    Each pair is returned once (``a.id < b.id``); no minimum count is applied, since even a
    single shared slide is often a real link.

    :param config: Neo4j connection settings.
    :param schema: Label/property names of the graph.
    :return: All concept pairs sharing at least one slide, with the shared count.
    """
    s = schema or GraphSchema()
    query = _COUNT.format(
        slide=s.slide_label, covers=s.covers_rel, concept=s.concept_label, idp=s.concept_id_prop
    )
    with GraphDatabase.driver(config.uri, auth=config.auth) as driver:
        with driver.session(database=config.database) as session:
            return session.run(query).data()


def write_edges(config: Neo4jConfig, found: list[dict], schema: GraphSchema | None = None) -> int:
    """
    Creates the ``(:Concept)-[:CO_OCCURS {slides: n}]->(:Concept)`` edges or updates ``slides``.

    One edge per pair is stored, so the relation has to be queried undirected. Uses MERGE,
    so reruns are harmless; stale edges are never deleted.

    :param config: Neo4j connection settings.
    :param found: Concept pairs as produced by :func:`concept_pairs`.
    :param schema: Label/property names of the graph.
    :return: Number of edges created or updated.
    """
    if not found:
        return 0
    s = schema or GraphSchema()
    query = _WRITE.format(concept=s.concept_label, idp=s.concept_id_prop, rel=CO_OCCURS_REL)
    with GraphDatabase.driver(config.uri, auth=config.auth) as driver:
        with driver.session(database=config.database) as session:
            return session.run(query, pairs=found).single()["written"]


def main() -> None:
    """
    Derives the :CO_OCCURS edges between concepts from shared slides.

    Usage: ``python -m Multiagent.GraphAccess.co_occurrence [--dry-run]``.
    """
    ap = argparse.ArgumentParser(
        description="Derives the :CO_OCCURS edges between concepts from shared slides."
    )
    ap.add_argument("--dry-run", action="store_true", help="only report, write nothing")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config = load_config()
    found = concept_pairs(config)
    distribution: dict[int, int] = {}
    for p in found:
        distribution[p["shared"]] = distribution.get(p["shared"], 0) + 1

    logger.info("%d concept pairs share at least one slide.", len(found))
    for n in sorted(distribution):
        logger.info("  %d shared slide(s): %d pairs", n, distribution[n])

    if args.dry_run:
        logger.info("\n--dry-run: nothing written. Strongest couplings:")
        for p in found[:10]:
            logger.info("  %2d  %s <-> %s", p["shared"], p["a"], p["b"])
        return

    count = write_edges(config, found)
    logger.info("%d :%s edges created or updated.", count, CO_OCCURS_REL)


if __name__ == "__main__":
    main()
