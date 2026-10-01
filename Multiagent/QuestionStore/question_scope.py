from __future__ import annotations

import argparse
import json
import logging
from collections import Counter

from neo4j import GraphDatabase

from Multiagent.GraphAccess.config import GraphSchema, Neo4jConfig, load_config
from Multiagent.QuestionGenerator.checks import mentioned_foreign_concepts

logger = logging.getLogger(__name__)

REQUIRES_REL = "REQUIRES"

_NEIGHBOURS = """
MATCH (c:{concept})-[:CO_OCCURS]-(o:{concept})
RETURN c.{idp} AS cid, collect({{id: o.{idp}, name: o.{namep}}}) AS neighbours
"""

_QUESTIONS = """
MATCH (c:{concept})<-[:{tests}]-(:{stem})-[:{variant}]->(q:{question})
RETURN c.{idp} AS cid, q.{idp} AS qid, q.payload AS payload
"""

_DELETE = "MATCH (:{question})-[r:{rel}]->(:{concept}) DELETE r"

_WRITE = """
UNWIND $pairs AS p
MATCH (q:{question} {{{idp}: p.qid}})
MATCH (o:{concept} {{{idp}: p.cid}})
MERGE (q)-[:{rel}]->(o)
"""


def collect_annotations(driver, config: Neo4jConfig, schema: GraphSchema) -> list[dict]:
    """
    Determines, read-only, which co-occurring neighbour concepts each generated question also tests.

    Detection uses :func:`~Multiagent.QuestionGenerator.checks.mentioned_foreign_concepts`,
    the same function behind the generator's quality warning, so warning and gating agree.
    Requires the :CO_OCCURS edges; without them there is nothing to annotate.

    :param driver: Open Neo4j driver.
    :param config: Connection settings (supplies the database name).
    :param schema: Label/property names of the graph.
    :return: One ``{"qid", "cid"}`` entry per edge to be written.
    """
    s = schema
    with driver.session(database=config.database) as session:
        neighbours = {
            r["cid"]: r["neighbours"]
            for r in session.run(
                _NEIGHBOURS.format(concept=s.concept_label, idp=s.concept_id_prop,
                                 namep=s.concept_name_prop)
            )
        }
        questions = session.run(
            _QUESTIONS.format(concept=s.concept_label, tests=s.tests_rel, stem=s.stem_label,
                           variant=s.has_variant_rel, question=s.generated_question_label,
                           idp=s.concept_id_prop)
        ).data()

    pairs: list[dict] = []
    for r in questions:
        candidates = neighbours.get(r["cid"]) or []
        if not candidates:
            continue
        names = [k["name"] for k in candidates]
        hits = set(mentioned_foreign_concepts(json.loads(r["payload"]), names))
        if not hits:
            continue
        for k in candidates:
            if k["name"] in hits:
                pairs.append({"qid": r["qid"], "cid": k["id"]})
    return pairs


def annotate_concept(driver, config: Neo4jConfig, schema: GraphSchema, concept_id: str) -> int:
    """
    Rewrites the ``(:GeneratedQuestion)-[:REQUIRES]->(:Concept)`` annotation for a single, freshly generated concept.

    Whether an item mentions a neighbour concept is only known after generation. Errors are
    handled by the caller: a missing annotation weakens gating but is no reason to discard
    a generated quiz.

    :param driver: Open Neo4j driver.
    :param config: Connection settings (supplies the database name).
    :param schema: Label/property names of the graph.
    :param concept_id: Concept whose questions are annotated.
    :return: Number of edges written.
    """
    s = schema
    with driver.session(database=config.database) as session:
        rec = session.run(
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $cid}})-[:CO_OCCURS]-"
            f"(o:{s.concept_label}) "
            f"RETURN collect({{id: o.{s.concept_id_prop}, name: o.{s.concept_name_prop}}}) "
            f"AS neighbours",
            cid=concept_id,
        ).single()
        candidates = (rec["neighbours"] if rec else None) or []
        if not candidates:
            return 0
        questions = session.run(
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $cid}})"
            f"<-[:{s.tests_rel}]-(:{s.stem_label})"
            f"-[:{s.has_variant_rel}]->(q:{s.generated_question_label}) "
            f"RETURN q.{s.concept_id_prop} AS qid, q.payload AS payload",
            cid=concept_id,
        ).data()

    names = [k["name"] for k in candidates]
    pairs = []
    for r in questions:
        hits = set(mentioned_foreign_concepts(json.loads(r["payload"]), names))
        pairs += [{"qid": r["qid"], "cid": k["id"]} for k in candidates if k["name"] in hits]

    with driver.session(database=config.database) as session:
        session.run(
            f"MATCH (c:{s.concept_label} {{{s.concept_id_prop}: $cid}})"
            f"<-[:{s.tests_rel}]-(:{s.stem_label})"
            f"-[:{s.has_variant_rel}]->(q:{s.generated_question_label}) "
            f"MATCH (q)-[r:{REQUIRES_REL}]->(:{s.concept_label}) DELETE r",
            cid=concept_id,
        )
        if pairs:
            session.run(
                _WRITE.format(question=s.generated_question_label, idp=s.concept_id_prop,
                                 concept=s.concept_label, rel=REQUIRES_REL),
                pairs=pairs,
            )
    return len(pairs)


def write_annotations(driver, config: Neo4jConfig, schema: GraphSchema, pairs: list[dict]) -> int:
    """
    Replaces all :REQUIRES edges with ``pairs`` (delete first, then write).

    Replacing instead of adding makes stale annotations disappear; otherwise a question
    would stay locked behind a concept it no longer mentions.

    :param driver: Open Neo4j driver.
    :param config: Connection settings (supplies the database name).
    :param schema: Label/property names of the graph.
    :param pairs: Edges to write, as produced by :func:`collect_annotations`.
    :return: Number of edges written.
    """
    s = schema
    with driver.session(database=config.database) as session:
        session.run(_DELETE.format(question=s.generated_question_label, rel=REQUIRES_REL,
                                    concept=s.concept_label))
        if pairs:
            session.run(
                _WRITE.format(question=s.generated_question_label, idp=s.concept_id_prop,
                                 concept=s.concept_label, rel=REQUIRES_REL),
                pairs=pairs,
            )
    return len(pairs)


def main() -> None:
    """
    Annotates all generated questions with the neighbour concepts they also test.

    Usage: ``python -m Multiagent.QuestionStore.question_scope [--dry-run]``.
    """
    parser = argparse.ArgumentParser(
        description="Annotates questions with the neighbouring concepts they also test (:REQUIRES)."
    )
    parser.add_argument("--dry-run", action="store_true", help="only report, write nothing")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    config, schema = load_config(), GraphSchema()
    driver = GraphDatabase.driver(config.uri, auth=config.auth)
    try:
        pairs = collect_annotations(driver, config, schema)
        per_question = Counter(p["qid"] for p in pairs)
        print(f"{len(per_question)} questions also test a neighbouring concept "
              f"({len(pairs)} :{REQUIRES_REL} edges).")
        distribution = Counter(per_question.values())
        for n in sorted(distribution):
            print(f"  {n} neighbouring concept(s): {distribution[n]} questions")
        if args.dry_run:
            print("--dry-run: nothing written.")
            return
        written = write_annotations(driver, config, schema, pairs)
        print(f"{written} :{REQUIRES_REL} edges written.")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
