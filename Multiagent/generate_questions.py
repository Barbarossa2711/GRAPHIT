from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm

from Multiagent.GraphAccess.config import GraphSchema, load_config
from Multiagent.QuestionGenerator import min_items
from Multiagent.QuestionGenerator.question_generator import QuestionGenerator
from Multiagent.Quiz.quiz_service import QuizService

SECONDS_PER_CONCEPT = 27.0
SECONDS_PER_ITEM = 9.1
ITEMS_PER_STEM = 2.26


def format_duration(seconds: float) -> str:
    """
    Formats a duration as seconds, minutes or hours, whichever is readable.

    :param seconds: Duration in seconds.
    :return: The duration as a readable h/min/s string.
    """
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds/60:.0f}min"
    return f"{seconds/3600:.1f}h"


class ConceptProgress:
    """
    Translates the generator's progress events into a tqdm bar per concept.

    The target is unknown until ``("planned", n)`` arrives after stem planning.
    ``("warning", n)`` counts quality findings, not items, and must not advance the bar.
    """

    def __init__(self, name: str, position: int = 1) -> None:
        """
        Creates the progress bar without a total.

        :param name: Concept name shown as bar label.
        :param position: Line of the bar in the terminal.
        """
        self.bar = tqdm(total=None, desc=f"  {name[:34]:34}", unit="item",
                        position=position, leave=False, dynamic_ncols=True)
        self.items = 0
        self.warning_count = 0
        self.phase = ""

    def __call__(self, event: str, count: int) -> None:
        """
        Handles a progress event; the total grows if phase 3 fills beyond the plan.

        :param event: ``planned``, ``item``, ``budget`` or ``warning``.
        :param count: Number of units of the event.
        """
        if event == "planned":
            self.bar.total = count
            self.phase = "variants"
            self.bar.set_postfix_str(self.phase)
            self.bar.refresh()
            return
        if event == "warning":
            self.warning_count += count
            self._postfix()
            return
        self.items += count
        if self.bar.total is not None and self.items > self.bar.total:
            self.bar.total = self.items
        if event == "budget":
            self.phase = "budget"
            self._postfix()
        self.bar.update(count)

    def _postfix(self) -> None:
        """
        Shows the current phase and the number of quality findings next to the bar.
        """
        parts = [t for t in (self.phase, f"{self.warning_count} warnings" if self.warning_count else "") if t]
        self.bar.set_postfix_str(" · ".join(parts))
        self.bar.refresh()

    def close(self) -> None:
        """
        Closes the progress bar.
        """
        self.bar.close()


def select_concepts(driver, database: str, schema: GraphSchema, args) -> list[dict]:
    """
    Selects the concepts to process, with slide count, chapter and number of stored stems.

    Uses a variable path length from chapter to concept (2 to 5 in the corpus).

    :param driver: Open Neo4j driver.
    :param database: Database to query.
    :param schema: Label/property names of the graph.
    :param args: Parsed command line arguments controlling the selection.
    :return: Concepts that have slides, including slide and stem count.
    """
    conditions, params = [], {}
    if args.concept:
        conditions.append("c.id IN $cids")
        params["cids"] = args.concept
    if args.chapter:
        conditions.append("ch.index IN $chs")
        params["chs"] = args.chapter
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""

    query = f"""
        MATCH (ch:{schema.chapter_label})
              -[:{schema.has_topic_rel}|{schema.has_subtopic_rel}|{schema.has_concept_rel}*1..5]->
              (c:{schema.concept_label})
        {where}
        OPTIONAL MATCH (sl:Slide)-[:COVERS]->(c)
        OPTIONAL MATCH (st:{schema.stem_label})-[:{schema.tests_rel}]->(c)
        WITH ch, c, count(DISTINCT sl) AS slides, count(DISTINCT st) AS stems
        WHERE slides > 0
        RETURN c.id AS cid, c.name AS name, ch.index AS chapter, ch.name AS chapter_name,
               slides, stems
        ORDER BY ch.index, c.id
    """
    with driver.session(database=database) as session:
        return session.run(query, **params).data()


def estimate(concepts: list[dict]) -> tuple[int, float]:
    """
    Estimates items and run time for the selection, only a rough figure for ``--dry-run``.

    Based on a measured run over 8 concepts: about 27 s per concept for stem planning plus
    9 s per item, and 2.26 items per stem.

    :param concepts: Concepts selected for the run.
    :return: Rough estimate as ``(items, seconds)``.
    """
    items = 0
    for k in concepts:
        _, max_stems = QuestionGenerator.slides_to_stem_range(k["slides"])
        items += max(min_items(), round(ITEMS_PER_STEM * max_stems))
    return items, SECONDS_PER_CONCEPT * len(concepts) + SECONDS_PER_ITEM * items


def main(argv: list[str] | None = None) -> int:
    """
    Generates questions for many concepts, chapter by chapter, with two progress bars.

    Usage: ``python -m Multiagent.generate_questions (--chapter N … | --concept ID … | --all)
    [--dry-run] [--keep-going]``. A normal run only adds: concepts with stored questions are
    skipped and curated lecture questions are never touched. ``--refresh`` replaces stored
    sets, the only deleting path, and therefore also requires ``--i-know-what-i-am-doing``.

    :param argv: Argument list; ``sys.argv`` when omitted.
    :return: Process exit code.
    """
    p = argparse.ArgumentParser(
        description="Generates quiz questions for many concepts (chapter by chapter, with progress).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    selection = p.add_argument_group("selection (at least one)")
    selection.add_argument("--chapter", "-k", type=int, nargs="+", metavar="N",
                         help="chapter index/indices, e.g. --chapter 4 5")
    selection.add_argument("--concept", "-c", nargs="+", metavar="ID",
                         help="individual concept ids")
    selection.add_argument("--all", action="store_true",
                         help="all concepts with slides")

    run_group = p.add_argument_group("run")
    run_group.add_argument("--database", "-d", default=None,
                      help="target database (default: NEO4J_DATABASE from .env)")
    run_group.add_argument("--dry-run", action="store_true",
                      help="only show selection and estimate, generate nothing")
    run_group.add_argument("--refresh", action="store_true",
                      help="replace stored questions instead of skipping them")
    run_group.add_argument("--limit", type=int, metavar="N",
                      help="process at most N concepts (for testing)")
    run_group.add_argument("--keep-going", action="store_true",
                      help="continue on errors instead of aborting")
    run_group.add_argument("--i-know-what-i-am-doing", dest="confirmed", action="store_true",
                      help="required for --refresh (replaces stored questions)")
    run_group.add_argument("--verbose", "-v", action="store_true",
                      help="show the generator's log output")

    args = p.parse_args(argv)
    if not (args.chapter or args.concept or args.all):
        p.error("Please pass --chapter, --concept or --all.")
    if args.refresh and not args.confirmed:
        p.error("--refresh replaces stored stems/variants and is the only deleting "
                "path. Only together with --i-know-what-i-am-doing.")

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")
    for name in ("neo4j", "neo4j.notifications", "httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.ERROR)

    config = load_config()
    if args.database:
        config = replace(config, database=args.database)
    schema = GraphSchema()
    driver = GraphDatabase.driver(config.uri, auth=config.auth)

    concepts = select_concepts(driver, config.database, schema, args)
    if not concepts:
        print("No concept matches the selection.")
        driver.close()
        return 1

    existing_count = [k for k in concepts if k["stems"] > 0]
    open_items = concepts if args.refresh else [k for k in concepts if k["stems"] == 0]
    if args.limit:
        open_items = open_items[: args.limit]

    with driver.session(database=config.database) as session:
        curated = session.run(
            "MATCH (q:Question) RETURN count(q) AS n").single()["n"]

    n_items, seconds = estimate(open_items)
    print(f"Database     : {config.database}  "
          f"({curated} curated :Question nodes, left untouched)")
    print(f"Selection    : {len(concepts)} concepts with slides")
    print(f"  already done: {len(existing_count)}"
          f"{' (will be replaced)' if args.refresh else ' (will be skipped)'}")
    print(f"  to generate : {len(open_items)}")
    print(f"Estimate     : ~{n_items} items, ~{format_duration(seconds)}")

    per_chapter: dict[tuple, int] = {}
    for k in open_items:
        per_chapter[(k["chapter"], k["chapter_name"])] = per_chapter.get(
            (k["chapter"], k["chapter_name"]), 0) + 1
    if len(per_chapter) > 1:
        print("\nper chapter:")
        for (idx, name), n in sorted(per_chapter.items()):
            chapter_items = [k for k in open_items if k["chapter"] == idx]
            _, secs = estimate(chapter_items)
            print(f"  {idx:>2} {str(name)[:44]:44} {n:4} concepts  ~{format_duration(secs)}")

    if args.dry_run or not open_items:
        driver.close()
        return 0

    service = QuizService(config=config, schema=schema)
    print(f"\nModel        : {service._generator.model} "
          f"({service._generator.base_url or 'api.openai.com'})\n")

    done, failed, items_total = 0, [], 0
    t0 = time.time()
    outer_bar = tqdm(open_items, desc="Concepts", unit="concept", position=0, dynamic_ncols=True)
    for k in outer_bar:
        outer_bar.set_postfix_str(f"K{k['chapter']} {k['name'][:24]}")
        inner_bar = ConceptProgress(k["name"])
        try:
            quiz = service.generate(k["cid"], refresh=args.refresh, progress=inner_bar)
            items_total += len(quiz)
            done += 1
            if len(quiz) < min_items():
                tqdm.write(f"  ! {k['cid']}: only {len(quiz)} items "
                           f"(budget {min_items()} missed)")
            if inner_bar.warning_count:
                tqdm.write(f"  ~ {k['cid']}: {inner_bar.warning_count} quality warnings")
        except KeyboardInterrupt:
            inner_bar.close()
            tqdm.write("\nAborted. Concepts generated so far are stored.")
            break
        except Exception as exc:
            failed.append((k["cid"], f"{type(exc).__name__}: {exc}"))
            tqdm.write(f"  ✗ {k['cid']}: {type(exc).__name__}: {str(exc)[:120]}")
            if not args.keep_going:
                inner_bar.close()
                tqdm.write("Aborted (use --keep-going to continue).")
                break
        finally:
            inner_bar.close()
    outer_bar.close()

    elapsed = time.time() - t0
    print(f"\n{done} concepts, {items_total} items in {format_duration(elapsed)}")
    if done:
        print(f"Average      : {elapsed/done:.0f}s per concept, "
              f"{elapsed/max(1, items_total):.1f}s per item")
    if failed:
        print(f"\n{len(failed)} failed:")
        for cid, message in failed:
            print(f"  {cid}: {message}")
    service.close()
    driver.close()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parents[1]))
    raise SystemExit(main())
