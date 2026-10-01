from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from neo4j import GraphDatabase
from tqdm import tqdm

from Multiagent.GraphAccess.concept_source import ConceptSource
from Multiagent.GraphAccess.config import GraphSchema, load_config
from Multiagent.QuestionGenerator.checks import semantic_errors
from Multiagent.QuestionGenerator.question_generator import QuestionGenerator
from Multiagent.QuestionStore import QuestionStore

logger = logging.getLogger(__name__)


def findings(qset) -> list[tuple[str, str, list[str]]]:
    """
    Collects all items of a set flagged by :func:`semantic_errors`.

    :param qset: Question set to inspect.
    :return: Flagged items as ``(stem_id, question_id, messages)``.
    """
    removed = []
    for stem in qset.stems:
        for question in stem.questions:
            payload = json.loads(question.payload) if isinstance(question.payload, str) else question.payload
            if failures := semantic_errors(question.type, payload):
                removed.append((stem.id, question.id, failures))
    return removed


def other_prompts(stem, except_id: str) -> list[str]:
    """
    Collects the question texts of the other items of the same stem, so the regenerated item does not duplicate them.

    :param stem: Stem whose other items are collected.
    :param except_id: Item to leave out — the one being repaired.
    :return: Prompt texts of the remaining items, used as ``avoid``.
    """
    texts = []
    for question in stem.questions:
        if question.id == except_id:
            continue
        p = json.loads(question.payload) if isinstance(question.payload, str) else question.payload
        if text := (p.get("prompt") or p.get("text")):
            texts.append(text)
    return texts


def main(argv: list[str] | None = None) -> int:
    """
    Regenerates flagged items in place, replacing only their payload.

    Usage: ``python -m Multiagent.repair_questions [--dry-run] [--limit N]``. Ids, stems and
    all other items stay untouched: ``QuestionStore.save`` merges on the id and only removes
    what is missing from the set. Deleting the broken variants would not help, since the
    concept would still count as done, and ``--refresh`` would discard sound items too.
    The affected items are backed up first; an item that cannot be regenerated is left as it
    is, so the finding stays reproducible.

    :param argv: Argument list; ``sys.argv`` when omitted.
    :return: Process exit code.
    """
    p = argparse.ArgumentParser(
        description="Regenerates flagged items without changing the other stored questions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--database", "-d", default=None,
                   help="target database (default: NEO4J_DATABASE from .env)")
    p.add_argument("--dry-run", action="store_true",
                   help="only show what is flagged; generate and write nothing")
    p.add_argument("--limit", type=int, metavar="N",
                   help="repair at most N concepts")
    p.add_argument("--concept", "-c", nargs="+", metavar="ID",
                   help="check only these concepts")
    p.add_argument("--versuche", dest="attempts", type=int, default=2, metavar="N",
                   help="regeneration attempts per item (default: 2)")
    p.add_argument("--backup", default=None, metavar="PATH",
                   help="backup of the affected items (default: automatic file name)")
    p.add_argument("--verbose", "-v", action="store_true")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s: %(message)s")
    for name in ("neo4j", "neo4j.notifications", "httpx", "httpcore", "openai"):
        logging.getLogger(name).setLevel(logging.ERROR)

    config = load_config()
    if args.database:
        config = replace(config, database=args.database)
    schema = GraphSchema()
    driver = GraphDatabase.driver(config.uri, auth=config.auth)
    store = QuestionStore(config=config, schema=schema, driver=driver)
    sources = ConceptSource(config=config, schema=schema)

    with driver.session(database=config.database) as session:
        if args.concept:
            cids = args.concept
        else:
            cids = session.run(
                f"MATCH (:{schema.stem_label})-[:{schema.tests_rel}]->"
                f"(c:{schema.concept_label}) RETURN DISTINCT c.{schema.concept_id_prop} AS id "
                f"ORDER BY id").value()

    print(f"Database  : {config.database}")
    print(f"Checked   : {len(cids)} concepts with stored questions")

    todo: dict[str, list] = {}
    n_items = 0
    for cid in tqdm(cids, desc="Checking", unit="concept", leave=False):
        qset = store.load(cid)
        if qset is None:
            continue
        n_items += qset.n_questions
        if hits := findings(qset):
            todo[cid] = hits

    affected = sum(len(v) for v in todo.values())
    print(f"Items     : {n_items}")
    print(f"Flagged   : {affected} items in {len(todo)} concepts\n")
    if not todo:
        print("Nothing to repair.")
        driver.close()
        return 0

    for cid, hits in list(todo.items())[:50]:
        for _, qid, failures in hits:
            print(f"  {qid}\n     {failures[0][:130]}")

    if args.dry_run:
        driver.close()
        return 0

    targets = list(todo.items())[: args.limit] if args.limit else list(todo.items())
    path_ = Path(args.backup or
                f"repair_backup_{datetime.now():%Y%m%d_%H%M%S}.json")
    backup_items = []
    for cid, hits in targets:
        qset = store.load(cid)
        for stem in qset.stems:
            for question in stem.questions:
                if any(question.id == qid for _, qid, _ in hits):
                    backup_items.append({"concept_id": cid, "stem_id": stem.id,
                                      "id": question.id, "type": question.type,
                                      "payload": question.payload})
    path_.write_text(json.dumps(backup_items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nBackup    : {path_}  ({len(backup_items)} items)")

    generator = QuestionGenerator()
    print(f"Model     : {generator.model} "
          f"({generator.base_url or 'api.openai.com'})\n")

    replaced, failed = 0, []
    progress_bar = tqdm(targets, desc="Repairing", unit="concept", dynamic_ncols=True)
    for cid, hits in progress_bar:
        progress_bar.set_postfix_str(cid)
        qset = store.load(cid)
        concept, slides = sources.load(cid)
        changed = False

        for stem_id, qid, _ in hits:
            stem = next((s for s in qset.stems if s.id == stem_id), None)
            question = next((q for q in (stem.questions if stem else []) if q.id == qid), None)
            if question is None:
                failed.append((qid, "item not found in store"))
                continue

            updated = None
            for _ in range(max(1, args.attempts)):
                updated = generator.generate_variant(
                    concept, stem.objective, question.type, slides,
                    avoid=other_prompts(stem, qid),
                )
                if updated is not None:
                    break
            if updated is None:
                failed.append((qid, "regeneration produced no valid item"))
                continue

            question.payload = json.dumps(updated, ensure_ascii=False)
            changed = True
            replaced += 1
            tqdm.write(f"  ✓ {qid}")

        if changed:
            store.save(qset, model=generator.model)
    progress_bar.close()

    rest = 0
    for cid, _ in targets:
        qset = store.load(cid)
        if qset:
            rest += len(findings(qset))

    print(f"\n{replaced} items replaced, {len(failed)} failed")
    print(f"Recheck   : {rest} remaining findings in the processed concepts")
    for qid, reason in failed:
        print(f"  ! {qid}: {reason}")
    driver.close()
    return 1 if (failed or rest) else 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parents[1]))
    raise SystemExit(main())
