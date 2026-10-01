from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Multiagent.QuestionGenerator import (
    Concept,
    QuestionGenerator,
    min_items,
    validate_question,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

CONCEPT = Concept(
    id="CH02_T03_S01",
    name="CAP-Theorem und NoSQL-Datenbanken",
    objective="Studierende verstehen das CAP-Theorem und ordnen NoSQL-Systeme ein.",
)

SLIDES = [
    "Das CAP-Theorem besagt, dass ein verteiltes System nicht gleichzeitig Consistency, "
    "Availability und Partition Tolerance vollständig garantieren kann. Bei einer "
    "Netzwerkpartition muss zwischen Konsistenz und Verfügbarkeit gewählt werden.",
    "NoSQL-Systeme lassen sich nach ihrem Datenmodell klassifizieren: Key-Value-Stores "
    "(Redis), Wide-Column-Stores (Cassandra, HBase), Dokumentenspeicher (MongoDB) und "
    "Graphdatenbanken (Neo4j).",
    "Cassandra ist ein AP-System (Availability + Partition Tolerance) und nutzt eine "
    "spaltenorientierte Architektur. HBase ist eher ein CP-System (Consistency + "
    "Partition Tolerance).",
    "Sharding verteilt Daten über mehrere Knoten (horizontale Skalierung), während "
    "Replikation Kopien der Daten vorhält, um Ausfallsicherheit und Leseskalierung zu "
    "erhöhen. Beide werden in verteilten NoSQL-Systemen kombiniert.",
]


EXISTING_QUESTIONS = [
    "Nennen Sie die drei Eigenschaften des CAP-Theorems und erklären Sie, warum bei einer "
    "Netzwerkpartition zwischen zwei davon gewählt werden muss.",
]


def main() -> None:
    """
    Generates a question set for a sample concept, prints it as JSON and re-validates every variant.

    Usage: ``python -m Multiagent.QuestionGenerator.example`` or run the file directly; the
    project root is put on the import path for the latter.
    """
    gen = QuestionGenerator()
    result = gen.generate(CONCEPT, SLIDES, existing_questions=EXISTING_QUESTIONS)

    print(json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False, indent=2))

    print("\n--- Summary ---")
    n_lecture = sum(1 for s in result.stems if s.source == "lecture")
    lo, hi = gen.slides_to_stem_range(result.n_slides)
    print(f"Slides: {result.n_slides}  ->  stem range: {lo}-{hi} "
          f"({n_lecture} of them from lecture questions) | generated stems: {len(result.stems)}")
    total = 0
    all_valid = True
    for i, stem in enumerate(result.stems, start=1):
        types = [q.type for q in stem.questions]
        print(f"  Stem {i} [{stem.source}] id={stem.id}: {len(stem.questions)} variants "
              f"{types} — {stem.objective}")
        for q in stem.questions:
            total += 1
            errs = validate_question(json.loads(q.payload))
            if errs:
                all_valid = False
                print(f"    ! Schema errors in '{q.id}': {errs}")
    target = min_items()
    print(f"Variants total: {total} | all schema-valid: {all_valid}")
    print(f"Item budget: {result.n_questions} of {target} required items "
          f"({'met' if result.n_questions >= target else 'NOT met'})")

    print("\n--- iter_questions() (ready for the graph import) ---")
    for rec in result.iter_questions():
        print(f"  {rec.question_id}  ({rec.type}, source={rec.source})  "
              f"-> payload {len(rec.payload)} characters")


if __name__ == "__main__":
    main()
