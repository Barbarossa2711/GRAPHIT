from __future__ import annotations

import argparse
import json
import logging
import sys

from .concept_source import ConceptSource

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main() -> None:
    """
    Loads a concept and its slide texts from Neo4j and optionally generates questions from them.

    Usage: ``python -m Multiagent.GraphAccess <concept_id> [--describe | --generate]``.
    The generator is imported lazily so that ``--describe`` and loading work without an
    OpenAI key.
    """
    parser = argparse.ArgumentParser(description="Loads a concept and its slides from Neo4j.")
    parser.add_argument("concept_id", help="The concept id (e.g. CH02_T03_S01).")
    parser.add_argument("--describe", action="store_true",
                        help="Discovery: print raw concept/slide properties (no JSON access).")
    parser.add_argument("--generate", action="store_true",
                        help="Pass the loaded data directly to the QuestionGenerator.")
    args = parser.parse_args()

    with ConceptSource() as src:
        if args.describe:
            src.describe(args.concept_id)
            return

        concept, slides = src.load(args.concept_id)
        print(f"Concept: id={concept.id!r} name={concept.name!r}")
        print(f"objective: {concept.objective!r}")
        print(f"Slide texts: {len(slides)}")
        for i, text in enumerate(slides, start=1):
            preview = text.replace("\n", " ")[:120]
            print(f"  [slide {i}] {preview}{'…' if len(text) > 120 else ''}")

    if not args.generate:
        return

    if not slides:
        print("\nNo slide texts extracted, generation skipped.")
        return

    from Multiagent.QuestionGenerator import QuestionGenerator

    print("\n--- Generating questions ---")
    gen = QuestionGenerator()
    result = gen.generate(concept, slides)
    print(json.dumps(result.model_dump(exclude_none=True), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
