from __future__ import annotations

from collections.abc import Mapping, Sequence

from .models import GeneratedStem, Question, QuestionSet

__all__ = [
    "select_next_question",
    "select_question_order",
    "stem_round",
]


def stem_round(stem_id: str, seen: Mapping[str, int]) -> int:
    """
    Returns the round a stem is in, i.e. how often it was delivered so far.

    :param stem_id: Stem being examined.
    :param seen: How often each stem was delivered so far.
    :return: The round this stem is currently in.
    """
    return seen.get(stem_id, 0)


def _variant_index(stem: GeneratedStem, seen_variants: Mapping[str, int]) -> int:
    """
    Returns the index of the least used variant of a stem, which rotates as ``k mod n``.

    :param stem: Stem whose next variant is due.
    :param seen_variants: How often each variant was delivered so far.
    :return: Index of the variant to deliver next.
    """
    counts = [seen_variants.get(q.id, 0) for q in stem.questions]
    return counts.index(min(counts))


def select_next_question(
    question_set: QuestionSet,
    seen_stems: Mapping[str, int] | None = None,
    seen_variants: Mapping[str, int] | None = None,
    *,
    exclude_stem_ids: Sequence[str] = (),
) -> tuple[GeneratedStem, Question] | None:
    """
    Selects the next ``(stem, variant)`` to deliver for a concept using round-robin over the stems.

    A stem is one testable fact and all its variants update the same mastery estimate, so a
    second variant of a stem only comes up once every stem was asked; otherwise ``s`` would
    count one fact twice. The least asked stem wins, ties are broken deterministically by
    stem id. If all stems are excluded, the exclusion is lifted.

    :param question_set: The stored set of the concept.
    :param seen_stems: Delivery counts per stem id.
    :param seen_variants: Delivery counts per question id.
    :param exclude_stem_ids: Stems that must not be delivered now, e.g. the one just asked.
    :return: The ``(stem, question)`` to deliver, or ``None`` if nothing is left.
    """
    seen_stems = seen_stems or {}
    seen_variants = seen_variants or {}
    excluded = set(exclude_stem_ids)

    candidates = [
        s for s in question_set.stems if s.questions and s.id not in excluded
    ]
    if not candidates:
        candidates = [s for s in question_set.stems if s.questions]
    if not candidates:
        return None

    stem = min(candidates, key=lambda s: (stem_round(s.id, seen_stems), s.id))
    return stem, stem.questions[_variant_index(stem, seen_variants)]


def select_question_order(
    question_set: QuestionSet, n: int
) -> list[tuple[GeneratedStem, Question]]:
    """
    Simulates the next ``n`` deliveries under the round-robin rule, never repeating the previous stem while alternatives exist.

    :param question_set: The stored set of the concept.
    :param n: How many deliveries to simulate.
    :return: The next ``n`` ``(stem, question)`` pairs under the round-robin rule.
    """
    seen_stems: dict[str, int] = {}
    seen_variants: dict[str, int] = {}
    last_stem: str | None = None
    order: list[tuple[GeneratedStem, Question]] = []

    for _ in range(n):
        chosen = select_next_question(
            question_set,
            seen_stems,
            seen_variants,
            exclude_stem_ids=(last_stem,) if last_stem else (),
        )
        if chosen is None:
            break
        stem, question = chosen
        seen_stems[stem.id] = seen_stems.get(stem.id, 0) + 1
        seen_variants[question.id] = seen_variants.get(question.id, 0) + 1
        last_stem = stem.id
        order.append((stem, question))

    return order
