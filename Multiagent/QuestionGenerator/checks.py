from __future__ import annotations

import re
import statistics
from collections import Counter

_PLACEHOLDER = re.compile(r"\{\{(\w+)\}\}")


def _norm(text: str) -> str:
    """
    Normalises text for comparison: collapses whitespace and lowercases.

    Punctuation is kept on purpose, since options like "10 <= k < 20" and "10 < k <= 20"
    differ only there.

    :param text: Text to normalise.
    :return: Normalised text.
    """
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def _ids(entries) -> set[str]:
    """
    Collects the ``id`` values of a list of payload entries.

    :param entries: Payload entries such as options, bank or left/right items.
    :return: The ids present in those entries.
    """
    return {e["id"] for e in entries or [] if isinstance(e, dict) and "id" in e}


def semantic_errors(qtype: str, payload: dict) -> list[str]:
    """
    Collects all blocking findings for an item, beyond what the JSON schema checks.

    The common ground is solvability: if the solution refers to something that does not
    exist or cannot be expressed in the frontend's answer format, no answer is graded as
    correct and the learner's ``f`` grows through no fault of their own. The messages go
    verbatim into the retry prompt and must say what to change.

    :param qtype: Question type (``single``, ``multiple``, ``cloze``, ``match``, ``order``).
    :param payload: Item payload to check.
    :return: Blocking findings; an empty list means the item is sound.
    """
    checker = {
        "single": _single_errors,
        "multiple": _multiple_errors,
        "cloze": _cloze_errors,
        "match": _match_errors,
        "order": _order_errors,
    }.get(qtype)
    return checker(payload) if checker else []


def _choice_references(payload: dict) -> list[str]:
    """
    Checks for single and multiple choice whether the solution points at existing options.

    :param payload: Item payload carrying options and solution.
    :return: A finding if the solution points at option ids that do not exist.
    """
    available = _ids(payload.get("options"))
    raw = payload.get("solution", {}).get("correct")
    chosen = [raw] if isinstance(raw, str) else list(raw or [])
    if missing_ := sorted(set(chosen) - available):
        return [
            f"Die Lösung verweist auf Options-IDs, die es nicht gibt: {missing_}. "
            f"Vorhanden sind {sorted(available)}."
        ]
    return []


def _duplicate_options(payload: dict) -> list[str]:
    """
    Detects options with identical wording, of which one would count as right and the other as wrong.

    :param payload: Item payload carrying the options.
    :return: A finding if two options share the same wording.
    """
    texts = [_norm(o.get("text", "")) for o in payload.get("options", [])]
    if duplicates := [t for t, n in Counter(texts).items() if n > 1 and t]:
        return [
            f"Es gibt mehrfach vorkommende Optionstexte ({duplicates[:2]}). Zwei gleiche "
            f"Optionen sind nicht unterscheidbar — formuliere sie inhaltlich verschieden."
        ]
    return []


def _single_errors(payload: dict) -> list[str]:
    """
    Blocking checks specific to single-choice items.

    :param payload: Item payload to check.
    :return: Findings for this item.
    """
    return _choice_references(payload) + _duplicate_options(payload)


def _multiple_errors(payload: dict) -> list[str]:
    """
    Blocking checks specific to multiple-choice items.

    :param payload: Item payload to check.
    :return: Findings for this item.
    """
    errors = _choice_references(payload) + _duplicate_options(payload)
    correct_ids = set(payload.get("solution", {}).get("correct", []))
    if len(correct_ids) < 2:
        errors.append(
            "multiple-Frage hat weniger als 2 korrekte Antworten. Erzeuge >= 2 korrekte "
            "Optionen oder formuliere die Frage so um, dass mehrere Antworten zutreffen."
        )
    if correct_ids and len(correct_ids) == len(payload.get("options", [])):
        errors.append(
            "Alle Optionen sind als korrekt markiert — es gibt keinen Distraktor. "
            "Ergänze mindestens eine fachlich falsche Option."
        )
    return errors


def _cloze_errors(payload: dict) -> list[str]:
    """
    Checks that text, ``blanks`` and ``solution`` refer to the same gaps and that the bank contains a distractor.

    :param payload: Item payload carrying text, blanks, bank and solution.
    :return: Findings if text, ``blanks`` and ``solution`` disagree about the gaps.
    """
    errors: list[str] = []
    in_text = set(_PLACEHOLDER.findall(payload.get("text", "")))
    blanks = set(payload.get("blanks") or [])
    solution = payload.get("solution") or {}
    keys = set(solution)

    if not (in_text == blanks == keys):
        errors.append(
            f"Die Lücken stimmen nicht überein — im Text {sorted(in_text)}, in 'blanks' "
            f"{sorted(blanks)}, in 'solution' {sorted(keys)}. Alle drei müssen dieselben "
            f"IDs enthalten; jeder Platzhalter {{{{id}}}} braucht genau einen Eintrag."
        )
    bank = _ids(payload.get("bank"))
    if missing_ := sorted({v for v in solution.values()} - bank):
        errors.append(
            f"Die Lösung verweist auf Begriffs-IDs, die in 'bank' fehlen: {missing_}."
        )
    elif bank and not (bank - set(solution.values())):
        errors.append(
            "Der Begriffs-Pool enthält keinen Distraktor — jeder Begriff ist Teil der "
            "Lösung. Ergänze mindestens einen plausiblen, aber falschen Begriff."
        )
    return errors


def _match_errors(payload: dict) -> list[str]:
    """
    Checks that a matching item assigns exactly one partner to each left element, using valid ids on both sides.

    The frontend returns the answer as a map ``leftId -> rightId``, so a solution with
    several partners per left element can never be answered correctly; the JSON schema
    does not catch this.

    :param payload: Item payload carrying both sides and the solution.
    :return: Findings if the assignment is not one partner per left-hand element.
    """
    solution = payload.get("solution")
    if not isinstance(solution, list):
        return []

    errors: list[str] = []
    concept_pairs = [m for m in solution if isinstance(m, dict) and "left" in m and "right" in m]

    counter = Counter(m["left"] for m in concept_pairs)
    if repeated := sorted(k for k, n in counter.items() if n > 1):
        errors.append(
            f"Die Lösung ordnet den Links-Elementen {repeated} jeweils mehrere "
            f"Rechts-Elemente zu. Erlaubt ist genau ein Partner je Links-Element. "
            f"Fasse die betroffenen Rechts-Elemente zu einer einzigen Aussage zusammen "
            f"oder verschiebe die überzähligen in die Distraktoren."
        )

    left_ids, right_ids = _ids(payload.get("left")), _ids(payload.get("right"))
    if unknown := sorted({m["left"] for m in concept_pairs} - left_ids):
        errors.append(f"Die Lösung verweist auf left-IDs, die es nicht gibt: {unknown}.")
    if unknown := sorted({m["right"] for m in concept_pairs} - right_ids):
        errors.append(f"Die Lösung verweist auf right-IDs, die es nicht gibt: {unknown}.")
    if missing_ := sorted(left_ids - {m["left"] for m in concept_pairs}):
        errors.append(
            f"Für die Links-Elemente {missing_} fehlt eine Zuordnung. Jedes Links-Element "
            f"braucht genau einen Partner; überzählige Rechts-Elemente sind Distraktoren."
        )
    return errors


def _order_errors(payload: dict) -> list[str]:
    """
    Checks that the solution is a permutation of the items, otherwise it cannot be reached.

    :param payload: Item payload carrying items and the expected order.
    :return: Findings if the solution is not a permutation of the items.
    """
    items = [o["id"] for o in payload.get("items", []) if isinstance(o, dict) and "id" in o]
    solution = list(payload.get("solution") or [])
    if sorted(solution) != sorted(items):
        missing_ = sorted(set(items) - set(solution))
        surplus = sorted(set(solution) - set(items))
        parts_ = []
        if missing_:
            parts_.append(f"in der Lösung fehlen {missing_}")
        if surplus:
            parts_.append(f"die Lösung nennt unbekannte IDs {surplus}")
        if not parts_:
            parts_.append("eine ID kommt mehrfach vor")
        return [
            "Die Lösung ist keine vollständige Reihenfolge der Items: "
            + ", ".join(parts_)
            + ". Sie muss jede Item-ID genau einmal enthalten."
        ]
    return []


LENGTH_RATIO_THRESHOLD = 1.6


_INTERNAL_IDS = re.compile(r"\b(?:[lrtbs][1-9])\b")

_META_VOCABULARY = re.compile(r"distraktor|item-budget|\bstem\b|isomorph", re.IGNORECASE)


def _visible_text(payload: dict) -> str:
    """
    Joins everything the learner gets to see, including the explanation shown after submission.

    Cloze placeholders like ``{{b1}}`` are removed, as they are legitimate there.

    :param payload: Item payload to read.
    :return: Everything the learner gets to see, including the explanation.
    """
    parts_ = [payload.get("prompt") or "", payload.get("explanation") or ""]
    if payload.get("type") == "cloze":
        parts_.append(_PLACEHOLDER.sub(" ", payload.get("text") or ""))
    for field_name in ("options", "bank", "left", "right", "items"):
        parts_ += [o.get("text", "") for o in payload.get(field_name) or [] if isinstance(o, dict)]
    return "\n".join(parts_)


def _leakage_warnings(payload: dict) -> list[str]:
    """
    Warns about traces of the generation process in student-visible text.

    Internal ids as assigned by the generator (l1/r1, t1, s1, b1) are matched case
    sensitively, since "L1-Cache" is a real term. Test construction vocabulary such as
    "Distraktor" is meant for the item author, not the learner.

    :param payload: Item payload to inspect.
    :return: Warnings about traces of the generation process in student-visible text.
    """
    text = _visible_text(payload)
    warnings = []
    if hits := sorted(set(_INTERNAL_IDS.findall(text))):
        warnings.append(
            f"Interne Bezeichner im sichtbaren Text: {hits}. Der Lernende sieht diese "
            f"IDs nie — benenne die Elemente stattdessen mit ihrem Text."
        )
    if hits := sorted({t.lower() for t in _META_VOCABULARY.findall(text)}):
        warnings.append(
            f"Vokabular der Testkonstruktion im sichtbaren Text: {hits}. Das richtet "
            f"sich an den Aufgabenersteller, nicht an den Lernenden."
        )
    return warnings


def mentioned_foreign_concepts(payload: dict, foreign_concepts: list[str] | None) -> list[str]:
    """
    Returns the neighbour concepts that are named in the visible text of an item.

    The only detection path, shared by :func:`foreign_concept_warnings` and the
    ``:REQUIRES`` annotation so that warning and quiz gating agree. Uses word boundaries
    (so "Volume" does not hit "Volumenmodell") and ignores case.

    :param payload: Item payload to inspect.
    :param foreign_concepts: Names of the neighbouring concepts to look for.
    :return: The neighbour names that occur in the visible text.
    """
    if not foreign_concepts:
        return []
    text = _visible_text(payload)
    return [
        name for name in foreign_concepts
        if name and re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE)
    ]


def foreign_concept_warnings(payload: dict, foreign_concepts: list[str] | None) -> list[str]:
    """
    Warns if an item seems to test a concept that does not belong to it.

    Only checked against the few :CO_OCCURS neighbours; matching against the whole corpus
    produces masses of false hits on generic names. Not blocking, since a neighbour may be
    named for contrast.

    :param payload: Item payload to inspect.
    :param foreign_concepts: Names of the neighbouring concepts to look for.
    :return: A non-blocking warning if the item names a neighbouring concept.
    """
    matched = mentioned_foreign_concepts(payload, foreign_concepts)
    if not matched:
        return []
    return [
        "Nennt das eigenständige Nachbarkonzept "
        + ", ".join(f"'{n}'" for n in matched)
        + " — prüfen, ob das Item noch das eigene Konzept testet oder bereits das fremde."
    ]


def quality_warnings(
    qtype: str, payload: dict, foreign_concepts: list[str] | None = None
) -> list[str]:
    """
    Collects non-blocking construction weaknesses, useful to evaluate a question pool.

    A correct option at least ``LENGTH_RATIO_THRESHOLD`` times as long as the distractors
    is flagged; in the measured pool the median ratio is 1.28, a systematic model tendency
    that belongs in the prompt rather than in a retry.

    :param qtype: Question type of the item.
    :param payload: Item payload to inspect.
    :param foreign_concepts: Names of neighbouring concepts, for the delimitation check.
    :return: Non-blocking findings about weak construction.
    """
    warnings: list[str] = _leakage_warnings(payload)
    warnings += foreign_concept_warnings(payload, foreign_concepts)
    if qtype == "single":
        if ratio := _length_ratio(payload):
            if ratio >= LENGTH_RATIO_THRESHOLD:
                warnings.append(
                    f"Die richtige Option ist {ratio:.1f}-mal so lang wie die "
                    f"Distraktoren im Mittel — die Länge verrät die Antwort."
                )
        if len(payload.get("options", [])) < 3:
            warnings.append(
                f"Nur {len(payload.get('options', []))} Optionen — die Ratewahrscheinlichkeit "
                f"liegt bei 50 %."
            )
    if qtype == "order" and len(payload.get("items", [])) < 3:
        warnings.append(
            "Nur 2 Schritte — es gibt lediglich zwei mögliche Reihenfolgen."
        )
    return warnings


def _length_ratio(payload: dict) -> float | None:
    """
    Computes the length of the correct option relative to the mean distractor length.

    :param payload: Item payload carrying the options.
    :return: Length of the correct option relative to the mean distractor length, or
        ``None`` if it cannot be computed.
    """
    correct_ids = payload.get("solution", {}).get("correct")
    option_list = payload.get("options", [])
    richtig = [o for o in option_list if o.get("id") == correct_ids]
    wrong = [o for o in option_list if o.get("id") != correct_ids]
    if not richtig or not wrong:
        return None
    mean_length = statistics.mean(len(o.get("text", "")) for o in wrong)
    return len(richtig[0].get("text", "")) / mean_length if mean_length else None
