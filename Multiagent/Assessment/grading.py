from __future__ import annotations

from typing import Any


def _grade_single(payload: dict, answer: Any) -> bool:
    """
    Grades a single-choice answer.

    :param payload: Item payload carrying options and solution.
    :param answer: The student's answer (an option id).
    :return: ``True`` if the chosen option is the correct one.
    """
    return isinstance(answer, str) and answer == payload["solution"]["correct"]


def _grade_multiple(payload: dict, answer: Any) -> bool:
    """
    Grades a multiple-choice answer; the set must match exactly.

    :param payload: Item payload carrying options and solution.
    :param answer: The student's answer (a collection of option ids).
    :return: ``True`` if exactly the correct options were selected.
    """
    if not isinstance(answer, (list, tuple, set)):
        return False
    return set(answer) == set(payload["solution"]["correct"])


def _grade_cloze(payload: dict, answer: Any) -> bool:
    """
    Grades a cloze answer; every blank must be filled correctly.

    :param payload: Item payload whose solution maps blank id -> term id.
    :param answer: The student's answer as ``blank id -> term id``.
    :return: ``True`` if all blanks match the solution.
    """
    if not isinstance(answer, dict):
        return False
    return dict(answer) == dict(payload["solution"])


def _pairs_from_match(value: Any) -> set[tuple[str, str]] | None:
    """
    Normalises a match answer or solution to a set of ``(left, right)`` pairs.

    :param value: A match answer or solution in any of the accepted shapes.
    :return: The pairs as a set, or ``None`` if the shape is not understood.
    """
    if isinstance(value, dict):
        return {(str(k), str(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        pairs: set[tuple[str, str]] = set()
        for item in value:
            if not isinstance(item, dict) or "left" not in item or "right" not in item:
                return None
            pairs.add((str(item["left"]), str(item["right"])))
        return pairs
    return None


def _grade_match(payload: dict, answer: Any) -> bool:
    """
    Grades a matching answer; all pairs must be correct.

    :param payload: Item payload carrying both sides and the solution.
    :param answer: The student's answer as ``left id -> right id``.
    :return: ``True`` if the pairs match the solution exactly.
    """
    sol = _pairs_from_match(payload["solution"])
    ans = _pairs_from_match(answer)
    return ans is not None and sol is not None and ans == sol


def _grade_order(payload: dict, answer: Any) -> bool:
    """
    Grades an ordering answer; the sequence must match exactly.

    :param payload: Item payload carrying items and the expected order.
    :param answer: The student's answer as an ordered list of item ids.
    :return: ``True`` if the order matches the solution.
    """
    if not isinstance(answer, (list, tuple)):
        return False
    return list(answer) == list(payload["solution"])


_GRADERS = {
    "single": _grade_single,
    "multiple": _grade_multiple,
    "cloze": _grade_cloze,
    "match": _grade_match,
    "order": _grade_order,
}


def grade_answer(payload: dict, answer: Any) -> bool:
    """
    Grades a student answer deterministically against the solution in the payload.

    Only ids are compared, the result is binary. Expected answer formats per type:
    ``single`` str, ``multiple`` list[str], ``cloze`` dict[str, str],
    ``match`` dict[str, str] or list of ``{"left", "right"}`` dicts, ``order`` list[str].

    :param payload: Full question JSON including its solution.
    :param answer: The student's answer in the type-specific format.
    :return: ``True`` if the answer is correct.
    :raises ValueError: If the question type is unknown.
    """
    qtype = payload.get("type")
    grader = _GRADERS.get(qtype)
    if grader is None:
        raise ValueError(f"Unbekannter Fragetyp: {qtype!r}")
    return grader(payload, answer)
