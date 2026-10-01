from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

SCHEMA_PATH = Path(__file__).with_name("question_schema.json")


@functools.lru_cache(maxsize=1)
def _validator() -> Draft7Validator:
    """
    Loads ``question_schema.json`` once and builds a reusable Draft-07 validator.

    :return: The cached validator.
    """
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft7Validator.check_schema(schema)
    return Draft7Validator(schema)


def validate_question(payload: dict[str, Any]) -> list[str]:
    """
    Validates a question payload against the JSON schema.

    :param payload: Item payload to validate.
    :return: Human-readable schema violations; an empty list means the payload is valid.
    """
    errors = sorted(_validator().iter_errors(payload), key=lambda e: list(e.path))
    return [f"{'/'.join(map(str, e.path)) or '<root>'}: {e.message}" for e in errors]


def is_valid(payload: dict[str, Any]) -> bool:
    """
    Boolean wrapper around :func:`validate_question`.

    :param payload: Item payload to validate.
    :return: ``True`` if the payload satisfies the schema.
    """
    return not validate_question(payload)
