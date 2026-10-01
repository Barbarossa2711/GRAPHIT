from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import dotenv

_DEFAULT_ENV_PATH = Path(__file__).parents[1] / ".env"
_DEFAULT_SLIDE_JSON = Path(__file__).parents[1] / "BDT2026_data" / "BDT26-chunks-from-pdf.json"
_DEFAULT_SLIDE_VERSION = "BDT 2026"


@dataclass(frozen=True)
class GraphSchema:
    """
    Label, relationship and property names of the graph, bundled so that a different schema is adapted in one place.

    Hierarchy: ``(:Lecture)-[:HAS_CHAPTER]->(:Chapter)-[:HAS_TOPIC]->(:Topic)
    -[:HAS_SUBTOPIC]->(:Subtopic)-[:HAS_CONCEPT]->(:Concept)``; slides:
    ``(:Slide)-[:COVERS]->(:Concept)``; generated questions:
    ``(:Concept)<-[:TESTS]-(:QuestionStem)-[:HAS_VARIANT]->(:GeneratedQuestion)``.
    Generated variants deliberately do not use the label :Question, which already belongs
    to the curated lecture questions, so that ``MATCH (q:Question)`` never mixes both.
    """

    concept_label: str = "Concept"
    slide_label: str = "Slide"
    covers_rel: str = "COVERS"
    same_as_rel: str = "SAME_AS"

    concept_id_prop: str = "id"
    concept_name_prop: str = "name"
    concept_objective_prop: str = "objective"

    slide_source_prop: str = "source"
    slide_page_prop: str = "pageNumber"

    lecture_label: str = "Lecture"
    chapter_label: str = "Chapter"
    topic_label: str = "Topic"
    subtopic_label: str = "Subtopic"

    has_chapter_rel: str = "HAS_CHAPTER"
    has_topic_rel: str = "HAS_TOPIC"
    has_subtopic_rel: str = "HAS_SUBTOPIC"
    has_concept_rel: str = "HAS_CONCEPT"

    stem_label: str = "QuestionStem"
    generated_question_label: str = "GeneratedQuestion"

    tests_rel: str = "TESTS"
    has_variant_rel: str = "HAS_VARIANT"


@dataclass(frozen=True)
class Neo4jConfig:
    """
    Connection settings and slide data location, loaded from the .env.

    ``database`` is the database name within the DBMS, not the user; ``slide_version`` names
    the slide set version cited below chat answers.
    """

    uri: str
    user: str
    password: str
    slide_content_json: Path
    database: str
    slide_version: str

    @property
    def auth(self) -> tuple[str, str]:
        """
        Returns the credentials in the form expected by the Neo4j driver.

        :return: ``(user, password)``.
        """
        return (self.user, self.password)


def load_config(env_path: str | Path | None = None) -> Neo4jConfig:
    """
    Reads ``NEO4J_*``, ``SLIDE_CONTENT_JSON`` and ``FOLIEN_STAND`` from the .env.

    ``SLIDE_CONTENT_JSON`` falls back to the bundled JSON generated from the slide PDFs,
    ``NEO4J_DATABASE`` to ``neo4j`` and ``FOLIEN_STAND`` to ``BDT 2026``.

    :param env_path: Path of the ``.env`` to read; the default location when omitted.
    :return: The Neo4j connection settings.
    :raises RuntimeError: If a mandatory variable is missing.
    """
    dotenv.load_dotenv(env_path or _DEFAULT_ENV_PATH)

    missing = [
        key
        for key in ("NEO4J_URI", "NEO4J_USER", "NEO4J_PASSWORD")
        if not os.getenv(key)
    ]
    if missing:
        raise RuntimeError(
            "Missing environment variables in the .env: "
            + ", ".join(missing)
            + f".\nPlease add them to {_DEFAULT_ENV_PATH} "
            "(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)."
        )

    slide_json = os.getenv("SLIDE_CONTENT_JSON")
    return Neo4jConfig(
        uri=os.environ["NEO4J_URI"],
        user=os.environ["NEO4J_USER"],
        password=os.environ["NEO4J_PASSWORD"],
        slide_content_json=Path(slide_json) if slide_json else _DEFAULT_SLIDE_JSON,
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        slide_version=(os.getenv("FOLIEN_STAND") or "").strip() or _DEFAULT_SLIDE_VERSION,
    )
