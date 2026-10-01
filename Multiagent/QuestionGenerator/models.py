from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

QuestionType = Literal["single", "multiple", "cloze", "match", "order"]


class Concept(BaseModel):
    """
    A concept (knowledge component) of a lecture, the input of the question generator.
    """

    id: str = Field(..., description="Unique concept id, e.g. 'CH02_T03_S01'.")
    name: str = Field(..., description="Short name / title of the concept.")
    objective: str | None = Field(
        default=None,
        description="Optional overarching learning objective of the concept.",
    )


class StemPlan(BaseModel):
    """
    Ein geplanter QuestionStem: ein einzelner testbarer Fakt / Lernziel.
    """

    objective: str = Field(
        ...,
        description="Der konkrete Fakt / das Lernziel, das dieser Stem abfragt.",
    )
    question_types: list[QuestionType] = Field(
        ...,
        min_length=2,
        description=(
            "Die für diesen Fakt geeigneten Fragetypen (mindestens zwei, damit derselbe "
            "Kern in verschiedenen Formaten abgefragt werden kann)."
        ),
    )


class StemPlanList(BaseModel):
    """
    Container für die Phase-1-Ausgabe des LLM.
    """

    stems: list[StemPlan]


class Option(BaseModel):
    """
    Eine ID-behaftete Antwort-/Begriffsoption.
    """

    id: str
    text: str


class SingleSolution(BaseModel):
    correct: str = Field(..., description="ID der einzig korrekten Option.")


class SingleChoiceQuestion(BaseModel):
    type: Literal["single"] = "single"
    prompt: str
    options: list[Option] = Field(..., min_length=2)
    solution: SingleSolution
    explanation: str | None = None


class MultipleSolution(BaseModel):
    correct: list[str] = Field(..., min_length=1, description="IDs aller korrekten Optionen.")


class MultipleChoiceQuestion(BaseModel):
    type: Literal["multiple"] = "multiple"
    prompt: str
    options: list[Option] = Field(..., min_length=2)
    solution: MultipleSolution
    grading: Literal["all_or_nothing"] = "all_or_nothing"
    explanation: str | None = None


class ClozeQuestion(BaseModel):
    type: Literal["cloze"] = "cloze"
    prompt: str = Field(..., description="Arbeitsanweisung, z. B. 'Fülle die Lücken mit den passenden Begriffen.'")
    text: str = Field(..., description="Text mit Platzhaltern der Form {{blankId}}.")
    blanks: list[str] = Field(..., min_length=1, description="Liste der Blank-IDs, z. B. ['b1','b2'].")
    bank: list[Option] = Field(..., min_length=2, description="Begriffs-Pool inkl. Distraktoren.")
    solution: dict[str, str] = Field(
        ..., description="Map Blank-ID -> Begriffs-ID, z. B. {'b1':'t1'}."
    )
    reuse: bool = Field(default=False, description="Darf ein Begriff mehrfach eingesetzt werden?")
    explanation: str | None = None


class MatchLeftRight(BaseModel):
    left: str
    right: str


class MatchQuestion(BaseModel):
    type: Literal["match"] = "match"
    prompt: str
    left: list[Option] = Field(..., min_length=2)
    right: list[Option] = Field(
        ..., min_length=2, description="Mehr rechte als linke Einträge = Distraktoren."
    )
    solution: list[MatchLeftRight] = Field(..., min_length=2)
    explanation: str | None = None


class OrderQuestion(BaseModel):
    type: Literal["order"] = "order"
    prompt: str
    items: list[Option] = Field(..., min_length=2, description="Elemente in beliebiger Anzeigereihenfolge.")
    solution: list[str] = Field(..., min_length=2, description="Item-IDs in korrekter Sequenz.")
    explanation: str | None = None


TYPE_TO_MODEL: dict[str, type[BaseModel]] = {
    "single": SingleChoiceQuestion,
    "multiple": MultipleChoiceQuestion,
    "cloze": ClozeQuestion,
    "match": MatchQuestion,
    "order": OrderQuestion,
}


StemSource = Literal["lecture", "generated"]


class Question(BaseModel):
    """
    A concrete, validated question variant, stored as a :GeneratedQuestion node.
    """

    id: str = Field(..., description="Unique question id, e.g. 'CH02_T03_S01_ST01_SINGLE'.")
    type: QuestionType
    payload: str = Field(
        ..., description="The complete, schema-valid question as a JSON string (Neo4j payload)."
    )


class GeneratedStem(BaseModel):
    """
    A stem (one testable fact) with its validated question variants, stored as a :QuestionStem node.
    """

    id: str = Field(..., description="Unique stem id, e.g. 'CH02_T03_S01_ST01'.")
    objective: str
    source: StemSource = Field(
        default="generated",
        description="'lecture' = derived from an existing lecture question, otherwise 'generated'.",
    )
    questions: list[Question] = Field(default_factory=list)


class QuestionRecord(BaseModel):
    """
    Flat record per question for graph import; ``payload`` is the question as JSON string.
    """

    concept_id: str
    stem_id: str
    stem_objective: str
    source: StemSource
    question_id: str
    type: QuestionType
    payload: str


class QualityWarning(BaseModel):
    """
    A non-blocking quality finding on a generated item.

    Deliberately not attached to :class:`Question`, which feeds the graph import; the finding
    describes the generation run, not the stored item.
    """

    question_id: str
    type: QuestionType
    message: str = Field(..., description="The finding verbatim from checks.quality_warnings.")


class QuestionSet(BaseModel):
    """
    Complete generation result for a concept.
    """

    concept_id: str
    concept_name: str
    n_slides: int
    stems: list[GeneratedStem] = Field(default_factory=list)
    warnings: list[QualityWarning] = Field(
        default_factory=list,
        description=(
            "Construction weaknesses of the generated items. Empty if the set comes from the "
            "store: the checks run during generation and are not stored, since they are "
            "deterministic and can be recomputed from the payload at any time."
        ),
    )

    @property
    def n_questions(self) -> int:
        """
        Counts the distinct items, the quantity checked against the item budget.

        :return: Number of questions over all stems.
        """
        return sum(len(stem.questions) for stem in self.stems)

    def iter_questions(self):
        """
        Iterates over all questions as flat records for graph import.

        :return: Generator of :class:`QuestionRecord`, one per question.
        """
        for stem in self.stems:
            for q in stem.questions:
                yield QuestionRecord(
                    concept_id=self.concept_id,
                    stem_id=stem.id,
                    stem_objective=stem.objective,
                    source=stem.source,
                    question_id=q.id,
                    type=q.type,
                    payload=q.payload,
                )
