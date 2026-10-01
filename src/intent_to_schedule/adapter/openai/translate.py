from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal

import openai
from pydantic import BaseModel, ConfigDict

from intent_to_schedule.application.command import ConstraintCommand, ElementCommand
from intent_to_schedule.application.translate import Ambiguous, CommandTranslator, TranslateResult, Utterance
from intent_to_schedule.domain.consistency import Validator
from intent_to_schedule.domain.constraint import ConstraintId
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule
from intent_to_schedule.domain.task import TaskId


class OutputModel(BaseModel):
    """Base of the OpenAI structured output models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class TimeIntervalOutput(OutputModel):
    """Structured output form of TimeInterval."""

    start: datetime
    end: datetime


class TaskOutput(OutputModel):
    """Structured output form of Task."""

    name: str
    duration: timedelta
    participant_ids: tuple[PersonId, ...]
    importance: Literal["low", "medium", "high"]
    required: bool
    stability: Literal["weak", "normal", "strong"]


class PointMeasureOutput(OutputModel):
    """Structured output form of PointMeasure."""

    kind: Literal["point"]
    task_id: TaskId


class IntervalMeasureOutput(OutputModel):
    """Structured output form of IntervalMeasure."""

    kind: Literal["interval"]
    task_id: TaskId


class DependencyMeasureOutput(OutputModel):
    """Structured output form of DependencyMeasure."""

    kind: Literal["dependency"]
    from_task_id: TaskId
    to_task_id: TaskId


class AggregateMeasureOutput(OutputModel):
    """Structured output form of AggregateMeasure."""

    kind: Literal["aggregate"]
    task_ids: tuple[TaskId, ...]
    quantity: Literal["count", "total_duration"]


type MeasureOutput = PointMeasureOutput | IntervalMeasureOutput | DependencyMeasureOutput | AggregateMeasureOutput
"""Structured output form of Measure."""


class InstantOutput(OutputModel):
    """Structured output form of a point in time."""

    kind: Literal["instant"]
    value: datetime


class DurationOutput(OutputModel):
    """Structured output form of a duration."""

    kind: Literal["duration"]
    value: timedelta


class CountOutput(OutputModel):
    """Structured output form of a count."""

    kind: Literal["count"]
    value: int


type QuantityOutput = InstantOutput | DurationOutput | CountOutput
"""Structured output form of Quantity."""


class DistanceOutput(OutputModel):
    """Structured output form of Distance."""

    kind: Literal["distance"]
    target: QuantityOutput


class IntrusionOutput(OutputModel):
    """Structured output form of Intrusion."""

    kind: Literal["intrusion"]
    region: tuple[TimeIntervalOutput, ...]


class ShortfallOutput(OutputModel):
    """Structured output form of Shortfall."""

    kind: Literal["shortfall"]
    lower: QuantityOutput


class ExcessOutput(OutputModel):
    """Structured output form of Excess."""

    kind: Literal["excess"]
    upper: QuantityOutput


type EvaluationOutput = DistanceOutput | IntrusionOutput | ShortfallOutput | ExcessOutput
"""Structured output form of Evaluation."""


class HardConstraintOutput(OutputModel):
    """Structured output form of HardConstraint."""

    kind: Literal["hard"]
    measure: MeasureOutput
    evaluation: EvaluationOutput


class SoftConstraintOutput(OutputModel):
    """Structured output form of SoftConstraint."""

    kind: Literal["soft"]
    measure: MeasureOutput
    evaluation: EvaluationOutput
    strength: Literal["weak", "normal", "strong"]


class AddTaskOutput(OutputModel):
    """Structured output form of AddTask."""

    kind: Literal["add_task"]
    task: TaskOutput


class ReplaceTaskOutput(OutputModel):
    """Structured output form of ReplaceTask."""

    kind: Literal["replace_task"]
    task_id: TaskId
    replacement: TaskOutput


class RemoveTaskOutput(OutputModel):
    """Structured output form of RemoveTask."""

    kind: Literal["remove_task"]
    task_id: TaskId


class AddConstraintOutput(OutputModel):
    """Structured output form of AddConstraint."""

    kind: Literal["add_constraint"]
    constraint: HardConstraintOutput | SoftConstraintOutput


class RemoveConstraintOutput(OutputModel):
    """Structured output form of RemoveConstraint."""

    kind: Literal["remove_constraint"]
    constraint_id: ConstraintId


type ElementCommandOutput = AddTaskOutput | ReplaceTaskOutput | RemoveTaskOutput
"""Structured output form of ElementCommand."""


type ConstraintCommandOutput = AddConstraintOutput | RemoveConstraintOutput
"""Structured output form of ConstraintCommand."""


class AmbiguousOutput(OutputModel):
    """Structured output form of Ambiguous."""

    kind: Literal["ambiguous"]
    question: str


class ElementCommandsOutput(OutputModel):
    """Element commands translated from an utterance."""

    kind: Literal["translated"]
    commands: tuple[ElementCommandOutput, ...]


class ConstraintCommandsOutput(OutputModel):
    """Constraint commands translated from an utterance."""

    kind: Literal["translated"]
    commands: tuple[ConstraintCommandOutput, ...]


class ElementTranslationOutput(OutputModel):
    """Root of the structured output for the element step: element commands or a clarifying question."""

    result: ElementCommandsOutput | AmbiguousOutput


class ConstraintTranslationOutput(OutputModel):
    """Root of the structured output for the constraint step: constraint commands or a clarifying question."""

    result: ConstraintCommandsOutput | AmbiguousOutput


def convert_element_output(output: ElementTranslationOutput) -> tuple[ElementCommand, ...] | Ambiguous:
    """Convert element step output to element commands, generating IDs for added Tasks."""
    ...


def convert_constraint_output(output: ConstraintTranslationOutput) -> tuple[ConstraintCommand, ...] | Ambiguous:
    """Convert constraint step output to constraint commands, generating IDs for added constraints."""
    ...


class OpenAICommandTranslator(CommandTranslator):
    """CommandTranslator backed by the OpenAI API."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        validators: Sequence[Validator],
    ) -> None: ...

    def translate(
        self,
        dialogue: Sequence[Utterance],
        problem: SchedulingProblem,
        previous: Schedule | None,
    ) -> TranslateResult:
        """Translate the latest utterance into element commands, then into constraint commands."""
        ...
