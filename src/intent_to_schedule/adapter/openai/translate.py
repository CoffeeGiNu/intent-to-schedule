from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timedelta
from enum import Enum
import json
from typing import Callable, Literal

import openai
from pydantic import BaseModel, ConfigDict

from intent_to_schedule.application.command import (
    AddConstraint,
    AddTask,
    ConstraintCommand,
    ElementCommand,
    Executed,
    Rejected,
    RemoveConstraint,
    RemoveTask,
    ReplaceTask,
    SchedulingCommand,
)
from intent_to_schedule.application.translate import Ambiguous, CommandTranslator, Translated, TranslateResult, Utterance
from intent_to_schedule.domain.calendar import TimeInterval
from intent_to_schedule.domain.consistency import ConsistencyError, Validator, Violations
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint, SoftConstraint
from intent_to_schedule.domain.evaluation import Distance, Evaluation, Excess, Intrusion, Shortfall
from intent_to_schedule.domain.measure import (
    AggregateMeasure,
    AggregateQuantity,
    DependencyMeasure,
    IntervalMeasure,
    Measure,
    PointMeasure,
)
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId


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
    question: str
    commands: tuple[ElementCommandOutput, ...]
    task: TaskOutput
    task_id: TaskId
    replacement: TaskOutput
    match output.result:
        case AmbiguousOutput(question=question):
            return Ambiguous(question)
        case ElementCommandsOutput(commands=commands):
            converted: list[ElementCommand] = []
            command: ElementCommandOutput
            for command in commands:
                match command:
                    case AddTaskOutput(task=task):
                        converted.append(AddTask(_convert_task(TaskId.generate(), task)))
                    case ReplaceTaskOutput(task_id=task_id, replacement=replacement):
                        converted.append(ReplaceTask(_convert_task(task_id, replacement)))
                    case RemoveTaskOutput(task_id=task_id):
                        converted.append(RemoveTask(task_id))
            return tuple(converted)


def convert_constraint_output(output: ConstraintTranslationOutput) -> tuple[ConstraintCommand, ...] | Ambiguous:
    """Convert constraint step output to constraint commands, generating IDs for added constraints."""
    question: str
    commands: tuple[ConstraintCommandOutput, ...]
    measure: MeasureOutput
    evaluation: EvaluationOutput
    strength: Literal["weak", "normal", "strong"]
    constraint_id: ConstraintId
    match output.result:
        case AmbiguousOutput(question=question):
            return Ambiguous(question)
        case ConstraintCommandsOutput(commands=commands):
            converted: list[ConstraintCommand] = []
            command: ConstraintCommandOutput
            for command in commands:
                match command:
                    case AddConstraintOutput(constraint=HardConstraintOutput(measure=measure, evaluation=evaluation)):
                        converted.append(
                            AddConstraint(HardConstraint(ConstraintId.generate(), _convert_measure(measure), _convert_evaluation(evaluation)))
                        )
                    case AddConstraintOutput(constraint=SoftConstraintOutput(measure=measure, evaluation=evaluation, strength=strength)):
                        converted.append(
                            AddConstraint(SoftConstraint(ConstraintId.generate(), _convert_measure(measure), _convert_evaluation(evaluation), Strength(strength)))
                        )
                    case RemoveConstraintOutput(constraint_id=constraint_id):
                        converted.append(RemoveConstraint(constraint_id))
            return tuple(converted)


def _convert_task(task_id: TaskId, output: TaskOutput) -> Task:
    """Convert a structured Task value."""
    return Task(task_id, output.name, output.duration, frozenset(output.participant_ids), Importance(output.importance), output.required, Strength(output.stability))


def _convert_measure(output: MeasureOutput) -> Measure:
    """Convert a structured measure value."""
    task_id: TaskId
    from_task_id: TaskId
    to_task_id: TaskId
    task_ids: tuple[TaskId, ...]
    quantity: Literal["count", "total_duration"]
    match output:
        case PointMeasureOutput(task_id=task_id):
            return PointMeasure(task_id)
        case IntervalMeasureOutput(task_id=task_id):
            return IntervalMeasure(task_id)
        case DependencyMeasureOutput(from_task_id=from_task_id, to_task_id=to_task_id):
            return DependencyMeasure(from_task_id, to_task_id)
        case AggregateMeasureOutput(task_ids=task_ids, quantity=quantity):
            return AggregateMeasure(frozenset(task_ids), AggregateQuantity(quantity))


def _convert_evaluation(output: EvaluationOutput) -> Evaluation:
    """Convert a structured evaluation value."""
    target: QuantityOutput
    region: tuple[TimeIntervalOutput, ...]
    lower: QuantityOutput
    upper: QuantityOutput
    match output:
        case DistanceOutput(target=target):
            return Distance(_convert_quantity(target))
        case IntrusionOutput(region=region):
            return Intrusion(tuple(TimeInterval(interval.start, interval.end) for interval in region))
        case ShortfallOutput(lower=lower):
            return Shortfall(_convert_quantity(lower))
        case ExcessOutput(upper=upper):
            return Excess(_convert_quantity(upper))


def _convert_quantity(output: QuantityOutput) -> datetime | timedelta | int:
    """Extract a structured quantity value."""
    value: datetime | timedelta | int
    match output:
        case InstantOutput(value=value) | DurationOutput(value=value) | CountOutput(value=value):
            return value


_ELEMENT_PROMPT: str = (
    "Translate the latest user utterance into commands that add, replace, or remove Tasks. "
    "Use only existing person, task, and constraint IDs from the snapshot. "
    "Ask a clarifying question with ambiguous when the meaning is not determined, such as with same-named people; do not guess. "
    "Importance low, medium, or high means how much it matters to do the Task at all. "
    "Required means the Task must be scheduled, including expressions such as 絶対. "
    "Stability means how strongly to keep the Task at its previous time. "
)

_CONSTRAINT_PROMPT: str = (
    "Translate the latest user utterance into commands that add or remove constraints. "
    "Use only existing IDs from the snapshot. A measure extracts a schedule value and an evaluation scores it. "
    "Supported pairs: point with distance to an instant; interval with intrusion into a region; "
    "dependency with distance or shortfall of a duration, measuring the gap from the end of from_task to the start of to_task; "
    "aggregate with excess per day, using a count or total duration. "
    "To express wanting a task inside A, use the complement of A within the horizon as the intrusion region. "
    "Use hard only for must requirements. For soft preferences choose weak, normal, or strong strength. "
    "All times and durations must be on the slot grid. Ask a clarifying question instead of guessing. "
)


def _json_default(value: object) -> object:
    """Serialize snapshot values for the model."""
    match value:
        case datetime():
            return value.isoformat()
        case timedelta():
            return value.total_seconds()
        case Enum():
            return value.value
        case frozenset():
            return [asdict(item) for item in value]
        case _:
            raise TypeError(f"Cannot serialize {type(value).__name__}")


def _snapshot(problem: SchedulingProblem, previous: Schedule | None) -> str:
    """Describe the current scheduling state."""
    snapshot: dict[str, object] = {
        "current_time": datetime.now(problem.calendar.grid.horizon.start.tzinfo).isoformat(),
        "problem": asdict(problem),
        "previous_schedule": asdict(previous) if previous is not None else None,
    }
    return json.dumps(snapshot, default=_json_default, ensure_ascii=False)


class OpenAICommandTranslator(CommandTranslator):
    """CommandTranslator backed by the OpenAI API."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        validators: Sequence[Validator],
    ) -> None:
        self._client: openai.OpenAI = client
        self._model: str = model
        self._validators: Sequence[Validator] = validators

    def translate(
        self,
        dialogue: Sequence[Utterance],
        problem: SchedulingProblem,
        previous: Schedule | None,
    ) -> TranslateResult:
        """Translate the latest utterance into element commands, then into constraint commands."""
        elements: tuple[tuple[ElementCommand, ...], SchedulingProblem] | Ambiguous = self._translate_step(
            dialogue, problem, previous, _ELEMENT_PROMPT, ElementTranslationOutput, convert_element_output
        )
        if isinstance(elements, Ambiguous):
            return elements
        element_commands: tuple[ElementCommand, ...]
        updated_problem: SchedulingProblem
        element_commands, updated_problem = elements
        constraints: tuple[tuple[ConstraintCommand, ...], SchedulingProblem] | Ambiguous = self._translate_step(
            dialogue, updated_problem, previous, _CONSTRAINT_PROMPT, ConstraintTranslationOutput, convert_constraint_output
        )
        if isinstance(constraints, Ambiguous):
            return constraints
        constraint_commands: tuple[ConstraintCommand, ...] = constraints[0]
        return Translated((*element_commands, *constraint_commands))

    def _translate_step[Output: OutputModel, Command: SchedulingCommand](
        self,
        dialogue: Sequence[Utterance],
        problem: SchedulingProblem,
        previous: Schedule | None,
        prompt: str,
        output_type: type[Output],
        convert: Callable[[Output], tuple[Command, ...] | Ambiguous],
    ) -> tuple[tuple[Command, ...], SchedulingProblem] | Ambiguous:
        """Request and validate one translation step."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": prompt + "\nSnapshot: " + _snapshot(problem, previous)},
            *({"role": utterance.speaker.value, "content": utterance.text} for utterance in dialogue),
        ]
        attempt: int
        error: ConsistencyError
        for attempt in range(3):
            response: openai.types.responses.ParsedResponse[Output] = self._client.responses.parse(
                model=self._model, input=messages, text_format=output_type
            )
            output: Output | None = response.output_parsed
            if output is None:
                raise ValueError("Model returned no parsed translation")
            converted: tuple[Command, ...] | Ambiguous = convert(output)
            if isinstance(converted, Ambiguous):
                return converted
            try:
                updated_problem: SchedulingProblem = self._apply_commands(converted, problem)
            except ConsistencyError as error:
                if attempt == 2:
                    raise
                messages.append({"role": "assistant", "content": output.model_dump_json()})
                messages.append({"role": "user", "content": "Correct these violations: " + "; ".join(item.message for item in error.violations.items)})
                continue
            return converted, updated_problem
        raise AssertionError("Translation retry loop ended unexpectedly")

    def _apply_commands(
        self, commands: Sequence[SchedulingCommand], problem: SchedulingProblem
    ) -> SchedulingProblem:
        """Apply commands and validate the resulting problem."""
        updated_problem: SchedulingProblem = problem
        command: SchedulingCommand
        for command in commands:
            result: Executed | Rejected = command.execute(updated_problem)
            if isinstance(result, Rejected):
                raise ConsistencyError(result.violations)
            updated_problem = result.problem
        violations: Violations = Violations(())
        validator: Validator
        for validator in self._validators:
            violations = violations.merge(validator.validate(updated_problem))
        if not violations.is_empty:
            raise ConsistencyError(violations)
        return updated_problem
