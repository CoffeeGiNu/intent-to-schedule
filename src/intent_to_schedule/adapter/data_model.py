from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from intent_to_schedule.application.command import AddConstraint, AddTask, RemoveConstraint, RemoveTask, ReplaceTask, SchedulingCommand
from intent_to_schedule.domain.calendar import TimeInterval
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint, SoftConstraint
from intent_to_schedule.domain.evaluation import Distance, Evaluation, Excess, Intrusion, Shortfall
from intent_to_schedule.domain.measure import AggregateMeasure, AggregateQuantity, DependencyMeasure, IntervalMeasure, Measure, PointMeasure
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId


class OutputModel(BaseModel):
    """Base of the structured command input models."""

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


type CommandOutput = AddTaskOutput | ReplaceTaskOutput | RemoveTaskOutput | AddConstraintOutput | RemoveConstraintOutput
"""Structured input form of SchedulingCommand."""


class CommandsInput(OutputModel):
    """Structured input containing commands of either kind."""

    commands: tuple[Annotated[CommandOutput, Field(discriminator="kind")], ...]


def convert_command(output: CommandOutput) -> SchedulingCommand:
    """Convert one structured command, generating an ID for an added object."""
    task: TaskOutput
    task_id: TaskId
    replacement: TaskOutput
    measure: MeasureOutput
    evaluation: EvaluationOutput
    strength: Literal["weak", "normal", "strong"]
    constraint_id: ConstraintId
    match output:
        case AddTaskOutput(task=task):
            return AddTask(convert_task(TaskId.generate(), task))
        case ReplaceTaskOutput(task_id=task_id, replacement=replacement):
            return ReplaceTask(convert_task(task_id, replacement))
        case RemoveTaskOutput(task_id=task_id):
            return RemoveTask(task_id)
        case AddConstraintOutput(constraint=HardConstraintOutput(measure=measure, evaluation=evaluation)):
            return AddConstraint(HardConstraint(ConstraintId.generate(), convert_measure(measure), convert_evaluation(evaluation)))
        case AddConstraintOutput(constraint=SoftConstraintOutput(measure=measure, evaluation=evaluation, strength=strength)):
            return AddConstraint(SoftConstraint(ConstraintId.generate(), convert_measure(measure), convert_evaluation(evaluation), Strength(strength)))
        case RemoveConstraintOutput(constraint_id=constraint_id):
            return RemoveConstraint(constraint_id)


def convert_commands_input(input: CommandsInput) -> tuple[SchedulingCommand, ...]:
    """Convert a structured list of commands to scheduling commands."""
    return tuple(convert_command(command) for command in input.commands)


def convert_task(task_id: TaskId, output: TaskOutput) -> Task:
    """Convert a structured Task value."""
    return Task(task_id, output.name, output.duration, frozenset(output.participant_ids), Importance(output.importance), output.required, Strength(output.stability))


def convert_measure(output: MeasureOutput) -> Measure:
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


def convert_evaluation(output: EvaluationOutput) -> Evaluation:
    """Convert a structured evaluation value."""
    target: QuantityOutput
    region: tuple[TimeIntervalOutput, ...]
    lower: QuantityOutput
    upper: QuantityOutput
    match output:
        case DistanceOutput(target=target):
            return Distance(convert_quantity(target))
        case IntrusionOutput(region=region):
            return Intrusion(tuple(TimeInterval(interval.start, interval.end) for interval in region))
        case ShortfallOutput(lower=lower):
            return Shortfall(convert_quantity(lower))
        case ExcessOutput(upper=upper):
            return Excess(convert_quantity(upper))


def convert_quantity(output: QuantityOutput) -> datetime | timedelta | int:
    """Extract a structured quantity value."""
    value: datetime | timedelta | int
    match output:
        case InstantOutput(value=value) | DurationOutput(value=value) | CountOutput(value=value):
            return value

