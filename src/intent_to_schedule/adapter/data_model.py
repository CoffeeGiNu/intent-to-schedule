from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, PlainSerializer, PlainValidator, WithJsonSchema

from intent_to_schedule.application.command import AddConstraint, AddTask, RemoveConstraint, RemoveTask, ReplaceTask, SchedulingCommand
from intent_to_schedule.domain.calendar import TimeInterval
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint, SoftConstraint
from intent_to_schedule.domain.evaluation import Distance, Evaluation, Excess, Intrusion, Shortfall
from intent_to_schedule.domain.measure import AggregateMeasure, AggregateQuantity, DependencyMeasure, IntervalMeasure, Measure, PointMeasure
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId


def _parse_id(value: object, id_type: type[TaskId] | type[PersonId] | type[ConstraintId]) -> TaskId | PersonId | ConstraintId:
    """Accept a string or an existing domain ID."""
    if isinstance(value, id_type):
        return value
    if isinstance(value, str):
        return id_type(value)
    raise ValueError("ID must be a string")


type TaskIdField = Annotated[TaskId, PlainValidator(lambda value: _parse_id(value, TaskId)), PlainSerializer(lambda value: value.value), WithJsonSchema({"type": "string"})]
"""Task ID encoded as a JSON string."""

type PersonIdField = Annotated[PersonId, PlainValidator(lambda value: _parse_id(value, PersonId)), PlainSerializer(lambda value: value.value), WithJsonSchema({"type": "string"})]
"""Person ID encoded as a JSON string."""

type ConstraintIdField = Annotated[ConstraintId, PlainValidator(lambda value: _parse_id(value, ConstraintId)), PlainSerializer(lambda value: value.value), WithJsonSchema({"type": "string"})]
"""Constraint ID encoded as a JSON string."""


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
    participant_ids: tuple[PersonIdField, ...]
    importance: Literal["low", "medium", "high"]
    required: bool
    stability: Literal["weak", "normal", "strong"]


class PointMeasureOutput(OutputModel):
    """Structured output form of PointMeasure."""

    kind: Literal["point"]
    task_id: TaskIdField


class IntervalMeasureOutput(OutputModel):
    """Structured output form of IntervalMeasure."""

    kind: Literal["interval"]
    task_id: TaskIdField


class DependencyMeasureOutput(OutputModel):
    """Structured output form of DependencyMeasure."""

    kind: Literal["dependency"]
    from_task_id: TaskIdField
    to_task_id: TaskIdField


class AggregateMeasureOutput(OutputModel):
    """Structured output form of AggregateMeasure."""

    kind: Literal["aggregate"]
    task_ids: tuple[TaskIdField, ...]
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
    task_id: TaskIdField
    replacement: TaskOutput


class RemoveTaskOutput(OutputModel):
    """Structured output form of RemoveTask."""

    kind: Literal["remove_task"]
    task_id: TaskIdField


class AddConstraintOutput(OutputModel):
    """Structured output form of AddConstraint."""

    kind: Literal["add_constraint"]
    constraint: HardConstraintOutput | SoftConstraintOutput


class RemoveConstraintOutput(OutputModel):
    """Structured output form of RemoveConstraint."""

    kind: Literal["remove_constraint"]
    constraint_id: ConstraintIdField


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


def to_time_interval_output(interval: TimeInterval) -> TimeIntervalOutput:
    """Convert a domain time interval to its data form."""
    return TimeIntervalOutput(start=interval.start, end=interval.end)


def to_task_output(task: Task) -> TaskOutput:
    """Convert a domain task to its data form."""
    return TaskOutput(
        name=task.name,
        duration=task.duration,
        participant_ids=tuple(sorted(task.participant_ids, key=lambda item: item.value)),
        importance=task.importance.value,
        required=task.required,
        stability=task.stability.value,
    )


def to_measure_output(measure: Measure) -> MeasureOutput:
    """Convert a domain measure to its data form."""
    match measure:
        case PointMeasure(task_id=task_id):
            return PointMeasureOutput(kind="point", task_id=task_id)
        case IntervalMeasure(task_id=task_id):
            return IntervalMeasureOutput(kind="interval", task_id=task_id)
        case DependencyMeasure(from_task_id=from_task_id, to_task_id=to_task_id):
            return DependencyMeasureOutput(kind="dependency", from_task_id=from_task_id, to_task_id=to_task_id)
        case AggregateMeasure(task_ids=task_ids, quantity=quantity):
            return AggregateMeasureOutput(
                kind="aggregate", task_ids=tuple(sorted(task_ids, key=lambda item: item.value)), quantity=quantity.value
            )


def to_quantity_output(value: datetime | timedelta | int) -> QuantityOutput:
    """Convert a domain quantity to its data form."""
    match value:
        case datetime():
            return InstantOutput(kind="instant", value=value)
        case timedelta():
            return DurationOutput(kind="duration", value=value)
        case int():
            return CountOutput(kind="count", value=value)


def to_evaluation_output(evaluation: Evaluation) -> EvaluationOutput:
    """Convert a domain evaluation to its data form."""
    match evaluation:
        case Distance(target=target):
            return DistanceOutput(kind="distance", target=to_quantity_output(target))
        case Intrusion(region=region):
            return IntrusionOutput(kind="intrusion", region=tuple(to_time_interval_output(item) for item in region))
        case Shortfall(lower=lower):
            return ShortfallOutput(kind="shortfall", lower=to_quantity_output(lower))
        case Excess(upper=upper):
            return ExcessOutput(kind="excess", upper=to_quantity_output(upper))
