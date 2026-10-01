from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    WithJsonSchema,
)

from intent_to_schedule.application.command import (
    AddConstraint,
    AddTask,
    RemoveConstraint,
    RemoveTask,
    ReplaceTask,
    SchedulingCommand,
)
from intent_to_schedule.domain.calendar import TimeInterval
from intent_to_schedule.domain.constraint import (
    ConstraintId,
    HardConstraint,
    SoftConstraint,
)
from intent_to_schedule.domain.evaluation import (
    Distance,
    Evaluation,
    Excess,
    Intrusion,
    Shortfall,
)
from intent_to_schedule.domain.measure import (
    AggregateMeasure,
    AggregateQuantity,
    DependencyMeasure,
    IntervalMeasure,
    Measure,
    PointMeasure,
)
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId


def _parse_id(
    value: object, id_type: type[TaskId] | type[PersonId] | type[ConstraintId]
) -> TaskId | PersonId | ConstraintId:
    """Accept a string or an existing domain ID."""
    if isinstance(value, id_type):
        return value
    if isinstance(value, str):
        return id_type(value)
    raise ValueError("ID must be a string")


type TaskIdField = Annotated[
    TaskId,
    PlainValidator(lambda value: _parse_id(value, TaskId)),
    PlainSerializer(lambda value: value.value),
    WithJsonSchema({"type": "string"}),
]
"""Task ID encoded as a JSON string."""

type PersonIdField = Annotated[
    PersonId,
    PlainValidator(lambda value: _parse_id(value, PersonId)),
    PlainSerializer(lambda value: value.value),
    WithJsonSchema({"type": "string"}),
]
"""Person ID encoded as a JSON string."""

type ConstraintIdField = Annotated[
    ConstraintId,
    PlainValidator(lambda value: _parse_id(value, ConstraintId)),
    PlainSerializer(lambda value: value.value),
    WithJsonSchema({"type": "string"}),
]
"""Constraint ID encoded as a JSON string."""


class DataModel(BaseModel):
    """Base of the JSON data models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class TimeIntervalData(DataModel):
    """JSON form of TimeInterval."""

    start: datetime
    end: datetime


class TaskData(DataModel):
    """JSON form of Task."""

    name: str
    duration: timedelta
    participant_ids: tuple[PersonIdField, ...]
    importance: Literal["low", "medium", "high"]
    required: bool
    stability: Literal["weak", "normal", "strong"]


class PointMeasureData(DataModel):
    """JSON form of PointMeasure."""

    kind: Literal["point"]
    task_id: TaskIdField


class IntervalMeasureData(DataModel):
    """JSON form of IntervalMeasure."""

    kind: Literal["interval"]
    task_id: TaskIdField


class DependencyMeasureData(DataModel):
    """JSON form of DependencyMeasure."""

    kind: Literal["dependency"]
    from_task_id: TaskIdField
    to_task_id: TaskIdField


class AggregateMeasureData(DataModel):
    """JSON form of AggregateMeasure."""

    kind: Literal["aggregate"]
    task_ids: tuple[TaskIdField, ...]
    quantity: Literal["count", "total_duration"]


type MeasureData = (
    PointMeasureData
    | IntervalMeasureData
    | DependencyMeasureData
    | AggregateMeasureData
)
"""JSON form of Measure."""


class InstantData(DataModel):
    """JSON form of a point in time."""

    kind: Literal["instant"]
    value: datetime


class DurationData(DataModel):
    """JSON form of a duration."""

    kind: Literal["duration"]
    value: timedelta


class CountData(DataModel):
    """JSON form of a count."""

    kind: Literal["count"]
    value: int


type QuantityData = InstantData | DurationData | CountData
"""JSON form of Quantity."""


class DistanceData(DataModel):
    """JSON form of Distance."""

    kind: Literal["distance"]
    target: QuantityData


class IntrusionData(DataModel):
    """JSON form of Intrusion."""

    kind: Literal["intrusion"]
    region: tuple[TimeIntervalData, ...]


class ShortfallData(DataModel):
    """JSON form of Shortfall."""

    kind: Literal["shortfall"]
    lower: QuantityData


class ExcessData(DataModel):
    """JSON form of Excess."""

    kind: Literal["excess"]
    upper: QuantityData


type EvaluationData = DistanceData | IntrusionData | ShortfallData | ExcessData
"""JSON form of Evaluation."""


class HardConstraintData(DataModel):
    """JSON form of HardConstraint."""

    kind: Literal["hard"]
    measure: MeasureData
    evaluation: EvaluationData


class SoftConstraintData(DataModel):
    """JSON form of SoftConstraint."""

    kind: Literal["soft"]
    measure: MeasureData
    evaluation: EvaluationData
    strength: Literal["weak", "normal", "strong"]


class AddTaskData(DataModel):
    """JSON form of AddTask."""

    kind: Literal["add_task"]
    task: TaskData


class ReplaceTaskData(DataModel):
    """JSON form of ReplaceTask."""

    kind: Literal["replace_task"]
    task_id: TaskIdField
    replacement: TaskData


class RemoveTaskData(DataModel):
    """JSON form of RemoveTask."""

    kind: Literal["remove_task"]
    task_id: TaskIdField


class AddConstraintData(DataModel):
    """JSON form of AddConstraint."""

    kind: Literal["add_constraint"]
    constraint: HardConstraintData | SoftConstraintData


class RemoveConstraintData(DataModel):
    """JSON form of RemoveConstraint."""

    kind: Literal["remove_constraint"]
    constraint_id: ConstraintIdField


type ElementCommandData = AddTaskData | ReplaceTaskData | RemoveTaskData
"""JSON form of ElementCommand."""


type ConstraintCommandData = AddConstraintData | RemoveConstraintData
"""JSON form of ConstraintCommand."""


type CommandData = (
    AddTaskData
    | ReplaceTaskData
    | RemoveTaskData
    | AddConstraintData
    | RemoveConstraintData
)
"""JSON form of SchedulingCommand."""


class CommandsData(DataModel):
    """JSON data containing commands of either kind."""

    commands: tuple[Annotated[CommandData, Field(discriminator="kind")], ...]


def convert_command(data: CommandData) -> SchedulingCommand:
    """Convert one structured command, generating an ID for an added object."""
    task: TaskData
    task_id: TaskId
    replacement: TaskData
    measure: MeasureData
    evaluation: EvaluationData
    strength: Literal["weak", "normal", "strong"]
    constraint_id: ConstraintId
    match data:
        case AddTaskData(task=task):
            return AddTask(convert_task(TaskId.generate(), task))
        case ReplaceTaskData(task_id=task_id, replacement=replacement):
            return ReplaceTask(convert_task(task_id, replacement))
        case RemoveTaskData(task_id=task_id):
            return RemoveTask(task_id)
        case AddConstraintData(
            constraint=HardConstraintData(measure=measure, evaluation=evaluation)
        ):
            return AddConstraint(
                HardConstraint(
                    ConstraintId.generate(),
                    convert_measure(measure),
                    convert_evaluation(evaluation),
                )
            )
        case AddConstraintData(
            constraint=SoftConstraintData(
                measure=measure, evaluation=evaluation, strength=strength
            )
        ):
            return AddConstraint(
                SoftConstraint(
                    ConstraintId.generate(),
                    convert_measure(measure),
                    convert_evaluation(evaluation),
                    Strength(strength),
                )
            )
        case RemoveConstraintData(constraint_id=constraint_id):
            return RemoveConstraint(constraint_id)


def convert_commands_input(data: CommandsData) -> tuple[SchedulingCommand, ...]:
    """Convert a structured list of commands to scheduling commands."""
    return tuple(convert_command(command) for command in data.commands)


def convert_task(task_id: TaskId, data: TaskData) -> Task:
    """Convert a structured Task value."""
    return Task(
        task_id,
        data.name,
        data.duration,
        frozenset(data.participant_ids),
        Importance(data.importance),
        data.required,
        Strength(data.stability),
    )


def convert_measure(data: MeasureData) -> Measure:
    """Convert a structured measure value."""
    task_id: TaskId
    from_task_id: TaskId
    to_task_id: TaskId
    task_ids: tuple[TaskId, ...]
    quantity: Literal["count", "total_duration"]
    match data:
        case PointMeasureData(task_id=task_id):
            return PointMeasure(task_id)
        case IntervalMeasureData(task_id=task_id):
            return IntervalMeasure(task_id)
        case DependencyMeasureData(from_task_id=from_task_id, to_task_id=to_task_id):
            return DependencyMeasure(from_task_id, to_task_id)
        case AggregateMeasureData(task_ids=task_ids, quantity=quantity):
            return AggregateMeasure(frozenset(task_ids), AggregateQuantity(quantity))


def convert_evaluation(data: EvaluationData) -> Evaluation:
    """Convert a structured evaluation value."""
    target: QuantityData
    region: tuple[TimeIntervalData, ...]
    lower: QuantityData
    upper: QuantityData
    match data:
        case DistanceData(target=target):
            return Distance(convert_quantity(target))
        case IntrusionData(region=region):
            return Intrusion(
                tuple(TimeInterval(interval.start, interval.end) for interval in region)
            )
        case ShortfallData(lower=lower):
            return Shortfall(convert_quantity(lower))
        case ExcessData(upper=upper):
            return Excess(convert_quantity(upper))


def convert_quantity(data: QuantityData) -> datetime | timedelta | int:
    """Extract a structured quantity value."""
    value: datetime | timedelta | int
    match data:
        case (
            InstantData(value=value)
            | DurationData(value=value)
            | CountData(value=value)
        ):
            return value


def to_time_interval_data(interval: TimeInterval) -> TimeIntervalData:
    """Convert a domain time interval to its data form."""
    return TimeIntervalData(start=interval.start, end=interval.end)


def to_task_data(task: Task) -> TaskData:
    """Convert a domain task to its data form."""
    return TaskData(
        name=task.name,
        duration=task.duration,
        participant_ids=tuple(
            sorted(task.participant_ids, key=lambda item: item.value)
        ),
        importance=task.importance.value,
        required=task.required,
        stability=task.stability.value,
    )


def to_measure_data(measure: Measure) -> MeasureData:
    """Convert a domain measure to its data form."""
    match measure:
        case PointMeasure(task_id=task_id):
            return PointMeasureData(kind="point", task_id=task_id)
        case IntervalMeasure(task_id=task_id):
            return IntervalMeasureData(kind="interval", task_id=task_id)
        case DependencyMeasure(from_task_id=from_task_id, to_task_id=to_task_id):
            return DependencyMeasureData(
                kind="dependency", from_task_id=from_task_id, to_task_id=to_task_id
            )
        case AggregateMeasure(task_ids=task_ids, quantity=quantity):
            return AggregateMeasureData(
                kind="aggregate",
                task_ids=tuple(sorted(task_ids, key=lambda item: item.value)),
                quantity=quantity.value,
            )


def to_quantity_data(value: datetime | timedelta | int) -> QuantityData:
    """Convert a domain quantity to its data form."""
    match value:
        case datetime():
            return InstantData(kind="instant", value=value)
        case timedelta():
            return DurationData(kind="duration", value=value)
        case int():
            return CountData(kind="count", value=value)


def to_evaluation_data(evaluation: Evaluation) -> EvaluationData:
    """Convert a domain evaluation to its data form."""
    match evaluation:
        case Distance(target=target):
            return DistanceData(kind="distance", target=to_quantity_data(target))
        case Intrusion(region=region):
            return IntrusionData(
                kind="intrusion",
                region=tuple(to_time_interval_data(item) for item in region),
            )
        case Shortfall(lower=lower):
            return ShortfallData(kind="shortfall", lower=to_quantity_data(lower))
        case Excess(upper=upper):
            return ExcessData(kind="excess", upper=to_quantity_data(upper))
