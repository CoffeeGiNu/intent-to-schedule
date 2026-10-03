from datetime import date, datetime, time, timedelta
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
    AddTimeConstraint,
    RemoveConstraint,
    RemoveTask,
    ReplaceTask,
    SchedulingCommand,
)
from intent_to_schedule.application.query import Answer, SchedulingQuery
from intent_to_schedule.application.time_windows import TimeRelation, TimeWindow
from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.constraint import (
    Constraint,
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
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId


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


class DateRangeData(DataModel):
    """JSON form of DateRange."""

    start: date
    end: date


class TimeRangeData(DataModel):
    """JSON form of TimeRange."""

    start: time
    end: time | None


class TimeWindowData(DataModel):
    """JSON form of TimeWindow."""

    date_range: DateRangeData | None = None
    weekdays: (
        tuple[
            Literal[
                "monday",
                "tuesday",
                "wednesday",
                "thursday",
                "friday",
                "saturday",
                "sunday",
            ],
            ...,
        ]
        | None
    ) = None
    time_range: TimeRangeData | None = None


class PersonData(DataModel):
    """JSON form of Person."""

    id: PersonIdField
    name: str


class NewTaskData(DataModel):
    """JSON form of a Task to add, before it has an ID."""

    name: str
    duration: timedelta
    participant_ids: tuple[PersonIdField, ...]
    importance: Literal["low", "medium", "high"]
    required: bool
    stability: Literal["weak", "normal", "strong"]


class TaskData(NewTaskData):
    """JSON form of Task."""

    id: TaskIdField


class NewFixedTaskData(DataModel):
    """JSON form of a FixedTask to add, before it has an ID."""

    name: str
    start: datetime
    duration: timedelta
    participant_ids: tuple[PersonIdField, ...]


class FixedTaskData(NewFixedTaskData):
    """JSON form of FixedTask."""

    id: TaskIdField


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


class NewHardConstraintData(DataModel):
    """JSON form of a HardConstraint to add, before it has an ID."""

    kind: Literal["hard"]
    measure: MeasureData
    evaluation: EvaluationData


class HardConstraintData(NewHardConstraintData):
    """JSON form of HardConstraint."""

    id: ConstraintIdField


class NewSoftConstraintData(DataModel):
    """JSON form of a SoftConstraint to add, before it has an ID."""

    kind: Literal["soft"]
    measure: MeasureData
    evaluation: EvaluationData
    strength: Literal["weak", "normal", "strong"]


class SoftConstraintData(NewSoftConstraintData):
    """JSON form of SoftConstraint."""

    id: ConstraintIdField


class AddTaskData(DataModel):
    """JSON form of AddTask."""

    kind: Literal["add_task"]
    task: NewTaskData


class ReplaceTaskData(DataModel):
    """JSON form of ReplaceTask."""

    kind: Literal["replace_task"]
    task: TaskData


class RemoveTaskData(DataModel):
    """JSON form of RemoveTask."""

    kind: Literal["remove_task"]
    task_id: TaskIdField


class AddConstraintData(DataModel):
    """JSON form of AddConstraint."""

    kind: Literal["add_constraint"]
    constraint: NewHardConstraintData | NewSoftConstraintData


class RemoveConstraintData(DataModel):
    """JSON form of RemoveConstraint."""

    kind: Literal["remove_constraint"]
    constraint_id: ConstraintIdField


class HardRequirementData(DataModel):
    """JSON form of a hard requirement."""

    kind: Literal["hard"]


class SoftRequirementData(DataModel):
    """JSON form of a soft requirement."""

    kind: Literal["soft"]
    strength: Literal["weak", "normal", "strong"]


class AddTimeConstraintData(DataModel):
    """JSON form of AddTimeConstraint."""

    kind: Literal["add_time_constraint"]
    task_id: TaskIdField
    relation: Literal["within", "avoid"]
    windows: tuple[TimeWindowData, ...]
    requirement: HardRequirementData | SoftRequirementData


type CommandData = (
    AddTaskData
    | ReplaceTaskData
    | RemoveTaskData
    | AddConstraintData
    | AddTimeConstraintData
    | RemoveConstraintData
)
"""JSON form of SchedulingCommand."""


class CommandsData(DataModel):
    """JSON data containing commands of either kind."""

    commands: tuple[Annotated[CommandData, Field(discriminator="kind")], ...]


class PeopleFilterData(DataModel):
    """JSON conditions for people."""

    person_ids: tuple[PersonIdField, ...] | None = None
    name_equals: str | None = None


class TasksFilterData(DataModel):
    """JSON conditions for Tasks and FixedTasks."""

    task_ids: tuple[TaskIdField, ...] | None = None
    type: Literal["task", "fixed"] | None = None
    name_equals: str | None = None
    participant_ids_all: tuple[PersonIdField, ...] | None = None
    start_range: TimeIntervalData | None = None


class ConstraintsFilterData(DataModel):
    """JSON conditions for constraints."""

    constraint_ids: tuple[ConstraintIdField, ...] | None = None
    task_ids: tuple[TaskIdField, ...] | None = None


class PreviousScheduleFilterData(DataModel):
    """JSON conditions for the previous schedule."""

    task_ids: tuple[TaskIdField, ...] | None = None
    start_range: TimeIntervalData | None = None


class SummaryQueryData(DataModel):
    """JSON form of SummaryQuery."""

    kind: Literal["summary"]


class PeopleQueryData(DataModel):
    """JSON form of PeopleQuery."""

    kind: Literal["people"]
    filter: PeopleFilterData
    select: Literal["all", "one"] = "all"
    limit: int = Field(20, ge=1, le=100)


class TasksQueryData(DataModel):
    """JSON form of TasksQuery."""

    kind: Literal["tasks"]
    filter: TasksFilterData
    select: Literal["all", "one"] = "all"
    limit: int = Field(20, ge=1, le=100)


class ConstraintsQueryData(DataModel):
    """JSON form of ConstraintsQuery."""

    kind: Literal["constraints"]
    filter: ConstraintsFilterData
    limit: int = Field(20, ge=1, le=100)


class PreviousScheduleQueryData(DataModel):
    """JSON form of PreviousScheduleQuery."""

    kind: Literal["previous_schedule"]
    filter: PreviousScheduleFilterData
    limit: int = Field(20, ge=1, le=100)


class AvailableStartsQueryData(DataModel):
    """JSON form of AvailableStartsQuery."""

    kind: Literal["available_starts"]
    participant_ids: tuple[PersonIdField, ...]
    duration: timedelta
    windows: tuple[TimeWindowData, ...] | None = None
    limit: int = Field(20, ge=1, le=100)


type QueryData = (
    SummaryQueryData
    | PeopleQueryData
    | TasksQueryData
    | ConstraintsQueryData
    | PreviousScheduleQueryData
    | AvailableStartsQueryData
)
"""JSON form of SchedulingQuery."""


def convert_query(data: QueryData) -> SchedulingQuery:
    """Convert a structured scheduling query."""
    raise NotImplementedError


def convert_time_window(data: TimeWindowData) -> TimeWindow:
    """Convert a structured time window."""
    raise NotImplementedError


def command_record(command: SchedulingCommand, grid: TimeGrid) -> dict[str, str]:
    """Describe the ID created or touched by an applied command."""
    task: Task
    task_id: TaskId
    constraint: Constraint
    constraint_id: ConstraintId
    match command:
        case AddTask(task=task):
            return {"kind": "add_task", "task_id": task.id.value, "name": task.name}
        case ReplaceTask(task=task):
            return {"kind": "replace_task", "task_id": task.id.value}
        case RemoveTask(task_id=task_id):
            return {"kind": "remove_task", "task_id": task_id.value}
        case AddConstraint(constraint=constraint):
            return {"kind": "add_constraint", "constraint_id": constraint.id.value}
        case AddTimeConstraint():
            # Record the constraint ID and a slot-rounding note from expand(...).rounded.
            raise NotImplementedError
        case RemoveConstraint(constraint_id=constraint_id):
            return {"kind": "remove_constraint", "constraint_id": constraint_id.value}


def answer_record(answer: Answer) -> dict[str, object]:
    """Convert a query answer to its JSON record."""
    raise NotImplementedError


def convert_command(data: CommandData) -> SchedulingCommand:
    """Convert one structured command, generating an ID for an added object."""
    new_task: NewTaskData
    task: TaskData
    task_id: TaskId
    constraint: NewHardConstraintData | NewSoftConstraintData
    constraint_id: ConstraintId
    relation: Literal["within", "avoid"]
    windows: tuple[TimeWindowData, ...]
    requirement: HardRequirementData | SoftRequirementData
    match data:
        case AddTaskData(task=new_task):
            return AddTask(convert_task(TaskId.generate(), new_task))
        case ReplaceTaskData(task=task):
            return ReplaceTask(convert_task(task.id, task))
        case RemoveTaskData(task_id=task_id):
            return RemoveTask(task_id)
        case AddConstraintData(constraint=constraint):
            return AddConstraint(convert_constraint(ConstraintId.generate(), constraint))
        case AddTimeConstraintData(
            task_id=task_id, relation=relation, windows=windows, requirement=requirement
        ):
            return AddTimeConstraint(
                ConstraintId.generate(),
                task_id,
                TimeRelation(relation),
                tuple(convert_time_window(window) for window in windows),
                Strength(requirement.strength)
                if isinstance(requirement, SoftRequirementData)
                else None,
            )
        case RemoveConstraintData(constraint_id=constraint_id):
            return RemoveConstraint(constraint_id)


def convert_commands_input(data: CommandsData) -> tuple[SchedulingCommand, ...]:
    """Convert a structured list of commands to scheduling commands."""
    return tuple(convert_command(command) for command in data.commands)


def convert_task(task_id: TaskId, data: NewTaskData) -> Task:
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


def convert_fixed_task(task_id: TaskId, data: NewFixedTaskData) -> FixedTask:
    """Convert a structured FixedTask value."""
    return FixedTask(
        task_id,
        data.name,
        data.start,
        data.duration,
        frozenset(data.participant_ids),
    )


def convert_constraint(
    constraint_id: ConstraintId, data: NewHardConstraintData | NewSoftConstraintData
) -> Constraint:
    """Convert a structured constraint value."""
    match data:
        case NewHardConstraintData():
            return HardConstraint(
                constraint_id,
                convert_measure(data.measure),
                convert_evaluation(data.evaluation),
            )
        case NewSoftConstraintData():
            return SoftConstraint(
                constraint_id,
                convert_measure(data.measure),
                convert_evaluation(data.evaluation),
                Strength(data.strength),
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
        id=task.id,
        name=task.name,
        duration=task.duration,
        participant_ids=tuple(
            sorted(task.participant_ids, key=lambda item: item.value)
        ),
        importance=task.importance.value,
        required=task.required,
        stability=task.stability.value,
    )


def to_fixed_task_data(task: FixedTask) -> FixedTaskData:
    """Convert a domain fixed task to its data form."""
    return FixedTaskData(
        id=task.id,
        name=task.name,
        start=task.start,
        duration=task.duration,
        participant_ids=tuple(sorted(task.participant_ids, key=lambda item: item.value)),
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


def to_constraint_data(
    constraint: Constraint,
) -> HardConstraintData | SoftConstraintData:
    """Convert a domain constraint to its data form."""
    measure: MeasureData = to_measure_data(constraint.measure)
    evaluation: EvaluationData = to_evaluation_data(constraint.evaluation)
    match constraint:
        case HardConstraint(id=identifier):
            return HardConstraintData(
                id=identifier, kind="hard", measure=measure, evaluation=evaluation
            )
        case SoftConstraint(id=identifier, strength=strength):
            return SoftConstraintData(
                id=identifier,
                kind="soft",
                measure=measure,
                evaluation=evaluation,
                strength=strength.value,
            )
