from calendar import Day
from datetime import date, datetime, time, timedelta
from typing import Annotated, Literal, cast

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    PlainValidator,
    StrictInt,
    TypeAdapter,
    ValidationInfo,
    WithJsonSchema,
    field_serializer,
    field_validator,
    model_validator,
)

from intent_to_schedule.application.command import (
    AddConstraint,
    AddTask,
    RemoveConstraint,
    RemoveTask,
    ReplaceConstraint,
    ReplaceTask,
    SchedulingCommand,
)
from intent_to_schedule.application.objective import ConstraintEvaluation, ScheduleSummary
from intent_to_schedule.application.query import (
    Answer,
    AvailableStartsAnswer,
    AvailableStartsQuery,
    ConstraintsAnswer,
    ConstraintsQuery,
    EvaluationAnswer,
    EvaluationQuery,
    ObjectivePolicyAnswer,
    ObjectivePolicyQuery,
    PeopleAnswer,
    PeopleQuery,
    PreviousScheduleAnswer,
    PreviousScheduleQuery,
    SchedulingQuery,
    Summary,
    SummaryQuery,
    TasksAnswer,
    TasksQuery,
    TaskType,
)
from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import (
    Condition,
    DailyLimitCondition,
    TaskGapCondition,
    TaskGapRelation,
    TimeBoundCondition,
    TimeBoundRelation,
    TimeWindowCondition,
)
from intent_to_schedule.domain.constraint import (
    Constraint,
    ConstraintId,
    HardConstraint,
    SoftConstraint,
)
from intent_to_schedule.domain.measure import AggregateQuantity, Boundary
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.schedule import DroppedTask, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId
from intent_to_schedule.domain.time_windows import (
    DateRange,
    Expansion,
    TimeRange,
    TimeRelation,
    TimeWindow,
    expand,
)
from intent_to_schedule.domain.violation import ViolationPart, ViolationUnit


def _parse_id(
    value: object, id_type: type[TaskId] | type[PersonId] | type[ConstraintId]
) -> TaskId | PersonId | ConstraintId:
    """Accept a string or an existing domain ID."""
    if isinstance(value, id_type):
        return value
    if isinstance(value, str) and value:
        return id_type(value)
    raise ValueError("ID must be a non-empty string")


type TaskIdField = Annotated[
    TaskId,
    PlainValidator(lambda value: _parse_id(value, TaskId)),
    PlainSerializer(lambda value: value.value),
    WithJsonSchema(
        {"type": "string", "description": "Non-empty identifier of a movable or fixed task."}
    ),
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
    WithJsonSchema(
        {"type": "string", "description": "Non-empty identifier of a constraint."}
    ),
]
"""Constraint ID encoded as a JSON string."""

type TimeOfDayField = Annotated[
    time,
    WithJsonSchema(
        {"type": "string", "pattern": "^([01][0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?$"}
    ),
]
"""Time of day without a UTC offset, encoded as HH:MM or HH:MM:SS."""


class DataModel(BaseModel):
    """Base of the JSON data models."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class ScheduledTaskData(DataModel):
    """JSON form of a scheduled task."""

    status: Literal["scheduled"]
    task_id: TaskIdField
    name: str
    start: AwareDatetime
    end: AwareDatetime

    @field_serializer("start", "end", when_used="json")
    def serialize_time(self, value: datetime) -> str:
        """Encode a scheduled time with its offset."""
        return value.isoformat()


class DroppedTaskData(DataModel):
    """JSON form of a dropped task."""

    status: Literal["dropped"]
    task_id: TaskIdField
    name: str


type ScheduleEntryData = Annotated[
    ScheduledTaskData | DroppedTaskData, Field(discriminator="status")
]
"""JSON form of a schedule entry."""


def to_schedule_entry_data(item: ScheduledTask | DroppedTask) -> ScheduleEntryData:
    """Convert a schedule entry to its JSON form."""
    if isinstance(item, ScheduledTask):
        return ScheduledTaskData(
            status="scheduled",
            task_id=item.task_id,
            name=item.name,
            start=item.start,
            end=item.end,
        )
    return DroppedTaskData(status="dropped", task_id=item.task_id, name=item.name)


class TimeIntervalData(DataModel):
    """JSON form of TimeInterval."""

    start: AwareDatetime
    end: AwareDatetime

    @model_validator(mode="after")
    def validate_times(self) -> "TimeIntervalData":
        """Validate the interval endpoints."""
        TimeInterval(self.start, self.end)
        return self


class DateRangeData(DataModel):
    """Calendar dates from an inclusive start to an exclusive end."""

    start: date = Field(description="First included date, written as YYYY-MM-DD.")
    end: date = Field(description="First excluded date, written as YYYY-MM-DD.")

    @model_validator(mode="after")
    def validate_dates(self) -> "DateRangeData":
        """Validate the date range endpoints."""
        DateRange(self.start, self.end)
        return self


class TimeRangeData(DataModel):
    """Time of day from an inclusive start to an exclusive end."""

    start: TimeOfDayField = Field(
        description="Included start as HH:MM or HH:MM:SS without an offset, in the calendar horizon's starting offset."
    )
    end: TimeOfDayField | None = Field(
        description="Excluded end in the same local time format, or null for the end of the day. Must be later than start; split overnight ranges into separate windows."
    )

    @model_validator(mode="after")
    def validate_times(self) -> "TimeRangeData":
        """Validate the times of day."""
        TimeRange(self.start, self.end)
        return self


type WeekdayField = Annotated[
    Literal[
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ],
    Field(description="Named weekday in the calendar horizon's starting offset."),
]
"""Weekday encoded as a JSON string."""


class TimeWindowData(DataModel):
    """Times matching all supplied date, weekday, and time of day fields."""

    date_range: DateRangeData | None = Field(
        default=None,
        description="Included start date and excluded end date. Omit or use null for the whole horizon."
    )
    weekdays: tuple[WeekdayField, ...] | None = Field(
        default=None,
        description="Allowed named weekdays: monday, tuesday, wednesday, thursday, friday, saturday, sunday. Omit or use null for every day; an empty array matches no day."
    )
    time_range: TimeRangeData | None = Field(
        default=None,
        description="Included start time and excluded end time on each matching date. Omit or use null for the whole day."
    )


class PersonData(DataModel):
    """JSON form of Person."""

    id: PersonIdField
    name: str


class TaskContentData(DataModel):
    """Fields shared by the added and stored JSON forms of Task."""

    name: str
    duration: timedelta
    participant_ids: tuple[PersonIdField, ...]
    importance: Literal["low", "medium", "high"]
    required: bool
    stability: Literal["weak", "normal", "strong"]


class NewTaskData(TaskContentData):
    """JSON form of a Task to add, with an optional given ID."""

    id: TaskIdField | None = Field(
        default=None,
        description="Identifier to use instead of a generated one; later commands in the same batch can reference it.",
    )


class TaskData(TaskContentData):
    """JSON form of Task."""

    id: TaskIdField


class FixedTaskContentData(DataModel):
    """Fields shared by the added and stored JSON forms of FixedTask."""

    name: str
    start: AwareDatetime
    duration: timedelta
    participant_ids: tuple[PersonIdField, ...]


class NewFixedTaskData(FixedTaskContentData):
    """JSON form of a FixedTask to add, with an optional given ID."""

    id: TaskIdField | None = Field(
        default=None,
        description="Identifier to use instead of a generated one; later commands in the same batch can reference it.",
    )


class FixedTaskData(FixedTaskContentData):
    """JSON form of FixedTask."""

    id: TaskIdField


class TimeWindowConditionData(DataModel):
    """Whole tasks kept within or away from the combined windows."""

    kind: Literal["time_window"] = Field(
        description="time_window restricts whole task intervals, rather than only their start times."
    )
    task_ids: tuple[TaskIdField, ...] = Field(
        min_length=1,
        description="Existing movable or fixed task identifiers sharing this condition. Unscheduled tasks have no violation."
    )
    relation: Literal["within", "avoid"] = Field(
        description="within keeps each whole task inside the combined windows; avoid keeps it from overlapping them. Soft violations sum hours outside for within, or overlapping for avoid, across tasks."
    )
    windows: tuple[TimeWindowData, ...] = Field(
        description="Alternative windows combined with or, clipped to the horizon and merged. within rounds inward to whole slots; avoid rounds outward to touched slots; an empty expansion is rejected."
    )


class TimeBoundConditionData(DataModel):
    """Start or end bounds for each scheduled task."""

    kind: Literal["time_bound"] = Field(
        description="time_bound compares each task's selected boundary with one date and time."
    )
    task_ids: tuple[TaskIdField, ...] = Field(
        min_length=1,
        description="Existing movable or fixed task identifiers sharing this bound. Unscheduled tasks have no violation; soft violations sum across scheduled tasks."
    )
    boundary: Literal["start", "end"] = Field(
        description="start is when the task begins; end is when it finishes. Fixed tasks use their real interval without rounding."
    )
    relation: Literal["at_or_before", "at_or_after", "at"] = Field(
        description="at_or_before is an inclusive latest time; at_or_after is an inclusive earliest time; at is exact equality. Soft violations are hours late, early, or away from the target, respectively."
    )
    at: AwareDatetime = Field(
        description="Target date and time, such as 2026-10-19T17:00:00+09:00; include the calendar offset. Compared without rounding; hard at is infeasible for a required movable task if the target is between slot boundaries."
    )


class TaskGapConditionData(DataModel):
    """Minimum or exact gap from one task's end to another's start."""

    kind: Literal["task_gap"] = Field(
        description="task_gap applies only when both tasks are scheduled; otherwise it has no violation."
    )
    from_task_id: TaskIdField = Field(
        description="Existing task that comes first; its end starts the gap. Fixed tasks use their real interval without rounding."
    )
    to_task_id: TaskIdField = Field(
        description="Existing task that comes second; its start ends the gap. Fixed tasks use their real interval without rounding."
    )
    relation: Literal["at_least", "exactly"] = Field(
        description="at_least sets an inclusive minimum gap; exactly sets an equal gap. Soft violations are hours short of the minimum or hours away from the exact gap, respectively."
    )
    gap: timedelta = Field(
        ge=timedelta(0),
        description="Non-negative duration, such as PT30M or PT1H, compared without rounding. PT0S with at_least orders tasks; PT0S with exactly asks for the second task immediately after the first."
    )


class DailyLimitConditionData(DataModel):
    """Inclusive daily maximum over the listed scheduled tasks."""

    kind: Literal["daily_limit"] = Field(
        description="daily_limit caps every calendar date intersecting the half-open horizon in its starting offset, even dates without slot starts. An end exactly at midnight excludes that following date. Tasks starting on other dates contribute zero. Soft violations sum daily excess counts or hours."
    )
    task_ids: tuple[TaskIdField, ...] = Field(
        min_length=1,
        description="Existing movable or fixed tasks to count; unscheduled tasks contribute zero. Only these tasks are included; list a person's meeting tasks to cap that person's new meetings."
    )
    quantity: Literal["count", "total_duration"] = Field(
        description="count counts tasks; total_duration sums their whole durations on their start date, even across midnight. Fixed tasks use their real durations and start dates without rounding. A fixed task counts when its start date is covered, even if its start time is outside the horizon; durations are not clipped."
    )
    maximum: StrictInt | timedelta = Field(
        description="Inclusive non-negative maximum: a JSON integer for count, or a duration string such as PT4H for total_duration. Zero is allowed; duration maxima need not align to slots."
    )

    @field_validator("maximum", mode="before")
    @classmethod
    def validate_maximum_type(
        cls, value: object, information: ValidationInfo
    ) -> object:
        """Require the maximum type for the selected quantity."""
        if information.data.get("quantity") == "count":
            if type(value) is not int:
                raise ValueError("Daily count maximum must be a non-negative integer.")
        elif not isinstance(value, (str, timedelta)):
            raise ValueError("Daily duration maximum must be a non-negative duration.")
        return value

    @field_validator("maximum")
    @classmethod
    def validate_maximum_sign(cls, value: int | timedelta) -> int | timedelta:
        """Require a non-negative maximum."""
        if (isinstance(value, int) and value < 0) or (
            isinstance(value, timedelta) and value < timedelta(0)
        ):
            raise ValueError("Daily maximum must be non-negative.")
        return value


type ConditionData = Annotated[
    TimeWindowConditionData
    | TimeBoundConditionData
    | TaskGapConditionData
    | DailyLimitConditionData,
    Field(description="One time window, time bound, task gap, or daily limit condition."),
]
"""A time window, time bound, task gap, or daily limit."""


class HardRequirementData(DataModel):
    """A condition that must have zero violation."""

    kind: Literal["hard"] = Field(
        description="hard requires the condition to hold for scheduled tasks. It does not require placement; use required: true on a movable task to require scheduling it."
    )


class SoftRequirementData(DataModel):
    """A preference whose weighted violation adds to the solve cost."""

    kind: Literal["soft"] = Field(
        description="soft permits violations and penalizes them alongside other preferences, dropping optional tasks, and moving previous placements. Unscheduled tasks have no condition violation."
    )
    strength: Literal["weak", "normal", "strong"] = Field(
        description="weak gives a low penalty weight; normal gives a medium weight; strong gives a high weight. These trade off total costs and do not make the condition hard."
    )


class ConstraintContentData(DataModel):
    """A scheduling condition with a hard requirement or soft preference."""

    label: str | None = Field(
        default=None,
        description="Optional text describing the request, returned by the constraints query. It has no effect on scheduling."
    )
    requirement: HardRequirementData | SoftRequirementData = Field(
        description="hard enforces zero violation; soft adds a violation penalty using its required strength. Neither requires optional tasks to be scheduled."
    )
    condition: ConditionData = Field(
        description="One time_window, time_bound, task_gap, or daily_limit condition. Stored as entered and returned by the constraints query, rather than replaced with rounded windows."
    )


class NewConstraintData(ConstraintContentData):
    """A constraint to add with an optional supplied identifier."""

    id: ConstraintIdField | None = Field(
        default=None,
        description="Identifier to use instead of a generated one; later commands in the same batch can reference it.",
    )


class ConstraintData(ConstraintContentData):
    """A stored constraint or complete replacement with its identifier."""

    id: ConstraintIdField = Field(
        description="Existing constraint identifier returned by apply or the constraints query. Required for replacement; it stays unchanged."
    )


class AddTaskData(DataModel):
    """JSON form of AddTask."""

    kind: Literal["add_task"]
    task: NewTaskData | NewFixedTaskData


class ReplaceTaskData(DataModel):
    """JSON form of ReplaceTask."""

    kind: Literal["replace_task"]
    task: TaskData | FixedTaskData


class RemoveTaskData(DataModel):
    """JSON form of RemoveTask."""

    kind: Literal["remove_task"]
    task_id: TaskIdField


class AddConstraintData(DataModel):
    """Add one hard constraint or soft preference."""

    kind: Literal["add_constraint"] = Field(description="add_constraint creates a constraint.")
    constraint: NewConstraintData = Field(
        description="New constraint with requirement and condition, plus optional id and label. A supplied identifier must be a non-empty string unique among constraints."
    )


class RemoveConstraintData(DataModel):
    """Remove an existing constraint by its identifier."""

    kind: Literal["remove_constraint"] = Field(description="remove_constraint deletes a constraint.")
    constraint_id: ConstraintIdField = Field(description="Identifier of the existing constraint to remove.")


class ReplaceConstraintData(DataModel):
    """Replace an existing constraint's complete content."""

    kind: Literal["replace_constraint"] = Field(description="replace_constraint edits a constraint with the same identifier.")
    constraint: ConstraintData = Field(
        description="Complete replacement with the existing id, requirement, and condition. Include label to retain it; omitting label clears it."
    )


type CommandData = (
    AddTaskData
    | ReplaceTaskData
    | RemoveTaskData
    | AddConstraintData
    | ReplaceConstraintData
    | RemoveConstraintData
)
"""JSON form of SchedulingCommand."""


class CommandsData(DataModel):
    """A batch of task and constraint commands applied in order."""

    commands: tuple[Annotated[CommandData, Field(discriminator="kind")], ...] = Field(
        description="Commands executed in order; the entire accepted batch is saved, or rejected without changing state."
    )


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
    """Filters selecting constraints by identifier and referenced tasks."""

    constraint_ids: tuple[ConstraintIdField, ...] | None = Field(
        default=None, description="Match any listed constraint identifier; omit or use null for all identifiers."
    )
    task_ids: tuple[TaskIdField, ...] | None = Field(
        default=None, description="Match constraints referencing any listed task; omit or use null for all tasks. Each matching constraint appears once."
    )


class PreviousScheduleFilterData(DataModel):
    """JSON conditions for the previous schedule."""

    task_ids: tuple[TaskIdField, ...] | None = None
    start_range: TimeIntervalData | None = None


class EvaluationFilterData(ConstraintsFilterData):
    """Filters selecting constraint evaluations."""

    violated_only: bool = Field(
        default=False,
        description="Return only constraints with a positive violation, including hard violations. Task filters select complete constraints and do not shorten breakdowns.",
    )


class EvaluationQueryData(DataModel):
    """Evaluate current constraints against the last saved solution."""

    kind: Literal["evaluation"] = Field(
        description="Returns has_previous, items, total, and truncated. With no saved solution, has_previous is false and items are empty. Each item has constraint_id, label, requirement, violation (amount and unit: hours or count), cost (null for hard), and breakdown (by task for time_window and time_bound, by date for daily_limit, empty for task_gap). Daily breakdowns include every calendar date intersecting the half-open horizon, even dates without slot starts and dates with zero violation. Hard violations come first, then soft costs highest first; satisfied hard constraints come last. Fixed tasks use current real intervals without rounding; missing movable tasks are unscheduled. Saved movable intervals are used even after task edits.",
    )
    filter: EvaluationFilterData = Field(
        default_factory=EvaluationFilterData,
        description="Optional filters combined with and; omitted filters match all constraints. Each selected constraint and its complete breakdown appears once.",
    )
    limit: int = Field(
        20,
        description="Limit from 1 to 100; out-of-range values are rejected. total counts matches before limiting; truncated indicates omitted items. Only constraints are limited; their breakdowns are complete.",
    )


class ObjectivePolicyQueryData(DataModel):
    """Read the default solver's current objective coefficients."""

    kind: Literal["objective_policy"] = Field(
        description="Returns drop_costs by importance, weights by strength, per_count scaling for daily count violations, and stability_drop_cost_ratio. Total cost adds dropped optional task costs, weighted soft violations in hours (counts scaled by per_count), and weighted hours moved capped at stability_drop_cost_ratio times drop cost per task. Required tasks also have capped stability costs. Hard constraints require zero violation; unscheduled tasks have no constraint or stability cost.",
    )


class SummaryQueryData(DataModel):
    """JSON form of SummaryQuery."""

    kind: Literal["summary"]


class PeopleQueryData(DataModel):
    """JSON form of PeopleQuery."""

    kind: Literal["people"]
    filter: PeopleFilterData = Field(default_factory=PeopleFilterData)
    select: Literal["all", "one"] = "all"
    limit: int = Field(
        20, description="Limit from 1 to 100; out-of-range values are rejected."
    )


class TasksQueryData(DataModel):
    """JSON form of TasksQuery."""

    kind: Literal["tasks"]
    filter: TasksFilterData = Field(default_factory=TasksFilterData)
    select: Literal["all", "one"] = "all"
    limit: int = Field(
        20, description="Limit from 1 to 100; out-of-range values are rejected."
    )


class ConstraintsQueryData(DataModel):
    """Read complete stored constraints with their entered conditions."""

    kind: Literal["constraints"] = Field(
        description="constraints returns id, label, requirement, and condition, including windows before rounding."
    )
    filter: ConstraintsFilterData = Field(
        default_factory=ConstraintsFilterData,
        description="Optional filters combined with and; omitted filters match all constraints."
    )
    limit: int = Field(
        20, description="Limit from 1 to 100; out-of-range values are rejected."
    )


class PreviousScheduleQueryData(DataModel):
    """JSON form of PreviousScheduleQuery."""

    kind: Literal["previous_schedule"]
    filter: PreviousScheduleFilterData = Field(
        default_factory=PreviousScheduleFilterData
    )
    limit: int = Field(
        20, description="Limit from 1 to 100; out-of-range values are rejected."
    )


class AvailableStartsQueryData(DataModel):
    """JSON form of AvailableStartsQuery."""

    kind: Literal["available_starts"]
    participant_ids: tuple[PersonIdField, ...]
    duration: timedelta
    windows: tuple[TimeWindowData, ...] | None = None
    limit: int = Field(
        20, description="Limit from 1 to 100; out-of-range values are rejected."
    )


type QueryData = (
    SummaryQueryData
    | PeopleQueryData
    | TasksQueryData
    | ConstraintsQueryData
    | PreviousScheduleQueryData
    | AvailableStartsQueryData
    | EvaluationQueryData
    | ObjectivePolicyQueryData
)
"""JSON form of SchedulingQuery."""


def convert_query(data: QueryData) -> SchedulingQuery:
    """Convert a structured scheduling query."""
    match data:
        case EvaluationQueryData():
            return EvaluationQuery(
                data.filter.violated_only,
                frozenset(data.filter.constraint_ids)
                if data.filter.constraint_ids is not None
                else None,
                frozenset(data.filter.task_ids)
                if data.filter.task_ids is not None
                else None,
                data.limit,
            )
        case ObjectivePolicyQueryData():
            return ObjectivePolicyQuery()
        case SummaryQueryData():
            return SummaryQuery()
        case PeopleQueryData():
            return PeopleQuery(
                frozenset(data.filter.person_ids)
                if data.filter.person_ids is not None
                else None,
                data.filter.name_equals,
                data.select == "one",
                data.limit,
            )
        case TasksQueryData():
            return TasksQuery(
                frozenset(data.filter.task_ids)
                if data.filter.task_ids is not None
                else None,
                TaskType(data.filter.type) if data.filter.type is not None else None,
                data.filter.name_equals,
                frozenset(data.filter.participant_ids_all)
                if data.filter.participant_ids_all is not None
                else None,
                TimeInterval(data.filter.start_range.start, data.filter.start_range.end)
                if data.filter.start_range is not None
                else None,
                data.select == "one",
                data.limit,
            )
        case ConstraintsQueryData():
            return ConstraintsQuery(
                frozenset(data.filter.constraint_ids)
                if data.filter.constraint_ids is not None
                else None,
                frozenset(data.filter.task_ids)
                if data.filter.task_ids is not None
                else None,
                data.limit,
            )
        case PreviousScheduleQueryData():
            return PreviousScheduleQuery(
                frozenset(data.filter.task_ids)
                if data.filter.task_ids is not None
                else None,
                TimeInterval(data.filter.start_range.start, data.filter.start_range.end)
                if data.filter.start_range is not None
                else None,
                data.limit,
            )
        case AvailableStartsQueryData():
            return AvailableStartsQuery(
                frozenset(data.participant_ids),
                data.duration,
                tuple(convert_time_window(window) for window in data.windows)
                if data.windows is not None
                else None,
                data.limit,
            )


def convert_time_window(data: TimeWindowData) -> TimeWindow:
    """Convert a structured time window."""
    return TimeWindow(
        DateRange(data.date_range.start, data.date_range.end)
        if data.date_range is not None
        else None,
        frozenset(Day[weekday.upper()] for weekday in data.weekdays)
        if data.weekdays is not None
        else None,
        TimeRange(data.time_range.start, data.time_range.end)
        if data.time_range is not None
        else None,
    )


def command_record(command: SchedulingCommand, grid: TimeGrid) -> dict[str, str]:
    """Describe the ID created or touched by an applied command."""
    task: Task | FixedTask
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
        case (
            AddConstraint(constraint=constraint)
            | ReplaceConstraint(constraint=constraint)
        ):
            record: dict[str, str] = {
                "kind": "add_constraint"
                if isinstance(command, AddConstraint)
                else "replace_constraint",
                "constraint_id": constraint.id.value,
            }
            condition: Condition = constraint.condition
            if isinstance(condition, TimeWindowCondition):
                expansion: Expansion = expand(
                    condition.windows, condition.relation, grid
                )
                if expansion.rounded:
                    direction: str = (
                        "inward"
                        if condition.relation is TimeRelation.WITHIN
                        else "outward"
                    )
                    record["note"] = (
                        f"Window times were rounded {direction} to calendar slots."
                    )
            return record
        case RemoveConstraint(constraint_id=constraint_id):
            return {"kind": "remove_constraint", "constraint_id": constraint_id.value}


def schedule_summary_record(summary: ScheduleSummary) -> dict[str, object]:
    """Convert a solved schedule summary to its JSON record."""
    return {
        "total_cost": summary.total_cost,
        "costs": {
            "dropped_tasks": summary.dropped_tasks_cost,
            "soft_constraints": summary.soft_constraints_cost,
            "stability": summary.stability_cost,
        },
        "counts": {
            "scheduled_tasks": summary.scheduled_tasks,
            "dropped_tasks": summary.dropped_tasks,
            "violated_soft_constraints": summary.violated_soft_constraints,
            "moved_tasks": summary.moved_tasks,
        },
    }


def _violation_record(amount: float, unit: ViolationUnit) -> dict[str, object]:
    """Describe a violation amount and its unit."""
    return {"amount": amount, "unit": unit}


def _violation_part_record(
    part: ViolationPart, item: ConstraintEvaluation
) -> dict[str, object]:
    """Describe a task or date's violation and cost."""
    record: dict[str, object] = {
        "violation": _violation_record(part.amount, item.violation.unit),
        "cost": part.amount * item.coefficient if item.coefficient is not None else None,
    }
    if part.task_id is not None:
        return {"task_id": part.task_id.value, **record}
    assert part.calendar_date is not None
    return {"date": part.calendar_date.isoformat(), **record}


def _evaluation_record(item: ConstraintEvaluation) -> dict[str, object]:
    """Describe a constraint evaluation with its breakdown."""
    constraint: Constraint = item.constraint
    requirement: dict[str, str] = (
        {"kind": "soft", "strength": constraint.strength.value}
        if isinstance(constraint, SoftConstraint)
        else {"kind": "hard"}
    )
    return {
        "constraint_id": constraint.id.value,
        "label": constraint.label,
        "requirement": requirement,
        "violation": _violation_record(item.violation.amount, item.violation.unit),
        "cost": item.cost,
        "breakdown": [
            _violation_part_record(part, item) for part in item.violation.breakdown
        ],
    }


def answer_record(answer: Answer) -> dict[str, object]:
    """Convert a query answer to its JSON record."""
    if isinstance(answer, ObjectivePolicyAnswer):
        return {
            "kind": "objective_policy",
            "drop_costs": {
                key.value: value for key, value in answer.policy.drop_costs.items()
            },
            "weights": {key.value: value for key, value in answer.policy.weights.items()},
            "per_count": answer.policy.per_count,
            "stability_drop_cost_ratio": answer.policy.stability_drop_cost_ratio,
        }
    if isinstance(answer, Summary):
        return {
            "kind": "summary",
            "grid": {
                "horizon": to_time_interval_data(answer.grid.horizon).model_dump(
                    mode="json"
                ),
                "slot": TypeAdapter(timedelta).dump_python(
                    answer.grid.slot, mode="json"
                ),
            },
            "counts": {
                "people": answer.people,
                "tasks": answer.tasks,
                "fixed_tasks": answer.fixed_tasks,
                "constraints": answer.constraints,
            },
            "has_previous": answer.has_previous,
        }
    record: dict[str, object] = {"total": answer.total, "truncated": answer.truncated}
    match answer:
        case EvaluationAnswer():
            record.update(
                kind="evaluation",
                has_previous=answer.has_previous,
                items=[_evaluation_record(item) for item in answer.items],
            )
        case PeopleAnswer():
            record.update(
                kind="people",
                items=[
                    PersonData(id=person.id, name=person.name).model_dump(mode="json")
                    for person in answer.items
                ],
            )
        case TasksAnswer():
            record.update(
                kind="tasks",
                items=[
                    (
                        to_task_data(task)
                        if isinstance(task, Task)
                        else to_fixed_task_data(task)
                    ).model_dump(mode="json")
                    for task in answer.items
                ],
            )
        case ConstraintsAnswer():
            record.update(
                kind="constraints",
                items=[
                    to_constraint_data(constraint).model_dump(mode="json")
                    for constraint in answer.items
                ],
            )
        case PreviousScheduleAnswer():
            record.update(
                kind="previous_schedule",
                has_previous=answer.has_previous,
                items=[
                    to_schedule_entry_data(item).model_dump(mode="json")
                    for item in answer.items
                ],
            )
        case AvailableStartsAnswer():
            record.update(
                kind="available_starts",
                items=[start.isoformat() for start in answer.items],
                note="Movable Tasks and constraints are not considered; use solve for the final schedule.",
            )
    return {"kind": record["kind"], **record}


def convert_command(data: CommandData) -> SchedulingCommand:
    """Convert one structured command, generating an ID for an added object without one."""
    new_task: NewTaskData | NewFixedTaskData
    task: TaskData | FixedTaskData
    task_id: TaskId
    constraint: NewConstraintData | ConstraintData
    constraint_id: ConstraintId
    match data:
        case AddTaskData(task=new_task):
            task_id = new_task.id if new_task.id is not None else TaskId.generate()
            return AddTask(
                convert_fixed_task(task_id, new_task)
                if isinstance(new_task, NewFixedTaskData)
                else convert_task(task_id, new_task)
            )
        case ReplaceTaskData(task=task):
            return ReplaceTask(
                convert_fixed_task(task.id, task)
                if isinstance(task, FixedTaskData)
                else convert_task(task.id, task)
            )
        case RemoveTaskData(task_id=task_id):
            return RemoveTask(task_id)
        case AddConstraintData(constraint=constraint):
            return AddConstraint(
                convert_constraint(
                    constraint.id
                    if constraint.id is not None
                    else ConstraintId.generate(),
                    constraint,
                )
            )
        case ReplaceConstraintData(constraint=constraint):
            return ReplaceConstraint(convert_constraint(constraint.id, constraint))
        case RemoveConstraintData(constraint_id=constraint_id):
            return RemoveConstraint(constraint_id)


def convert_commands_input(data: CommandsData) -> tuple[SchedulingCommand, ...]:
    """Convert a structured list of commands to scheduling commands."""
    return tuple(convert_command(command) for command in data.commands)


def convert_task(task_id: TaskId, data: TaskContentData) -> Task:
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


def convert_fixed_task(task_id: TaskId, data: FixedTaskContentData) -> FixedTask:
    """Convert a structured FixedTask value."""
    return FixedTask(
        task_id,
        data.name,
        data.start,
        data.duration,
        frozenset(data.participant_ids),
    )


def convert_constraint(
    constraint_id: ConstraintId, data: ConstraintContentData
) -> Constraint:
    """Convert a structured constraint value."""
    condition: Condition = convert_condition(data.condition)
    if isinstance(data.requirement, SoftRequirementData):
        return SoftConstraint(
            constraint_id, condition, Strength(data.requirement.strength), data.label
        )
    return HardConstraint(constraint_id, condition, data.label)


def convert_condition(data: ConditionData) -> Condition:
    """Convert a structured condition value."""
    match data:
        case TimeWindowConditionData():
            return TimeWindowCondition(
                frozenset(data.task_ids),
                TimeRelation(data.relation),
                tuple(convert_time_window(window) for window in data.windows),
            )
        case TimeBoundConditionData():
            return TimeBoundCondition(
                frozenset(data.task_ids),
                Boundary(data.boundary),
                TimeBoundRelation(data.relation),
                data.at,
            )
        case TaskGapConditionData():
            return TaskGapCondition(
                data.from_task_id,
                data.to_task_id,
                TaskGapRelation(data.relation),
                data.gap,
            )
        case DailyLimitConditionData():
            return DailyLimitCondition(
                frozenset(data.task_ids), AggregateQuantity(data.quantity), data.maximum
            )


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
        participant_ids=tuple(
            sorted(task.participant_ids, key=lambda item: item.value)
        ),
    )


def to_time_window_data(window: TimeWindow) -> TimeWindowData:
    """Convert a domain time window to its data form."""
    return TimeWindowData(
        date_range=DateRangeData(
            start=window.date_range.start, end=window.date_range.end
        )
        if window.date_range is not None
        else None,
        weekdays=tuple(
            cast(WeekdayField, day.name.lower()) for day in sorted(window.weekdays)
        )
        if window.weekdays is not None
        else None,
        time_range=TimeRangeData(
            start=window.time_range.start, end=window.time_range.end
        )
        if window.time_range is not None
        else None,
    )


def to_condition_data(condition: Condition) -> ConditionData:
    """Convert a domain condition to its data form."""
    task_ids: tuple[TaskId, ...] = tuple(
        sorted(condition.task_ids, key=lambda item: item.value)
    )
    match condition:
        case TimeWindowCondition():
            return TimeWindowConditionData(
                kind="time_window",
                task_ids=task_ids,
                relation=condition.relation.value,
                windows=tuple(
                    to_time_window_data(window) for window in condition.windows
                ),
            )
        case TimeBoundCondition():
            return TimeBoundConditionData(
                kind="time_bound",
                task_ids=task_ids,
                boundary=condition.boundary.value,
                relation=condition.relation.value,
                at=condition.at,
            )
        case TaskGapCondition():
            return TaskGapConditionData(
                kind="task_gap",
                from_task_id=condition.from_task_id,
                to_task_id=condition.to_task_id,
                relation=condition.relation.value,
                gap=condition.gap,
            )
        case DailyLimitCondition():
            return DailyLimitConditionData(
                kind="daily_limit",
                task_ids=task_ids,
                quantity=condition.quantity.value,
                maximum=condition.maximum,
            )


def to_constraint_data(constraint: Constraint) -> ConstraintData:
    """Convert a domain constraint to its data form."""
    requirement: HardRequirementData | SoftRequirementData = (
        SoftRequirementData(kind="soft", strength=constraint.strength.value)
        if isinstance(constraint, SoftConstraint)
        else HardRequirementData(kind="hard")
    )
    return ConstraintData(
        id=constraint.id,
        label=constraint.label,
        requirement=requirement,
        condition=to_condition_data(constraint.condition),
    )
