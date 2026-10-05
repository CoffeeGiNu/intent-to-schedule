"""Tests for the JSON data models and their conversions."""

import json
from calendar import Day
from datetime import date, datetime, time, timedelta, timezone
from typing import Annotated, Any, cast
from unittest.mock import Mock, patch

import pytest
from pydantic import Field, TypeAdapter, ValidationError

from intent_to_schedule.adapter.data_model import (
    AddConstraintData,
    AddTaskData,
    CommandData,
    CommandsData,
    DataModel,
    DateRangeData,
    FixedTaskContentData,
    FixedTaskData,
    NewFixedTaskData,
    NewTaskData,
    QueryData,
    ReplaceTaskData,
    ScheduledTaskData,
    TaskContentData,
    TaskData,
    TimeBoundConditionData,
    TimeIntervalData,
    TimeWindowData,
    answer_record,
    command_record,
    convert_command,
    convert_commands_input,
    convert_fixed_task,
    convert_query,
    convert_time_window,
    to_fixed_task_data,
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
from intent_to_schedule.application.policy import DEFAULT_POLICY
from intent_to_schedule.application.query import (
    Answered,
    AnswerResult,
    AvailableStartsAnswer,
    AvailableStartsQuery,
    ConstraintsAnswer,
    ObjectivePolicyAnswer,
    PeopleAnswer,
    PeopleQuery,
    PreviousScheduleAnswer,
    SchedulingQuery,
    Summary,
    TasksAnswer,
    TasksQuery,
    TaskType,
)
from intent_to_schedule.domain.calendar import (
    Availability,
    Calendar,
    TimeGrid,
    TimeInterval,
)
from intent_to_schedule.domain.condition import (
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
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import DroppedTask, Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId
from intent_to_schedule.domain.time_windows import (
    DateRange,
    TimeRange,
    TimeRelation,
    TimeWindow,
)

START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone(timedelta(hours=9)))
HOUR: timedelta = timedelta(hours=1)
GRID: TimeGrid = TimeGrid(TimeInterval(START, START + 8 * HOUR), HOUR)
EMPTY_PROBLEM: SchedulingProblem = SchedulingProblem(Calendar(GRID, ()), (), (), (), ())
PERSON: PersonId = PersonId("person")
QUERY_ADAPTER: TypeAdapter[QueryData] = TypeAdapter(
    Annotated[QueryData, Field(discriminator="kind")]
)
CONDITIONS: tuple[dict[str, object], ...] = (
    {
        "kind": "time_window",
        "task_ids": ["a", "b"],
        "relation": "within",
        "windows": [{"time_range": {"start": "09:10", "end": "15:20"}}],
    },
    {
        "kind": "time_bound",
        "task_ids": ["a", "b"],
        "boundary": "end",
        "relation": "at_or_before",
        "at": "2026-10-29T15:00:00+09:00",
    },
    {
        "kind": "task_gap",
        "from_task_id": "a",
        "to_task_id": "b",
        "relation": "at_least",
        "gap": "PT0S",
    },
    {"kind": "daily_limit", "task_ids": ["a", "b"], "quantity": "count", "maximum": 2},
    {
        "kind": "daily_limit",
        "task_ids": ["a", "b"],
        "quantity": "total_duration",
        "maximum": "PT4H",
    },
)


def at(hour: int, minute: int = 0) -> datetime:
    """Build a time on the example calendar day."""
    return START.replace(hour=hour, minute=minute)


def parse_query(value: dict[str, object]) -> SchedulingQuery:
    """Parse and convert a JSON query."""
    return convert_query(QUERY_ADAPTER.validate_python(value))


def constraint_command(
    condition: dict[str, object], identifier: str = "c", kind: str = "add_constraint"
) -> dict[str, object]:
    """Build a constraint command with a given identifier."""
    return {
        "kind": kind,
        "constraint": {
            "id": identifier,
            "label": "Entered request",
            "requirement": {"kind": "hard"},
            "condition": condition,
        },
    }


def task_input(fixed: bool) -> dict[str, Any]:
    """Build the selected task input shape."""
    data: dict[str, Any] = {
        "name": "Health check",
        "duration": "PT1H30M",
        "participant_ids": ["ito"],
    }
    if fixed:
        data["start"] = "2026-10-19T10:30:00+09:00"
    else:
        data.update(importance="high", required=True, stability="normal")
    return data


def test_commands_input_converts_mixed_commands() -> None:
    """Convert task and constraint commands in input order."""
    start: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    task_data: dict[str, object] = {
        "name": "Review",
        "duration": 3600,
        "participant_ids": [],
        "importance": "high",
        "required": True,
        "stability": "weak",
    }
    data: CommandsData = CommandsData.model_validate(
        {
            "commands": [
                {"kind": "add_task", "task": {"id": "new", **task_data}},
                {"kind": "replace_task", "task": {"id": "old", **task_data}},
                {"kind": "remove_task", "task_id": "old"},
                {
                    "kind": "add_constraint",
                    "constraint": {
                        "id": "c1",
                        "requirement": {"kind": "hard"},
                        "condition": {
                            "kind": "time_bound",
                            "task_ids": ["old"],
                            "boundary": "start",
                            "relation": "at",
                            "at": start.isoformat(),
                        },
                    },
                },
                {"kind": "remove_constraint", "constraint_id": "obsolete"},
            ]
        }
    )
    commands: tuple[SchedulingCommand, ...] = convert_commands_input(data)
    task: Task = Task(
        TaskId("new"),
        "Review",
        timedelta(hours=1),
        frozenset(),
        Importance.HIGH,
        True,
        Strength.WEAK,
    )
    replacement: Task = Task(
        TaskId("old"),
        "Review",
        timedelta(hours=1),
        frozenset(),
        Importance.HIGH,
        True,
        Strength.WEAK,
    )
    assert commands == (
        AddTask(task),
        ReplaceTask(replacement),
        RemoveTask(TaskId("old")),
        AddConstraint(
            HardConstraint(
                ConstraintId("c1"),
                TimeBoundCondition(
                    frozenset({TaskId("old")}),
                    Boundary.START,
                    TimeBoundRelation.AT,
                    start,
                ),
            )
        ),
        RemoveConstraint(ConstraintId("obsolete")),
    )


def test_fixed_task_data_converts_with_supplied_id_and_round_trips() -> None:
    """Convert a fixed task with its supplied identifier and round trip its data."""
    start: datetime = datetime(2026, 10, 1, 9, 15, tzinfo=timezone.utc)
    data: NewFixedTaskData = NewFixedTaskData.model_validate(
        {
            "name": "Existing meeting",
            "start": start.isoformat(),
            "duration": "PT30M",
            "participant_ids": ["alice", "alice"],
        }
    )
    expected: FixedTask = FixedTask(
        TaskId("fixed"),
        "Existing meeting",
        start,
        timedelta(minutes=30),
        frozenset({PersonId("alice")}),
    )
    assert convert_fixed_task(expected.id, data) == expected
    output: FixedTaskData = to_fixed_task_data(expected)
    assert output.model_dump(mode="json")["participant_ids"] == ["alice"]
    assert output.id == expected.id
    restored: FixedTaskData = FixedTaskData.model_validate_json(
        output.model_dump_json()
    )
    assert convert_fixed_task(restored.id, restored) == expected


@pytest.mark.parametrize("kind", ["add_task", "replace_task"])
@pytest.mark.parametrize("fixed", [False, True])
def test_json_task_shape_and_identifier_generation(kind: str, fixed: bool) -> None:
    """Parse each task shape and generate added identifiers once."""
    task: dict[str, Any] = task_input(fixed)
    if kind == "replace_task":
        task["id"] = "existing"
    data: CommandsData = CommandsData.model_validate_json(
        json.dumps(
            {
                "commands": [{"kind": kind, "task": task}],
            }
        )
    )
    expected_type: type[TaskContentData | FixedTaskContentData] = (
        (FixedTaskData if fixed else TaskData)
        if kind == "replace_task"
        else (NewFixedTaskData if fixed else NewTaskData)
    )
    parsed: CommandData = data.commands[0]
    assert isinstance(parsed, (AddTaskData, ReplaceTaskData))
    assert isinstance(parsed.task, expected_type)
    generation: Mock
    with patch.object(
        TaskId, "generate", return_value=TaskId("generated")
    ) as generation:
        commands: tuple[SchedulingCommand, ...] = convert_commands_input(data)
    command: SchedulingCommand = commands[0]
    assert isinstance(command, (AddTask, ReplaceTask))
    assert isinstance(command, AddTask if kind == "add_task" else ReplaceTask)
    assert isinstance(command.task, FixedTask if fixed else Task)
    assert command.task.id == TaskId("generated" if kind == "add_task" else "existing")
    assert generation.call_count == (1 if kind == "add_task" else 0)
    if kind == "add_task":
        assert command_record(command, EMPTY_PROBLEM) == {
            "kind": "add_task",
            "task_id": "generated",
            "name": "Health check",
        }


@pytest.mark.parametrize("kind", ["add_task", "replace_task"])
@pytest.mark.parametrize(
    "movable_fields",
    [
        {"importance": "high"},
        {"required": True},
        {"stability": "normal"},
        {"importance": "high", "required": True, "stability": "normal"},
    ],
)
def test_json_rejects_mixed_task_shapes(
    kind: str,
    movable_fields: dict[str, object],
) -> None:
    """Reject movable fields mixed into fixed appointments."""
    task: dict[str, Any] = task_input(True)
    task.update(movable_fields)
    if kind == "replace_task":
        task["id"] = "existing"
    with pytest.raises(ValidationError):
        CommandsData.model_validate_json(
            json.dumps({"commands": [{"kind": kind, "task": task}]})
        )


@pytest.mark.parametrize(
    "condition, expected",
    tuple(
        zip(
            CONDITIONS,
            (
                TimeWindowCondition,
                TimeBoundCondition,
                TaskGapCondition,
                DailyLimitCondition,
                DailyLimitCondition,
            ),
            strict=True,
        )
    ),
)
def test_parse_conditions(condition: dict[str, object], expected: type[object]) -> None:
    """Parse every condition and preserve its label."""
    commands: tuple[object, ...] = convert_commands_input(
        CommandsData.model_validate({"commands": [constraint_command(condition)]})
    )
    command: object = commands[0]
    assert isinstance(command, AddConstraint)
    assert isinstance(command.constraint.condition, expected)
    assert command.constraint.label == "Entered request"


@pytest.mark.parametrize(
    "quantity, maximum",
    [
        ("count", True),
        ("count", "PT4H"),
        ("count", "2"),
        ("count", 2.0),
        ("count", -1),
        ("total_duration", 2),
        ("total_duration", False),
        ("total_duration", "-PT1H"),
    ],
)
def test_daily_limit_rejects_wrong_maximum(quantity: str, maximum: object) -> None:
    """Reject a maximum with the wrong type or sign."""
    condition: dict[str, object] = {
        "kind": "daily_limit",
        "task_ids": ["a"],
        "quantity": quantity,
        "maximum": maximum,
    }
    with pytest.raises(ValidationError, match="maximum"):
        CommandsData.model_validate({"commands": [constraint_command(condition)]})


@pytest.mark.parametrize(
    "condition", [CONDITIONS[0], CONDITIONS[1], CONDITIONS[3], CONDITIONS[4]]
)
def test_condition_rejects_empty_task_ids(condition: dict[str, object]) -> None:
    """Reject conditions without task identifiers."""
    with pytest.raises(ValidationError, match="task_ids"):
        CommandsData.model_validate(
            {"commands": [constraint_command({**condition, "task_ids": []})]}
        )


def test_replace_constraint_requires_identifier() -> None:
    """Reject a replacement without an identifier."""
    with pytest.raises(ValidationError):
        CommandsData.model_validate(
            {
                "commands": [
                    {
                        "kind": "replace_constraint",
                        "constraint": {
                            "condition": CONDITIONS[1],
                            "requirement": {"kind": "hard"},
                        },
                    }
                ]
            }
        )


def test_replace_constraint_conversion_keeps_identifier_requirement_and_label() -> None:
    """Convert a replacement without generating a new identifier."""
    data: CommandsData = CommandsData.model_validate(
        {
            "commands": [
                {
                    "kind": "replace_constraint",
                    "constraint": {
                        "id": "first",
                        "label": "Changed request",
                        "requirement": {"kind": "soft", "strength": "strong"},
                        "condition": CONDITIONS[4],
                    },
                }
            ]
        }
    )
    commands: tuple[SchedulingCommand, ...] = convert_commands_input(data)
    assert commands == (
        ReplaceConstraint(
            SoftConstraint(
                ConstraintId("first"),
                DailyLimitCondition(
                    frozenset({TaskId("a"), TaskId("b")}),
                    AggregateQuantity.TOTAL_DURATION,
                    timedelta(hours=4),
                ),
                Strength.STRONG,
                "Changed request",
            )
        ),
    )


@pytest.mark.parametrize("kind", ["add_constraint", "replace_constraint"])
@pytest.mark.parametrize(
    "relation, direction", [("within", "inward"), ("avoid", "outward")]
)
def test_window_rounding_note_on_add_and_replace(
    kind: str, relation: str, direction: str
) -> None:
    """Report the rounding direction for either window command."""
    condition: dict[str, object] = {**CONDITIONS[0], "relation": relation}
    command: object = convert_commands_input(
        CommandsData.model_validate(
            {"commands": [constraint_command(condition, kind=kind)]}
        )
    )[0]
    assert isinstance(command, (AddConstraint, ReplaceConstraint))
    assert command_record(command, EMPTY_PROBLEM) == {
        "kind": kind,
        "constraint_id": "c",
        "note": f"Window times were rounded {direction} to calendar slots.",
    }


@pytest.mark.parametrize("relation", ["within", "avoid"])
def test_window_record_omits_note_for_aligned_times(relation: str) -> None:
    """Report no rounding when window times fall on calendar slots."""
    condition: dict[str, object] = {
        **CONDITIONS[0],
        "relation": relation,
        "windows": [{"time_range": {"start": "09:00", "end": "15:00"}}],
    }
    command: SchedulingCommand = convert_commands_input(
        CommandsData.model_validate({"commands": [constraint_command(condition)]})
    )[0]
    assert command_record(command, EMPTY_PROBLEM) == {
        "kind": "add_constraint",
        "constraint_id": "c",
    }


def test_constraints_record_keeps_entered_windows_before_rounding() -> None:
    """Return the entered condition, not its slot-rounded windows."""
    command: SchedulingCommand = convert_commands_input(
        CommandsData.model_validate({"commands": [constraint_command(CONDITIONS[0])]})
    )[0]
    assert isinstance(command, AddConstraint)
    assert answer_record(ConstraintsAnswer((command.constraint,), 1)) == {
        "kind": "constraints",
        "items": [
            {
                "id": "c",
                "label": "Entered request",
                "requirement": {"kind": "hard"},
                "condition": {
                    **CONDITIONS[0],
                    "windows": [
                        {
                            "date_range": None,
                            "weekdays": None,
                            "time_range": {"start": "09:10:00", "end": "15:20:00"},
                        }
                    ],
                },
            }
        ],
        "total": 1,
        "truncated": False,
    }


@pytest.mark.parametrize(
    "requirement,expected_strength",
    [
        ({"kind": "hard"}, None),
        ({"kind": "soft", "strength": "weak"}, "weak"),
        ({"kind": "soft", "strength": "normal"}, "normal"),
        ({"kind": "soft", "strength": "strong"}, "strong"),
    ],
)
def test_time_window_constraint_conversion_generates_id_once(
    requirement: dict[str, str],
    expected_strength: str | None,
) -> None:
    data: AddConstraintData = AddConstraintData.model_validate(
        {
            "kind": "add_constraint",
            "constraint": {
                "requirement": requirement,
                "condition": {
                    "kind": "time_window",
                    "task_ids": ["review"],
                    "relation": "within",
                    "windows": [
                        {
                            "date_range": {"start": "2026-10-12", "end": "2026-10-17"},
                            "weekdays": ["friday", "friday"],
                            "time_range": {"start": "13:00", "end": None},
                        }
                    ],
                },
            },
        }
    )
    grid: TimeGrid = TimeGrid(
        TimeInterval(
            datetime(2026, 10, 12, tzinfo=timezone.utc),
            datetime(2026, 10, 17, tzinfo=timezone.utc),
        ),
        timedelta(hours=1),
    )
    generation: Mock
    with patch.object(
        ConstraintId, "generate", return_value=ConstraintId("created")
    ) as generation:
        command: SchedulingCommand = convert_command(data)
        assert isinstance(command, AddConstraint)
        assert command.constraint.id == ConstraintId("created")
        assert command.constraint.condition.task_ids == frozenset({TaskId("review")})
        assert isinstance(command.constraint.condition, TimeWindowCondition)
        assert command.constraint.condition.relation is TimeRelation.WITHIN
        assert isinstance(
            command.constraint, SoftConstraint if expected_strength else HardConstraint
        )
        if isinstance(command.constraint, SoftConstraint):
            assert command.constraint.strength == Strength(expected_strength)
        assert command.constraint.condition.windows == (
            TimeWindow(
                DateRange(date(2026, 10, 12), date(2026, 10, 17)),
                frozenset({Day.FRIDAY}),
                TimeRange(time(13), None),
            ),
        )
        assert command_record(
            command, SchedulingProblem(Calendar(grid, ()), (), (), (), ())
        ) == {
            "kind": "add_constraint",
            "constraint_id": "created",
        }
        command.execute(SchedulingProblem(Calendar(grid, ()), (), (), (), ()))
    generation.assert_called_once_with()


def test_time_window_constraint_conversion_preserves_omitted_and_empty_window_fields() -> (
    None
):
    assert convert_time_window(TimeWindowData()) == TimeWindow(None, None, None)
    assert convert_time_window(TimeWindowData(weekdays=())) == TimeWindow(
        None, frozenset(), None
    )


@pytest.mark.parametrize("end", ["13:00", "12:00"])
def test_time_window_constraint_input_rejects_nonincreasing_times(end: str) -> None:
    with pytest.raises(ValidationError) as error:
        AddConstraintData.model_validate(
            {
                "kind": "add_constraint",
                "constraint": {
                    "requirement": {"kind": "hard"},
                    "condition": {
                        "kind": "time_window",
                        "task_ids": ["review"],
                        "relation": "within",
                        "windows": [{"time_range": {"start": "13:00", "end": end}}],
                    },
                },
            }
        )
    message: str = str(error.value)
    assert "windows" in message and "time_range" in message
    assert "13:00" in message and end in message and "after" in message


@pytest.mark.parametrize("start,end", [("13:00+09:00", None), ("13:00", "18:00+09:00")])
def test_time_window_constraint_input_rejects_zoned_times(
    start: str, end: str | None
) -> None:
    with pytest.raises(ValidationError) as error:
        AddConstraintData.model_validate(
            {
                "kind": "add_constraint",
                "constraint": {
                    "requirement": {"kind": "soft", "strength": "normal"},
                    "condition": {
                        "kind": "time_window",
                        "task_ids": ["review"],
                        "relation": "avoid",
                        "windows": [{"time_range": {"start": start, "end": end}}],
                    },
                },
            }
        )
    message: str = str(error.value)
    assert "time_range" in message and "+09:00" in message
    assert "without a time zone" in message


def test_convert_query_shared_time_window() -> None:
    """Convert windows through the shared time window adapter."""
    assert parse_query(
        {
            "kind": "available_starts",
            "participant_ids": [],
            "duration": "PT1H",
            "windows": [{"time_range": {"start": "10:00", "end": "12:00"}}],
        }
    ) == AvailableStartsQuery(
        frozenset(),
        timedelta(hours=1),
        (TimeWindow(None, None, TimeRange(time(10), time(12))),),
        20,
    )


@pytest.mark.parametrize(
    "model, data, field",
    [
        (
            TimeIntervalData,
            {"start": "2026-10-05T09:00:00Z", "end": "2026-10-05T17:00:00Z"},
            "start",
        ),
        (
            TimeIntervalData,
            {"start": "2026-10-05T09:00:00Z", "end": "2026-10-05T17:00:00Z"},
            "end",
        ),
        (
            NewFixedTaskData,
            {
                "name": "Fixed",
                "start": "2026-10-05T17:00:00Z",
                "duration": "PT1H",
                "participant_ids": [],
            },
            "start",
        ),
        (
            TimeBoundConditionData,
            {
                "kind": "time_bound",
                "task_ids": ["task"],
                "boundary": "start",
                "relation": "at",
                "at": "2026-10-05T17:00:00Z",
            },
            "at",
        ),
        (
            ScheduledTaskData,
            {
                "status": "scheduled",
                "task_id": "task",
                "name": "Task",
                "start": "2026-10-05T09:00:00Z",
                "end": "2026-10-05T17:00:00Z",
            },
            "start",
        ),
        (
            ScheduledTaskData,
            {
                "status": "scheduled",
                "task_id": "task",
                "name": "Task",
                "start": "2026-10-05T09:00:00Z",
                "end": "2026-10-05T17:00:00Z",
            },
            "end",
        ),
    ],
)
def test_json_datetimes_require_utc_offsets(
    model: type[DataModel], data: dict[str, object], field: str
) -> None:
    """Reject offset-free datetimes at each JSON field."""
    invalid: dict[str, object] = {**data, field: "2026-10-05T17:00:00"}
    with pytest.raises(ValidationError, match="timezone"):
        model.model_validate_json(json.dumps(invalid))
    assert model.model_validate_json(json.dumps(data))


def test_time_interval_data_rejects_reversed_range() -> None:
    """Reject reversed intervals during JSON validation."""
    with pytest.raises(ValidationError, match="start.*end"):
        TimeIntervalData.model_validate_json(
            '{"start":"2026-10-05T17:00:00Z","end":"2026-10-05T09:00:00Z"}'
        )


def test_time_interval_data_allows_zero_length() -> None:
    """Accept equal interval endpoints in JSON."""
    interval: TimeIntervalData = TimeIntervalData.model_validate_json(
        '{"start":"2026-10-05T09:00:00Z","end":"2026-10-05T09:00:00Z"}'
    )
    assert interval.start == interval.end


@pytest.mark.parametrize("end", ["2026-10-04", "2026-10-05"])
def test_date_range_data_rejects_nonincreasing_range(end: str) -> None:
    """Reject reversed or empty date ranges during JSON validation."""
    with pytest.raises(ValidationError, match="Date range end.*must be after start"):
        DateRangeData.model_validate_json(
            json.dumps({"start": "2026-10-05", "end": end})
        )


@pytest.mark.parametrize(
    "kind", ["people", "tasks", "constraints", "previous_schedule"]
)
def test_filter_and_limit_defaults(kind: str) -> None:
    """Allow omitted filters and default the limit to twenty."""
    query: SchedulingQuery = parse_query({"kind": kind})
    assert getattr(query, "limit") == 20


def test_people_query_conversion_and_record() -> None:
    """Convert people filters and serialize people in answer order."""
    assert parse_query(
        {
            "kind": "people",
            "filter": {"person_ids": ["bob", "alice-two"], "name_equals": "Alice"},
        }
    ) == PeopleQuery(
        frozenset({PersonId("bob"), PersonId("alice-two")}), "Alice", False, 20
    )
    assert answer_record(
        PeopleAnswer(
            (
                Person(PersonId("alice"), "Alice"),
                Person(PersonId("alice-two"), "Alice"),
            ),
            3,
        )
    ) == {
        "kind": "people",
        "items": [
            {"id": "alice", "name": "Alice"},
            {"id": "alice-two", "name": "Alice"},
        ],
        "total": 3,
        "truncated": True,
    }


def test_tasks_query_conversion() -> None:
    """Convert every task filter and the selection mode."""
    assert parse_query(
        {
            "kind": "tasks",
            "filter": {
                "task_ids": ["start"],
                "type": "fixed",
                "name_equals": "Team meeting",
                "participant_ids_all": ["alice", "bob"],
                "start_range": {"start": at(10), "end": at(12)},
            },
            "select": "one",
            "limit": 3,
        }
    ) == TasksQuery(
        frozenset({TaskId("start")}),
        TaskType.FIXED,
        "Team meeting",
        frozenset({PersonId("alice"), PersonId("bob")}),
        TimeInterval(at(10), at(12)),
        True,
        3,
    )


def test_tasks_record_keeps_fixed_and_movable_shapes() -> None:
    """Serialize fixed and movable tasks with their own fields."""
    appointment: FixedTask = FixedTask(
        TaskId("fixed-first"),
        "Team meeting",
        at(10),
        timedelta(minutes=30),
        frozenset({PersonId("bob"), PersonId("alice")}),
    )
    movable: Task = Task(
        TaskId("task-one"),
        "Alpha",
        HOUR,
        frozenset({PersonId("alice")}),
        Importance.HIGH,
        True,
    )
    assert answer_record(TasksAnswer((appointment, movable), 2)) == {
        "kind": "tasks",
        "items": [
            {
                "id": "fixed-first",
                "name": "Team meeting",
                "start": "2026-10-01T10:00:00+09:00",
                "duration": "PT30M",
                "participant_ids": ["alice", "bob"],
            },
            {
                "id": "task-one",
                "name": "Alpha",
                "duration": "PT1H",
                "participant_ids": ["alice"],
                "importance": "high",
                "required": True,
                "stability": "normal",
            },
        ],
        "total": 2,
        "truncated": False,
    }


@pytest.mark.parametrize("count", [0, 3])
def test_time_window_query_keeps_all_windows(count: int) -> None:
    """Keep entered windows complete."""
    constraint: Constraint = HardConstraint(
        ConstraintId("constraint"),
        TimeWindowCondition(
            frozenset({TaskId("task-one")}),
            TimeRelation.AVOID,
            (
                TimeWindow(
                    DateRange(date(2026, 10, 1), date(2026, 10, 2)),
                    None,
                    TimeRange(time(9), time(10)),
                ),
            )
            * count,
        ),
    )
    window: dict[str, object] = {
        "date_range": {"start": "2026-10-01", "end": "2026-10-02"},
        "weekdays": None,
        "time_range": {"start": "09:00:00", "end": "10:00:00"},
    }
    assert answer_record(ConstraintsAnswer((constraint,), 1)) == {
        "kind": "constraints",
        "items": [
            {
                "id": "constraint",
                "label": None,
                "requirement": {"kind": "hard"},
                "condition": {
                    "kind": "time_window",
                    "task_ids": ["task-one"],
                    "relation": "avoid",
                    "windows": [window] * count,
                },
            }
        ],
        "total": 1,
        "truncated": False,
    }


def test_previous_schedule_record_lists_scheduled_then_dropped() -> None:
    """Serialize scheduled and dropped entries and whether a schedule exists."""
    answer: PreviousScheduleAnswer = PreviousScheduleAnswer(
        (
            ScheduledTask(TaskId("old-a"), "Old A", at(10), at(11)),
            ScheduledTask(TaskId("old-late"), "Old late", at(12), at(13)),
            DroppedTask(TaskId("dropped-a"), "Dropped A"),
        ),
        5,
        True,
    )
    assert answer_record(answer) == {
        "kind": "previous_schedule",
        "items": [
            {
                "status": "scheduled",
                "task_id": "old-a",
                "name": "Old A",
                "start": at(10).isoformat(),
                "end": at(11).isoformat(),
            },
            {
                "status": "scheduled",
                "task_id": "old-late",
                "name": "Old late",
                "start": at(12).isoformat(),
                "end": at(13).isoformat(),
            },
            {"status": "dropped", "task_id": "dropped-a", "name": "Dropped A"},
        ],
        "total": 5,
        "truncated": True,
        "has_previous": True,
    }
    assert answer_record(PreviousScheduleAnswer((), 0, False)) == {
        "kind": "previous_schedule",
        "items": [],
        "total": 0,
        "truncated": False,
        "has_previous": False,
    }


def test_summary_record() -> None:
    """Serialize the grid and element counts."""
    assert answer_record(
        Summary(
            TimeGrid(TimeInterval(at(9), at(13)), timedelta(minutes=30)),
            3,
            2,
            3,
            0,
            True,
        )
    ) == {
        "kind": "summary",
        "grid": {
            "horizon": {
                "start": "2026-10-01T09:00:00+09:00",
                "end": "2026-10-01T13:00:00+09:00",
            },
            "slot": "PT30M",
        },
        "counts": {"people": 3, "tasks": 2, "fixed_tasks": 3, "constraints": 0},
        "has_previous": True,
    }


def test_available_starts_record_notes_what_is_ignored() -> None:
    """Serialize start times with a note on what they do not consider."""
    assert answer_record(AvailableStartsAnswer((at(10),), 3)) == {
        "kind": "available_starts",
        "items": [at(10).isoformat()],
        "total": 3,
        "truncated": True,
        "note": "Movable Tasks and constraints are not considered; use solve for the final schedule.",
    }


def test_objective_policy_record_lists_every_coefficient() -> None:
    """Expose every coefficient used by the default solver."""
    assert answer_record(ObjectivePolicyAnswer(DEFAULT_POLICY)) == {
        "kind": "objective_policy",
        "drop_costs": {"low": 5.0, "medium": 20.0, "high": 100.0},
        "weights": {"weak": 1.0, "normal": 5.0, "strong": 20.0},
        "per_count": 1.0,
        "stability_drop_cost_ratio": 0.5,
    }


def task(identifier: str, required: bool = True) -> Task:
    """Build a movable task."""
    return Task(
        TaskId(identifier),
        identifier,
        HOUR,
        frozenset({PERSON}),
        Importance.LOW,
        required,
    )


def evaluation_problem() -> tuple[SchedulingProblem, Schedule]:
    """Build current constraints and an older schedule."""
    first: Task = task("first")
    missing: Task = task("missing")
    dropped: Task = task("dropped", False)
    fixed: FixedTask = FixedTask(
        TaskId("fixed"), "Fixed", START + timedelta(minutes=10), HOUR, frozenset()
    )
    identifiers: frozenset[TaskId] = frozenset(
        {first.id, missing.id, dropped.id, fixed.id}
    )
    deadline: TimeBoundCondition = TimeBoundCondition(
        identifiers,
        Boundary.END,
        TimeBoundRelation.AT_OR_BEFORE,
        START + timedelta(minutes=30),
    )
    constraints: tuple[Constraint, ...] = (
        SoftConstraint(ConstraintId("weak"), deadline, Strength.WEAK, "Deadline"),
        SoftConstraint(
            ConstraintId("strong"),
            DailyLimitCondition(identifiers, AggregateQuantity.COUNT, 1),
            Strength.STRONG,
        ),
        HardConstraint(ConstraintId("hard"), deadline, "Required deadline"),
        SoftConstraint(
            ConstraintId("duration"),
            DailyLimitCondition(
                identifiers, AggregateQuantity.TOTAL_DURATION, timedelta(minutes=30)
            ),
            Strength.NORMAL,
        ),
        SoftConstraint(
            ConstraintId("window"),
            TimeWindowCondition(
                identifiers,
                TimeRelation.WITHIN,
                (TimeWindow(None, None, TimeRange(time(9, 15), time(10, 45))),),
            ),
            Strength.NORMAL,
        ),
        SoftConstraint(
            ConstraintId("gap"),
            TaskGapCondition(fixed.id, first.id, TaskGapRelation.AT_LEAST, HOUR),
            Strength.NORMAL,
        ),
        HardConstraint(
            ConstraintId("satisfied"),
            TimeBoundCondition(
                frozenset({missing.id}), Boundary.START, TimeBoundRelation.AT, START
            ),
        ),
    )
    grid: TimeGrid = TimeGrid(TimeInterval(START, START + 4 * HOUR), HOUR)
    problem: SchedulingProblem = SchedulingProblem(
        Calendar(grid, (Availability(PERSON, (grid.horizon,)),)),
        (Person(PERSON, "Person"),),
        (first, missing, dropped),
        (fixed,),
        constraints,
    )
    previous: Schedule = Schedule(
        (
            ScheduledTask(first.id, first.name, START + HOUR, START + 2 * HOUR),
            ScheduledTask(TaskId("removed"), "Removed", START, START + HOUR),
        ),
        (DroppedTask(dropped.id, dropped.name),),
    )
    return problem, previous


def query_record(
    source: dict[str, object], value: SchedulingProblem, previous: Schedule | None
) -> dict[str, object]:
    """Answer a query through its JSON forms."""
    result: AnswerResult = parse_query(source).answer(value, previous)
    assert isinstance(result, Answered)
    return answer_record(result.answer)


def test_evaluation_amounts_units_costs_breakdowns_and_ordering() -> None:
    """Evaluate current constraints against recorded and fixed placements."""
    value: SchedulingProblem
    previous: Schedule
    value, previous = evaluation_problem()
    record: dict[str, object] = query_record({"kind": "evaluation"}, value, previous)
    assert record["has_previous"] is True
    assert record["total"] == 7 and record["truncated"] is False
    items: list[dict[str, object]] = cast(list[dict[str, object]], record["items"])
    assert [item["constraint_id"] for item in items] == [
        "hard",
        "strong",
        "window",
        "duration",
        "gap",
        "weak",
        "satisfied",
    ]
    by_identifier: dict[str, dict[str, object]] = {
        str(item["constraint_id"]): item for item in items
    }
    assert by_identifier["hard"]["cost"] is None
    assert by_identifier["hard"]["requirement"] == {"kind": "hard"}
    hard_breakdown: list[dict[str, object]] = cast(
        list[dict[str, object]], by_identifier["hard"]["breakdown"]
    )
    assert all(part["cost"] is None for part in hard_breakdown)
    assert by_identifier["weak"]["label"] == "Deadline"
    assert by_identifier["weak"]["violation"] == {"amount": 13 / 6, "unit": "hours"}
    assert by_identifier["weak"]["cost"] == 13 / 6
    assert by_identifier["weak"]["requirement"] == {"kind": "soft", "strength": "weak"}
    assert by_identifier["weak"]["breakdown"] == [
        {
            "task_id": "dropped",
            "violation": {"amount": 0.0, "unit": "hours"},
            "cost": 0.0,
        },
        {
            "task_id": "first",
            "violation": {"amount": 1.5, "unit": "hours"},
            "cost": 1.5,
        },
        {
            "task_id": "fixed",
            "violation": {"amount": 2 / 3, "unit": "hours"},
            "cost": 2 / 3,
        },
        {
            "task_id": "missing",
            "violation": {"amount": 0.0, "unit": "hours"},
            "cost": 0.0,
        },
    ]
    assert by_identifier["strong"]["violation"] == {"amount": 1.0, "unit": "count"}
    assert by_identifier["strong"]["cost"] == 20.0
    assert by_identifier["strong"]["breakdown"] == [
        {
            "date": "2026-10-01",
            "violation": {"amount": 1.0, "unit": "count"},
            "cost": 20.0,
        }
    ]
    assert by_identifier["duration"]["violation"] == {"amount": 1.5, "unit": "hours"}
    assert by_identifier["duration"]["cost"] == 7.5
    assert by_identifier["window"]["violation"] == {"amount": 2.0, "unit": "hours"}
    assert by_identifier["gap"]["violation"] == {"amount": 7 / 6, "unit": "hours"}
    assert by_identifier["gap"]["breakdown"] == []


@pytest.mark.parametrize(
    "filters,limit,expected,total",
    [
        ({"violated_only": True}, 2, ["hard", "strong"], 6),
        ({"constraint_ids": ["weak", "hard"]}, 20, ["hard", "weak"], 2),
        (
            {"task_ids": ["first"], "constraint_ids": ["weak", "satisfied"]},
            20,
            ["weak"],
            1,
        ),
        ({"task_ids": []}, 20, [], 0),
        ({"constraint_ids": []}, 20, [], 0),
    ],
)
def test_evaluation_filters_and_limit(
    filters: dict[str, object], limit: int, expected: list[str], total: int
) -> None:
    """Filter complete constraint evaluations before limiting."""
    value: SchedulingProblem
    previous: Schedule
    value, previous = evaluation_problem()
    record: dict[str, object] = query_record(
        {"kind": "evaluation", "filter": filters, "limit": limit}, value, previous
    )
    items: list[dict[str, object]] = cast(list[dict[str, object]], record["items"])
    assert [item["constraint_id"] for item in items] == expected
    assert record["total"] == total
    assert record["truncated"] == (total > limit)
    if items and items[-1]["constraint_id"] == "weak":
        assert items[-1]["violation"] == {"amount": 13 / 6, "unit": "hours"}


def test_evaluation_without_previous_and_with_empty_previous() -> None:
    """Distinguish a missing saved solution from an empty one."""
    value: SchedulingProblem = evaluation_problem()[0]
    assert query_record({"kind": "evaluation"}, value, None) == {
        "kind": "evaluation",
        "has_previous": False,
        "items": [],
        "total": 0,
        "truncated": False,
    }
    record: dict[str, object] = query_record(
        {"kind": "evaluation", "filter": {"violated_only": True}},
        value,
        Schedule((), ()),
    )
    assert record["has_previous"] is True
    assert record["total"] == 4
