import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from intent_to_schedule.adapter.command_line_interface.state import (
    CalendarInput,
    ScheduleState,
)
from intent_to_schedule.adapter.data_model import (
    CommandsData,
    DataModel,
    DateRangeData,
    FixedTaskData,
    NewFixedTaskData,
    ScheduledTaskData,
    TimeBoundConditionData,
    TimeIntervalData,
    convert_commands_input,
    convert_fixed_task,
    to_fixed_task_data,
)
from intent_to_schedule.application.command import (
    AddConstraint,
    AddTask,
    RemoveConstraint,
    RemoveTask,
    ReplaceTask,
    SchedulingCommand,
)
from intent_to_schedule.domain.condition import TimeBoundCondition, TimeBoundRelation
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint
from intent_to_schedule.domain.measure import Boundary
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId


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
                {"kind": "add_task", "task": task_data},
                {"kind": "replace_task", "task": {"id": "old", **task_data}},
                {"kind": "remove_task", "task_id": "old"},
                {
                    "kind": "add_constraint",
                    "constraint": {
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
    with (
        patch.object(TaskId, "generate", return_value=TaskId("new")),
        patch.object(ConstraintId, "generate", return_value=ConstraintId("c1")),
    ):
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
                "name": "Fixed", "start": "2026-10-05T17:00:00Z",
                "duration": "PT1H", "participant_ids": [],
            },
            "start",
        ),
        (
            TimeBoundConditionData,
            {
                "kind": "time_bound", "task_ids": ["task"], "boundary": "start",
                "relation": "at", "at": "2026-10-05T17:00:00Z",
            },
            "at",
        ),
        (
            ScheduledTaskData,
            {
                "status": "scheduled", "task_id": "task", "name": "Task",
                "start": "2026-10-05T09:00:00Z", "end": "2026-10-05T17:00:00Z",
            },
            "start",
        ),
        (
            ScheduledTaskData,
            {
                "status": "scheduled", "task_id": "task", "name": "Task",
                "start": "2026-10-05T09:00:00Z", "end": "2026-10-05T17:00:00Z",
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


@pytest.mark.parametrize("model", [CalendarInput, ScheduleState])
def test_state_json_datetimes_require_utc_offsets(model: type[DataModel]) -> None:
    """Reject offset-free datetimes in calendar and schedule state."""
    data: dict[str, object] = (
        {
            "horizon": {
                "start": "2026-10-05T09:00:00", "end": "2026-10-05T17:00:00Z"
            },
            "slot": "PT1H", "availabilities": [], "people": [], "fixed_tasks": [],
        }
        if model is CalendarInput
        else {
            "items": [
                {
                    "status": "scheduled", "task_id": "task", "name": "Task",
                    "start": "2026-10-05T09:00:00Z", "end": "2026-10-05T17:00:00",
                }
            ]
        }
    )
    with pytest.raises(ValidationError, match="timezone"):
        model.model_validate_json(json.dumps(data))


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
