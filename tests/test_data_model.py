from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from intent_to_schedule.adapter.data_model import (
    CommandsData,
    FixedTaskData,
    NewFixedTaskData,
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
