from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from intent_to_schedule.adapter.data_model import CommandsData, convert_commands_input
from intent_to_schedule.application.command import AddConstraint, AddTask, RemoveConstraint, RemoveTask, ReplaceTask
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint
from intent_to_schedule.domain.evaluation import Distance
from intent_to_schedule.domain.measure import PointMeasure
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId


def test_commands_input_converts_mixed_commands() -> None:
    """Parse all command kinds and convert them in input order."""
    start: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    task_data: dict[str, object] = {
        "name": "Review",
        "duration": 3600,
        "participant_ids": [],
        "importance": "high",
        "required": True,
        "stability": "weak",
    }
    data: CommandsData = CommandsData.model_validate({"commands": [
        {"kind": "add_task", "task": task_data},
        {"kind": "replace_task", "task_id": "old", "replacement": task_data},
        {"kind": "remove_task", "task_id": "old"},
        {"kind": "add_constraint", "constraint": {
            "kind": "hard",
            "measure": {"kind": "point", "task_id": "old"},
            "evaluation": {"kind": "distance", "target": {"kind": "instant", "value": start.isoformat()}},
        }},
        {"kind": "remove_constraint", "constraint_id": "obsolete"},
    ]})
    with patch.object(TaskId, "generate", return_value=TaskId("new")), patch.object(
        ConstraintId, "generate", return_value=ConstraintId("c1")
    ):
        commands: tuple[AddTask | ReplaceTask | RemoveTask | AddConstraint | RemoveConstraint, ...] = convert_commands_input(data)
    task: Task = Task(TaskId("new"), "Review", timedelta(hours=1), frozenset(), Importance.HIGH, True, Strength.WEAK)
    replacement: Task = Task(TaskId("old"), "Review", timedelta(hours=1), frozenset(), Importance.HIGH, True, Strength.WEAK)
    assert commands == (
        AddTask(task),
        ReplaceTask(replacement),
        RemoveTask(TaskId("old")),
        AddConstraint(HardConstraint(ConstraintId("c1"), PointMeasure(TaskId("old")), Distance(start))),
        RemoveConstraint(ConstraintId("obsolete")),
    )
