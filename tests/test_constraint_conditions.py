import io
import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from intent_to_schedule.adapter.command_line_interface.main import main
from intent_to_schedule.adapter.command_line_interface.state import (
    State,
    load_state,
    save_state,
    to_problem,
    to_problem_state,
)
from intent_to_schedule.adapter.data_model import (
    CommandsData,
    answer_record,
    command_record,
    convert_commands_input,
)
from intent_to_schedule.application.command import (
    AddConstraint,
    Executed,
    Rejected,
    RemoveTask,
    ReplaceConstraint,
)
from intent_to_schedule.application.query import Answered, ConstraintsQuery
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import (
    DailyLimitCondition,
    TaskGapCondition,
    TimeBoundCondition,
    TimeWindowCondition,
)
from intent_to_schedule.domain.constraint import (
    Constraint,
    ConstraintId,
    SoftConstraint,
)
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId

START: datetime = datetime(2026, 10, 29, 9, tzinfo=timezone(timedelta(hours=9)))
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


def problem() -> SchedulingProblem:
    """Build a problem with two tasks."""
    grid: TimeGrid = TimeGrid(
        TimeInterval(START, START + timedelta(hours=8)), timedelta(hours=1)
    )
    tasks: tuple[Task, ...] = tuple(
        Task(TaskId(name), name, timedelta(hours=1), frozenset(), Importance.LOW, True)
        for name in ("a", "b")
    )
    return SchedulingProblem(Calendar(grid, ()), (), tasks, (), ())


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


def test_state_round_trip_every_condition(tmp_path: Path) -> None:
    """Preserve every entered condition in a state file."""
    original: SchedulingProblem = problem()
    constraints: tuple[Constraint, ...] = tuple(
        command.constraint
        for command in convert_commands_input(
            CommandsData.model_validate(
                {
                    "commands": [
                        constraint_command(condition, str(index))
                        for index, condition in enumerate(CONDITIONS)
                    ]
                }
            )
        )
        if isinstance(command, AddConstraint)
    )
    constraints = tuple(
        SoftConstraint(
            constraint.id, constraint.condition, Strength.STRONG, constraint.label
        )
        if index % 2
        else constraint
        for index, constraint in enumerate(constraints)
    )
    original = replace(original, constraints=constraints)
    path: Path = tmp_path / "state.json"
    save_state(
        path, State(problem=to_problem_state(original), previous=None, dialogue=())
    )
    assert to_problem(load_state(path).problem) == original
    stored: dict[str, object] = json.loads(path.read_text())
    assert "measure" not in json.dumps(stored)
    assert "09:10:00" in json.dumps(stored)


def test_replace_missing_identifier_and_required_identifier() -> None:
    """Reject replacement without an existing identifier."""
    command: object = convert_commands_input(
        CommandsData.model_validate(
            {"commands": [constraint_command(CONDITIONS[1], kind="replace_constraint")]}
        )
    )[0]
    assert isinstance(command, ReplaceConstraint)
    result: Executed | Rejected = command.execute(problem())
    assert isinstance(result, Rejected)
    assert result.violations.items[0].message == "Constraint c does not exist"
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


def test_replace_preserves_order_and_changes_requirement_and_label() -> None:
    """Replace a constraint's content in its existing position."""
    commands: tuple[object, ...] = convert_commands_input(
        CommandsData.model_validate(
            {
                "commands": [
                    constraint_command(CONDITIONS[1], "first"),
                    constraint_command(CONDITIONS[3], "second"),
                ]
            }
        )
    )
    original: SchedulingProblem = replace(
        problem(),
        constraints=tuple(
            command.constraint
            for command in commands
            if isinstance(command, AddConstraint)
        ),
    )
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
    with patch.object(
        ConstraintId,
        "generate",
        side_effect=AssertionError("Replacement generated an identifier"),
    ):
        replacement: object = convert_commands_input(data)[0]
    assert isinstance(replacement, ReplaceConstraint)
    result: Executed | Rejected = replacement.execute(original)
    assert isinstance(result, Executed)
    assert result.problem.constraints == (
        replacement.constraint,
        original.constraints[1],
    )
    assert isinstance(replacement.constraint, SoftConstraint)
    assert replacement.constraint.strength is Strength.STRONG
    assert replacement.constraint.label == "Changed request"
    assert original.constraints[0].label == "Entered request"


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
    assert command_record(command, problem().calendar.grid) == {
        "kind": kind,
        "constraint_id": "c",
        "note": f"Window times were rounded {direction} to calendar slots.",
    }


def test_remove_task_prunes_conditions() -> None:
    """Shrink conditions and delete gaps and empty conditions."""
    original: SchedulingProblem = problem()
    commands: tuple[object, ...] = convert_commands_input(
        CommandsData.model_validate(
            {
                "commands": [
                    constraint_command(condition, str(index))
                    for index, condition in enumerate(CONDITIONS)
                ]
            }
        )
    )
    original = replace(
        original,
        constraints=tuple(
            command.constraint
            for command in commands
            if isinstance(command, AddConstraint)
        ),
    )
    result: Executed | Rejected = RemoveTask(TaskId("a")).execute(original)
    assert isinstance(result, Executed)
    assert [item.id.value for item in result.problem.constraints] == [
        "0",
        "1",
        "3",
        "4",
    ]
    assert all(
        item.condition.task_ids == frozenset({TaskId("b")})
        for item in result.problem.constraints
    )
    result = RemoveTask(TaskId("b")).execute(result.problem)
    assert isinstance(result, Executed) and result.problem.constraints == ()


def test_constraints_query_stored_values_and_filters() -> None:
    """Return entered conditions and filter by identifiers and any task."""
    commands: tuple[object, ...] = convert_commands_input(
        CommandsData.model_validate(
            {
                "commands": [
                    constraint_command(condition, str(index))
                    for index, condition in enumerate(CONDITIONS)
                ]
            }
        )
    )
    original: SchedulingProblem = replace(
        problem(),
        constraints=tuple(
            command.constraint
            for command in commands
            if isinstance(command, AddConstraint)
        ),
    )
    result: object = ConstraintsQuery(None, frozenset({TaskId("b")}), 20).answer(
        original, None
    )
    assert isinstance(result, Answered)
    record: dict[str, object] = answer_record(result.answer)
    items: object = record["items"]
    assert isinstance(items, list) and len(items) == 5
    assert items[0] == {
        "id": "0",
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
    result = ConstraintsQuery(
        frozenset({ConstraintId("2")}), frozenset({TaskId("a"), TaskId("unknown")}), 20
    ).answer(original, None)
    assert isinstance(result, Answered) and result.answer.items == (
        original.constraints[2],
    )
    result = ConstraintsQuery(None, frozenset({TaskId("unknown")}), 20).answer(
        original, None
    )
    assert isinstance(result, Answered) and result.answer.items == ()


@pytest.mark.parametrize("kind", ["add_constraint", "replace_constraint"])
@pytest.mark.parametrize("relation", ["within", "avoid"])
def test_empty_windows_rejected_atomically(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    relation: str,
) -> None:
    """Reject empty window expansion on addition and replacement."""
    original: SchedulingProblem = problem()
    command: object = convert_commands_input(
        CommandsData.model_validate({"commands": [constraint_command(CONDITIONS[1])]})
    )[0]
    assert isinstance(command, AddConstraint)
    original = replace(
        original,
        constraints=(command.constraint,) if kind == "replace_constraint" else (),
    )
    path: Path = tmp_path / "state.json"
    save_state(
        path, State(problem=to_problem_state(original), previous=None, dialogue=())
    )
    before: bytes = path.read_bytes()
    condition: dict[str, object] = {
        "kind": "time_window",
        "task_ids": ["a"],
        "relation": relation,
        "windows": [{"date_range": {"start": "2027-01-01", "end": "2027-01-02"}}],
    }
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps({"commands": [constraint_command(condition, kind=kind)]})
        ),
    )
    assert main(["--state", str(path), "apply"]) == 1
    output: dict[str, object] = json.loads(capsys.readouterr().out)
    covered: str = "whole slot" if relation == "within" else "time"
    assert output == {
        "rejected": [
            f"Time constraint on task a: the {relation} windows cover no {covered} of the calendar horizon {START.isoformat()} to {(START + timedelta(hours=8)).isoformat()} (slot 1:00:00); widen or move the windows."
        ]
    }
    assert path.read_bytes() == before


def test_command_line_add_solve_replace_solve(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Solve a batch with given identifiers and solve after replacement."""
    path: Path = tmp_path / "state.json"
    save_state(
        path,
        State(
            problem=to_problem_state(replace(problem(), tasks=())),
            previous=None,
            dialogue=(),
        ),
    )
    task: dict[str, object] = {
        "id": "a",
        "name": "Review",
        "duration": "PT1H",
        "participant_ids": [],
        "importance": "low",
        "required": True,
        "stability": "weak",
    }
    condition: dict[str, object] = {
        "kind": "time_bound",
        "task_ids": ["a"],
        "boundary": "start",
        "relation": "at",
        "at": START.isoformat(),
    }
    window: dict[str, object] = {
        "kind": "time_window",
        "task_ids": ["a"],
        "relation": "within",
        "windows": [{"time_range": {"start": "09:00", "end": "16:20"}}],
    }
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "commands": [
                        {"kind": "add_task", "task": task},
                        constraint_command(condition, "bound"),
                        constraint_command(window, "window"),
                    ]
                }
            )
        ),
    )
    assert main(["--state", str(path), "apply"]) == 0
    output: dict[str, object] = json.loads(capsys.readouterr().out)
    assert output["executed"] == [
        {"kind": "add_task", "task_id": "a", "name": "Review"},
        {"kind": "add_constraint", "constraint_id": "bound"},
        {
            "kind": "add_constraint",
            "constraint_id": "window",
            "note": "Window times were rounded inward to calendar slots.",
        },
    ]
    assert main(["--state", str(path), "solve"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["items"][0]["start"] == START.isoformat()
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "commands": [
                        constraint_command(
                            {
                                **condition,
                                "at": (START + timedelta(hours=2)).isoformat(),
                            },
                            "bound",
                            "replace_constraint",
                        )
                    ]
                }
            )
        ),
    )
    assert main(["--state", str(path), "apply"]) == 0
    assert json.loads(capsys.readouterr().out)["executed"] == [
        {"kind": "replace_constraint", "constraint_id": "bound"}
    ]
    assert main(["--state", str(path), "solve"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["items"][0]["start"] == (START + timedelta(hours=2)).isoformat()
