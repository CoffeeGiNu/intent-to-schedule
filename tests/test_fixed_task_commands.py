"""Tests for commands that add and replace fixed appointments."""

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from intent_to_schedule.adapter.command_line_interface.main import main
from intent_to_schedule.adapter.command_line_interface.state import (
    load_state,
    to_problem,
)
from intent_to_schedule.adapter.data_model import (
    AddTaskData,
    CommandData,
    CommandsData,
    FixedTaskData,
    NewFixedTaskData,
    NewTaskData,
    ReplaceTaskData,
    TaskData,
    command_record,
    convert_commands_input,
)
from intent_to_schedule.adapter.mathopt.solve import MathOptSchedulingSolver
from intent_to_schedule.application.command import (
    AddTask,
    Executed,
    Rejected,
    RemoveTask,
    ReplaceTask,
    SchedulingCommand,
)
from intent_to_schedule.application.policy import DEFAULT_POLICY
from intent_to_schedule.application.query import (
    Answered,
    AvailableStartsAnswer,
    AvailableStartsQuery,
)
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import Solved, SolveResult
from intent_to_schedule.domain.availability import free_slots
from intent_to_schedule.domain.calendar import (
    Availability,
    Calendar,
    TimeGrid,
    TimeInterval,
)
from intent_to_schedule.domain.consistency import (
    AlignedToSlots,
    AllOf,
    AvailabilityForEveryone,
    ReferencesExist,
    SupportedCombinations,
    UniqueIds,
)
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint
from intent_to_schedule.domain.evaluation import Distance
from intent_to_schedule.domain.measure import PointMeasure
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import ScheduledTask
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId


START: datetime = datetime.fromisoformat("2026-10-19T09:00:00+09:00")
PARTICIPANT: PersonId = PersonId("ito")


@pytest.fixture
def problem() -> SchedulingProblem:
    """Build a morning calendar with one movable task."""
    horizon: TimeInterval = TimeInterval(START, START + timedelta(hours=4))
    calendar: Calendar = Calendar(
        TimeGrid(horizon, timedelta(minutes=30)),
        (Availability(PARTICIPANT, (horizon,)),),
    )
    task: Task = Task(
        TaskId("review"),
        "Review",
        timedelta(hours=1),
        frozenset({PARTICIPANT}),
        Importance.HIGH,
        True,
    )
    return SchedulingProblem(calendar, (Person(PARTICIPANT, "Ito"),), (task,), (), ())


@pytest.fixture
def scheduling() -> Scheduling:
    """Build the scheduling service with domain validators."""
    return Scheduling(
        MathOptSchedulingSolver(DEFAULT_POLICY),
        AllOf(
            UniqueIds(),
            ReferencesExist(),
            AvailabilityForEveryone(),
            AlignedToSlots(),
            SupportedCombinations(),
        ),
    )


def appointment() -> FixedTask:
    """Build a fixed health check appointment."""
    return FixedTask(
        TaskId("health-check"),
        "健康診断",
        START + timedelta(minutes=90),
        timedelta(minutes=90),
        frozenset({PARTICIPANT}),
    )


@pytest.mark.parametrize("offset, duration", [(90, 90), (95, 80)])
def test_add_fixed_task_blocks_queries_solver_and_later_tasks(
    problem: SchedulingProblem,
    scheduling: Scheduling,
    offset: int,
    duration: int,
) -> None:
    """Block all touched slots for present and future tasks."""
    fixed: FixedTask = replace(
        appointment(),
        start=START + timedelta(minutes=offset),
        duration=timedelta(minutes=duration),
    )
    added: Executed | Rejected = scheduling.execute(problem, (AddTask(fixed),))
    assert isinstance(added, Executed)
    assert added.problem.fixed_tasks == (fixed,)
    assert added.problem.tasks == problem.tasks
    assert free_slots(added.problem, PARTICIPANT) == (
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
    )
    expected: tuple[datetime, ...] = (
        START,
        START + timedelta(minutes=30),
        START + timedelta(hours=3),
    )
    query: AvailableStartsQuery = AvailableStartsQuery(
        frozenset({PARTICIPANT}),
        timedelta(hours=1),
        None,
        100,
    )
    assert query.answer(added.problem, None) == Answered(
        AvailableStartsAnswer(expected, len(expected))
    )
    first_solution: SolveResult = scheduling.solve(added.problem, None)
    assert isinstance(first_solution, Solved)
    assert first_solution.schedule.scheduled[0].start in expected
    later: Task = replace(problem.tasks[0], id=TaskId("later"))
    with_later: Executed | Rejected = scheduling.execute(
        added.problem, (AddTask(later),)
    )
    assert isinstance(with_later, Executed)
    solution: SolveResult = scheduling.solve(with_later.problem, None)
    assert isinstance(solution, Solved)
    assert {item.task_id for item in solution.schedule.scheduled} == {
        problem.tasks[0].id,
        later.id,
    }
    item: ScheduledTask
    for item in solution.schedule.scheduled:
        assert item.start in expected
    assert problem.fixed_tasks == ()


def test_remove_added_fixed_task_restores_free_slots(
    problem: SchedulingProblem,
    scheduling: Scheduling,
) -> None:
    """Restore availability after removing a fixed appointment."""
    fixed: FixedTask = appointment()
    added: Executed | Rejected = scheduling.execute(problem, (AddTask(fixed),))
    assert isinstance(added, Executed)
    removed: Executed | Rejected = scheduling.execute(
        added.problem, (RemoveTask(fixed.id),)
    )
    assert isinstance(removed, Executed)
    assert removed.problem == problem
    assert free_slots(removed.problem, PARTICIPANT) == free_slots(problem, PARTICIPANT)
    query: AvailableStartsQuery = AvailableStartsQuery(
        frozenset({PARTICIPANT}),
        timedelta(hours=1),
        None,
        100,
    )
    assert query.answer(removed.problem, None) == query.answer(problem, None)


def test_replace_task_changes_fixedness_and_keeps_constraints(
    problem: SchedulingProblem,
    scheduling: Scheduling,
) -> None:
    """Fix and release a task while retaining its constraints."""
    movable: Task = problem.tasks[0]
    fixed: FixedTask = replace(appointment(), id=movable.id)
    constraint: HardConstraint = HardConstraint(
        ConstraintId("start"),
        PointMeasure(movable.id),
        Distance(fixed.start),
    )
    original: SchedulingProblem = replace(problem, constraints=(constraint,))
    result: Executed | Rejected = scheduling.execute(original, (ReplaceTask(fixed),))
    assert isinstance(result, Executed)
    assert result.problem.tasks == ()
    assert result.problem.fixed_tasks == (fixed,)
    assert result.problem.constraints == original.constraints
    assert isinstance(scheduling.solve(result.problem, None), Solved)
    released: Executed | Rejected = scheduling.execute(
        result.problem, (ReplaceTask(movable),)
    )
    assert isinstance(released, Executed)
    assert released.problem == original
    solution: SolveResult = scheduling.solve(released.problem, None)
    assert isinstance(solution, Solved)
    assert solution.schedule.scheduled == (ScheduledTask(movable.id, fixed.start),)


def test_replace_fixed_task_preserves_other_tasks_and_order(
    problem: SchedulingProblem,
    scheduling: Scheduling,
) -> None:
    """Replace one fixed appointment without disturbing its neighbors."""
    fixed: FixedTask = appointment()
    other: FixedTask = replace(fixed, id=TaskId("other"), participant_ids=frozenset())
    original: SchedulingProblem = replace(problem, fixed_tasks=(fixed, other))
    replacement: FixedTask = replace(fixed, start=START + timedelta(minutes=15))
    result: Executed | Rejected = scheduling.execute(
        original, (ReplaceTask(replacement),)
    )
    assert isinstance(result, Executed)
    assert result.problem.fixed_tasks == (replacement, other)
    assert result.problem.tasks == original.tasks
    assert original.fixed_tasks == (fixed, other)


def test_fixed_task_commands_reject_duplicate_and_missing_identifiers(
    problem: SchedulingProblem,
) -> None:
    """Check fixed identifiers against both task collections."""
    fixed: FixedTask = appointment()
    original: SchedulingProblem = replace(problem, fixed_tasks=(fixed,))
    assert isinstance(AddTask(fixed).execute(original), Rejected)
    assert isinstance(
        AddTask(replace(fixed, id=problem.tasks[0].id)).execute(original), Rejected
    )
    assert isinstance(
        ReplaceTask(replace(fixed, id=TaskId("missing"))).execute(original), Rejected
    )


@pytest.mark.parametrize("kind", ["add_task", "replace_task"])
def test_fixed_task_commands_validate_participant_references(
    problem: SchedulingProblem,
    scheduling: Scheduling,
    kind: str,
) -> None:
    """Reject fixed appointments with unknown participants."""
    fixed: FixedTask = replace(
        appointment(),
        id=problem.tasks[0].id if kind == "replace_task" else TaskId("new"),
        participant_ids=frozenset({PersonId("missing")}),
    )
    command: SchedulingCommand = (
        AddTask(fixed) if kind == "add_task" else ReplaceTask(fixed)
    )
    result: Executed | Rejected = scheduling.execute(problem, (command,))
    assert isinstance(result, Rejected)
    assert "missing person id missing" in result.violations.items[0].message


def task_input(fixed: bool) -> dict[str, Any]:
    """Build the selected task input shape."""
    data: dict[str, Any] = {
        "name": "健康診断",
        "duration": "PT1H30M",
        "participant_ids": ["ito"],
    }
    if fixed:
        data["start"] = "2026-10-19T10:30:00+09:00"
    else:
        data.update(importance="high", required=True, stability="normal")
    return data


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
    expected_type: type[NewTaskData | NewFixedTaskData] = (
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
        grid: TimeGrid = TimeGrid(
            TimeInterval(START, START + timedelta(hours=4)), timedelta(minutes=30)
        )
        assert command_record(command, grid) == {
            "kind": "add_task",
            "task_id": "generated",
            "name": "健康診断",
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


def test_command_schema_exposes_disjoint_task_shapes(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Expose both strict task shapes through the schema command."""
    assert main(["schema"]) == 0
    schema: dict[str, Any] = json.loads(capsys.readouterr().out)
    name: str
    movable: str
    fixed: str
    for name, movable, fixed in (
        ("AddTaskData", "NewTaskData", "NewFixedTaskData"),
        ("ReplaceTaskData", "TaskData", "FixedTaskData"),
    ):
        assert schema["$defs"][name]["properties"]["task"]["anyOf"] == [
            {"$ref": f"#/$defs/{movable}"},
            {"$ref": f"#/$defs/{fixed}"},
        ]
        assert "start" not in schema["$defs"][movable]["properties"]
        assert "start" in schema["$defs"][fixed]["required"]
        assert schema["$defs"][movable]["additionalProperties"] is False
        assert schema["$defs"][fixed]["additionalProperties"] is False


def test_apply_fixed_task_persists_and_reports_generated_identifier(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Persist an added appointment and return its generated identifier."""
    calendar: Path = tmp_path / "calendar.json"
    state: Path = tmp_path / "state.json"
    commands: Path = tmp_path / "commands.json"
    calendar.write_text(
        json.dumps(
            {
                "horizon": {
                    "start": START.isoformat(),
                    "end": (START + timedelta(hours=4)).isoformat(),
                },
                "slot": "PT30M",
                "people": [{"id": "ito", "name": "Ito"}],
                "availabilities": [
                    {
                        "person_id": "ito",
                        "intervals": [
                            {
                                "start": START.isoformat(),
                                "end": (START + timedelta(hours=4)).isoformat(),
                            }
                        ],
                    }
                ],
                "fixed_tasks": [],
            }
        ),
        encoding="utf-8",
    )
    assert main(["init", "--state", str(state), "--calendar", str(calendar)]) == 0
    capsys.readouterr()
    commands.write_text(
        json.dumps({"commands": [{"kind": "add_task", "task": task_input(True)}]}),
        encoding="utf-8",
    )
    assert main(["apply", "--state", str(state), "--file", str(commands)]) == 0
    output: dict[str, Any] = json.loads(capsys.readouterr().out)
    problem: SchedulingProblem = to_problem(load_state(state).problem)
    fixed: FixedTask = problem.fixed_tasks[0]
    assert output == {
        "executed": [
            {"kind": "add_task", "task_id": fixed.id.value, "name": fixed.name}
        ]
    }
    assert fixed.start == appointment().start
    assert fixed.duration == appointment().duration
    assert problem.tasks == ()
