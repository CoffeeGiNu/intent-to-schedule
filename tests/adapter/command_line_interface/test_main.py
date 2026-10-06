"""Tests for the JSON command line adapter."""

import io
import json
import sys
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import openai
import pytest

import intent_to_schedule.adapter.command_line_interface.main as main_module
import intent_to_schedule.application.converse as converse
from intent_to_schedule.adapter.command_line_interface.main import chat, main
from intent_to_schedule.adapter.local.state import (
    ProblemState,
    State,
    UtteranceState,
    to_problem,
    to_problem_state,
    to_schedule,
    to_schedule_state,
)
from intent_to_schedule.adapter.local.store import (
    LocalDialogueStore,
    LocalStateStore,
    load_state,
    save_state,
)
from intent_to_schedule.adapter.data_model import (
    APPLY_OPERATION_DESCRIPTION,
    CalendarInputData,
    ConstraintData,
    QUERY_OPERATION_DESCRIPTION,
    SCHEDULE_OPERATION_DESCRIPTION,
    SoftRequirementData,
    TimeWindowConditionData,
    convert_calendar_input,
)
from intent_to_schedule.adapter.mathopt.solve import MathOptSchedulingSolver
from intent_to_schedule.application.command import AddTask
from intent_to_schedule.application.objective import ScheduleSummary
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import (
    Conflicts,
    ConflictsNotFound,
    DroppedRequiredTask,
    DropReason,
    FeasibleSolution,
    NoFeasibleSolution,
    OptimalSolution,
    Solution,
    SolutionNotFound,
)
from intent_to_schedule.application.translate import (
    ApplyStep,
    MessageStep,
    ScheduleStep,
    Speaker,
    Step,
    Utterance,
)
from intent_to_schedule.domain.calendar import (
    Availability,
    Calendar,
    TimeGrid,
    TimeInterval,
)
from intent_to_schedule.domain.condition import TimeBoundCondition, TimeBoundRelation
from intent_to_schedule.domain.consistency import AllOf
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint
from intent_to_schedule.domain.measure import Boundary
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId

START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
HOUR: timedelta = timedelta(hours=1)
SLOT: timedelta = timedelta(minutes=30)
PERSON: PersonId = PersonId("person")
WEEKDAY_NAMES: list[str] = [
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
]


def task(identifier: str, duration: timedelta = HOUR) -> Task:
    """Build a required low-importance task named after its identifier."""
    return Task(
        TaskId(identifier), identifier, duration, frozenset(), Importance.LOW, True
    )


def problem(
    *tasks: Task,
    hours: int = 8,
    slot: timedelta = HOUR,
    people: tuple[PersonId, ...] = (),
) -> SchedulingProblem:
    """Build a problem whose people are available over the whole horizon."""
    grid: TimeGrid = TimeGrid(TimeInterval(START, START + hours * HOUR), slot)
    return SchedulingProblem(
        Calendar(
            grid,
            tuple(Availability(person_id, (grid.horizon,)) for person_id in people),
        ),
        tuple(Person(person_id, person_id.value) for person_id in people),
        tasks,
        (),
        (),
    )


def save_problem(
    path: Path, value: SchedulingProblem, previous: Schedule | None = None
) -> None:
    """Write a state file holding a problem and an optional previous schedule."""
    save_state(
        path,
        State(
            problem=to_problem_state(value),
            previous=to_schedule_state(previous) if previous is not None else None,
            dialogue=(),
        ),
    )


def constraint_command(
    condition: dict[str, object], identifier: str = "c", kind: str = "add_constraint"
) -> dict[str, object]:
    """Build a hard constraint command with a given identifier."""
    return {
        "kind": kind,
        "constraint": {
            "id": identifier,
            "label": "Entered request",
            "requirement": {"kind": "hard"},
            "condition": condition,
        },
    }


def fixed_task_input() -> dict[str, object]:
    """Build an appointment for the calendar's person."""
    return {
        "name": "Health check",
        "start": (START + timedelta(minutes=90)).isoformat(),
        "duration": "PT1H30M",
        "participant_ids": ["alice"],
    }


def calendar_data() -> dict[str, object]:
    """Build a small valid calendar file."""
    return {
        "horizon": {
            "start": "2026-10-01T09:00:00+00:00",
            "end": "2026-10-01T12:00:00+00:00",
        },
        "slot": "PT1H",
        "people": [{"id": "alice", "name": "Alice"}],
        "availabilities": [
            {
                "person_id": "alice",
                "intervals": [
                    {
                        "start": "2026-10-01T09:00:00+00:00",
                        "end": "2026-10-01T12:00:00+00:00",
                    }
                ],
            }
        ],
        "fixed_tasks": [],
    }


def invoke(
    capsys: pytest.CaptureFixture[str], *argv: str
) -> tuple[int, dict[str, object]]:
    """Call the entry point and parse its single JSON output."""
    status: int = main(list(argv))
    output: str = capsys.readouterr().out
    return status, json.loads(output)


def schedule_summary(
    scheduled_tasks: int = 0,
    dropped_tasks: int = 0,
    dropped_cost: float = 0.0,
    stability_cost: float = 0.0,
    moved_tasks: int = 0,
) -> dict[str, object]:
    """Build an expected schedule summary without soft violations."""
    return {
        "total_cost": dropped_cost + stability_cost,
        "costs": {
            "dropped_tasks": dropped_cost,
            "soft_constraints": 0.0,
            "stability": stability_cost,
        },
        "counts": {
            "scheduled_tasks": scheduled_tasks,
            "dropped_tasks": dropped_tasks,
            "violated_soft_constraints": 0,
            "moved_tasks": moved_tasks,
        },
    }


def initialized(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """Create a state file from a calendar."""
    calendar_path: Path = tmp_path / "calendar.json"
    state_path: Path = tmp_path / "nested" / "state.json"
    calendar_path.write_text(json.dumps(calendar_data()), encoding="utf-8")
    status: int
    output: dict[str, object]
    status, output = invoke(
        capsys, "--state", str(state_path), "init", "--calendar", str(calendar_path)
    )
    assert status == 0
    assert output == {
        "kind": "summary",
        "grid": {
            "horizon": {"start": "2026-10-01T09:00:00Z", "end": "2026-10-01T12:00:00Z"},
            "slot": "PT1H",
        },
        "counts": {"people": 1, "tasks": 0, "fixed_tasks": 0, "constraints": 0},
        "has_previous": False,
    }
    assert state_path.exists()
    return state_path


@pytest.mark.parametrize(
    ("command", "required_text"),
    [
        (
            "schedule",
            (
                "optimal",
                "feasible",
                "no_feasible_solution",
                "solution_not_found",
                "Exit codes: 0, 2, and 3",
                "{status, summary, items}",
                "{status, conflicts}",
                "time_limit",
                "previous schedule",
                "--no-stability",
            ),
        ),
        (
            "chat",
            (
                "schedule",
                "optimal",
                "feasible",
                "no_feasible_solution",
                "solution_not_found",
                "Exit codes: 0, 2, and 3",
                "No feasible solution.",
                "Solution not found.",
            ),
        ),
    ],
)
def test_schedule_and_chat_help_describe_results_and_chat_policy(
    capsys: pytest.CaptureFixture[str],
    command: str,
    required_text: tuple[str, ...],
) -> None:
    """Describe schedule output and retain chat-specific policy in help."""
    system_exit: pytest.ExceptionInfo[SystemExit]
    with pytest.raises(SystemExit) as system_exit:
        main([command, "--help"])
    assert system_exit.value.code == 0
    output: str = capsys.readouterr().out
    flattened_output: str = " ".join(output.split())
    phrase: str
    for phrase in required_text:
        assert phrase in flattened_output


def test_solve_command_has_no_alias(capsys: pytest.CaptureFixture[str]) -> None:
    """Reject the command name that was replaced by schedule."""
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "solve")
    assert status == 1
    assert "invalid choice" in str(output["error"])
    assert "schedule" in str(output["error"])


def test_cli_query_uses_scheduling_policy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Return the same policy configured for command line scheduling."""
    path: Path = initialized(tmp_path, capsys)
    query_path: Path = tmp_path / "query.json"
    query_path.write_text('{"kind":"objective_policy"}', encoding="utf-8")
    policy: ObjectivePolicy = replace(DEFAULT_POLICY, per_count=3.0)
    with patch(
        "intent_to_schedule.adapter.command_line_interface.main.DEFAULT_POLICY", policy
    ):
        status: int
        output: dict[str, object]
        status, output = invoke(
            capsys, "--state", str(path), "query", "--file", str(query_path)
        )
    assert status == 0
    assert output["per_count"] == 3.0


def test_cli_naive_time_bound_returns_validation_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report offset-free bounds as JSON errors without changing state."""
    path: Path = initialized(tmp_path, capsys)
    before: str = path.read_text(encoding="utf-8")
    input_path: Path = tmp_path / "commands.json"
    input_path.write_text(
        json.dumps(
            {
                "commands": [
                    {
                        "kind": "add_constraint",
                        "constraint": {
                            "requirement": {"kind": "hard"},
                            "condition": {
                                "kind": "time_bound",
                                "task_ids": ["missing"],
                                "boundary": "start",
                                "relation": "at",
                                "at": "2026-10-05T17:00:00",
                            },
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    status: int
    output: dict[str, object]
    status, output = invoke(
        capsys, "--state", str(path), "apply", "--file", str(input_path)
    )
    assert status == 1
    assert "error" in output
    assert "timezone" in str(output["error"])
    assert path.read_text(encoding="utf-8") == before


def test_init_show_schema_and_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Initialize state, show it, expose the schema, and round trip the problem."""
    path: Path = initialized(tmp_path, capsys)
    state_before: State = load_state(path)
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "show", "--state", str(path))
    assert status == 0
    assert output == state_before.model_dump(mode="json")
    assert to_problem(load_state(path).problem) == to_problem(state_before.problem)
    status, output = invoke(capsys, "schema")
    assert status == 0
    assert output["title"] == "CommandsData"
    assert output["required"] == ["commands"]


@pytest.mark.parametrize(
    ("command", "description"),
    [
        ("apply", APPLY_OPERATION_DESCRIPTION),
        ("query", QUERY_OPERATION_DESCRIPTION),
    ],
)
def test_schema_uses_the_shared_operation_description(
    capsys: pytest.CaptureFixture[str], command: str, description: str
) -> None:
    """Expose the shared apply or query description at the schema root."""
    status: int
    output: dict[str, Any]
    status, output = invoke(capsys, "schema", command)
    assert status == 0
    assert output["description"] == description


@pytest.mark.parametrize(
    ("command", "description"),
    [
        ("apply", APPLY_OPERATION_DESCRIPTION),
        ("query", QUERY_OPERATION_DESCRIPTION),
        ("schedule", SCHEDULE_OPERATION_DESCRIPTION),
    ],
)
def test_command_help_uses_the_shared_operation_description(
    capsys: pytest.CaptureFixture[str], command: str, description: str
) -> None:
    """Show the unchanged shared description in each operation's help."""
    error: pytest.ExceptionInfo[SystemExit]
    with pytest.raises(SystemExit) as error:
        main(["help", command])
    assert error.value.code == 0
    output: str = " ".join(capsys.readouterr().out.split())
    assert description in output


def test_init_rejects_invalid_calendar_without_writing_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Report validator errors without creating a state file."""
    calendar: dict[str, object] = calendar_data()
    calendar["availabilities"] = []
    calendar_path: Path = tmp_path / "calendar.json"
    state_path: Path = tmp_path / "state.json"
    calendar_path.write_text(json.dumps(calendar), encoding="utf-8")
    status: int
    output: dict[str, object]
    status, output = invoke(
        capsys, "--state", str(state_path), "init", "--calendar", str(calendar_path)
    )
    assert status == 1
    assert output == {"rejected": ["Person alice has no availability."]}
    assert not state_path.exists()


def test_init_fixed_tasks_persists_ids_and_round_trips(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Initialize fixed events with generated IDs and preserve their exact times."""
    calendar: dict[str, object] = calendar_data()
    calendar["fixed_tasks"] = [
        {
            "name": "Existing review",
            "start": "2026-10-01T09:15:00+00:00",
            "duration": "PT30M",
            "participant_ids": ["alice"],
        }
    ]
    calendar_path: Path = tmp_path / "calendar.json"
    path: Path = tmp_path / "state.json"
    calendar_path.write_text(json.dumps(calendar), encoding="utf-8")
    status: int
    output: dict[str, object]
    status, output = invoke(
        capsys, "init", "--state", str(path), "--calendar", str(calendar_path)
    )
    assert status == 0
    assert output["counts"] == {
        "people": 1,
        "tasks": 0,
        "fixed_tasks": 1,
        "constraints": 0,
    }
    state: State = load_state(path)
    assert state.problem.fixed_tasks[0].id.value
    problem: SchedulingProblem = to_problem(state.problem)
    fixed: FixedTask = problem.fixed_tasks[0]
    assert fixed.start == datetime(2026, 10, 1, 9, 15, tzinfo=timezone.utc)
    assert fixed.duration == timedelta(minutes=30)
    assert to_problem(to_problem_state(problem)) == problem
    status, output = invoke(capsys, "schedule", "--state", str(path))
    assert status == 0
    assert output == {"status": "optimal", "summary": schedule_summary(), "items": []}


def test_init_rejects_fixed_task_with_unknown_participant(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reject fixed events that reference a missing person."""
    calendar: dict[str, object] = calendar_data()
    calendar["fixed_tasks"] = [
        {
            "id": "existing-review",
            "name": "Existing review",
            "start": "2026-10-01T09:15:00+00:00",
            "duration": "PT30M",
            "participant_ids": ["missing"],
        }
    ]
    calendar_path: Path = tmp_path / "calendar.json"
    path: Path = tmp_path / "state.json"
    calendar_path.write_text(json.dumps(calendar), encoding="utf-8")
    status: int
    output: dict[str, object]
    status, output = invoke(
        capsys, "init", "--state", str(path), "--calendar", str(calendar_path)
    )
    assert status == 1
    assert output == {
        "rejected": ["Task existing-review references missing person id missing."]
    }
    assert not path.exists()


def test_apply_reject_and_schedule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Apply commands, preserve rejected state, and keep the second solve stable."""
    path: Path = initialized(tmp_path, capsys)
    task_file: Path = tmp_path / "task.json"
    task_file.write_text(
        json.dumps(
            {
                "commands": [
                    {
                        "kind": "add_task",
                        "task": {
                            "name": "Review",
                            "duration": "PT1H",
                            "participant_ids": ["alice"],
                            "importance": "high",
                            "required": True,
                            "stability": "normal",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    status: int
    output: dict[str, object]
    status, output = invoke(
        capsys, "--state", str(path), "apply", "--file", str(task_file)
    )
    assert status == 0
    task_id: str = load_state(path).problem.tasks[0].id.value
    assert output == {
        "executed": [{"kind": "add_task", "task_id": task_id, "name": "Review"}]
    }

    constraint_input: dict[str, object] = {
        "commands": [
            {
                "kind": "add_constraint",
                "constraint": {
                    "requirement": {"kind": "hard"},
                    "condition": {
                        "kind": "time_bound",
                        "task_ids": [task_id],
                        "boundary": "start",
                        "relation": "at",
                        "at": datetime(2026, 10, 1, 9, tzinfo=timezone.utc).isoformat(),
                    },
                },
            }
        ]
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(constraint_input)))
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 0
    current: State = load_state(path)
    assert len(current.problem.constraints) == 1
    assert output == {
        "executed": [
            {
                "kind": "add_constraint",
                "constraint_id": current.problem.constraints[0].id.value,
            }
        ]
    }
    domain_problem: SchedulingProblem = to_problem(current.problem)
    save_state(
        path, current.model_copy(update={"problem": to_problem_state(domain_problem)})
    )
    assert to_problem(load_state(path).problem) == domain_problem

    unchanged: str = path.read_text(encoding="utf-8")
    rejected_file: Path = tmp_path / "rejected.json"
    rejected_file.write_text(
        json.dumps({"commands": [{"kind": "remove_task", "task_id": "missing"}]}),
        encoding="utf-8",
    )
    status, output = invoke(
        capsys, "--state", str(path), "apply", "--file", str(rejected_file)
    )
    assert status == 1
    assert output == {"rejected": ["Task missing does not exist"]}
    assert path.read_text(encoding="utf-8") == unchanged

    status, output = invoke(capsys, "--state", str(path), "schedule")
    assert status == 0
    assert output == {
        "status": "optimal",
        "summary": schedule_summary(scheduled_tasks=1),
        "items": [
            {
                "status": "scheduled",
                "task_id": task_id,
                "name": "Review",
                "start": "2026-10-01T09:00:00+00:00",
                "end": "2026-10-01T10:00:00+00:00",
                "participant_ids": ["alice"],
            }
        ],
    }
    assert load_state(path).previous is not None
    second: dict[str, object]
    status, second = invoke(capsys, "--state", str(path), "schedule")
    assert status == 0
    assert second == output


def test_apply_uses_given_ids_within_the_same_batch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Add a task and a constraint with given IDs and reference the task in the same batch."""
    path: Path = initialized(tmp_path, capsys)
    commands_file: Path = tmp_path / "commands.json"
    commands_file.write_text(
        json.dumps(
            {
                "commands": [
                    {
                        "kind": "add_task",
                        "task": {
                            "id": "review",
                            "name": "Review",
                            "duration": "PT1H",
                            "participant_ids": ["alice"],
                            "importance": "high",
                            "required": True,
                            "stability": "normal",
                        },
                    },
                    {
                        "kind": "add_constraint",
                        "constraint": {
                            "id": "review-at-ten",
                            "requirement": {"kind": "soft", "strength": "normal"},
                            "condition": {
                                "kind": "time_bound",
                                "task_ids": ["review"],
                                "boundary": "start",
                                "relation": "at",
                                "at": "2026-10-01T10:00:00+00:00",
                            },
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    status: int
    output: dict[str, object]
    status, output = invoke(
        capsys, "--state", str(path), "apply", "--file", str(commands_file)
    )
    assert status == 0
    assert output == {
        "executed": [
            {"kind": "add_task", "task_id": "review", "name": "Review"},
            {"kind": "add_constraint", "constraint_id": "review-at-ten"},
        ]
    }
    status, output = invoke(capsys, "--state", str(path), "schedule")
    assert status == 0
    assert output["items"] == [
        {
            "status": "scheduled",
            "task_id": "review",
            "name": "Review",
            "start": "2026-10-01T10:00:00+00:00",
            "end": "2026-10-01T11:00:00+00:00",
            "participant_ids": ["alice"],
        }
    ]


@pytest.mark.parametrize(
    "commands,message",
    [
        (
            [
                {
                    "kind": "add_task",
                    "task": {
                        "id": "review",
                        "name": "Review",
                        "duration": "PT1H",
                        "participant_ids": ["alice"],
                        "importance": "high",
                        "required": True,
                        "stability": "normal",
                    },
                }
            ]
            * 2,
            "Task review already exists",
        ),
        (
            [
                {
                    "kind": "add_task",
                    "task": {
                        "id": "",
                        "name": "Review",
                        "duration": "PT1H",
                        "participant_ids": ["alice"],
                        "importance": "high",
                        "required": True,
                        "stability": "normal",
                    },
                }
            ],
            "ID must be a non-empty string",
        ),
    ],
)
def test_apply_rejects_duplicate_or_empty_given_id_without_saving(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    commands: list[dict[str, object]],
    message: str,
) -> None:
    """Reject a given ID that already exists or is empty, leaving the state unchanged."""
    path: Path = initialized(tmp_path, capsys)
    unchanged: str = path.read_text(encoding="utf-8")
    commands_file: Path = tmp_path / "commands.json"
    commands_file.write_text(json.dumps({"commands": commands}), encoding="utf-8")
    status: int
    output: dict[str, object]
    status, output = invoke(
        capsys, "--state", str(path), "apply", "--file", str(commands_file)
    )
    assert status == 1
    assert message in json.dumps(output)
    assert path.read_text(encoding="utf-8") == unchanged


@pytest.mark.parametrize(
    ("relation_value", "direction"), [("within", "inward"), ("avoid", "outward")]
)
def test_apply_time_window_constraint_persists_windows_and_reports_rounding(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    relation_value: str,
    direction: str,
) -> None:
    path: Path = initialized(tmp_path, capsys)
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "commands": [
                        {
                            "kind": "add_task",
                            "task": {
                                "name": "Review",
                                "duration": "PT1H",
                                "participant_ids": ["alice"],
                                "importance": "high",
                                "required": False,
                                "stability": "normal",
                            },
                        }
                    ]
                }
            )
        ),
    )
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 0
    task_id: str = load_state(path).problem.tasks[0].id.value
    assert output == {
        "executed": [{"kind": "add_task", "task_id": task_id, "name": "Review"}]
    }
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "commands": [
                        {
                            "kind": "add_constraint",
                            "constraint": {
                                "requirement": {"kind": "soft", "strength": "strong"},
                                "condition": {
                                    "kind": "time_window",
                                    "task_ids": [task_id],
                                    "relation": relation_value,
                                    "windows": [
                                        {
                                            "time_range": {
                                                "start": "09:10",
                                                "end": "11:10",
                                            }
                                        }
                                    ],
                                },
                            },
                        }
                    ]
                }
            )
        ),
    )
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 0
    state: State = load_state(path)
    constraint: ConstraintData = state.problem.constraints[0]
    record: dict[str, object] = {
        "kind": "add_constraint",
        "constraint_id": constraint.id.value,
        "note": f"Window times were rounded {direction} to calendar slots.",
    }
    if relation_value == "avoid":
        record["warnings"] = [window_warning(task_id, relation_value, VIOLATED)]
    assert output == {"executed": [record]}
    assert isinstance(constraint.requirement, SoftRequirementData)
    assert constraint.requirement.strength == "strong"
    assert isinstance(constraint.condition, TimeWindowConditionData)
    assert {value.value for value in constraint.condition.task_ids} == {task_id}
    assert not state.problem.tasks[0].required
    assert constraint.condition.model_dump(mode="json") == {
        "kind": "time_window",
        "task_ids": [task_id],
        "relation": relation_value,
        "windows": [
            {
                "date_range": "horizon",
                "weekdays": WEEKDAY_NAMES,
                "time_range": {"start": "09:10", "end": "11:10"},
            }
        ],
    }


def test_show_and_constraints_query_write_omitted_window_fields_explicitly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Write the defaults of omitted window fields instead of null."""
    path: Path = tmp_path / "state.json"
    save_problem(path, problem(task("review")))
    condition: dict[str, object] = {
        "kind": "time_window",
        "task_ids": ["review"],
        "relation": "within",
        "windows": [{}, {"date_range": "horizon", "weekdays": []}],
    }
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"commands": [constraint_command(condition)]})),
    )
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 0
    whole_day: dict[str, str] = {"start": "00:00", "end": "24:00"}
    windows: list[dict[str, object]] = [
        {"date_range": "horizon", "weekdays": WEEKDAY_NAMES, "time_range": whole_day},
        {"date_range": "horizon", "weekdays": [], "time_range": whole_day},
    ]
    status, output = invoke(capsys, "--state", str(path), "show")
    assert status == 0
    shown: dict[str, Any] = cast(dict[str, Any], output)["problem"]["constraints"][0]
    assert shown["condition"]["windows"] == windows
    assert "null" not in json.dumps(shown["condition"])
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"kind": "constraints"})))
    status, output = invoke(capsys, "--state", str(path), "query")
    assert status == 0
    listed: dict[str, Any] = cast(list[dict[str, Any]], output["items"])[0]
    assert listed["condition"]["windows"] == windows


UNSCHEDULED: str = "this hard constraint keeps the task unscheduled."
INFEASIBLE: str = (
    "this hard constraint makes scheduling infeasible because the task is required."
)
VIOLATED: str = "this soft constraint is violated whenever the task is scheduled."


def late_start_problem(required: bool = False) -> SchedulingProblem:
    """Build a three-hour review for a person working from 11:00 to the 17:00 horizon end."""
    value: SchedulingProblem = problem(people=(PERSON,))
    review: Task = Task(
        TaskId("review"),
        "Review",
        3 * HOUR,
        frozenset({PERSON}),
        Importance.LOW,
        required,
    )
    working: Availability = Availability(
        PERSON, (TimeInterval(START + 2 * HOUR, START + 8 * HOUR),)
    )
    return replace(
        value,
        calendar=replace(value.calendar, availabilities=(working,)),
        tasks=(review,),
    )


def window_warning(task_id: str, relation: str, consequence: str) -> str:
    """Build the warning for a task named Review without a satisfying start."""
    placement: str = (
        "within the windows" if relation == "within" else "out of the windows"
    )
    return (
        f"Task {task_id} (Review) has no available start (all participants available "
        f"and free of fixed tasks) that keeps the whole task {placement} after "
        f"rounding; {consequence}"
    )


@pytest.mark.parametrize("kind", ["add_constraint", "replace_constraint"])
@pytest.mark.parametrize(
    "relation, time_range, warned",
    [
        ("within", {"start": "00:00", "end": "12:00"}, True),
        ("within", {"start": "00:00", "end": "14:00"}, False),
        ("avoid", {"start": "10:00", "end": "24:00"}, True),
        ("avoid", {"start": "14:00", "end": "24:00"}, False),
    ],
)
def test_apply_warns_when_no_available_start_satisfies_time_window(
    kind: str,
    relation: str,
    time_range: dict[str, str],
    warned: bool,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Warn about a window outside working hours while saving the batch."""
    value: SchedulingProblem = late_start_problem()
    if kind == "replace_constraint":
        existing: HardConstraint = HardConstraint(
            ConstraintId("window"),
            TimeBoundCondition(
                frozenset({TaskId("review")}),
                Boundary.END,
                TimeBoundRelation.AT_OR_BEFORE,
                START + 8 * HOUR,
            ),
        )
        value = replace(value, constraints=(existing,))
    path: Path = tmp_path / "state.json"
    save_problem(path, value)
    condition: dict[str, object] = {
        "kind": "time_window",
        "task_ids": ["review"],
        "relation": relation,
        "windows": [{"time_range": time_range}],
    }
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps({"commands": [constraint_command(condition, "window", kind)]})
        ),
    )
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 0
    record: dict[str, object] = {"kind": kind, "constraint_id": "window"}
    if warned:
        record["warnings"] = [window_warning("review", relation, UNSCHEDULED)]
    assert output == {"executed": [record]}
    saved: ConstraintData = load_state(path).problem.constraints[0]
    assert isinstance(saved.condition, TimeWindowConditionData)
    assert saved.condition.relation == relation


@pytest.mark.parametrize(
    "requirement, required, consequence",
    [
        ({"kind": "hard"}, True, INFEASIBLE),
        ({"kind": "soft", "strength": "strong"}, False, VIOLATED),
        ({"kind": "soft", "strength": "weak"}, True, VIOLATED),
    ],
)
def test_window_warning_states_the_consequence_of_the_requirement(
    requirement: dict[str, str],
    required: bool,
    consequence: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explain what an unsatisfiable window does for hard and soft requirements."""
    path: Path = tmp_path / "state.json"
    save_problem(path, late_start_problem(required))
    command: dict[str, Any] = constraint_command(
        {
            "kind": "time_window",
            "task_ids": ["review"],
            "relation": "within",
            "windows": [{"time_range": {"start": "00:00", "end": "12:00"}}],
        },
        "window",
    )
    command["constraint"]["requirement"] = requirement
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"commands": [command]})))
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 0
    assert output == {
        "executed": [
            {
                "kind": "add_constraint",
                "constraint_id": "window",
                "warnings": [window_warning("review", "within", consequence)],
            }
        ]
    }


def test_apply_warns_using_fixed_tasks_added_later_in_the_batch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Judge windows against the whole batch and skip fixed tasks in the condition."""
    path: Path = tmp_path / "state.json"
    save_problem(path, late_start_problem())
    window: dict[str, object] = {
        "kind": "time_window",
        "relation": "within",
        "windows": [{"time_range": {"start": "00:00", "end": "15:00"}}],
    }
    absence: dict[str, object] = {
        "id": "absence",
        "name": "Absence",
        "start": (START + 4 * HOUR).isoformat(),
        "duration": "PT3H",
        "participant_ids": [PERSON.value],
    }
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "commands": [
                        constraint_command(
                            {**window, "task_ids": ["review"]}, "review-window"
                        ),
                        {"kind": "add_task", "task": absence},
                        constraint_command(
                            {**window, "task_ids": ["absence"]}, "absence-window"
                        ),
                    ]
                }
            )
        ),
    )
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 0
    assert output == {
        "executed": [
            {
                "kind": "add_constraint",
                "constraint_id": "review-window",
                "warnings": [window_warning("review", "within", UNSCHEDULED)],
            },
            {"kind": "add_task", "task_id": "absence", "name": "Absence"},
            {"kind": "add_constraint", "constraint_id": "absence-window"},
        ]
    }
    assert len(load_state(path).problem.constraints) == 2


def window_until(end: str, kind: str = "add_constraint") -> dict[str, object]:
    """Build a hard window command keeping the review between midnight and an end time."""
    return constraint_command(
        {
            "kind": "time_window",
            "task_ids": ["review"],
            "relation": "within",
            "windows": [{"time_range": {"start": "00:00", "end": end}}],
        },
        "window",
        kind,
    )


@pytest.mark.parametrize(
    "commands, warned",
    [
        ([window_until("12:00"), window_until("17:00", "replace_constraint")], False),
        (
            [
                window_until("12:00"),
                {"kind": "remove_constraint", "constraint_id": "window"},
            ],
            False,
        ),
        (
            [window_until("17:00"), window_until("12:00", "replace_constraint")],
            True,
        ),
    ],
)
def test_apply_warns_only_for_the_constraint_the_batch_keeps(
    commands: list[dict[str, object]],
    warned: bool,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip warnings for windows a later command in the batch replaces or removes."""
    path: Path = tmp_path / "state.json"
    save_problem(path, late_start_problem(required=True))
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"commands": commands})))
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 0
    expected: list[dict[str, object]] = [
        {"kind": command["kind"], "constraint_id": "window"} for command in commands
    ]
    if warned:
        expected[-1]["warnings"] = [window_warning("review", "within", INFEASIBLE)]
    assert output == {"executed": expected}


def query_state(tmp_path: Path) -> Path:
    """Create query state without applying commands or solving."""
    calendar: CalendarInputData = CalendarInputData.model_validate(calendar_data())
    path: Path = tmp_path / "query-state.json"
    save_state(
        path,
        State(
            problem=to_problem_state(convert_calendar_input(calendar)),
            previous=None,
            dialogue=(),
        ),
    )
    return path


def test_query_file_and_standard_input_preserve_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Read queries from files and standard input without writing state."""
    path: Path = query_state(tmp_path)
    unchanged: bytes = path.read_bytes()
    query_path: Path = tmp_path / "query.json"
    query_path.write_text(json.dumps({"kind": "summary"}), encoding="utf-8")
    status: int
    output: dict[str, object]
    status, output = invoke(
        capsys, "query", "--state", str(path), "--file", str(query_path)
    )
    assert status == 0 and output["kind"] == "summary"
    assert output["counts"] == {
        "people": 1,
        "tasks": 0,
        "fixed_tasks": 0,
        "constraints": 0,
    }
    assert output["has_previous"] is False
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"kind": "people"})))
    status, output = invoke(capsys, "--state", str(path), "query")
    assert status == 0
    assert output == {
        "kind": "people",
        "items": [{"id": "alice", "name": "Alice"}],
        "total": 1,
        "truncated": False,
    }
    assert path.read_bytes() == unchanged


@pytest.mark.parametrize(
    "value",
    [
        {"kind": "people", "limit": 0},
        {"kind": "tasks", "limit": 101},
        {"kind": "people", "select": "one", "filter": {"person_ids": ["missing"]}},
        {
            "kind": "available_starts",
            "participant_ids": ["missing"],
            "duration": "PT1H",
        },
        {"kind": "agenda", "person_id": "missing"},
    ],
)
def test_query_rejection_exits_one(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    value: dict[str, object],
) -> None:
    """Return rejected answers with exit status one and preserve state."""
    path: Path = query_state(tmp_path)
    unchanged: bytes = path.read_bytes()
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(value)))
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "query", "--state", str(path))
    assert status == 1 and set(output) == {"rejected"}
    assert output["rejected"]
    assert path.read_bytes() == unchanged


def test_query_invalid_json_reports_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report invalid query input without changing state."""
    path: Path = query_state(tmp_path)
    unchanged: bytes = path.read_bytes()
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"kind": "unknown"}'))
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "query", "--state", str(path))
    assert status == 1 and "error" in output
    assert path.read_bytes() == unchanged


def test_query_previous_schedule_and_schema(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Expose query schema and answer a missing previous schedule."""
    path: Path = query_state(tmp_path)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"kind": "previous_schedule"}'))
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "query", "--state", str(path))
    assert status == 0
    assert output == {
        "kind": "previous_schedule",
        "items": [],
        "total": 0,
        "truncated": False,
        "has_previous": False,
    }
    status, output = invoke(capsys, "schema", "query")
    assert status == 0
    assert output["discriminator"] == {
        "propertyName": "kind",
        "mapping": {
            "summary": "#/$defs/SummaryQueryData",
            "people": "#/$defs/PeopleQueryData",
            "tasks": "#/$defs/TasksQueryData",
            "constraints": "#/$defs/ConstraintsQueryData",
            "previous_schedule": "#/$defs/PreviousScheduleQueryData",
            "available_starts": "#/$defs/AvailableStartsQueryData",
            "evaluation": "#/$defs/EvaluationQueryData",
            "objective_policy": "#/$defs/ObjectivePolicyQueryData",
            "agenda": "#/$defs/AgendaQueryData",
        },
    }


@pytest.mark.parametrize(
    ("outcome", "stability", "explicit_now"),
    [
        ("message", True, False),
        ("exhausted", True, True),
        ("optimal", True, False),
        ("feasible", False, True),
        ("no_feasible_solution", True, False),
        ("solution_not_found", True, True),
    ],
)
def test_chat_persists_schedule_outcomes_and_uses_clock(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    explicit_now: bool,
    stability: bool,
) -> None:
    """Persist dialogue and accepted changes for every turn."""
    start: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    horizon: TimeInterval = TimeInterval(start, start + timedelta(hours=3))
    existing: Task = Task(
        TaskId("existing"),
        "Existing",
        timedelta(hours=1),
        frozenset({PersonId("alice")}),
        Importance.HIGH,
        True,
        Strength.NORMAL,
    )
    added: Task = Task(
        TaskId("added"),
        "Added",
        timedelta(hours=1),
        frozenset({PersonId("alice")}),
        Importance.MEDIUM,
        False,
        Strength.NORMAL,
    )
    problem: SchedulingProblem = SchedulingProblem(
        Calendar(
            TimeGrid(horizon, timedelta(hours=1)),
            (Availability(PersonId("alice"), (horizon,)),),
        ),
        (Person(PersonId("alice"), "Alice"),),
        (existing,),
        (),
        (),
    )
    previous: Schedule = Schedule(
        (
            ScheduledTask(
                existing.id,
                existing.name,
                start,
                start + existing.duration,
                existing.participant_ids,
            ),
        ),
        (),
    )
    replacement: Schedule = Schedule(
        (
            ScheduledTask(
                added.id,
                added.name,
                start + timedelta(hours=2),
                start + timedelta(hours=3),
                added.participant_ids,
            ),
            ScheduledTask(
                existing.id,
                existing.name,
                start + timedelta(hours=1),
                start + timedelta(hours=2),
                existing.participant_ids,
            ),
        ),
        (),
    )
    path: Path = tmp_path / "state.json"
    state: State = State(
        problem=to_problem_state(problem),
        previous=to_schedule_state(previous),
        dialogue=(
            UtteranceState(speaker="user", text="earlier request"),
            UtteranceState(speaker="assistant", text="earlier answer"),
        ),
    )
    save_state(path, state)
    terminal: Step = (
        MessageStep("When works for you?")
        if outcome == "message"
        else ScheduleStep(stability)
    )
    steps: list[Step] = [ApplyStep((AddTask(added),)), terminal]
    if outcome == "exhausted":
        steps = [
            ApplyStep((AddTask(added),)),
            *(ApplyStep(()) for _ in range(converse.STEP_LIMIT - 1)),
        ]
    translator: MagicMock = MagicMock()
    translator.translate.side_effect = steps
    factory: MagicMock = MagicMock(return_value=translator)
    monkeypatch.setattr(main_module, "OpenAIStepTranslator", factory)
    client: MagicMock = MagicMock()
    monkeypatch.setattr(openai, "OpenAI", lambda: client)
    solver: MagicMock = MagicMock()
    summary_value: ScheduleSummary = ScheduleSummary(
        0.0,
        0.0,
        5 / (1 + 5 / 50) if stability else 0.0,
        2,
        0,
        0,
        1,
    )
    schedule_result: OptimalSolution | FeasibleSolution | NoFeasibleSolution | SolutionNotFound = (
        NoFeasibleSolution(
            Conflicts(
                (), (DroppedRequiredTask(added.id, added.name, DropReason.CONFLICT),)
            )
        )
        if outcome == "no_feasible_solution"
        else SolutionNotFound("time_limit")
        if outcome == "solution_not_found"
        else FeasibleSolution(replacement, summary_value)
        if outcome == "feasible"
        else OptimalSolution(replacement, summary_value)
    )
    solver.solve.return_value = schedule_result
    now: datetime | None = start - timedelta(days=2) if explicit_now else None
    state_store: LocalStateStore = LocalStateStore(path)
    dialogue_store: LocalDialogueStore = LocalDialogueStore(path)
    service: Scheduling = Scheduling(solver, AllOf(), state_store)
    status: int = chat(
        "new request", "demo-model", now, service, dialogue_store
    )
    output: dict[str, object] = json.loads(capsys.readouterr().out)
    persisted: State = load_state(path)
    assert persisted.dialogue[:2] == state.dialogue
    assert persisted.dialogue[2] == UtteranceState(speaker="user", text="new request")
    assert persisted.dialogue[3].speaker == "assistant"
    assert translator.translate.call_args_list[0].args[0] == (
        Utterance(Speaker.USER, "earlier request"),
        Utterance(Speaker.ASSISTANT, "earlier answer"),
        Utterance(Speaker.USER, "new request"),
    )
    assert factory.call_args.args[:2] == (client, "demo-model")
    assert factory.call_args.args[2]() == (now if explicit_now else start)
    assert translator.translate.call_args_list[1].args[1].tasks == 2
    assert to_problem(persisted.problem).tasks == (existing, added)
    if outcome in {"message", "exhausted"}:
        assert persisted.previous == state.previous
        solver.solve.assert_not_called()
        if outcome == "message":
            assert status == 0
            assert output == {"message": "When works for you?"}
            assert persisted.dialogue[-1].text == "When works for you?"
        else:
            assert status == 1
            assert output == {"exhausted": True}
            assert persisted.dialogue[-1].text == "Step limit reached."
    else:
        solver.solve.assert_called_once()
        if outcome == "no_feasible_solution":
            assert status == 2
            assert output == {
                "status": "no_feasible_solution",
                "conflicts": {
                    "status": "found",
                    "constraints": [],
                    "dropped_required_tasks": [
                        {"task_id": "added", "name": "Added", "reason": "conflict"}
                    ],
                },
            }
            assert persisted.previous == state.previous
            assert persisted.dialogue[-1].text == "No feasible solution."
        elif outcome == "solution_not_found":
            assert status == 3
            assert output == {
                "status": "solution_not_found",
                "reason": "time_limit",
            }
            assert persisted.previous == state.previous
            assert persisted.dialogue[-1].text == "Solution not found."
        else:
            assert status == 0
            assert persisted.previous == to_schedule_state(replacement)
            assert output == {
                "status": outcome,
                "summary": schedule_summary(
                    scheduled_tasks=2,
                    stability_cost=5 / (1 + 5 / 50) if stability else 0.0,
                    moved_tasks=1,
                ),
                "items": [
                    {
                        "status": "scheduled",
                        "task_id": "existing",
                        "name": "Existing",
                        "start": (start + timedelta(hours=1)).isoformat(),
                        "end": (start + timedelta(hours=2)).isoformat(),
                        "participant_ids": ["alice"],
                    },
                    {
                        "status": "scheduled",
                        "task_id": "added",
                        "name": "Added",
                        "start": (start + timedelta(hours=2)).isoformat(),
                        "end": (start + timedelta(hours=3)).isoformat(),
                        "participant_ids": ["alice"],
                    },
                ],
            }
            assert persisted.dialogue[-1].text == "Scheduled."


def test_chat_saves_accepted_apply_before_later_translator_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep an accepted apply saved when a later translation raises."""
    path: Path = tmp_path / "state.json"
    original: SchedulingProblem = problem()
    added: Task = task("added")
    save_problem(path, original)
    translator: MagicMock = MagicMock()
    translator.translate.side_effect = [
        ApplyStep((AddTask(added),)),
        RuntimeError("translator failed"),
    ]
    factory: MagicMock = MagicMock(return_value=translator)
    monkeypatch.setattr(main_module, "OpenAIStepTranslator", factory)
    client: MagicMock = MagicMock()
    monkeypatch.setattr(openai, "OpenAI", lambda: client)
    service: Scheduling = Scheduling(
        MagicMock(), AllOf(), LocalStateStore(path)
    )

    with pytest.raises(RuntimeError, match="translator failed"):
        chat(
            "new request",
            "demo-model",
            None,
            service,
            LocalDialogueStore(path),
        )

    assert to_problem(load_state(path).problem).tasks == (added,)


@pytest.mark.parametrize(
    ("outcome", "exit_status", "saves_schedule"),
    [
        ("optimal", 0, True),
        ("feasible", 0, True),
        ("no_feasible_solution", 2, False),
        ("solution_not_found", 3, False),
    ],
)
def test_schedule_reports_each_outcome_and_persists_only_solutions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    exit_status: int,
    saves_schedule: bool,
) -> None:
    """Report each solution result and save only schedules with a solution."""
    item: Task = task("review")
    value: SchedulingProblem = problem(item)
    previous: Schedule = Schedule((), ())
    scheduled: Schedule = Schedule(
        (
            ScheduledTask(
                item.id,
                item.name,
                START,
                START + item.duration,
                item.participant_ids,
            ),
        ),
        (),
    )
    summary_value: ScheduleSummary = ScheduleSummary(0.0, 0.0, 0.0, 1, 0, 0, 0)
    result: Solution = (
        OptimalSolution(scheduled, summary_value)
        if outcome == "optimal"
        else FeasibleSolution(scheduled, summary_value)
        if outcome == "feasible"
        else NoFeasibleSolution(ConflictsNotFound("time_limit"))
        if outcome == "no_feasible_solution"
        else SolutionNotFound("time_limit")
    )
    path: Path = tmp_path / "state.json"
    save_problem(path, value, previous)
    before: State = load_state(path)
    monkeypatch.setattr(
        MathOptSchedulingSolver, "solve", MagicMock(return_value=result)
    )

    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(path), "schedule")

    assert status == exit_status
    expected: dict[str, object] = (
        {
            "status": outcome,
            "summary": schedule_summary(scheduled_tasks=1),
            "items": [
                {
                    "status": "scheduled",
                    "task_id": "review",
                    "name": "review",
                    "start": START.isoformat(),
                    "end": (START + HOUR).isoformat(),
                    "participant_ids": [],
                }
            ],
        }
        if outcome in {"optimal", "feasible"}
        else {
            "status": "no_feasible_solution",
            "conflicts": {"status": "not_found", "reason": "time_limit"},
        }
        if outcome == "no_feasible_solution"
        else {"status": "solution_not_found", "reason": "time_limit"}
    )
    assert output == expected
    after: State = load_state(path)
    assert after.problem == before.problem
    assert after.dialogue == before.dialogue
    assert after.previous == (
        to_schedule_state(scheduled) if saves_schedule else before.previous
    )


def test_schedule_entries_survive_task_changes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve solved names, times, and participants after tasks change or disappear."""
    path: Path = initialized(tmp_path, capsys)
    task: dict[str, object] = {
        "id": "review",
        "name": "Original review",
        "duration": "PT1H",
        "participant_ids": ["alice"],
        "importance": "low",
        "required": True,
        "stability": "normal",
    }
    dropped_task: dict[str, object] = {
        **task,
        "id": "dropped",
        "name": "Original dropped",
        "duration": "PT4H",
        "required": False,
    }
    commands: list[dict[str, object]] = [
        {"kind": "add_task", "task": task},
        {"kind": "add_task", "task": dropped_task},
        {
            "kind": "add_constraint",
            "constraint": {
                "requirement": {"kind": "hard"},
                "condition": {
                    "kind": "time_bound",
                    "task_ids": ["review"],
                    "boundary": "start",
                    "relation": "at",
                    "at": "2026-10-01T09:00:00+00:00",
                },
            },
        },
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"commands": commands})))
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 0
    status, output = invoke(capsys, "--state", str(path), "schedule")
    expected: list[dict[str, str | list[str]]] = [
        {
            "status": "scheduled",
            "task_id": "review",
            "name": "Original review",
            "start": "2026-10-01T09:00:00+00:00",
            "end": "2026-10-01T10:00:00+00:00",
            "participant_ids": ["alice"],
        },
        {"status": "dropped", "task_id": "dropped", "name": "Original dropped"},
    ]
    assert status == 0
    assert output == {
        "status": "optimal",
        "summary": schedule_summary(scheduled_tasks=1, dropped_tasks=1, dropped_cost=5.0),
        "items": expected,
    }
    persisted: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["previous"] == {"items": expected}
    changes: tuple[list[dict[str, object]], ...] = (
        [
            {
                "kind": "replace_task",
                "task": {
                    **task,
                    "name": "Renamed review",
                    "duration": "PT2H",
                    "participant_ids": [],
                },
            },
            {
                "kind": "replace_task",
                "task": {**dropped_task, "name": "Renamed dropped"},
            },
        ],
        [
            {"kind": "remove_task", "task_id": "review"},
            {"kind": "remove_task", "task_id": "dropped"},
        ],
    )
    change: list[dict[str, object]]
    for change in changes:
        monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"commands": change})))
        status, output = invoke(capsys, "--state", str(path), "apply")
        assert status == 0
        monkeypatch.setattr(sys, "stdin", io.StringIO('{"kind":"previous_schedule"}'))
        status, output = invoke(capsys, "--state", str(path), "query")
        assert status == 0
        assert output == {
            "kind": "previous_schedule",
            "items": expected,
            "total": 2,
            "truncated": False,
            "has_previous": True,
        }
        monkeypatch.setattr(
            sys, "stdin", io.StringIO('{"kind":"agenda","person_id":"alice"}')
        )
        status, output = invoke(capsys, "--state", str(path), "query")
        assert status == 0
        assert output == {
            "kind": "agenda",
            "person_id": "alice",
            "has_previous": True,
            "days": [
                {
                    "date": "2026-10-01",
                    "working": [
                        {
                            "start": "2026-10-01T09:00:00+00:00",
                            "end": "2026-10-01T12:00:00+00:00",
                        }
                    ],
                    "items": [
                        {
                            "type": "scheduled",
                            "task_id": "review",
                            "name": "Original review",
                            "start": "2026-10-01T09:00:00+00:00",
                            "end": "2026-10-01T10:00:00+00:00",
                            "participant_ids": ["alice"],
                        }
                    ],
                    "free": [
                        {
                            "start": "2026-10-01T10:00:00+00:00",
                            "end": "2026-10-01T12:00:00+00:00",
                        }
                    ],
                }
            ],
        }
        assert (
            json.loads(path.read_text(encoding="utf-8"))["previous"]
            == persisted["previous"]
        )


@pytest.mark.parametrize("stability", [False, True])
def test_schedule_option_passes_previous_schedule(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    stability: bool,
) -> None:
    """Pass the saved schedule and whether stability is enabled."""
    path: Path = initialized(tmp_path, capsys)
    state: State = load_state(path)
    previous: Schedule = Schedule((), ())
    save_state(path, state.model_copy(update={"previous": to_schedule_state(previous)}))
    summary_value: ScheduleSummary = ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)
    solver: MagicMock = MagicMock(
        return_value=OptimalSolution(previous, summary_value)
    )
    monkeypatch.setattr(MathOptSchedulingSolver, "solve", solver)
    arguments: tuple[str, ...] = () if stability else ("--no-stability",)
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(path), "schedule", *arguments)
    assert status == 0
    assert output == {"status": "optimal", "summary": schedule_summary(), "items": []}
    solver.assert_called_once_with(
        to_problem(state.problem), DEFAULT_POLICY, previous, stability
    )
    assert to_schedule(load_state(path).previous) == previous


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
    existing: HardConstraint = HardConstraint(
        ConstraintId("c"),
        TimeBoundCondition(
            frozenset({TaskId("a")}),
            Boundary.END,
            TimeBoundRelation.AT_OR_BEFORE,
            START + 6 * HOUR,
        ),
    )
    original: SchedulingProblem = replace(
        problem(task("a"), task("b")),
        constraints=(existing,) if kind == "replace_constraint" else (),
    )
    path: Path = tmp_path / "state.json"
    save_problem(path, original)
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


def test_command_line_add_schedule_replace_schedule(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Solve a batch with given identifiers and solve after replacement."""
    path: Path = tmp_path / "state.json"
    save_problem(path, problem())
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
    assert main(["--state", str(path), "schedule"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["items"] == [
        {
            "status": "scheduled",
            "task_id": "a",
            "name": "Review",
            "start": START.isoformat(),
            "end": (START + timedelta(hours=1)).isoformat(),
            "participant_ids": [],
        }
    ]
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
    assert main(["--state", str(path), "schedule"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["items"] == [
        {
            "status": "scheduled",
            "task_id": "a",
            "name": "Review",
            "start": (START + timedelta(hours=2)).isoformat(),
            "end": (START + timedelta(hours=3)).isoformat(),
            "participant_ids": [],
        }
    ]


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


@pytest.mark.parametrize("command", ["apply", "query"])
def test_schema_describes_every_model_and_field(
    capsys: pytest.CaptureFixture[str], command: str
) -> None:
    """Describe every model and field of the printed input schema."""
    assert main(["schema", command]) == 0
    schema: dict[str, Any] = json.loads(capsys.readouterr().out)
    models: dict[str, dict[str, Any]] = dict(schema["$defs"])
    if "properties" in schema:
        models[schema["title"]] = schema
    missing: list[str] = [
        name for name, model in models.items() if not model.get("description")
    ] + [
        f"{name}.{field}"
        for name, model in models.items()
        for field, value in model.get("properties", {}).items()
        if not value.get("description")
    ]
    assert missing == []
    assert "JSON form of" not in json.dumps(schema)


def test_apply_fixed_task_persists_and_reports_generated_identifier(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Persist an added appointment and return its generated identifier."""
    state: Path = initialized(tmp_path, capsys)
    commands: Path = tmp_path / "commands.json"
    commands.write_text(
        json.dumps({"commands": [{"kind": "add_task", "task": fixed_task_input()}]}),
        encoding="utf-8",
    )
    status: int
    output: dict[str, object]
    status, output = invoke(
        capsys, "apply", "--state", str(state), "--file", str(commands)
    )
    assert status == 0
    stored: SchedulingProblem = to_problem(load_state(state).problem)
    fixed: FixedTask = stored.fixed_tasks[0]
    assert output == {
        "executed": [
            {"kind": "add_task", "task_id": fixed.id.value, "name": fixed.name}
        ]
    }
    assert fixed.start == START + timedelta(minutes=90)
    assert fixed.duration == timedelta(minutes=90)
    assert stored.tasks == ()


@pytest.mark.parametrize("command", ["init", "apply"])
@pytest.mark.parametrize("duration", ["-PT1M", "PT0S"])
def test_fixed_duration_uses_consistency_channel(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    command: str,
    duration: str,
) -> None:
    """Reject negative durations and accept zero through both commands."""
    fixed: dict[str, object] = {
        **fixed_task_input(),
        "id": "fixed",
        "duration": duration,
    }
    input_path: Path = tmp_path / "input.json"
    state_path: Path
    arguments: list[str]
    if command == "init":
        state_path = tmp_path / "state.json"
        input_path.write_text(
            json.dumps({**calendar_data(), "fixed_tasks": [fixed]}), encoding="utf-8"
        )
        arguments = ["init", "--calendar", str(input_path)]
    else:
        state_path = initialized(tmp_path, capsys)
        input_path.write_text(
            json.dumps({"commands": [{"kind": "add_task", "task": fixed}]}),
            encoding="utf-8",
        )
        arguments = ["apply", "--file", str(input_path)]
    before: bytes | None = state_path.read_bytes() if state_path.exists() else None
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(state_path), *arguments)
    if duration == "PT0S":
        assert status == 0
        return
    assert status == 1
    assert output == {"rejected": ["Fixed task fixed duration must not be negative."]}
    assert (state_path.read_bytes() if state_path.exists() else None) == before


@pytest.mark.parametrize("task_ids", [["first", "second"], [], ["first", "missing"]])
def test_apply_and_query_multi_task_constraint(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    task_ids: list[str],
) -> None:
    """Persist one constraint or reject invalid targets atomically."""
    path: Path = tmp_path / "state.json"
    save_problem(
        path, problem(task("first", SLOT), task("second", 2 * SLOT), hours=3, slot=SLOT)
    )
    before: bytes = path.read_bytes()
    command: dict[str, object] = {
        "kind": "add_constraint",
        "constraint": {
            "requirement": {"kind": "soft", "strength": "strong"},
            "condition": {
                "kind": "time_window",
                "task_ids": task_ids,
                "relation": "avoid",
                "windows": [{"time_range": {"start": "09:00", "end": "10:00"}}],
            },
        },
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"commands": [command]})))
    status: int = main(["--state", str(path), "apply"])
    output: dict[str, object] = json.loads(capsys.readouterr().out)
    if not task_ids or "missing" in task_ids:
        assert status == 1
        assert (
            "at least" if not task_ids else "missing task id missing"
        ) in json.dumps(output)
        assert path.read_bytes() == before
        return
    assert status == 0
    executed: object = output["executed"]
    assert isinstance(executed, list) and len(executed) == 1
    record: object = executed[0]
    assert isinstance(record, dict)
    constraint_id: object = record["constraint_id"]
    assert isinstance(constraint_id, str)
    stored: SchedulingProblem = to_problem(load_state(path).problem)
    assert len(stored.constraints) == 1
    assert stored.constraints[0].id.value == constraint_id
    assert stored.constraints[0].condition.task_ids == frozenset(
        TaskId(value) for value in task_ids
    )
    selected: list[str]
    for selected in (["first"], ["second"], ["first", "second"]):
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO(
                json.dumps({"kind": "constraints", "filter": {"task_ids": selected}})
            ),
        )
        assert main(["--state", str(path), "query"]) == 0
        output = json.loads(capsys.readouterr().out)
        items: object = output["items"]
        assert output["total"] == 1
        assert isinstance(items, list) and len(items) == 1
        record = items[0]
        assert isinstance(record, dict)
        assert record["id"] == constraint_id
        assert record["condition"] == {
            "kind": "time_window",
            "task_ids": ["first", "second"],
            "relation": "avoid",
            "windows": [
                {
                    "date_range": "horizon",
                    "weekdays": WEEKDAY_NAMES,
                    "time_range": {"start": "09:00", "end": "10:00"},
                }
            ],
        }


@pytest.mark.parametrize("stability", [False, True])
def test_command_line_summary_and_queries_use_saved_solution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], stability: bool
) -> None:
    """Expose measured summaries and queries through the command line."""
    item: Task = replace(task("task"), participant_ids=frozenset({PERSON}))
    previous: Schedule = Schedule(
        (
            ScheduledTask(
                item.id, item.name, START, START + item.duration, item.participant_ids
            ),
        ),
        (),
    )
    condition: TimeBoundCondition = TimeBoundCondition(
        frozenset({item.id}), Boundary.START, TimeBoundRelation.AT, START + 3 * HOUR
    )
    value: SchedulingProblem = replace(
        problem(item, hours=4, people=(PERSON,)),
        constraints=(HardConstraint(ConstraintId("moved"), condition),),
    )
    path: Path = tmp_path / "state.json"
    save_problem(path, value, previous)
    arguments: list[str] = ["--state", str(path), "schedule"] + (
        [] if stability else ["--no-stability"]
    )
    assert main(arguments) == 0
    output: dict[str, object] = json.loads(capsys.readouterr().out)
    assert output["summary"] == {
        "total_cost": 15 / 7 if stability else 0.0,
        "costs": {
            "dropped_tasks": 0.0,
            "soft_constraints": 0.0,
            "stability": 15 / 7 if stability else 0.0,
        },
        "counts": {
            "scheduled_tasks": 1,
            "dropped_tasks": 0,
            "violated_soft_constraints": 0,
            "moved_tasks": 1,
        },
    }
    saved: State = load_state(path)
    assert saved.previous is not None
    assert saved.previous.model_dump(mode="json") == {"items": output["items"]}
    query_path: Path = tmp_path / "query.json"
    query_path.write_text('{"kind":"evaluation","filter":{"violated_only":true}}')
    assert main(["--state", str(path), "query", "--file", str(query_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "kind": "evaluation",
        "has_previous": True,
        "items": [],
        "total": 0,
        "truncated": False,
    }
    query_path.write_text('{"kind":"objective_policy"}')
    assert main(["--state", str(path), "query", "--file", str(query_path)]) == 0
    assert json.loads(capsys.readouterr().out)["per_count"] == DEFAULT_POLICY.per_count


def test_no_feasible_schedule_reports_conflicts_and_keeps_previous(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Report relaxed conflicts with exit status 2 without changing the state."""
    path: Path = initialized(tmp_path, capsys)
    review: dict[str, object] = {
        "id": "review",
        "name": "Review",
        "duration": "PT1H",
        "participant_ids": ["alice"],
        "importance": "low",
        "required": True,
        "stability": "normal",
    }
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(json.dumps({"commands": [{"kind": "add_task", "task": review}]})),
    )
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 0
    status, output = invoke(capsys, "--state", str(path), "schedule")
    assert status == 0
    commands: list[dict[str, object]] = [
        {
            "kind": "add_task",
            "task": {**review, "id": "long", "name": "Long", "duration": "PT4H"},
        },
        constraint_command(
            {
                "kind": "time_bound",
                "task_ids": ["review"],
                "boundary": "start",
                "relation": "at_or_after",
                "at": "2026-10-01T12:00:00+00:00",
            },
            "late",
        ),
        constraint_command(
            {
                "kind": "time_bound",
                "task_ids": ["review"],
                "boundary": "end",
                "relation": "at_or_before",
                "at": "2026-10-01T12:00:00+00:00",
            },
            "early",
        ),
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"commands": commands})))
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 0
    before: str = path.read_text(encoding="utf-8")
    status, output = invoke(capsys, "--state", str(path), "schedule")
    assert status == 2
    hour: dict[str, object] = {"amount": 1.0, "unit": "hours"}
    assert output == {
        "status": "no_feasible_solution",
        "conflicts": {
            "status": "found",
            "constraints": [
                {
                    "constraint_id": "late",
                    "label": "Entered request",
                    "requirement": {"kind": "hard"},
                    "violation": hour,
                    "breakdown": [{"task_id": "review", "violation": hour}],
                    "related_constraint_ids": ["early"],
                }
            ],
            "dropped_required_tasks": [
                {"task_id": "long", "name": "Long", "reason": "no_free_start"}
            ],
        },
    }
    assert path.read_text(encoding="utf-8") == before


@pytest.mark.parametrize("reason", ["time_limit", "numerical_error"])
@pytest.mark.parametrize("command", ["schedule", "chat"])
def test_no_feasible_solution_without_relaxed_schedule_prints_reason(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    reason: str,
) -> None:
    """Exit 2 with conflicts not found and the reason the relaxed solve gave."""
    path: Path = initialized(tmp_path, capsys)
    before: State = load_state(path)
    monkeypatch.setattr(
        MathOptSchedulingSolver,
        "solve",
        MagicMock(return_value=NoFeasibleSolution(ConflictsNotFound(reason))),
    )
    translator: MagicMock = MagicMock()
    translator.translate.return_value = ScheduleStep(True)
    monkeypatch.setattr(
        main_module, "OpenAIStepTranslator", MagicMock(return_value=translator)
    )
    monkeypatch.setattr(openai, "OpenAI", MagicMock())
    arguments: tuple[str, ...] = (
        ("chat", "Schedule it", "--model", "demo-model")
        if command == "chat"
        else ("schedule",)
    )
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(path), *arguments)
    assert status == 2
    assert output == {
        "status": "no_feasible_solution",
        "conflicts": {"status": "not_found", "reason": reason},
    }
    after: State = load_state(path)
    assert (after.problem, after.previous) == (before.problem, before.previous)
