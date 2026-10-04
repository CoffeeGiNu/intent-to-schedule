"""Tests for the JSON command line adapter."""

from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys

import pytest

from intent_to_schedule.adapter.command_line_interface.main import main
from intent_to_schedule.adapter.command_line_interface.state import State, load_state, save_state, to_problem, to_problem_state
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import FixedTask


def calendar_data() -> dict[str, object]:
    """Build a small valid calendar file."""
    return {
        "horizon": {"start": "2026-10-01T09:00:00+00:00", "end": "2026-10-01T12:00:00+00:00"},
        "slot": "PT1H",
        "people": [{"id": "alice", "name": "Alice"}],
        "availabilities": [{
            "person_id": "alice",
            "intervals": [{"start": "2026-10-01T09:00:00+00:00", "end": "2026-10-01T12:00:00+00:00"}],
        }],
        "fixed_tasks": [],
    }


def invoke(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, object]]:
    """Call the entry point and parse its single JSON output."""
    status: int = main(list(argv))
    output: str = capsys.readouterr().out
    return status, json.loads(output)


def initialized(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> Path:
    """Create a state file from a calendar."""
    calendar_path: Path = tmp_path / "calendar.json"
    state_path: Path = tmp_path / "nested" / "state.json"
    calendar_path.write_text(json.dumps(calendar_data()), encoding="utf-8")
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(state_path), "init", "--calendar", str(calendar_path))
    assert status == 0
    assert output["previous"] is None
    assert output["problem"]["people"] == [{"id": "alice", "name": "Alice"}]
    assert state_path.exists()
    return state_path


def test_init_show_schema_and_round_trip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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
    assert "commands" in output["properties"]


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
    status, output = invoke(capsys, "--state", str(state_path), "init", "--calendar", str(calendar_path))
    assert status == 1
    assert output == {"rejected": ["Person alice has no availability."]}
    assert not state_path.exists()


def test_init_fixed_tasks_persists_ids_and_round_trips(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Initialize fixed events with generated IDs and preserve their exact times."""
    calendar: dict[str, object] = calendar_data()
    calendar["fixed_tasks"] = [{"name": "Existing review", "start": "2026-10-01T09:15:00+00:00", "duration": "PT30M", "participant_ids": ["alice"]}]
    calendar_path: Path = tmp_path / "calendar.json"
    path: Path = tmp_path / "state.json"
    calendar_path.write_text(json.dumps(calendar), encoding="utf-8")
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "init", "--state", str(path), "--calendar", str(calendar_path))
    assert status == 0
    assert output["problem"]["fixed_tasks"][0]["id"]
    assert set(output["problem"]["calendar"]) == {"horizon", "slot", "availabilities"}
    state: State = load_state(path)
    problem: SchedulingProblem = to_problem(state.problem)
    fixed: FixedTask = problem.fixed_tasks[0]
    assert fixed.start == datetime(2026, 10, 1, 9, 15, tzinfo=timezone.utc)
    assert fixed.duration == timedelta(minutes=30)
    assert to_problem(to_problem_state(problem)) == problem
    status, output = invoke(capsys, "solve", "--state", str(path))
    assert status == 0
    assert output == {"scheduled": [], "dropped": []}


def test_init_rejects_fixed_task_with_unknown_participant(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Reject fixed events that reference a missing person."""
    calendar: dict[str, object] = calendar_data()
    calendar["fixed_tasks"] = [{"name": "Existing review", "start": "2026-10-01T09:15:00+00:00", "duration": "PT30M", "participant_ids": ["missing"]}]
    calendar_path: Path = tmp_path / "calendar.json"
    path: Path = tmp_path / "state.json"
    calendar_path.write_text(json.dumps(calendar), encoding="utf-8")
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "init", "--state", str(path), "--calendar", str(calendar_path))
    assert status == 1
    assert "missing person id missing" in output["rejected"][0]
    assert not path.exists()


def test_apply_reject_and_solve(tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    """Apply commands, preserve rejected state, and keep the second solve stable."""
    path: Path = initialized(tmp_path, capsys)
    task_file: Path = tmp_path / "task.json"
    task_file.write_text(json.dumps({"commands": [{"kind": "add_task", "task": {
        "name": "Review", "duration": "PT1H", "participant_ids": ["alice"],
        "importance": "high", "required": True, "stability": "normal",
    }}]}), encoding="utf-8")
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(path), "apply", "--file", str(task_file))
    assert status == 0
    task_id: str = output["executed"][0]["task_id"]
    assert output["executed"][0]["name"] == "Review"

    constraint_input: dict[str, object] = {"commands": [{"kind": "add_constraint", "constraint": {
        "kind": "hard",
        "measure": {"kind": "point", "task_id": task_id},
        "evaluation": {"kind": "distance", "target": {
            "kind": "instant", "value": datetime(2026, 10, 1, 9, tzinfo=timezone.utc).isoformat(),
        }},
    }}]}
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(constraint_input)))
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 0
    assert output["executed"][0]["constraint_id"]
    assert len(load_state(path).problem.constraints) == 1
    current: State = load_state(path)
    domain_problem: SchedulingProblem = to_problem(current.problem)
    save_state(path, current.model_copy(update={"problem": to_problem_state(domain_problem)}))
    assert to_problem(load_state(path).problem) == domain_problem

    unchanged: str = path.read_text(encoding="utf-8")
    rejected_file: Path = tmp_path / "rejected.json"
    rejected_file.write_text(json.dumps({"commands": [{"kind": "remove_task", "task_id": "missing"}]}), encoding="utf-8")
    status, output = invoke(capsys, "--state", str(path), "apply", "--file", str(rejected_file))
    assert status == 1
    assert output == {"rejected": ["Task missing does not exist"]}
    assert path.read_text(encoding="utf-8") == unchanged

    status, output = invoke(capsys, "--state", str(path), "solve")
    assert status == 0
    assert output == {"scheduled": [{
        "task_id": task_id, "name": "Review", "start": "2026-10-01T09:00:00+00:00",
        "end": "2026-10-01T10:00:00+00:00",
    }], "dropped": []}
    assert load_state(path).previous is not None
    second: dict[str, object]
    status, second = invoke(capsys, "--state", str(path), "solve")
    assert status == 0
    assert second == output


@pytest.mark.parametrize(
    "requirement,expected_strength",
    [
        ({"kind": "hard"}, None),
        ({"kind": "soft", "strength": "weak"}, "weak"),
        ({"kind": "soft", "strength": "normal"}, "normal"),
        ({"kind": "soft", "strength": "strong"}, "strong"),
    ],
)
def test_add_time_constraint_conversion_generates_id_once(
    requirement: dict[str, str],
    expected_strength: str | None,
) -> None:
    from calendar import Day
    from datetime import date, time
    from unittest.mock import Mock, patch

    from intent_to_schedule.adapter.data_model import (
        AddTimeConstraintData,
        command_record,
        convert_command,
    )
    from intent_to_schedule.application.command import (
        AddTimeConstraint,
        SchedulingCommand,
    )
    from intent_to_schedule.application.time_windows import (
        DateRange,
        Expansion,
        TimeRange,
        TimeRelation,
        TimeWindow,
    )
    from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
    from intent_to_schedule.domain.constraint import ConstraintId
    from intent_to_schedule.domain.strength import Strength
    from intent_to_schedule.domain.task import TaskId

    data: AddTimeConstraintData = AddTimeConstraintData.model_validate(
        {
            "kind": "add_time_constraint",
            "task_ids": ["review"],
            "relation": "within",
            "windows": [
                {
                    "date_range": {"start": "2026-10-12", "end": "2026-10-17"},
                    "weekdays": ["friday", "friday"],
                    "time_range": {"start": "13:00", "end": None},
                }
            ],
            "requirement": requirement,
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
        assert isinstance(command, AddTimeConstraint)
        assert command.constraint_id == ConstraintId("created")
        assert command.task_ids == frozenset({TaskId("review")})
        assert command.relation is TimeRelation.WITHIN
        assert command.strength == (
            Strength(expected_strength) if expected_strength else None
        )
        assert command.windows == (
            TimeWindow(
                DateRange(date(2026, 10, 12), date(2026, 10, 17)),
                frozenset({Day.FRIDAY}),
                TimeRange(time(13), None),
            ),
        )
        with patch(
            "intent_to_schedule.adapter.data_model.expand",
            return_value=Expansion((grid.horizon,), False),
        ):
            assert command_record(command, grid) == {
                "kind": "add_time_constraint",
                "constraint_id": "created",
            }
        with patch(
            "intent_to_schedule.application.command.expand",
            return_value=Expansion((grid.horizon,), False),
        ):
            command.execute(SchedulingProblem(Calendar(grid, ()), (), (), (), ()))
    generation.assert_called_once_with()


def test_add_time_constraint_conversion_preserves_omitted_and_empty_window_fields() -> (
    None
):
    from intent_to_schedule.adapter.data_model import (
        TimeWindowData,
        convert_time_window,
    )
    from intent_to_schedule.application.time_windows import TimeWindow

    assert convert_time_window(TimeWindowData()) == TimeWindow(None, None, None)
    assert convert_time_window(TimeWindowData(weekdays=())) == TimeWindow(
        None, frozenset(), None
    )


@pytest.mark.parametrize("end", ["13:00", "12:00"])
def test_add_time_constraint_input_rejects_nonincreasing_times(end: str) -> None:
    from pydantic import ValidationError

    from intent_to_schedule.adapter.data_model import AddTimeConstraintData

    with pytest.raises(ValidationError) as error:
        AddTimeConstraintData.model_validate(
            {
                "kind": "add_time_constraint",
                "task_ids": ["review"],
                "relation": "within",
                "windows": [{"time_range": {"start": "13:00", "end": end}}],
                "requirement": {"kind": "hard"},
            }
        )
    message: str = str(error.value)
    assert "windows" in message and "time_range" in message
    assert "13:00" in message and end in message and "after" in message


@pytest.mark.parametrize("start,end", [("13:00+09:00", None), ("13:00", "18:00+09:00")])
def test_add_time_constraint_input_rejects_zoned_times(
    start: str, end: str | None
) -> None:
    from pydantic import ValidationError

    from intent_to_schedule.adapter.data_model import AddTimeConstraintData

    with pytest.raises(ValidationError) as error:
        AddTimeConstraintData.model_validate(
            {
                "kind": "add_time_constraint",
                "task_ids": ["review"],
                "relation": "avoid",
                "windows": [{"time_range": {"start": start, "end": end}}],
                "requirement": {"kind": "soft", "strength": "normal"},
            }
        )
    message: str = str(error.value)
    assert "time_range" in message and "+09:00" in message
    assert "without a time zone" in message


@pytest.mark.parametrize("rounded", [False, True])
@pytest.mark.parametrize("relation_value", ["within", "avoid"])
def test_add_time_constraint_record_has_id_and_only_changed_rounding_note(
    rounded: bool, relation_value: str
) -> None:
    from unittest.mock import Mock, patch

    from intent_to_schedule.adapter.data_model import command_record
    from intent_to_schedule.application.command import AddTimeConstraint
    from intent_to_schedule.application.time_windows import (
        Expansion,
        TimeRelation,
        TimeWindow,
    )
    from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
    from intent_to_schedule.domain.constraint import ConstraintId
    from intent_to_schedule.domain.task import TaskId

    grid: TimeGrid = TimeGrid(
        TimeInterval(datetime(2026, 10, 1, 9), datetime(2026, 10, 1, 12)),
        timedelta(hours=1),
    )
    command: AddTimeConstraint = AddTimeConstraint(
        ConstraintId("created"),
        frozenset({TaskId("review")}),
        TimeRelation(relation_value),
        (TimeWindow(None, None, None),),
        None,
    )
    expansion: Mock
    with patch(
        "intent_to_schedule.adapter.data_model.expand",
        return_value=Expansion((grid.horizon,), rounded),
    ) as expansion:
        record: dict[str, str] = command_record(command, grid)
    expansion.assert_called_once_with(command.windows, command.relation, grid)
    assert record["constraint_id"] == "created"
    assert record["kind"] == "add_time_constraint"
    assert set(record) == (
        {"kind", "constraint_id", "note"} if rounded else {"kind", "constraint_id"}
    )
    if rounded:
        assert "round" in record["note"].lower()
        assert ("inward" if relation_value == "within" else "outward") in record[
            "note"
        ].lower()


@pytest.mark.parametrize("relation_value", ["within", "avoid"])
def test_apply_add_time_constraint_persists_region_and_reports_rounding(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    relation_value: str,
) -> None:
    from intent_to_schedule.adapter.data_model import (
        HardConstraintData,
        SoftConstraintData,
    )

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
    task_id: str = output["executed"][0]["task_id"]
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "commands": [
                        {
                            "kind": "add_time_constraint",
                            "task_ids": [task_id],
                            "relation": relation_value,
                            "windows": [
                                {"time_range": {"start": "09:10", "end": "11:10"}}
                            ],
                            "requirement": {"kind": "soft", "strength": "strong"},
                        }
                    ]
                }
            )
        ),
    )
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 0
    record: dict[str, str] = output["executed"][0]
    assert set(record) == {"kind", "constraint_id", "note"}
    state: State = load_state(path)
    constraint: HardConstraintData | SoftConstraintData = state.problem.constraints[0]
    assert constraint.id.value == record["constraint_id"]
    assert constraint.kind == "soft" and constraint.strength == "strong"
    assert {value.value for value in constraint.measure.task_ids} == {task_id}
    assert not state.problem.tasks[0].required
    expected: list[dict[str, str]] = (
        [
            {"start": "2026-10-01T09:00:00Z", "end": "2026-10-01T10:00:00Z"},
            {"start": "2026-10-01T11:00:00Z", "end": "2026-10-01T12:00:00Z"},
        ]
        if relation_value == "within"
        else [
            {"start": "2026-10-01T09:00:00Z", "end": "2026-10-01T12:00:00Z"},
        ]
    )
    assert constraint.evaluation.model_dump(mode="json")["region"] == expected


def test_apply_add_time_constraint_rejects_empty_expansion_without_saving(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path: Path = initialized(tmp_path, capsys)
    unchanged: str = path.read_text(encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "stdin",
        io.StringIO(
            json.dumps(
                {
                    "commands": [
                        {
                            "kind": "add_time_constraint",
                            "task_ids": ["review"],
                            "relation": "within",
                            "windows": [],
                            "requirement": {"kind": "hard"},
                        }
                    ]
                }
            )
        ),
    )
    status: int
    output: dict[str, object]
    status, output = invoke(capsys, "--state", str(path), "apply")
    assert status == 1
    assert "windows" in output["rejected"][0]
    assert "review" in output["rejected"][0]
    assert path.read_text(encoding="utf-8") == unchanged


def query_state(tmp_path: Path) -> Path:
    """Create query state without applying commands or solving."""
    from intent_to_schedule.adapter.command_line_interface.state import (
        CalendarInput,
        ProblemState,
    )

    calendar: CalendarInput = CalendarInput.model_validate(calendar_data())
    path: Path = tmp_path / "query-state.json"
    save_state(
        path,
        State(
            problem=ProblemState(
                calendar=calendar,
                people=calendar.people,
                tasks=(),
                fixed_tasks=(),
                constraints=(),
            ),
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


@pytest.mark.parametrize("value", [{"kind": "people", "limit": 0}, {"kind": "tasks", "limit": 101}, {"kind": "people", "select": "one", "filter": {"person_ids": ["missing"]}}, {"kind": "available_starts", "participant_ids": ["missing"], "duration": "PT1H"}])
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
    assert output["discriminator"]["propertyName"] == "kind"
    assert set(output["discriminator"]["mapping"]) == {
        "summary",
        "people",
        "tasks",
        "constraints",
        "previous_schedule",
        "available_starts",
    }


@pytest.mark.parametrize("outcome", ["message", "exhausted", "solved", "infeasible"])
@pytest.mark.parametrize("explicit_now", [False, True])
def test_chat_persists_only_solve_changes_and_uses_clock(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    outcome: str,
    explicit_now: bool,
) -> None:
    """Persist dialogue for every turn and problem changes only after solve."""
    from typing import cast
    from unittest.mock import MagicMock

    import openai
    import intent_to_schedule.adapter.command_line_interface.main as main_module
    import intent_to_schedule.application.converse as converse
    from intent_to_schedule.adapter.command_line_interface.main import chat
    from intent_to_schedule.adapter.command_line_interface.state import (
        to_schedule_state,
        UtteranceState,
    )
    from intent_to_schedule.application.command import AddTask
    from intent_to_schedule.application.schedule import Scheduling
    from intent_to_schedule.application.solve import Infeasible, Solved
    from intent_to_schedule.application.translate import (
        ApplyStep,
        MessageStep,
        SolveStep,
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
    from intent_to_schedule.domain.consistency import AllOf
    from intent_to_schedule.domain.person import Person, PersonId
    from intent_to_schedule.domain.schedule import Schedule, ScheduledTask
    from intent_to_schedule.domain.strength import Strength
    from intent_to_schedule.domain.task import Importance, Task, TaskId

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
    previous: Schedule = Schedule((ScheduledTask(existing.id, start),), frozenset())
    replacement: Schedule = Schedule(
        (
            ScheduledTask(existing.id, start + timedelta(hours=1)),
            ScheduledTask(added.id, start + timedelta(hours=2)),
        ),
        frozenset(),
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
        MessageStep("When works for you?") if outcome == "message" else SolveStep(True)
    )
    steps: list[Step] = [ApplyStep((AddTask(added),)), terminal]
    if outcome == "exhausted":
        steps = [ApplyStep((AddTask(added),))] + [ApplyStep(())] * (
            converse.STEP_LIMIT - 1
        )
    translator: MagicMock = MagicMock()
    translator.translate.side_effect = steps
    factory: MagicMock = MagicMock(return_value=translator)
    monkeypatch.setattr(main_module, "OpenAIStepTranslator", factory)
    client: MagicMock = MagicMock()
    monkeypatch.setattr(openai, "OpenAI", lambda: client)
    solver: MagicMock = MagicMock()
    solver.solve.return_value = (
        Infeasible() if outcome == "infeasible" else Solved(replacement)
    )
    now: datetime | None = start - timedelta(days=2) if explicit_now else None
    status: int = chat(
        path, "new request", "demo-model", now, Scheduling(solver, AllOf())
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
    if outcome in {"message", "exhausted"}:
        assert persisted.problem == state.problem
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
        assert to_problem(persisted.problem).tasks == (existing, added)
        solver.solve.assert_called_once()
        if outcome == "infeasible":
            assert status == 2
            assert output == {"infeasible": True}
            assert persisted.previous == state.previous
            assert persisted.dialogue[-1].text == "Infeasible."
        else:
            assert status == 0
            assert persisted.previous == to_schedule_state(replacement)
            assert len(cast(list[object], output["scheduled"])) == 2
            assert persisted.dialogue[-1].text == "Scheduled."
