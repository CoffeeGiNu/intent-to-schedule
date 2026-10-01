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
    status, second = invoke(capsys, "--state", str(path), "solve")
    assert status == 0
    assert second == output
