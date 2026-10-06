"""Tests for local state and dialogue stores."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from intent_to_schedule.adapter.local.state import (
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
from intent_to_schedule.application.translate import Speaker, Utterance
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule, ScheduledTask
from intent_to_schedule.domain.task import Importance, Task, TaskId


START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)


def make_problem(task_value: str | None = None) -> SchedulingProblem:
    """Build a problem with an optional required task."""
    tasks: tuple[Task, ...] = (
        (
            Task(
                TaskId(task_value),
                task_value,
                timedelta(hours=1),
                frozenset(),
                Importance.LOW,
                True,
            ),
        )
        if task_value is not None
        else ()
    )
    return SchedulingProblem(
        Calendar(
            TimeGrid(
                TimeInterval(START, START + timedelta(hours=4)),
                timedelta(hours=1),
            ),
            (),
        ),
        (),
        tasks,
        (),
        (),
    )


def make_schedule(identifier: str = "task") -> Schedule:
    """Build a previous schedule with one scheduled entry."""
    task_id: TaskId = TaskId(identifier)
    return Schedule(
        (
            ScheduledTask(
                task_id,
                identifier,
                START,
                START + timedelta(hours=1),
                frozenset({PersonId("alice")}),
            ),
        ),
        (),
    )


def seeded_state(path: Path) -> State:
    """Write a state containing all three persisted parts."""
    state: State = State(
        problem=to_problem_state(make_problem("old")),
        previous=to_schedule_state(make_schedule()),
        dialogue=(UtteranceState(speaker="user", text="old dialogue"),),
    )
    save_state(path, state)
    return state


def test_saving_problem_preserves_previous_and_dialogue(tmp_path: Path) -> None:
    """Replace the problem without changing the other stored parts."""
    path: Path = tmp_path / "state.json"
    original: State = seeded_state(path)
    store: LocalStateStore = LocalStateStore(path)
    replacement: SchedulingProblem = make_problem("new")

    store.save_problem(replacement)

    updated: State = load_state(path)
    assert to_problem(updated.problem) == replacement
    assert updated.previous == original.previous
    assert updated.dialogue == original.dialogue


def test_saving_previous_preserves_problem_and_dialogue(tmp_path: Path) -> None:
    """Replace the previous schedule without changing other stored parts."""
    path: Path = tmp_path / "state.json"
    original: State = seeded_state(path)
    store: LocalStateStore = LocalStateStore(path)
    replacement: Schedule = make_schedule("new")

    store.save_previous(replacement)

    updated: State = load_state(path)
    assert updated.problem == original.problem
    assert to_schedule(updated.previous) == replacement
    assert updated.dialogue == original.dialogue


def test_saving_dialogue_preserves_problem_and_previous(tmp_path: Path) -> None:
    """Replace the dialogue without changing problem or schedule state."""
    path: Path = tmp_path / "state.json"
    original: State = seeded_state(path)
    store: LocalDialogueStore = LocalDialogueStore(path)
    replacement: tuple[Utterance, ...] = (
        Utterance(Speaker.USER, "new request"),
        Utterance(Speaker.ASSISTANT, "new answer"),
    )

    store.save(replacement)

    updated: State = load_state(path)
    assert updated.problem == original.problem
    assert updated.previous == original.previous
    assert store.load() == replacement


def test_save_problem_creates_missing_state_file(tmp_path: Path) -> None:
    """Create a state file with empty schedule and dialogue parts."""
    path: Path = tmp_path / "nested" / "state.json"
    problem: SchedulingProblem = make_problem()
    store: LocalStateStore = LocalStateStore(path)

    store.save_problem(problem)

    assert path.exists()
    state: State = load_state(path)
    assert to_problem(state.problem) == problem
    assert state.previous is None
    assert state.dialogue == ()


def test_load_todays_state_file_shape(tmp_path: Path) -> None:
    """Load a file using the existing problem, previous, and dialogue fields."""
    path: Path = tmp_path / "state.json"
    state_data: dict[str, object] = {
        "problem": {
            "calendar": {
                "horizon": {
                    "start": "2026-10-01T09:00:00+00:00",
                    "end": "2026-10-01T13:00:00+00:00",
                },
                "slot": "PT1H",
                "availabilities": [],
            },
            "people": [],
            "tasks": [],
            "fixed_tasks": [],
            "constraints": [],
        },
        "previous": {
            "items": [
                {
                    "status": "scheduled",
                    "task_id": "task",
                    "name": "task",
                    "start": "2026-10-01T09:00:00+00:00",
                    "end": "2026-10-01T10:00:00+00:00",
                    "participant_ids": [],
                }
            ]
        },
        "dialogue": [{"speaker": "user", "text": "old request"}],
    }
    path.write_text(json.dumps(state_data), encoding="utf-8")
    state_store: LocalStateStore = LocalStateStore(path)
    dialogue_store: LocalDialogueStore = LocalDialogueStore(path)

    assert state_store.load_problem() == make_problem()
    assert state_store.load_previous() == make_schedule_with_no_participants()
    assert dialogue_store.load() == (Utterance(Speaker.USER, "old request"),)


def make_schedule_with_no_participants() -> Schedule:
    """Build the schedule encoded by the compatibility fixture."""
    return Schedule(
        (
            ScheduledTask(
                TaskId("task"),
                "task",
                START,
                START + timedelta(hours=1),
                frozenset(),
            ),
        ),
        (),
    )
