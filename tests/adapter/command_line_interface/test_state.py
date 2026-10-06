from calendar import Day
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from intent_to_schedule.adapter.command_line_interface.state import (
    CalendarInput,
    ScheduleState,
    State,
    load_state,
    save_state,
    to_problem,
    to_problem_state,
    to_schedule,
    to_schedule_state,
)
from intent_to_schedule.adapter.data_model import DataModel
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import (
    Condition,
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
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId
from intent_to_schedule.domain.time_windows import (
    TimeRange,
    TimeRelation,
    TimeWindow,
    WholeHorizon,
)

START: datetime = datetime(2026, 10, 29, 9, tzinfo=timezone(timedelta(hours=9)))


def test_state_round_trip_every_condition(tmp_path: Path) -> None:
    """Preserve every entered condition in a state file."""
    tasks: tuple[Task, ...] = tuple(
        Task(TaskId(name), name, timedelta(hours=1), frozenset(), Importance.LOW, True)
        for name in ("a", "b")
    )
    task_ids: frozenset[TaskId] = frozenset(task.id for task in tasks)
    conditions: tuple[Condition, ...] = (
        TimeWindowCondition(
            task_ids,
            TimeRelation.WITHIN,
            (TimeWindow(WholeHorizon(), frozenset(Day), TimeRange(timedelta(hours=9, minutes=10), timedelta(hours=15, minutes=20))),),
        ),
        TimeBoundCondition(
            task_ids,
            Boundary.END,
            TimeBoundRelation.AT_OR_BEFORE,
            START + timedelta(hours=6),
        ),
        TaskGapCondition(
            TaskId("a"), TaskId("b"), TaskGapRelation.AT_LEAST, timedelta(0)
        ),
        DailyLimitCondition(task_ids, AggregateQuantity.COUNT, 2),
        DailyLimitCondition(
            task_ids, AggregateQuantity.TOTAL_DURATION, timedelta(hours=4)
        ),
    )
    constraints: tuple[Constraint, ...] = tuple(
        SoftConstraint(
            ConstraintId(str(index)), condition, Strength.STRONG, "Entered request"
        )
        if index % 2
        else HardConstraint(ConstraintId(str(index)), condition, "Entered request")
        for index, condition in enumerate(conditions)
    )
    original: SchedulingProblem = SchedulingProblem(
        Calendar(
            TimeGrid(
                TimeInterval(START, START + timedelta(hours=8)), timedelta(hours=1)
            ),
            (),
        ),
        (),
        tasks,
        (),
        constraints,
    )
    path: Path = tmp_path / "state.json"
    save_state(
        path, State(problem=to_problem_state(original), previous=None, dialogue=())
    )
    assert to_problem(load_state(path).problem) == original
    stored: str = path.read_text()
    assert "measure" not in stored
    assert "09:10:00" in stored


@pytest.mark.parametrize("model", [CalendarInput, ScheduleState])
def test_state_json_datetimes_require_utc_offsets(model: type[DataModel]) -> None:
    """Reject offset-free datetimes in calendar and schedule state."""
    data: dict[str, object] = (
        {
            "horizon": {"start": "2026-10-05T09:00:00", "end": "2026-10-05T17:00:00Z"},
            "slot": "PT1H",
            "availabilities": [],
            "people": [],
            "fixed_tasks": [],
        }
        if model is CalendarInput
        else {
            "items": [
                {
                    "status": "scheduled",
                    "task_id": "task",
                    "name": "Task",
                    "start": "2026-10-05T09:00:00Z",
                    "end": "2026-10-05T17:00:00",
                    "participant_ids": [],
                }
            ]
        }
    )
    with pytest.raises(ValidationError, match="timezone"):
        model.model_validate_json(json.dumps(data))


def test_schedule_state_keeps_participants_and_requires_them() -> None:
    """Round-trip recorded participants and reject scheduled entries without them."""
    end: datetime = START + timedelta(hours=1)
    schedule: Schedule = Schedule(
        (
            ScheduledTask(
                TaskId("a"), "A", START, end, frozenset({PersonId("bob"), PersonId("alice")})
            ),
        ),
        (),
    )
    form: ScheduleState | None = to_schedule_state(schedule)
    assert form is not None
    assert form.model_dump(mode="json")["items"][0]["participant_ids"] == ["alice", "bob"]
    assert to_schedule(ScheduleState.model_validate_json(form.model_dump_json())) == schedule
    entry: dict[str, object] = {
        "status": "scheduled",
        "task_id": "a",
        "name": "A",
        "start": START.isoformat(),
        "end": end.isoformat(),
    }
    with pytest.raises(ValidationError, match="participant_ids"):
        ScheduleState.model_validate_json(json.dumps({"items": [entry]}))
    assert to_schedule(
        ScheduleState.model_validate({"items": [{**entry, "participant_ids": []}]})
    ) == Schedule((ScheduledTask(TaskId("a"), "A", START, end, frozenset()),), ())
