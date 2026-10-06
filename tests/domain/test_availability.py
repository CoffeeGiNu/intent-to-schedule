from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from intent_to_schedule.domain.availability import available_start_slots, free_slots
from intent_to_schedule.domain.calendar import (
    Availability,
    Calendar,
    TimeGrid,
    TimeInterval,
)
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId

START: datetime = datetime(2026, 10, 1, 9, 10, tzinfo=timezone(timedelta(hours=9)))
SLOT: timedelta = timedelta(minutes=30)
GRID: TimeGrid = TimeGrid(TimeInterval(START, START + 4 * SLOT), SLOT)
PERSON: PersonId = PersonId("person")


def interval(first: int, last: int) -> TimeInterval:
    return TimeInterval(
        START + timedelta(minutes=first), START + timedelta(minutes=last)
    )


def problem(
    periods: tuple[TimeInterval, ...] = (GRID.horizon,),
    fixed_tasks: tuple[FixedTask, ...] = (),
) -> SchedulingProblem:
    return SchedulingProblem(
        Calendar(GRID, (Availability(PERSON, periods),)),
        (Person(PERSON, "Person"),),
        (),
        fixed_tasks,
        (),
    )


def test_free_slots_unions_availability_and_ignores_other_people() -> None:
    value: SchedulingProblem = problem(
        (interval(-30, 30), interval(0, 60), interval(90, 150))
    )
    value = replace(
        value,
        calendar=replace(
            value.calendar,
            availabilities=(
                *value.calendar.availabilities,
                Availability(PersonId("other"), (GRID.horizon,)),
            ),
        ),
    )
    assert free_slots(value, PERSON) == (True, True, False, True)
    assert free_slots(value, PersonId("missing")) == (False,) * 4
    assert free_slots(problem(()), PERSON) == (False,) * 4


def test_free_slots_uses_only_whole_available_slots() -> None:
    assert free_slots(problem((interval(10, 100),)), PERSON) == (
        False,
        True,
        True,
        False,
    )


def test_intervals_before_horizon_leave_free_slots_unchanged() -> None:
    before: TimeInterval = interval(-60, -30)
    baseline: SchedulingProblem = problem()
    with_availability: SchedulingProblem = problem((before, GRID.horizon))
    fixed: FixedTask = FixedTask(
        TaskId("before"),
        "Before",
        before.start,
        before.end - before.start,
        frozenset({PERSON}),
    )
    assert free_slots(baseline, PERSON) == (True,) * 4
    assert free_slots(with_availability, PERSON) == free_slots(baseline, PERSON)
    assert free_slots(problem(fixed_tasks=(fixed,)), PERSON) == free_slots(
        baseline, PERSON
    )


@pytest.mark.parametrize(
    "offset,duration,expected",
    [
        (-60, 30, (True, True)),
        (60, 30, (True, True)),
        (-10, 20, (False, True)),
        (50, 20, (True, False)),
        (-10, 80, (False, False)),
        (10, 30, (False, False)),
        (30, 30, (True, False)),
        (10, 0, (True, True)),
    ],
)
def test_fixed_tasks_round_outward_and_clip_to_horizon(
    offset: int, duration: int, expected: tuple[bool, ...]
) -> None:
    fixed: FixedTask = FixedTask(
        TaskId("fixed"),
        "Existing",
        START + timedelta(minutes=offset),
        timedelta(minutes=duration),
        frozenset({PERSON}),
    )
    short_grid: TimeGrid = TimeGrid(interval(0, 60), SLOT)
    value: SchedulingProblem = problem(fixed_tasks=(fixed,))
    value = replace(value, calendar=replace(value.calendar, grid=short_grid))
    assert free_slots(value, PERSON) == expected


def test_free_slots_ignores_movable_tasks_and_unrelated_fixed_tasks() -> None:
    fixed: FixedTask = FixedTask(
        TaskId("fixed"), "Other", START, 4 * SLOT, frozenset({PersonId("other")})
    )
    movable: Task = Task(
        TaskId("movable"),
        "Movable",
        4 * SLOT,
        frozenset({PERSON}),
        Importance.LOW,
        True,
    )
    assert (
        free_slots(replace(problem(fixed_tasks=(fixed,)), tasks=(movable,)), PERSON)
        == (True,) * 4
    )


def test_available_starts_requires_every_participant_for_whole_duration() -> None:
    participants: tuple[tuple[bool, ...], ...] = (
        (True, True, True, True),
        (False, True, True, False),
    )
    assert available_start_slots(GRID, participants, SLOT) == (1, 2)
    assert available_start_slots(GRID, participants, 2 * SLOT) == (1,)
    assert available_start_slots(GRID, participants, 3 * SLOT) == ()


@pytest.mark.parametrize(
    "duration,expected",
    [(SLOT, (0, 1, 2, 3)), (2 * SLOT, (0, 1, 2)), (4 * SLOT, (0,)), (5 * SLOT, ())],
)
def test_no_participants_returns_all_fitting_horizon_starts(
    duration: timedelta, expected: tuple[int, ...]
) -> None:
    assert available_start_slots(GRID, (), duration) == expected


def test_duration_longer_than_horizon_has_no_start_for_participants() -> None:
    assert available_start_slots(GRID, ((True,) * 4,), 5 * SLOT) == ()


def test_duration_longer_than_horizon_does_not_require_a_representable_end() -> None:
    start: datetime = datetime(9999, 12, 31, 23, tzinfo=timezone.utc)
    grid: TimeGrid = TimeGrid(TimeInterval(start, start + SLOT), SLOT)
    assert available_start_slots(grid, (), timedelta(days=1)) == ()


@pytest.mark.parametrize("duration", [timedelta(0), -SLOT, timedelta(minutes=10)])
def test_available_starts_rejects_invalid_duration(duration: timedelta) -> None:
    with pytest.raises(ValueError, match="duration"):
        available_start_slots(GRID, (), duration)
