from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from intent_to_schedule.domain.availability import available_start_slots, free_slots
from intent_to_schedule.domain.calendar import Availability, Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId


START: datetime = datetime(2026, 10, 1, 9, 10, tzinfo=timezone(timedelta(hours=9)))
SLOT: timedelta = timedelta(minutes=30)
GRID: TimeGrid = TimeGrid(TimeInterval(START, START + 4 * SLOT), SLOT)
PERSON: PersonId = PersonId("person")


def interval(first: int, last: int) -> TimeInterval:
    return TimeInterval(START + timedelta(minutes=first), START + timedelta(minutes=last))


def problem(
    periods: tuple[TimeInterval, ...] = (GRID.horizon,),
    fixed_tasks: tuple[FixedTask, ...] = (),
) -> SchedulingProblem:
    return SchedulingProblem(
        Calendar(GRID, (Availability(PERSON, periods),)),
        (Person(PERSON, "Person"),), (), fixed_tasks, (),
    )


@pytest.mark.parametrize("slot", [timedelta(0), -SLOT])
def test_grid_rejects_nonpositive_slot(slot: timedelta) -> None:
    with pytest.raises(ValueError, match="slot.*positive"):
        TimeGrid(GRID.horizon, slot)


@pytest.mark.parametrize(
    ("end", "message"), [(START, "horizon.*start.*end"), (START - SLOT, "start.*end")]
)
def test_grid_rejects_nonpositive_horizon(end: datetime, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TimeGrid(TimeInterval(START, end), SLOT)


def test_grid_rejects_unaligned_horizon_end() -> None:
    with pytest.raises(ValueError, match="horizon end.*aligned"):
        TimeGrid(interval(0, 31), SLOT)


@pytest.mark.parametrize("index", [-2, 0, 1, 4, 6])
def test_grid_coordinates_extend_beyond_horizon(index: int) -> None:
    assert GRID.slot_count == 4
    assert GRID.time_at(index) == START + index * SLOT
    assert GRID.is_aligned(GRID.time_at(index))


def test_grid_alignment_preserves_microsecond_precision() -> None:
    tiny: TimeGrid = TimeGrid(TimeInterval(START, START + timedelta(microseconds=12)), timedelta(microseconds=3))
    assert tiny.slot_count == 4
    assert not tiny.is_aligned(START + timedelta(microseconds=10))


@pytest.mark.parametrize("first,last,expected", [
    (10, 70, range(1, 2)), (0, 60, range(0, 2)), (-70, -10, range(0)),
    (110, 160, range(0)), (-40, 160, range(0, 4)),
    (-10, 40, range(0, 1)), (80, 140, range(3, 4)),
])
def test_slots_within_uses_origin_and_clips_to_horizon(first: int, last: int, expected: range) -> None:
    assert GRID.slots_within(interval(first, last)) == expected


@pytest.mark.parametrize("first,last", [(10, 20), (10, 40), (0, 0), (10, 10), (30, 30)])
def test_slots_within_returns_empty_when_no_slot_fits(first: int, last: int) -> None:
    assert GRID.slots_within(interval(first, last)) == range(0)


@pytest.mark.parametrize("first,last,expected", [
    (10, 40, range(0, 2)), (0, 60, range(0, 2)), (-40, -10, range(0)),
    (110, 140, range(3, 4)), (-10, 130, range(0, 4)),
    (-10, 10, range(0, 1)), (120, 150, range(0)), (-30, 0, range(0)),
    (150, 180, range(0)), (30, 60, range(1, 2)),
])
def test_slots_touching_uses_origin_and_clips_to_horizon(first: int, last: int, expected: range) -> None:
    assert GRID.slots_touching(interval(first, last)) == expected


@pytest.mark.parametrize("offset", [-10, 0, 10, 120, 130])
def test_slots_touching_returns_empty_for_zero_length_interval(offset: int) -> None:
    assert GRID.slots_touching(interval(offset, offset)) == range(0)


@pytest.mark.parametrize("count", [-2, 0, 1, 4, 6])
def test_duration_slots_are_independent_of_horizon(count: int) -> None:
    duration: timedelta = count * SLOT
    assert GRID.is_whole_slots(duration)
    assert GRID.slots_of(duration) == count


@pytest.mark.parametrize("duration", [timedelta(minutes=1), -timedelta(minutes=1), SLOT + timedelta(microseconds=1)])
def test_duration_slots_reject_partial_slots(duration: timedelta) -> None:
    assert not GRID.is_whole_slots(duration)
    with pytest.raises(ValueError, match="[Dd]uration.*aligned|[Dd]uration.*multiple"):
        GRID.slots_of(duration)


def test_duration_slots_preserve_microsecond_precision_without_datetime_arithmetic() -> None:
    tiny: TimeGrid = TimeGrid(TimeInterval(START, START + timedelta(microseconds=12)), timedelta(microseconds=3))
    assert tiny.slots_of(timedelta(microseconds=9)) == 3
    assert not tiny.is_whole_slots(timedelta(microseconds=10))
    assert GRID.slots_of(timedelta(days=999999999)) == 47999999952


@pytest.mark.parametrize("slots", [range(0, 4), range(1, 3), range(3, 4)])
def test_interval_of_covers_nonempty_slots(slots: range) -> None:
    assert GRID.interval_of(slots) == TimeInterval(GRID.time_at(slots.start), GRID.time_at(slots.stop))


@pytest.mark.parametrize("slots", [range(0), range(2, 2), range(3, 1)])
def test_interval_of_rejects_empty_slots(slots: range) -> None:
    with pytest.raises(ValueError, match="empty|non-empty"):
        GRID.interval_of(slots)


def test_date_of_uses_horizon_start_offset() -> None:
    at: datetime = datetime(2026, 10, 1, 23, tzinfo=timezone.utc)
    assert GRID.date_of(at) == date(2026, 10, 2)


def test_dates_are_sorted_unique_horizon_dates() -> None:
    """Include every horizon date regardless of slot starts."""
    start: datetime = datetime(2026, 10, 1, 23, tzinfo=START.tzinfo)
    grid: TimeGrid = TimeGrid(TimeInterval(start, start + timedelta(hours=25)), timedelta(hours=1))
    assert grid.dates == (date(2026, 10, 1), date(2026, 10, 2))
    sparse: TimeGrid = TimeGrid(TimeInterval(start, start + timedelta(days=4)), timedelta(days=2))
    assert sparse.dates == (
        date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3),
        date(2026, 10, 4), date(2026, 10, 5),
    )


@pytest.mark.parametrize("duration,expected", [
    (timedelta(minutes=10), (date(2026, 10, 1),)),
    (timedelta(minutes=10, microseconds=1), (date(2026, 10, 1), date(2026, 10, 2))),
    (SLOT, (date(2026, 10, 1), date(2026, 10, 2))),
])
def test_dates_cover_half_open_horizon_in_start_offset(
    duration: timedelta, expected: tuple[date, ...]
) -> None:
    """Use the start offset and exclude an end at midnight."""
    start: datetime = START.replace(hour=23, minute=50)
    end: datetime = (start + duration).astimezone(timezone.utc)
    grid: TimeGrid = TimeGrid(TimeInterval(start, end), duration)
    assert grid.dates == expected


@pytest.mark.parametrize("slot", [SLOT, timedelta(hours=1), timedelta(days=2)])
def test_dates_do_not_depend_on_slot_size(slot: timedelta) -> None:
    """Keep daily scope constant across scheduling resolutions."""
    start: datetime = START.replace(hour=23, minute=50)
    grid: TimeGrid = TimeGrid(TimeInterval(start, start + timedelta(days=4)), slot)
    assert grid.dates == (
        date(2026, 10, 1), date(2026, 10, 2), date(2026, 10, 3),
        date(2026, 10, 4), date(2026, 10, 5),
    )


@pytest.mark.parametrize("end", [
    datetime.min.replace(tzinfo=timezone.utc) + timedelta(microseconds=1),
    datetime.max.replace(tzinfo=timezone.utc),
])
def test_dates_preserve_datetime_extremes(end: datetime) -> None:
    """Enumerate dates without overflowing datetime boundaries."""
    grid: TimeGrid = TimeGrid(
        TimeInterval(end - timedelta(microseconds=1), end), timedelta(microseconds=1)
    )
    assert grid.dates == (end.date(),)


def test_free_slots_unions_availability_and_ignores_other_people() -> None:
    value: SchedulingProblem = problem((interval(-30, 30), interval(0, 60), interval(90, 150)))
    value = replace(value, calendar=replace(value.calendar, availabilities=(*value.calendar.availabilities, Availability(PersonId("other"), (GRID.horizon,)))))
    assert free_slots(value, PERSON) == (True, True, False, True)
    assert free_slots(value, PersonId("missing")) == (False,) * 4
    assert free_slots(problem(()), PERSON) == (False,) * 4


def test_free_slots_uses_only_whole_available_slots() -> None:
    assert free_slots(problem((interval(10, 100),)), PERSON) == (False, True, True, False)


def test_intervals_before_horizon_leave_free_slots_unchanged() -> None:
    before: TimeInterval = interval(-60, -30)
    baseline: SchedulingProblem = problem()
    with_availability: SchedulingProblem = problem((before, GRID.horizon))
    fixed: FixedTask = FixedTask(TaskId("before"), "Before", before.start, before.end - before.start, frozenset({PERSON}))
    assert free_slots(baseline, PERSON) == (True,) * 4
    assert free_slots(with_availability, PERSON) == free_slots(baseline, PERSON)
    assert free_slots(problem(fixed_tasks=(fixed,)), PERSON) == free_slots(baseline, PERSON)


@pytest.mark.parametrize("offset,duration,expected", [
    (-60, 30, (True, True)), (60, 30, (True, True)),
    (-10, 20, (False, True)), (50, 20, (True, False)),
    (-10, 80, (False, False)), (10, 30, (False, False)),
    (30, 30, (True, False)), (10, 0, (True, True)),
])
def test_fixed_tasks_round_outward_and_clip_to_horizon(offset: int, duration: int, expected: tuple[bool, ...]) -> None:
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Existing", START + timedelta(minutes=offset), timedelta(minutes=duration), frozenset({PERSON}))
    short_grid: TimeGrid = TimeGrid(interval(0, 60), SLOT)
    value: SchedulingProblem = problem(fixed_tasks=(fixed,))
    value = replace(value, calendar=replace(value.calendar, grid=short_grid))
    assert free_slots(value, PERSON) == expected


def test_free_slots_ignores_movable_tasks_and_unrelated_fixed_tasks() -> None:
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Other", START, 4 * SLOT, frozenset({PersonId("other")}))
    movable: Task = Task(TaskId("movable"), "Movable", 4 * SLOT, frozenset({PERSON}), Importance.LOW, True)
    assert free_slots(replace(problem(fixed_tasks=(fixed,)), tasks=(movable,)), PERSON) == (True,) * 4


def test_available_starts_requires_every_participant_for_whole_duration() -> None:
    participants: tuple[tuple[bool, ...], ...] = ((True, True, True, True), (False, True, True, False))
    assert available_start_slots(GRID, participants, SLOT) == (1, 2)
    assert available_start_slots(GRID, participants, 2 * SLOT) == (1,)
    assert available_start_slots(GRID, participants, 3 * SLOT) == ()


@pytest.mark.parametrize("duration,expected", [(SLOT, (0, 1, 2, 3)), (2 * SLOT, (0, 1, 2)), (4 * SLOT, (0,)), (5 * SLOT, ())])
def test_no_participants_returns_all_fitting_horizon_starts(duration: timedelta, expected: tuple[int, ...]) -> None:
    assert available_start_slots(GRID, (), duration) == expected


def test_duration_longer_than_horizon_has_no_start_for_participants() -> None:
    assert available_start_slots(GRID, ((True,) * 4,), 5 * SLOT) == ()


def test_duration_longer_than_horizon_does_not_require_a_representable_end() -> None:
    start: datetime = datetime(9999, 12, 31, 23)
    grid: TimeGrid = TimeGrid(TimeInterval(start, start + SLOT), SLOT)
    assert available_start_slots(grid, (), timedelta(days=1)) == ()


@pytest.mark.parametrize("duration", [timedelta(0), -SLOT, timedelta(minutes=10)])
def test_available_starts_rejects_invalid_duration(duration: timedelta) -> None:
    with pytest.raises(ValueError, match="duration"):
        available_start_slots(GRID, (), duration)
