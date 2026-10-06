from datetime import date, datetime, timedelta, timezone

import pytest

from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval

START: datetime = datetime(2026, 10, 1, 9, 10, tzinfo=timezone(timedelta(hours=9)))
SLOT: timedelta = timedelta(minutes=30)
GRID: TimeGrid = TimeGrid(TimeInterval(START, START + 4 * SLOT), SLOT)


def interval(first: int, last: int) -> TimeInterval:
    return TimeInterval(
        START + timedelta(minutes=first), START + timedelta(minutes=last)
    )


def test_time_interval_rejects_reversed_endpoints() -> None:
    """Reject an interval whose end precedes its start."""
    start: datetime = datetime(2026, 10, 5, 9)
    with pytest.raises(ValueError, match="start.*end"):
        TimeInterval(start, start - timedelta(minutes=1))


def test_time_interval_allows_equal_endpoints() -> None:
    """Allow zero-length domain intervals."""
    start: datetime = datetime(2026, 10, 5, 9)
    assert TimeInterval(start, start).end == start


@pytest.mark.parametrize("duration", [timedelta(0), timedelta(hours=1, microseconds=1)])
def test_time_interval_duration(duration: timedelta) -> None:
    """Measure interval duration exactly."""
    start: datetime = datetime(2026, 10, 5, 9)
    assert TimeInterval(start, start + duration).duration == duration


@pytest.mark.parametrize(
    "first,last,expected",
    [
        (0, 60, True),
        (10, 50, True),
        (0, 0, True),
        (60, 60, True),
        (-1, 30, False),
        (30, 61, False),
        (-1, 61, False),
    ],
)
def test_time_interval_contains_whole_interval(
    first: int, last: int, expected: bool
) -> None:
    """Contain intervals including equal and empty endpoints."""
    start: datetime = datetime(2026, 10, 5, 9)
    outer: TimeInterval = TimeInterval(start, start + timedelta(hours=1))
    inner: TimeInterval = TimeInterval(
        start + timedelta(minutes=first), start + timedelta(minutes=last)
    )
    assert outer.contains(inner) is expected


@pytest.mark.parametrize(
    "offset,expected", [(-1, False), (0, True), (30, True), (60, False), (61, False)]
)
def test_time_interval_includes_half_open_points(offset: int, expected: bool) -> None:
    """Include the start and exclude the end."""
    start: datetime = datetime(2026, 10, 5, 9)
    assert (
        TimeInterval(start, start + timedelta(hours=1)).includes(
            start + timedelta(minutes=offset)
        )
        is expected
    )
    assert not TimeInterval(start, start).includes(start)


@pytest.mark.parametrize(
    "first,last,minutes",
    [
        (-60, -1, 0),
        (-30, 0, 0),
        (60, 90, 0),
        (70, 90, 0),
        (-10, 10, 10),
        (30, 90, 30),
        (10, 50, 40),
        (-10, 70, 60),
        (0, 60, 60),
        (30, 30, 0),
    ],
)
def test_time_interval_overlap_is_symmetric_and_nonnegative(
    first: int, last: int, minutes: int
) -> None:
    """Measure overlapping time without counting touching endpoints."""
    start: datetime = datetime(2026, 10, 5, 9)
    outer: TimeInterval = TimeInterval(start, start + timedelta(hours=1))
    other: TimeInterval = TimeInterval(
        start + timedelta(minutes=first), start + timedelta(minutes=last)
    )
    assert outer.overlap(other) == timedelta(minutes=minutes)
    assert other.overlap(outer) == timedelta(minutes=minutes)


def test_time_interval_operations_compare_instants_across_offsets() -> None:
    """Compare equivalent times with different offsets."""
    start: datetime = datetime(2026, 10, 5, 9, tzinfo=timezone(timedelta(hours=9)))
    outer: TimeInterval = TimeInterval(start, start + timedelta(hours=1))
    other: TimeInterval = TimeInterval(
        start.astimezone(timezone.utc), outer.end.astimezone(timezone.utc)
    )
    assert outer.contains(other)
    assert outer.includes(other.start)
    assert outer.overlap(other) == timedelta(hours=1)


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


def test_grid_rejects_horizon_without_offset() -> None:
    naive: datetime = START.replace(tzinfo=None)
    with pytest.raises(ValueError, match="horizon.*offset"):
        TimeGrid(TimeInterval(naive, naive + 4 * SLOT), SLOT)


def test_grid_starting_offset_is_the_fixed_offset_of_the_horizon_start() -> None:
    grid: TimeGrid = TimeGrid(
        TimeInterval(START, (START + 4 * SLOT).astimezone(timezone.utc)), SLOT
    )
    assert grid.starting_offset == timezone(timedelta(hours=9))


def test_grid_rejects_unaligned_horizon_end() -> None:
    with pytest.raises(ValueError, match="horizon end.*aligned"):
        TimeGrid(interval(0, 31), SLOT)


@pytest.mark.parametrize("index", [-2, 0, 1, 4, 6])
def test_grid_coordinates_extend_beyond_horizon(index: int) -> None:
    assert GRID.slot_count == 4
    assert GRID.time_at(index) == START + index * SLOT
    assert GRID.is_aligned(GRID.time_at(index))


def test_grid_alignment_preserves_microsecond_precision() -> None:
    tiny: TimeGrid = TimeGrid(
        TimeInterval(START, START + timedelta(microseconds=12)),
        timedelta(microseconds=3),
    )
    assert tiny.slot_count == 4
    assert not tiny.is_aligned(START + timedelta(microseconds=10))


@pytest.mark.parametrize(
    "first,last,expected",
    [
        (10, 70, range(1, 2)),
        (0, 60, range(0, 2)),
        (-70, -10, range(0)),
        (110, 160, range(0)),
        (-40, 160, range(0, 4)),
        (-10, 40, range(0, 1)),
        (80, 140, range(3, 4)),
    ],
)
def test_slots_within_uses_origin_and_clips_to_horizon(
    first: int, last: int, expected: range
) -> None:
    assert GRID.slots_within(interval(first, last)) == expected


@pytest.mark.parametrize("first,last", [(10, 20), (10, 40), (0, 0), (10, 10), (30, 30)])
def test_slots_within_returns_empty_when_no_slot_fits(first: int, last: int) -> None:
    assert GRID.slots_within(interval(first, last)) == range(0)


@pytest.mark.parametrize(
    "first,last,expected",
    [
        (10, 40, range(0, 2)),
        (0, 60, range(0, 2)),
        (-40, -10, range(0)),
        (110, 140, range(3, 4)),
        (-10, 130, range(0, 4)),
        (-10, 10, range(0, 1)),
        (120, 150, range(0)),
        (-30, 0, range(0)),
        (150, 180, range(0)),
        (30, 60, range(1, 2)),
    ],
)
def test_slots_touching_uses_origin_and_clips_to_horizon(
    first: int, last: int, expected: range
) -> None:
    assert GRID.slots_touching(interval(first, last)) == expected


@pytest.mark.parametrize("offset", [-10, 0, 10, 120, 130])
def test_slots_touching_returns_empty_for_zero_length_interval(offset: int) -> None:
    assert GRID.slots_touching(interval(offset, offset)) == range(0)


@pytest.mark.parametrize("count", [-2, 0, 1, 4, 6])
def test_duration_slots_are_independent_of_horizon(count: int) -> None:
    duration: timedelta = count * SLOT
    assert GRID.is_whole_slots(duration)
    assert GRID.slots_of(duration) == count


@pytest.mark.parametrize(
    "duration",
    [timedelta(minutes=1), -timedelta(minutes=1), SLOT + timedelta(microseconds=1)],
)
def test_duration_slots_reject_partial_slots(duration: timedelta) -> None:
    assert not GRID.is_whole_slots(duration)
    with pytest.raises(ValueError, match="[Dd]uration.*aligned|[Dd]uration.*multiple"):
        GRID.slots_of(duration)


def test_duration_slots_preserve_microsecond_precision_without_datetime_arithmetic() -> (
    None
):
    tiny: TimeGrid = TimeGrid(
        TimeInterval(START, START + timedelta(microseconds=12)),
        timedelta(microseconds=3),
    )
    assert tiny.slots_of(timedelta(microseconds=9)) == 3
    assert not tiny.is_whole_slots(timedelta(microseconds=10))
    assert GRID.slots_of(timedelta(days=999999999)) == 47999999952


@pytest.mark.parametrize("slots", [range(0, 4), range(1, 3), range(3, 4)])
def test_interval_of_covers_nonempty_slots(slots: range) -> None:
    assert GRID.interval_of(slots) == TimeInterval(
        GRID.time_at(slots.start), GRID.time_at(slots.stop)
    )


@pytest.mark.parametrize("slots", [range(0), range(2, 2), range(3, 1)])
def test_interval_of_rejects_empty_slots(slots: range) -> None:
    with pytest.raises(ValueError, match="empty|non-empty"):
        GRID.interval_of(slots)


def test_date_of_uses_horizon_start_offset() -> None:
    at: datetime = datetime(2026, 10, 1, 23, tzinfo=timezone.utc)
    assert GRID.date_of(at) == date(2026, 10, 2)


@pytest.mark.parametrize(
    "duration,expected",
    [
        (timedelta(minutes=10), (date(2026, 10, 1),)),
        (timedelta(minutes=10, microseconds=1), (date(2026, 10, 1), date(2026, 10, 2))),
        (SLOT, (date(2026, 10, 1), date(2026, 10, 2))),
    ],
)
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
        date(2026, 10, 1),
        date(2026, 10, 2),
        date(2026, 10, 3),
        date(2026, 10, 4),
        date(2026, 10, 5),
    )


@pytest.mark.parametrize(
    "end",
    [
        datetime.min.replace(tzinfo=timezone.utc) + timedelta(microseconds=1),
        datetime.max.replace(tzinfo=timezone.utc),
    ],
)
def test_dates_preserve_datetime_extremes(end: datetime) -> None:
    """Enumerate dates without overflowing datetime boundaries."""
    grid: TimeGrid = TimeGrid(
        TimeInterval(end - timedelta(microseconds=1), end), timedelta(microseconds=1)
    )
    assert grid.dates == (end.date(),)
