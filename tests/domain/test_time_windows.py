from calendar import Day
from datetime import date, datetime, timedelta, timezone

import pytest

from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.time_windows import (
    DateRange,
    Expansion,
    TimeRange,
    TimeRelation,
    TimeWindow,
    WholeHorizon,
    complement,
    expand,
    window_times,
)

OFFSET: timezone = timezone(timedelta(hours=9))
HOUR: timedelta = timedelta(hours=1)
EVERY_DAY: frozenset[Day] = frozenset(Day)
WHOLE_DAY: TimeRange = TimeRange(timedelta(0), timedelta(hours=24))


def at(day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=OFFSET)


def interval(day: int, start: int, end: int) -> TimeInterval:
    return TimeInterval(at(day, start), at(day, end))


def clock(hour: int, minute: int = 0) -> timedelta:
    return timedelta(hours=hour, minutes=minute)


def daily(start: timedelta, end: timedelta) -> TimeWindow:
    return TimeWindow(WholeHorizon(), EVERY_DAY, TimeRange(start, end))


def test_window_fields_are_intersected_and_windows_are_unioned() -> None:
    horizon: TimeInterval = TimeInterval(at(12), at(20))
    windows: tuple[TimeWindow, ...] = (
        TimeWindow(
            DateRange(date(2026, 10, 12), date(2026, 10, 17)),
            frozenset({Day.FRIDAY, Day.SATURDAY}),
            TimeRange(clock(13), clock(18)),
        ),
        TimeWindow(
            DateRange(date(2026, 10, 19), date(2026, 10, 20)),
            frozenset({Day.MONDAY}),
            TimeRange(clock(9), clock(10)),
        ),
    )
    assert window_times(windows, TimeGrid(horizon, HOUR)) == (
        interval(16, 13, 18),
        interval(19, 9, 10),
    )


def test_whole_horizon_every_day_and_whole_day_cover_the_whole_horizon() -> None:
    horizon: TimeInterval = TimeInterval(at(12, 11), at(14, 15))
    window: TimeWindow = TimeWindow(WholeHorizon(), EVERY_DAY, WHOLE_DAY)
    assert window_times((window,), TimeGrid(horizon, HOUR)) == (horizon,)


def test_whole_horizon_repeats_the_time_on_every_day() -> None:
    horizon: TimeInterval = TimeInterval(at(12), at(15))
    window: TimeWindow = daily(clock(13), clock(18))
    assert window_times((window,), TimeGrid(horizon, HOUR)) == tuple(
        interval(day, 13, 18) for day in (12, 13, 14)
    )


def test_whole_day_covers_selected_whole_days() -> None:
    horizon: TimeInterval = TimeInterval(at(12), at(20))
    window: TimeWindow = TimeWindow(WholeHorizon(), frozenset({Day.FRIDAY}), WHOLE_DAY)
    assert window_times((window,), TimeGrid(horizon, HOUR)) == (
        TimeInterval(at(16), at(17)),
    )


def test_time_end_at_24_hours_covers_the_rest_of_each_day() -> None:
    horizon: TimeInterval = TimeInterval(at(12), at(14))
    window: TimeWindow = daily(clock(22), clock(24))
    assert window_times((window,), TimeGrid(horizon, HOUR)) == (
        TimeInterval(at(12, 22), at(13)),
        TimeInterval(at(13, 22), at(14)),
    )


def test_windows_are_clipped_to_the_horizon() -> None:
    horizon: TimeInterval = TimeInterval(at(12, 14), at(13, 15))
    window: TimeWindow = TimeWindow(
        DateRange(date(2026, 10, 1), date(2026, 11, 1)),
        EVERY_DAY,
        TimeRange(clock(13), clock(18)),
    )
    assert window_times((window,), TimeGrid(horizon, HOUR)) == (
        interval(12, 14, 18),
        interval(13, 13, 15),
    )


def test_window_time_end_and_horizon_end_are_excluded() -> None:
    grid: TimeGrid = TimeGrid(interval(12, 9, 10), HOUR)
    assert window_times((daily(clock(10), clock(11)),), grid) == ()
    assert window_times((daily(clock(8), clock(9)),), grid) == ()


def test_window_dates_use_the_horizon_start_offset() -> None:
    horizon: TimeInterval = TimeInterval(at(12, 23), at(13, 2).astimezone(timezone.utc))
    window: TimeWindow = TimeWindow(
        DateRange(date(2026, 10, 13), date(2026, 10, 14)),
        frozenset({Day.TUESDAY}),
        TimeRange(clock(1), clock(2)),
    )
    result: tuple[TimeInterval, ...] = window_times((window,), TimeGrid(horizon, HOUR))
    assert result == (interval(13, 1, 2),)
    assert result[0].start.utcoffset() == timedelta(hours=9)


@pytest.mark.parametrize("weekdays", [EVERY_DAY, frozenset()])
def test_empty_windows_and_empty_weekdays(weekdays: frozenset[Day]) -> None:
    horizon: TimeInterval = TimeInterval(at(12), at(13))
    assert window_times((), TimeGrid(horizon, HOUR)) == ()
    window: TimeWindow = TimeWindow(WholeHorizon(), weekdays, WHOLE_DAY)
    assert window_times((window,), TimeGrid(horizon, HOUR)) == (
        (horizon,) if weekdays else ()
    )


def test_overlapping_and_adjacent_windows_are_merged_in_time_order() -> None:
    horizon: TimeInterval = TimeInterval(at(12), at(13))
    windows: tuple[TimeWindow, ...] = tuple(
        daily(clock(start), clock(end))
        for start, end in ((11, 13), (9, 10), (10, 12), (15, 16), (9, 10))
    )
    assert window_times(windows, TimeGrid(horizon, HOUR)) == (
        interval(12, 9, 13),
        interval(12, 15, 16),
    )


@pytest.mark.parametrize("relation", list(TimeRelation))
def test_empty_expansion_needs_no_rounding(relation: TimeRelation) -> None:
    grid: TimeGrid = TimeGrid(interval(12, 9, 12), timedelta(hours=1))
    assert expand((), relation, grid) == Expansion((), False)


@pytest.mark.parametrize(
    ("relation", "expected"),
    [
        (TimeRelation.WITHIN, (interval(12, 10, 11),)),
        (TimeRelation.AVOID, (interval(12, 9, 12),)),
    ],
)
def test_expansion_merges_before_selecting_grid_slots(
    relation: TimeRelation, expected: tuple[TimeInterval, ...]
) -> None:
    """Find a whole slot that no single window contains."""
    grid: TimeGrid = TimeGrid(interval(12, 9, 13), timedelta(hours=1))
    windows: tuple[TimeWindow, ...] = (
        daily(clock(9, 30), clock(10, 15)),
        daily(clock(10, 15), clock(11, 30)),
        daily(clock(10), clock(10, 30)),
    )
    assert expand(windows, relation, grid) == Expansion(expected, True)


def test_avoid_expansion_merges_again_after_selecting_slots() -> None:
    grid: TimeGrid = TimeGrid(interval(12, 9, 13), timedelta(hours=1))
    windows: tuple[TimeWindow, ...] = (
        daily(clock(9, 10), clock(9, 20)),
        daily(clock(10, 10), clock(10, 20)),
        daily(clock(10, 40), clock(11, 10)),
    )
    result: Expansion = expand(windows, TimeRelation.AVOID, grid)
    assert result == Expansion((interval(12, 9, 12),), True)


@pytest.mark.parametrize("relation", list(TimeRelation))
def test_expansion_reports_unchanged_aligned_slot_times(relation: TimeRelation) -> None:
    grid: TimeGrid = TimeGrid(interval(12, 9, 12), timedelta(hours=1))
    assert expand((daily(clock(10), clock(11)),), relation, grid) == Expansion(
        (interval(12, 10, 11),), False
    )


def test_within_expansion_drops_empty_slot_ranges() -> None:
    grid: TimeGrid = TimeGrid(interval(12, 9, 12), timedelta(hours=1))
    assert expand(
        (daily(clock(9, 10), clock(9, 20)),), TimeRelation.WITHIN, grid
    ) == Expansion((), True)


@pytest.mark.parametrize(
    ("relation", "expected"),
    [
        (TimeRelation.WITHIN, (interval(12, 10, 11),)),
        (TimeRelation.AVOID, (interval(12, 9, 12),)),
    ],
)
def test_real_grid_rounds_fully_inside_or_touching_slots(
    relation: TimeRelation, expected: tuple[TimeInterval, ...]
) -> None:
    grid: TimeGrid = TimeGrid(interval(12, 9, 13), timedelta(hours=1))
    window: TimeWindow = daily(clock(9, 10), clock(11, 10))
    assert expand((window,), relation, grid) == Expansion(expected, True)


def test_real_grid_rounding_uses_the_horizon_start() -> None:
    grid: TimeGrid = TimeGrid(
        TimeInterval(at(12, 9, 15), at(12, 12, 15)), timedelta(hours=1)
    )
    windows: tuple[TimeWindow, ...] = (
        daily(clock(8), clock(10, 15)),
        daily(clock(10, 15), clock(13)),
    )
    assert expand(windows, TimeRelation.WITHIN, grid) == Expansion(
        (grid.horizon,), False
    )


def test_complement_merges_clips_and_returns_ordered_gaps() -> None:
    horizon: TimeInterval = interval(12, 9, 17)
    intervals: tuple[TimeInterval, ...] = (
        interval(12, 14, 16),
        interval(12, 8, 10),
        interval(12, 12, 14),
        interval(12, 13, 15),
        interval(12, 18, 20),
        interval(12, 11, 11),
    )
    assert complement(intervals, horizon) == (
        interval(12, 10, 12),
        interval(12, 16, 17),
    )
    assert complement((), horizon) == (horizon,)
    assert complement((interval(12, 8, 18),), horizon) == ()


@pytest.mark.parametrize("end", [clock(13), clock(12)])
def test_time_range_rejects_nonincreasing_times_with_values(end: timedelta) -> None:
    with pytest.raises(ValueError) as error:
        TimeRange(clock(13), end)
    message: str = str(error.value)
    assert f"end {end // HOUR:02}:00" in message
    assert "start 13:00" in message
    assert "after" in message


@pytest.mark.parametrize(
    "start,end",
    [
        (-timedelta(minutes=1), clock(9)),
        (clock(23), clock(24, 1)),
        (clock(24), clock(25)),
    ],
)
def test_time_range_rejects_times_outside_the_day(
    start: timedelta, end: timedelta
) -> None:
    with pytest.raises(ValueError, match="00:00.*24:00"):
        TimeRange(start, end)


@pytest.mark.parametrize("end", [date(2026, 10, 4), date(2026, 10, 5)])
def test_date_range_rejects_nonincreasing_endpoints(end: date) -> None:
    """Reject date ranges whose end is not after their start."""
    with pytest.raises(ValueError, match="Date range end.*must be after start"):
        DateRange(date(2026, 10, 5), end)
