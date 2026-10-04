from calendar import Day
from datetime import date, datetime, time, timedelta, timezone
from unittest.mock import Mock, call, patch

import pytest

from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.time_windows import (
    DateRange,
    Expansion,
    TimeRange,
    TimeRelation,
    TimeWindow,
    complement,
    expand,
    window_times,
)

OFFSET: timezone = timezone(timedelta(hours=9))


def at(day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(2026, 10, day, hour, minute, tzinfo=OFFSET)


def interval(day: int, start: int, end: int) -> TimeInterval:
    return TimeInterval(at(day, start), at(day, end))


def test_window_fields_are_intersected_and_windows_are_unioned() -> None:
    horizon: TimeInterval = TimeInterval(at(12), at(20))
    windows: tuple[TimeWindow, ...] = (
        TimeWindow(
            DateRange(date(2026, 10, 12), date(2026, 10, 17)),
            frozenset({Day.FRIDAY, Day.SATURDAY}),
            TimeRange(time(13), time(18)),
        ),
        TimeWindow(
            DateRange(date(2026, 10, 19), date(2026, 10, 20)),
            frozenset({Day.MONDAY}),
            TimeRange(time(9), time(10)),
        ),
    )
    assert window_times(windows, horizon) == (interval(16, 13, 18), interval(19, 9, 10))


def test_omitted_fields_cover_the_whole_horizon() -> None:
    horizon: TimeInterval = TimeInterval(at(12, 11), at(14, 15))
    assert window_times((TimeWindow(None, None, None),), horizon) == (horizon,)


def test_omitted_dates_repeat_the_time_on_every_day() -> None:
    horizon: TimeInterval = TimeInterval(at(12), at(15))
    window: TimeWindow = TimeWindow(None, None, TimeRange(time(13), time(18)))
    assert window_times((window,), horizon) == tuple(
        interval(day, 13, 18) for day in (12, 13, 14)
    )


def test_omitted_time_covers_selected_whole_days() -> None:
    horizon: TimeInterval = TimeInterval(at(12), at(20))
    window: TimeWindow = TimeWindow(None, frozenset({Day.FRIDAY}), None)
    assert window_times((window,), horizon) == (TimeInterval(at(16), at(17)),)


def test_null_time_end_covers_the_rest_of_each_day() -> None:
    horizon: TimeInterval = TimeInterval(at(12), at(14))
    window: TimeWindow = TimeWindow(None, None, TimeRange(time(22), None))
    assert window_times((window,), horizon) == (
        TimeInterval(at(12, 22), at(13)),
        TimeInterval(at(13, 22), at(14)),
    )


def test_windows_are_clipped_to_the_horizon() -> None:
    horizon: TimeInterval = TimeInterval(at(12, 14), at(13, 15))
    window: TimeWindow = TimeWindow(
        DateRange(date(2026, 10, 1), date(2026, 11, 1)),
        None,
        TimeRange(time(13), time(18)),
    )
    assert window_times((window,), horizon) == (
        interval(12, 14, 18),
        interval(13, 13, 15),
    )


def test_window_time_end_and_horizon_end_are_excluded() -> None:
    horizon: TimeInterval = interval(12, 9, 10)
    assert (
        window_times((TimeWindow(None, None, TimeRange(time(10), time(11))),), horizon)
        == ()
    )
    assert (
        window_times((TimeWindow(None, None, TimeRange(time(8), time(9))),), horizon)
        == ()
    )


def test_window_dates_use_the_horizon_start_offset() -> None:
    horizon: TimeInterval = TimeInterval(at(12, 23), at(13, 2).astimezone(timezone.utc))
    window: TimeWindow = TimeWindow(
        DateRange(date(2026, 10, 13), date(2026, 10, 14)),
        frozenset({Day.TUESDAY}),
        TimeRange(time(1), time(2)),
    )
    result: tuple[TimeInterval, ...] = window_times((window,), horizon)
    assert result == (interval(13, 1, 2),)
    assert result[0].start.utcoffset() == timedelta(hours=9)


@pytest.mark.parametrize("weekdays", [None, frozenset()])
def test_empty_windows_and_empty_weekdays(weekdays: frozenset[Day] | None) -> None:
    horizon: TimeInterval = TimeInterval(at(12), at(13))
    assert window_times((), horizon) == ()
    window: TimeWindow = TimeWindow(None, weekdays, None)
    assert window_times((window,), horizon) == ((horizon,) if weekdays is None else ())


def test_overlapping_and_adjacent_windows_are_merged_in_time_order() -> None:
    horizon: TimeInterval = TimeInterval(at(12), at(13))
    windows: tuple[TimeWindow, ...] = tuple(
        TimeWindow(None, None, TimeRange(time(start), time(end)))
        for start, end in ((11, 13), (9, 10), (10, 12), (15, 16), (9, 10))
    )
    assert window_times(windows, horizon) == (interval(12, 9, 13), interval(12, 15, 16))


@pytest.mark.parametrize("relation", list(TimeRelation))
def test_empty_expansion_needs_no_rounding(relation: TimeRelation) -> None:
    grid: TimeGrid = TimeGrid(interval(12, 9, 12), timedelta(hours=1))
    assert expand((), relation, grid) == Expansion((), False)


@pytest.mark.parametrize("relation", list(TimeRelation))
def test_expansion_merges_before_selecting_grid_slots(relation: TimeRelation) -> None:
    grid: TimeGrid = TimeGrid(interval(12, 9, 12), timedelta(hours=1))
    windows: tuple[TimeWindow, ...] = (
        TimeWindow(None, None, TimeRange(time(9, 10), time(9, 30))),
        TimeWindow(None, None, TimeRange(time(9, 30), time(10, 10))),
        TimeWindow(None, None, TimeRange(time(9, 20), time(10))),
    )
    expected: TimeInterval = (
        interval(12, 9, 11) if relation is TimeRelation.AVOID else interval(12, 10, 10)
    )
    rounding: Mock
    with patch.object(
        TimeGrid,
        "slots_touching" if relation is TimeRelation.AVOID else "slots_within",
        return_value=range(0, 2) if relation is TimeRelation.AVOID else range(0),
    ) as rounding:
        result: Expansion = expand(windows, relation, grid)
    rounding.assert_called_once_with(TimeInterval(at(12, 9, 10), at(12, 10, 10)))
    assert result == Expansion(
        (expected,) if expected.start < expected.end else (), True
    )


def test_avoid_expansion_merges_again_after_selecting_slots() -> None:
    grid: TimeGrid = TimeGrid(interval(12, 9, 13), timedelta(hours=1))
    windows: tuple[TimeWindow, ...] = (
        TimeWindow(None, None, TimeRange(time(9, 10), time(9, 20))),
        TimeWindow(None, None, TimeRange(time(10, 10), time(10, 20))),
        TimeWindow(None, None, TimeRange(time(10, 40), time(11, 10))),
    )
    rounding: Mock
    with patch.object(
        TimeGrid,
        "slots_touching",
        side_effect=(range(0, 1), range(1, 2), range(1, 3)),
    ) as rounding:
        result: Expansion = expand(windows, TimeRelation.AVOID, grid)
    assert rounding.call_args_list == [
        call(TimeInterval(at(12, 9, 10), at(12, 9, 20))),
        call(TimeInterval(at(12, 10, 10), at(12, 10, 20))),
        call(TimeInterval(at(12, 10, 40), at(12, 11, 10))),
    ]
    assert result == Expansion((interval(12, 9, 12),), True)


@pytest.mark.parametrize("relation", list(TimeRelation))
def test_expansion_reports_unchanged_aligned_slot_times(relation: TimeRelation) -> None:
    grid: TimeGrid = TimeGrid(interval(12, 9, 12), timedelta(hours=1))
    with patch.object(
        TimeGrid,
        "slots_touching" if relation is TimeRelation.AVOID else "slots_within",
        return_value=range(1, 2),
    ):
        assert expand(
            (TimeWindow(None, None, TimeRange(time(10), time(11))),), relation, grid
        ) == Expansion((interval(12, 10, 11),), False)


def test_within_expansion_drops_empty_slot_ranges() -> None:
    grid: TimeGrid = TimeGrid(interval(12, 9, 12), timedelta(hours=1))
    with patch.object(TimeGrid, "slots_within", return_value=range(0)):
        assert expand(
            (TimeWindow(None, None, TimeRange(time(9, 10), time(9, 20))),),
            TimeRelation.WITHIN,
            grid,
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
    window: TimeWindow = TimeWindow(None, None, TimeRange(time(9, 10), time(11, 10)))
    assert expand((window,), relation, grid) == Expansion(expected, True)


def test_real_grid_rounding_uses_the_horizon_start() -> None:
    grid: TimeGrid = TimeGrid(
        TimeInterval(at(12, 9, 15), at(12, 12, 15)), timedelta(hours=1)
    )
    windows: tuple[TimeWindow, ...] = (
        TimeWindow(None, None, TimeRange(time(8), time(10, 15))),
        TimeWindow(None, None, TimeRange(time(10, 15), time(13))),
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


@pytest.mark.parametrize("end", [time(13), time(12)])
def test_time_range_rejects_nonincreasing_times_with_values(end: time) -> None:
    with pytest.raises(ValueError) as error:
        TimeRange(time(13), end)
    assert "end" in str(error.value).lower()
    assert end.isoformat() in str(error.value)
    assert "13:00:00" in str(error.value)
    assert "after" in str(error.value)


@pytest.mark.parametrize(
    "start,end", [(time(13, tzinfo=OFFSET), None), (time(13), time(18, tzinfo=OFFSET))]
)
def test_time_range_rejects_zoned_times_with_values(
    start: time, end: time | None
) -> None:
    with pytest.raises(ValueError) as error:
        TimeRange(start, end)
    assert "+09:00" in str(error.value)
    assert "without a time zone" in str(error.value)


@pytest.mark.parametrize("end", [date(2026, 10, 4), date(2026, 10, 5)])
def test_date_range_rejects_nonincreasing_endpoints(end: date) -> None:
    """Reject date ranges whose end is not after their start."""
    with pytest.raises(ValueError, match="Date range end.*must be after start"):
        DateRange(date(2026, 10, 5), end)
