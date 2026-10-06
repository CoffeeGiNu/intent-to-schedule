from calendar import Day
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum

from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval


class TimeRelation(Enum):
    """How a Task relates to time windows."""

    WITHIN = "within"
    AVOID = "avoid"


@dataclass(frozen=True)
class DateRange:
    """Dates from start up to but not including end."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError(
                f"Date range end {self.end.isoformat()} must be after start "
                f"{self.start.isoformat()}."
            )


@dataclass(frozen=True)
class TimeRange:
    """Times of day from start up to but not including end."""

    start: time
    end: time | None
    """End time, or None for the end of the day."""

    def __post_init__(self) -> None:
        if self.start.tzinfo is not None or (
            self.end is not None and self.end.tzinfo is not None
        ):
            raise ValueError(
                f"Time range start {self.start.isoformat()} and end "
                f"{self.end.isoformat() if self.end is not None else 'null'} "
                "must be times without a time zone."
            )
        if self.end is not None and self.end <= self.start:
            raise ValueError(
                f"Time range end {self.end.isoformat()} must be after start "
                f"{self.start.isoformat()}; use null for the end of the day."
            )


@dataclass(frozen=True)
class TimeWindow:
    """Times on the given dates and weekdays within a time of day."""

    date_range: DateRange | None
    """Dates covered, or None for the whole horizon."""
    weekdays: frozenset[Day] | None
    """Weekdays covered, or None for every day."""
    time_range: TimeRange | None
    """Time of day covered, or None for the whole day."""


@dataclass(frozen=True)
class Expansion:
    """Slot-aligned times covered by time windows."""

    intervals: tuple[TimeInterval, ...]
    rounded: bool
    """Whether rounding to slots changed the times."""


def window_times(
    windows: Sequence[TimeWindow], grid: TimeGrid
) -> tuple[TimeInterval, ...]:
    """Times in the horizon covered by any window, merged into disjoint intervals."""
    calendar_timezone: timezone = grid.starting_offset
    intervals: list[TimeInterval] = []
    window: TimeWindow
    for window in windows:
        current: date = grid.date_of(grid.horizon.start)
        if window.date_range is not None:
            current = max(current, window.date_range.start)
        while current <= grid.date_of(grid.horizon.end):
            if window.date_range is not None and current >= window.date_range.end:
                break
            following: date = current + timedelta(days=1)
            if window.weekdays is None or Day(current.weekday()) in window.weekdays:
                start: datetime = datetime.combine(
                    current,
                    window.time_range.start
                    if window.time_range is not None
                    else time.min,
                    calendar_timezone,
                )
                end: datetime = (
                    datetime.combine(current, window.time_range.end, calendar_timezone)
                    if window.time_range is not None
                    and window.time_range.end is not None
                    else datetime.combine(following, time.min, calendar_timezone)
                )
                intervals.append(TimeInterval(start, end))
            current = following
    return _merge(intervals, grid.horizon)


def expand(
    windows: Sequence[TimeWindow], relation: TimeRelation, grid: TimeGrid
) -> Expansion:
    """Merge windows and round them to slots in the direction the relation needs."""
    times: tuple[TimeInterval, ...] = window_times(windows, grid)
    intervals: list[TimeInterval] = []
    interval: TimeInterval
    for interval in times:
        slots: range = (
            grid.slots_within(interval)
            if relation is TimeRelation.WITHIN
            else grid.slots_touching(interval)
        )
        if slots:
            intervals.append(grid.interval_of(slots))
    merged: tuple[TimeInterval, ...] = _merge(intervals, grid.horizon)
    return Expansion(merged, merged != times)


def complement(
    intervals: Sequence[TimeInterval], horizon: TimeInterval
) -> tuple[TimeInterval, ...]:
    """Times in the horizon outside the intervals."""
    merged: tuple[TimeInterval, ...] = _merge(intervals, horizon)
    gaps: list[TimeInterval] = []
    start: datetime = horizon.start
    interval: TimeInterval
    for interval in merged:
        if start < interval.start:
            gaps.append(TimeInterval(start, interval.start))
        start = interval.end
    if start < horizon.end:
        gaps.append(TimeInterval(start, horizon.end))
    return tuple(gaps)


def _merge(
    intervals: Sequence[TimeInterval], horizon: TimeInterval
) -> tuple[TimeInterval, ...]:
    """Clip intervals to the horizon and merge overlapping or adjacent times."""
    merged: list[TimeInterval] = []
    interval: TimeInterval
    for interval in sorted(intervals, key=lambda item: item.start):
        start: datetime = max(interval.start, horizon.start)
        end: datetime = min(interval.end, horizon.end)
        if start >= end:
            continue
        if merged and start <= merged[-1].end:
            merged[-1] = TimeInterval(merged[-1].start, max(merged[-1].end, end))
        else:
            merged.append(TimeInterval(start, end))
    return tuple(merged)
