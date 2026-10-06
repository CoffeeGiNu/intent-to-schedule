from calendar import Day
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
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

    def includes(self, day: date) -> bool:
        """Whether a date lies in the range."""
        return self.start <= day < self.end


@dataclass(frozen=True)
class WholeHorizon:
    """Every date of the calendar horizon."""

    def includes(self, day: date) -> bool:
        """Whether a date lies in the range; every horizon date does."""
        return True


def format_time_of_day(value: timedelta) -> str:
    """Time since midnight written as HH:MM."""
    minutes: int = value // timedelta(minutes=1)
    return f"{minutes // 60:02}:{minutes % 60:02}"


@dataclass(frozen=True)
class TimeRange:
    """Times of day from start up to but not including end, as time since midnight."""

    start: timedelta
    end: timedelta

    def __post_init__(self) -> None:
        start: str = format_time_of_day(self.start)
        end: str = format_time_of_day(self.end)
        if self.start < timedelta(0) or self.end > timedelta(days=1):
            raise ValueError(
                f"Time range start {start} and end {end} must lie between 00:00 and 24:00."
            )
        if self.end <= self.start:
            raise ValueError(
                f"Time range end {end} must be after start {start}; "
                "split overnight ranges into separate windows."
            )


@dataclass(frozen=True)
class TimeWindow:
    """Times on the given dates and weekdays within a time of day."""

    date_range: DateRange | WholeHorizon
    weekdays: frozenset[Day]
    """Weekdays covered; an empty set covers no day."""
    time_range: TimeRange


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
    intervals: list[TimeInterval] = []
    window: TimeWindow
    day: date
    for window in windows:
        for day in grid.dates:
            if window.date_range.includes(day) and Day(day.weekday()) in window.weekdays:
                midnight: datetime = datetime.combine(day, time.min, grid.starting_offset)
                intervals.append(
                    TimeInterval(
                        midnight + window.time_range.start,
                        midnight + window.time_range.end,
                    )
                )
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
