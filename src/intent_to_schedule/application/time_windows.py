from calendar import Day
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, time
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
            raise ValueError("Time range times must not have a time zone.")
        if self.end is not None and self.end <= self.start:
            raise ValueError(
                "Time range end must be after start; use null for the end of the day."
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
    windows: Sequence[TimeWindow], horizon: TimeInterval
) -> tuple[TimeInterval, ...]:
    """Times in the horizon covered by any window, merged into disjoint intervals."""
    raise NotImplementedError


def expand(
    windows: Sequence[TimeWindow], relation: TimeRelation, grid: TimeGrid
) -> Expansion:
    """Merge windows and round them to slots in the direction the relation needs."""
    raise NotImplementedError


def complement(
    intervals: Sequence[TimeInterval], horizon: TimeInterval
) -> tuple[TimeInterval, ...]:
    """Times in the horizon outside the intervals."""
    raise NotImplementedError
