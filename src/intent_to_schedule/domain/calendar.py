from dataclasses import dataclass
from datetime import datetime, timedelta

from intent_to_schedule.domain.person import PersonId


@dataclass(frozen=True)
class TimeInterval:
    """Half-open time interval from start to end."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(
                f"Time interval start {self.start.isoformat()} must not be "
                f"after end {self.end.isoformat()}."
            )


@dataclass(frozen=True)
class TimeGrid:
    """Planning horizon divided into slots."""

    horizon: TimeInterval
    slot: timedelta

    def __post_init__(self) -> None:
        if self.slot <= timedelta(0):
            raise ValueError("TimeGrid slot must be positive.")
        if self.horizon.start >= self.horizon.end:
            raise ValueError("TimeGrid horizon start must precede its end.")
        if not self.is_aligned(self.horizon.end):
            raise ValueError("TimeGrid horizon end must be aligned to the time grid.")

    @property
    def slot_count(self) -> int:
        """Number of slots in the horizon."""
        return self.index_of(self.horizon.end)

    def index_of(self, at: datetime) -> int:
        """Slot index of a time on a slot boundary."""
        if not self.is_aligned(at):
            raise ValueError(f"Time {at.isoformat()} is not aligned to the time grid.")
        return (at - self.horizon.start) // self.slot

    def time_at(self, index: int) -> datetime:
        """Start time of a slot index."""
        return self.horizon.start + index * self.slot

    def round_outward(self, interval: TimeInterval) -> TimeInterval:
        """Smallest slot-aligned interval that contains the interval."""
        first: int = (interval.start - self.horizon.start) // self.slot
        if interval.start == interval.end:
            return TimeInterval(self.time_at(first), self.time_at(first))
        last: int = -((self.horizon.start - interval.end) // self.slot)
        return TimeInterval(self.time_at(first), self.time_at(last))

    def round_inward(self, interval: TimeInterval) -> TimeInterval | None:
        """Largest slot-aligned interval inside the interval, or None if none fits."""
        first: int = -((self.horizon.start - interval.start) // self.slot)
        last: int = (interval.end - self.horizon.start) // self.slot
        if first >= last:
            return None
        return TimeInterval(self.time_at(first), self.time_at(last))

    def is_aligned(self, at: datetime) -> bool:
        """Whether a time falls on a slot boundary."""
        return (at - self.horizon.start) % self.slot == timedelta(0)


@dataclass(frozen=True)
class Availability:
    """Times when a person can take Tasks."""

    person_id: PersonId
    intervals: tuple[TimeInterval, ...]


@dataclass(frozen=True)
class Calendar:
    """Time grid and availabilities."""

    grid: TimeGrid
    availabilities: tuple[Availability, ...]
