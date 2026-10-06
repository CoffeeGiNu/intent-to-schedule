from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone

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

    @property
    def duration(self) -> timedelta:
        """Length of the interval."""
        return self.end - self.start

    def contains(self, other: "TimeInterval") -> bool:
        """Whether another interval lies entirely inside this interval."""
        return other.start >= self.start and other.end <= self.end

    def includes(self, at: datetime) -> bool:
        """Whether a time lies inside this half-open interval."""
        return self.start <= at < self.end

    def overlap(self, other: "TimeInterval") -> timedelta:
        """Length of the overlap with another interval."""
        return max(
            min(self.end, other.end) - max(self.start, other.start), timedelta(0)
        )


# TODO: decide how to handle daylight saving time. Datetimes sharing a zone subtract by
# wall clock, so the solver and the plain evaluation can disagree; consider detecting
# such zones and asking the user back.
@dataclass(frozen=True)
class TimeGrid:
    """Planning horizon divided into slots."""

    horizon: TimeInterval
    slot: timedelta

    def __post_init__(self) -> None:
        if self.slot <= timedelta(0):
            raise ValueError("TimeGrid slot must be positive.")
        if self.horizon.start.utcoffset() is None:
            raise ValueError("TimeGrid horizon must have an offset.")
        if self.horizon.start >= self.horizon.end:
            raise ValueError("TimeGrid horizon start must precede its end.")
        if not self.is_aligned(self.horizon.end):
            raise ValueError("TimeGrid horizon end must be aligned to the time grid.")

    @property
    def slot_count(self) -> int:
        """Number of slots in the horizon."""
        return self.slots_of(self.horizon.duration)

    def slots_of(self, duration: timedelta) -> int:
        """Number of whole slots in a duration."""
        if not self.is_whole_slots(duration):
            raise ValueError("Duration must be a multiple of the time grid slot.")
        return duration // self.slot

    def is_whole_slots(self, duration: timedelta) -> bool:
        """Whether a duration is a whole number of slots."""
        return duration % self.slot == timedelta(0)

    def slots_within(self, interval: TimeInterval) -> range:
        """Slots lying entirely inside the interval and horizon."""
        first: int = max(0, -((self.horizon.start - interval.start) // self.slot))
        last: int = min(
            self.slot_count, (interval.end - self.horizon.start) // self.slot
        )
        return range(first, last) if first < last else range(0)

    def slots_touching(self, interval: TimeInterval) -> range:
        """Slots in the horizon that overlap the interval."""
        if interval.duration == timedelta(0):
            return range(0)
        first: int = max(0, (interval.start - self.horizon.start) // self.slot)
        last: int = min(
            self.slot_count, -((self.horizon.start - interval.end) // self.slot)
        )
        return range(first, last) if first < last else range(0)

    def interval_of(self, slots: range) -> TimeInterval:
        """Time covered by a non-empty range of slots."""
        if not slots:
            raise ValueError("Slot range must not be empty.")
        return TimeInterval(self.time_at(slots.start), self.time_at(slots.stop))

    @property
    def starting_offset(self) -> timezone:
        """Fixed offset of the horizon start."""
        offset: timedelta | None = self.horizon.start.utcoffset()
        assert offset is not None
        return timezone(offset)

    def date_of(self, at: datetime) -> date:
        """Calendar date of a time in the horizon's starting offset."""
        return at.astimezone(self.starting_offset).date()

    @property
    def dates(self) -> tuple[date, ...]:
        """Sorted calendar dates intersecting the half-open horizon."""
        first: date = self.date_of(self.horizon.start)
        last: date = self.date_of(self.horizon.end - datetime.resolution)
        return tuple(
            first + timedelta(days=index) for index in range((last - first).days + 1)
        )

    def time_at(self, index: int) -> datetime:
        """Start time of a slot index."""
        return self.horizon.start + index * self.slot

    def is_aligned(self, at: datetime) -> bool:
        """Whether a time falls on a slot boundary."""
        return self.is_whole_slots(at - self.horizon.start)


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
