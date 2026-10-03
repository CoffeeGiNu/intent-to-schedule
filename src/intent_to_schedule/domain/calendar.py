from dataclasses import dataclass
from datetime import datetime, timedelta

from intent_to_schedule.domain.person import PersonId


@dataclass(frozen=True)
class TimeInterval:
    """Half-open time interval from start to end."""

    start: datetime
    end: datetime


@dataclass(frozen=True)
class TimeGrid:
    """Planning horizon divided into slots."""

    horizon: TimeInterval
    slot: timedelta

    @property
    def slot_count(self) -> int:
        """Number of slots in the horizon."""
        raise NotImplementedError

    def index_of(self, at: datetime) -> int:
        """Slot index of a time on a slot boundary."""
        raise NotImplementedError

    def time_at(self, index: int) -> datetime:
        """Start time of a slot index."""
        raise NotImplementedError

    def round_outward(self, interval: TimeInterval) -> TimeInterval:
        """Smallest slot-aligned interval that contains the interval."""
        raise NotImplementedError

    def round_inward(self, interval: TimeInterval) -> TimeInterval | None:
        """Largest slot-aligned interval inside the interval, or None if none fits."""
        raise NotImplementedError

    def is_aligned(self, at: datetime) -> bool:
        """Whether a time falls on a slot boundary."""
        raise NotImplementedError


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
