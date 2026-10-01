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
