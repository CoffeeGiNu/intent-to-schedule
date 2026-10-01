import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Self

from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.strength import Strength


@dataclass(frozen=True)
class TaskId:
    """Identifier of a Task."""

    value: str

    @classmethod
    def generate(cls) -> Self:
        """Create a new unique TaskId."""
        return cls(uuid.uuid4().hex)


class Importance(Enum):
    """Importance of a Task."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class Task:
    """Unit of work to be scheduled in time."""

    id: TaskId
    name: str
    duration: timedelta
    participant_ids: frozenset[PersonId]
    importance: Importance
    required: bool
    """Whether the Task must be scheduled."""
    stability: Strength = Strength.NORMAL
    """Strength of the preference to keep the Task at its previous start."""


@dataclass(frozen=True)
class FixedTask:
    """Existing event whose start is fixed."""

    id: TaskId
    name: str
    start: datetime
    duration: timedelta
    participant_ids: frozenset[PersonId]
