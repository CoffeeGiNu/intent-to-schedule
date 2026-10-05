from dataclasses import dataclass
from datetime import datetime

from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.task import TaskId


@dataclass(frozen=True)
class ScheduledTask:
    """Task name, scheduled interval, and participants recorded at solve time."""

    task_id: TaskId
    name: str
    start: datetime
    end: datetime
    participant_ids: frozenset[PersonId]


@dataclass(frozen=True)
class DroppedTask:
    """Dropped task name recorded at solve time."""

    task_id: TaskId
    name: str


@dataclass(frozen=True)
class Schedule:
    """Scheduled and dropped Tasks."""

    scheduled: tuple[ScheduledTask, ...]
    dropped: tuple[DroppedTask, ...]
