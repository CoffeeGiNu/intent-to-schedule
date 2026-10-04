from dataclasses import dataclass
from datetime import datetime

from intent_to_schedule.domain.task import TaskId


@dataclass(frozen=True)
class ScheduledTask:
    """Task name and scheduled interval recorded at solve time."""

    task_id: TaskId
    name: str
    start: datetime
    end: datetime


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
