from dataclasses import dataclass
from datetime import datetime

from intent_to_schedule.domain.task import TaskId


@dataclass(frozen=True)
class ScheduledTask:
    """Task scheduled at a start time."""

    task_id: TaskId
    start: datetime


@dataclass(frozen=True)
class Schedule:
    """Scheduled and dropped Tasks."""

    scheduled: tuple[ScheduledTask, ...]
    dropped_task_ids: frozenset[TaskId]
