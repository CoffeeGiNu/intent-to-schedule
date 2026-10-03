from dataclasses import dataclass, replace
from enum import Enum
from typing import Self

from intent_to_schedule.domain.task import TaskId


@dataclass(frozen=True)
class PointMeasure:
    """Start time of a Task."""

    task_id: TaskId

    @property
    def task_ids(self) -> frozenset[TaskId]:
        return frozenset({self.task_id})

    def without_task(self, task_id: TaskId) -> Self | None:
        """Measure left after removing a Task, or None if it no longer makes sense."""
        return self if task_id not in self.task_ids else None


@dataclass(frozen=True)
class IntervalMeasure:
    """Time spans occupied by a non-empty set of Tasks."""

    task_ids: frozenset[TaskId]

    def __post_init__(self) -> None:
        if not self.task_ids:
            raise ValueError("IntervalMeasure must reference at least one Task.")

    def without_task(self, task_id: TaskId) -> Self | None:
        """Remove a Task and return the remaining measure."""
        task_ids: frozenset[TaskId] = self.task_ids - {task_id}
        return replace(self, task_ids=task_ids) if task_ids else None


@dataclass(frozen=True)
class DependencyMeasure:
    """Gap from the end of one Task to the start of another."""

    from_task_id: TaskId
    to_task_id: TaskId

    @property
    def task_ids(self) -> frozenset[TaskId]:
        return frozenset({self.from_task_id, self.to_task_id})

    def without_task(self, task_id: TaskId) -> Self | None:
        """Measure left after removing a Task, or None if it no longer makes sense."""
        return self if task_id not in self.task_ids else None


class AggregateQuantity(Enum):
    """What an AggregateMeasure aggregates: number of Tasks or total duration."""

    COUNT = "count"
    TOTAL_DURATION = "total_duration"


@dataclass(frozen=True)
class AggregateMeasure:
    """Daily count or total duration of Tasks."""

    task_ids: frozenset[TaskId]
    quantity: AggregateQuantity

    def without_task(self, task_id: TaskId) -> Self | None:
        """Measure left after removing a Task, or None if it no longer makes sense."""
        task_ids: frozenset[TaskId] = self.task_ids - {task_id}
        return replace(self, task_ids=task_ids) if task_ids else None


type Measure = PointMeasure | IntervalMeasure | DependencyMeasure | AggregateMeasure
"""Value measured from a schedule."""
