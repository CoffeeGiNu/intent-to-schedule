from dataclasses import dataclass
from enum import Enum

from intent_to_schedule.domain.task import TaskId


@dataclass(frozen=True)
class PointMeasure:
    """Start time of a Task."""

    task_id: TaskId

    @property
    def task_ids(self) -> frozenset[TaskId]:
        return frozenset({self.task_id})


@dataclass(frozen=True)
class IntervalMeasure:
    """Time span occupied by a Task."""

    task_id: TaskId

    @property
    def task_ids(self) -> frozenset[TaskId]:
        return frozenset({self.task_id})


@dataclass(frozen=True)
class DependencyMeasure:
    """Gap from the end of one Task to the start of another."""

    from_task_id: TaskId
    to_task_id: TaskId

    @property
    def task_ids(self) -> frozenset[TaskId]:
        return frozenset({self.from_task_id, self.to_task_id})


class AggregateQuantity(Enum):
    """What an AggregateMeasure aggregates: number of Tasks or total duration."""

    COUNT = "count"
    TOTAL_DURATION = "total_duration"


@dataclass(frozen=True)
class AggregateMeasure:
    """Daily count or total duration of Tasks."""

    task_ids: frozenset[TaskId]
    quantity: AggregateQuantity


type Measure = PointMeasure | IntervalMeasure | DependencyMeasure | AggregateMeasure
"""Value measured from a schedule."""
