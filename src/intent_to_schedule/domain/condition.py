from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import Enum
from typing import Self

from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.evaluation import (
    Distance,
    Evaluation,
    Excess,
    Intrusion,
    Shortfall,
)
from intent_to_schedule.domain.measure import (
    AggregateMeasure,
    AggregateQuantity,
    Boundary,
    DependencyMeasure,
    IntervalMeasure,
    Measure,
    PointMeasure,
)
from intent_to_schedule.domain.task import TaskId
from intent_to_schedule.domain.time_windows import (
    Expansion,
    TimeRelation,
    TimeWindow,
    complement,
    expand,
)


@dataclass(frozen=True)
class Criterion:
    """Measure and evaluation of a condition's violation."""

    measure: Measure
    evaluation: Evaluation


@dataclass(frozen=True)
class TimeWindowCondition:
    """Tasks kept within or away from time windows."""

    task_ids: frozenset[TaskId]
    relation: TimeRelation
    windows: tuple[TimeWindow, ...]

    def __post_init__(self) -> None:
        if not self.task_ids:
            raise ValueError("TimeWindowCondition must reference at least one Task.")

    def without_task(self, task_id: TaskId) -> Self | None:
        """Remove a Task and return the remaining condition."""
        task_ids: frozenset[TaskId] = self.task_ids - {task_id}
        return replace(self, task_ids=task_ids) if task_ids else None

    def criteria(self, grid: TimeGrid) -> tuple[Criterion, ...]:
        """Translate the time windows into a violation criterion."""
        expansion: Expansion = expand(self.windows, self.relation, grid)
        region: tuple[TimeInterval, ...] = (
            complement(expansion.intervals, grid.horizon)
            if self.relation is TimeRelation.WITHIN
            else expansion.intervals
        )
        return (Criterion(IntervalMeasure(self.task_ids), Intrusion(region)),)


class TimeBoundRelation(Enum):
    """How a Task boundary relates to a time."""

    AT_OR_BEFORE = "at_or_before"
    AT_OR_AFTER = "at_or_after"
    AT = "at"


@dataclass(frozen=True)
class TimeBoundCondition:
    """Task starts or ends compared with a time."""

    task_ids: frozenset[TaskId]
    boundary: Boundary
    relation: TimeBoundRelation
    at: datetime

    def __post_init__(self) -> None:
        if not self.task_ids:
            raise ValueError("TimeBoundCondition must reference at least one Task.")

    def without_task(self, task_id: TaskId) -> Self | None:
        """Remove a Task and return the remaining condition."""
        task_ids: frozenset[TaskId] = self.task_ids - {task_id}
        return replace(self, task_ids=task_ids) if task_ids else None

    def criteria(self, grid: TimeGrid) -> tuple[Criterion, ...]:
        """Translate each Task boundary into a violation criterion."""
        del grid
        evaluation: Evaluation
        match self.relation:
            case TimeBoundRelation.AT_OR_BEFORE:
                evaluation = Excess(self.at)
            case TimeBoundRelation.AT_OR_AFTER:
                evaluation = Shortfall(self.at)
            case TimeBoundRelation.AT:
                evaluation = Distance(self.at)
        return tuple(
            Criterion(PointMeasure(task_id, self.boundary), evaluation)
            for task_id in sorted(self.task_ids, key=lambda item: item.value)
        )


class TaskGapRelation(Enum):
    """How the gap between Tasks relates to a duration."""

    AT_LEAST = "at_least"
    EXACTLY = "exactly"


@dataclass(frozen=True)
class TaskGapCondition:
    """Minimum or exact gap between two Tasks."""

    from_task_id: TaskId
    to_task_id: TaskId
    relation: TaskGapRelation
    gap: timedelta

    def __post_init__(self) -> None:
        if self.gap < timedelta(0):
            raise ValueError("Task gap must be a non-negative duration.")

    @property
    def task_ids(self) -> frozenset[TaskId]:
        """Tasks at either end of the gap."""
        return frozenset({self.from_task_id, self.to_task_id})

    def without_task(self, task_id: TaskId) -> Self | None:
        """Return the condition unless either Task is removed."""
        return self if task_id not in self.task_ids else None

    def criteria(self, grid: TimeGrid) -> tuple[Criterion, ...]:
        """Translate the Task gap into a violation criterion."""
        del grid
        evaluation: Evaluation = (
            Shortfall(self.gap)
            if self.relation is TaskGapRelation.AT_LEAST
            else Distance(self.gap)
        )
        return (
            Criterion(
                DependencyMeasure(self.from_task_id, self.to_task_id), evaluation
            ),
        )


@dataclass(frozen=True)
class DailyLimitCondition:
    """Maximum daily count or total duration of Tasks."""

    task_ids: frozenset[TaskId]
    quantity: AggregateQuantity
    maximum: int | timedelta

    def __post_init__(self) -> None:
        if not self.task_ids:
            raise ValueError("DailyLimitCondition must reference at least one Task.")
        if self.quantity is AggregateQuantity.COUNT:
            if (
                not isinstance(self.maximum, int)
                or isinstance(self.maximum, bool)
                or self.maximum < 0
            ):
                raise ValueError("Daily count maximum must be a non-negative integer.")
        elif not isinstance(self.maximum, timedelta) or self.maximum < timedelta(0):
            raise ValueError("Daily duration maximum must be a non-negative duration.")

    def without_task(self, task_id: TaskId) -> Self | None:
        """Remove a Task and return the remaining condition."""
        task_ids: frozenset[TaskId] = self.task_ids - {task_id}
        return replace(self, task_ids=task_ids) if task_ids else None

    def criteria(self, grid: TimeGrid) -> tuple[Criterion, ...]:
        """Translate the daily maximum into a violation criterion."""
        del grid
        return (
            Criterion(
                AggregateMeasure(self.task_ids, self.quantity), Excess(self.maximum)
            ),
        )


type Condition = (
    TimeWindowCondition | TimeBoundCondition | TaskGapCondition | DailyLimitCondition
)
"""Scheduling requirement translated into violation criteria."""
