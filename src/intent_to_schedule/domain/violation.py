"""Violation measurements from task placements."""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Literal

from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import Criterion
from intent_to_schedule.domain.evaluation import (
    Distance,
    Excess,
    Intrusion,
    Quantity,
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

type ViolationUnit = Literal["hours", "count"]
"""Unit of a measured violation."""


@dataclass(frozen=True)
class TaskViolationPart:
    """Violation attributed to a task."""

    amount: float
    task_id: TaskId


@dataclass(frozen=True)
class DateViolationPart:
    """Violation attributed to a calendar date."""

    amount: float
    calendar_date: date


type ViolationPart = TaskViolationPart | DateViolationPart
"""Violation attributed to a task or calendar date."""


@dataclass(frozen=True)
class CriterionViolation:
    """Violation amount, unit, and attribution of a criterion."""

    amount: float
    unit: ViolationUnit
    breakdown: tuple[ViolationPart, ...]


def _difference(
    value: datetime | timedelta | int, target: datetime | timedelta | int
) -> float:
    """Measure a difference in hours or counts."""
    match value, target:
        case datetime(), datetime():
            return (value - target) / timedelta(hours=1)
        case timedelta(), timedelta():
            return (value - target) / timedelta(hours=1)
        case int(), int():
            return float(value - target)
        case _:
            raise ValueError("Unsupported evaluation quantities")


def _evaluate(value: datetime | timedelta | int, criterion: Criterion) -> float:
    """Measure a scalar criterion's violation."""
    target: Quantity
    upper: Quantity
    lower: Quantity
    match criterion.evaluation:
        case Distance(target=target):
            return abs(_difference(value, target))
        case Excess(upper=upper):
            return max(_difference(value, upper), 0.0)
        case Shortfall(lower=lower):
            return max(-_difference(value, lower), 0.0)
        case _:
            raise ValueError("Unsupported scalar evaluation")


def measure_criterion(
    criterion: Criterion,
    placements: Mapping[TaskId, TimeInterval],
    grid: TimeGrid,
) -> CriterionViolation:
    """Measure a criterion against scheduled task intervals."""
    measure: Measure = criterion.measure
    parts: list[ViolationPart] = []
    task_id: TaskId
    interval: TimeInterval | None
    amount: float
    match measure:
        case PointMeasure():
            interval = placements.get(measure.task_id)
            amount = (
                _evaluate(
                    interval.start
                    if measure.boundary is Boundary.START
                    else interval.end,
                    criterion,
                )
                if interval is not None
                else 0.0
            )
            parts.append(TaskViolationPart(amount, measure.task_id))
        case IntervalMeasure():
            if not isinstance(criterion.evaluation, Intrusion):
                raise ValueError("Unsupported interval evaluation")
            region_indices: set[int] = {
                slot
                for region in criterion.evaluation.region
                for slot in grid.slots_within(region)
            }
            region_slots: tuple[TimeInterval, ...] = tuple(
                TimeInterval(grid.time_at(slot), grid.time_at(slot + 1))
                for slot in sorted(region_indices)
            )
            for task_id in sorted(measure.task_ids, key=lambda item: item.value):
                interval = placements.get(task_id)
                amount = (
                    sum(
                        interval.overlap(slot) / timedelta(hours=1)
                        for slot in region_slots
                    )
                    if interval is not None
                    else 0.0
                )
                parts.append(TaskViolationPart(amount, task_id))
        case DependencyMeasure():
            from_interval: TimeInterval | None = placements.get(measure.from_task_id)
            to_interval: TimeInterval | None = placements.get(measure.to_task_id)
            amount = (
                _evaluate(to_interval.start - from_interval.end, criterion)
                if from_interval is not None and to_interval is not None
                else 0.0
            )
            return CriterionViolation(amount, "hours", ())
        case AggregateMeasure():
            dates: tuple[date, ...] = grid.dates
            totals: dict[date, int | timedelta] = {
                day: 0 if measure.quantity is AggregateQuantity.COUNT else timedelta(0)
                for day in dates
            }
            for task_id in measure.task_ids:
                interval = placements.get(task_id)
                if interval is None:
                    continue
                day: date = grid.date_of(interval.start)
                if day not in totals:
                    continue
                current: int | timedelta = totals[day]
                if isinstance(current, int):
                    totals[day] = current + 1
                else:
                    totals[day] = current + interval.duration
            parts = [
                DateViolationPart(_evaluate(totals[day], criterion), day)
                for day in dates
            ]
    unit: ViolationUnit = (
        "count"
        if isinstance(measure, AggregateMeasure)
        and measure.quantity is AggregateQuantity.COUNT
        else "hours"
    )
    return CriterionViolation(sum(part.amount for part in parts), unit, tuple(parts))
