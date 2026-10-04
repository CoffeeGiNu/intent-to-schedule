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
class ViolationPart:
    """Violation attributed to a task or calendar date."""

    amount: float
    task_id: TaskId | None = None
    calendar_date: date | None = None


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
            parts.append(ViolationPart(amount, task_id=measure.task_id))
        case IntervalMeasure():
            if not isinstance(criterion.evaluation, Intrusion):
                raise ValueError("Unsupported interval evaluation")
            region_slots: tuple[TimeInterval, ...] = tuple(
                TimeInterval(grid.time_at(slot), grid.time_at(slot + 1))
                for slot in range(grid.slot_count)
                if any(
                    region.start <= grid.time_at(slot)
                    and grid.time_at(slot + 1) <= region.end
                    for region in criterion.evaluation.region
                )
            )
            for task_id in sorted(measure.task_ids, key=lambda item: item.value):
                interval = placements.get(task_id)
                amount = (
                    sum(
                        max(
                            min(interval.end, slot.end)
                            - max(interval.start, slot.start),
                            timedelta(0),
                        )
                        / timedelta(hours=1)
                        for slot in region_slots
                    )
                    if interval is not None
                    else 0.0
                )
                parts.append(ViolationPart(amount, task_id=task_id))
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
            dates: set[date] = {
                grid.time_at(slot).date() for slot in range(grid.slot_count)
            }
            totals: dict[date, int | timedelta] = {
                day: 0 if measure.quantity is AggregateQuantity.COUNT else timedelta(0)
                for day in dates
            }
            for task_id in measure.task_ids:
                interval = placements.get(task_id)
                if interval is None:
                    continue
                day: date = interval.start.astimezone(grid.horizon.start.tzinfo).date()
                if day not in totals:
                    continue
                current: int | timedelta = totals[day]
                if isinstance(current, int):
                    totals[day] = current + 1
                else:
                    totals[day] = current + interval.end - interval.start
            parts = [
                ViolationPart(_evaluate(totals[day], criterion), calendar_date=day)
                for day in sorted(dates)
            ]
    unit: ViolationUnit = (
        "count"
        if isinstance(measure, AggregateMeasure)
        and measure.quantity is AggregateQuantity.COUNT
        else "hours"
    )
    return CriterionViolation(sum(part.amount for part in parts), unit, tuple(parts))
