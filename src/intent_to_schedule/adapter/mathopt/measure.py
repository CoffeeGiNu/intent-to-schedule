from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from ortools.math_opt.python import mathopt

from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.measure import (
    AggregateMeasure,
    AggregateQuantity,
    Boundary,
    DependencyMeasure,
    IntervalMeasure,
    Measure,
    PointMeasure,
)
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import TaskId


@dataclass(frozen=True)
class PointExpression:
    """Placement choices and the offset of a task boundary."""

    placements: Mapping[int, mathopt.Variable]
    offset: timedelta = timedelta(0)


@dataclass(frozen=True)
class ConstantPointExpression:
    """Real constant time of a fixed task boundary."""

    value: datetime


@dataclass(frozen=True)
class IntervalExpression:
    """Variable and constant task occupancy for each slot."""

    occupancy: Mapping[int, mathopt.LinearBase | float]


@dataclass(frozen=True)
class DependencyExpression:
    """Task gap and the presences that activate it."""

    gap: mathopt.LinearBase | float
    from_presence: mathopt.Variable | float
    to_presence: mathopt.Variable | float
    bound: float


@dataclass(frozen=True)
class DailyVectorExpression:
    """Measured value for each calendar date."""

    values: Mapping[date, mathopt.LinearBase | float]
    quantity: AggregateQuantity


type MeasureExpression = (
    PointExpression
    | ConstantPointExpression
    | IntervalExpression
    | DependencyExpression
    | DailyVectorExpression
)
"""Measure built as MathOpt expressions."""


def compile_measure(
    measure: Measure,
    problem: SchedulingProblem,
    model: mathopt.Model,
    starts: Mapping[TaskId, mathopt.Variable],
    presences: Mapping[TaskId, mathopt.Variable],
    placements: Mapping[TaskId, Mapping[int, mathopt.Variable]],
) -> MeasureExpression:
    """Build a measure expression from a Measure."""
    del model
    grid: TimeGrid = problem.calendar.grid
    durations: dict[TaskId, int] = {
        task.id: grid.slots_of(task.duration) for task in problem.tasks
    }
    fixed_intervals: dict[TaskId, TimeInterval] = {
        task.id: task.interval for task in problem.fixed_tasks
    }

    match measure:
        case PointMeasure():
            fixed: TimeInterval | None = fixed_intervals.get(measure.task_id)
            if fixed is not None:
                return ConstantPointExpression(
                    fixed.end if measure.boundary is Boundary.END else fixed.start
                )
            offset: timedelta = (
                durations[measure.task_id] * grid.slot
                if measure.boundary is Boundary.END
                else timedelta(0)
            )
            return PointExpression(placements[measure.task_id], offset)
        case IntervalMeasure():
            occupied: dict[int, list[mathopt.Variable]] = {}
            task_id: TaskId
            duration: int
            start: int
            variable: mathopt.Variable
            slot_index: int
            for task_id in sorted(measure.task_ids, key=lambda item: item.value):
                if task_id in fixed_intervals:
                    continue
                duration = durations[task_id]
                for start, variable in placements[task_id].items():
                    for slot_index in range(start, start + duration):
                        occupied.setdefault(slot_index, []).append(variable)
            occupancy: dict[int, mathopt.LinearBase | float] = {
                slot_index: mathopt.LinearSum(occupied.get(slot_index, ()))
                + sum(
                    fixed_intervals[task_id].overlap(
                        TimeInterval(
                            grid.time_at(slot_index), grid.time_at(slot_index + 1)
                        )
                    )
                    / grid.slot
                    for task_id in measure.task_ids
                    if task_id in fixed_intervals
                )
                for slot_index in range(grid.slot_count)
            }
            return IntervalExpression(occupancy)
        case DependencyMeasure():
            from_fixed: TimeInterval | None = fixed_intervals.get(measure.from_task_id)
            to_fixed: TimeInterval | None = fixed_intervals.get(measure.to_task_id)
            from_end: mathopt.LinearBase | float
            from_presence: mathopt.Variable | float
            from_lower: float
            from_upper: float
            if from_fixed is not None:
                from_end = (from_fixed.end - grid.horizon.start) / grid.slot
                from_presence = 1.0
                from_lower = from_upper = from_end
            else:
                from_start: mathopt.Variable = starts[measure.from_task_id]
                from_duration: int = durations[measure.from_task_id]
                from_presence = presences[measure.from_task_id]
                from_end = from_start + from_duration * from_presence
                from_lower = (
                    from_start.lower_bound + from_duration * from_presence.lower_bound
                )
                from_upper = (
                    from_start.upper_bound + from_duration * from_presence.upper_bound
                )
            to_start: mathopt.Variable | float
            to_presence: mathopt.Variable | float
            to_lower: float
            to_upper: float
            if to_fixed is not None:
                to_start = (to_fixed.start - grid.horizon.start) / grid.slot
                to_presence = 1.0
                to_lower = to_upper = to_start
            else:
                to_start = starts[measure.to_task_id]
                to_presence = presences[measure.to_task_id]
                to_lower = to_start.lower_bound
                to_upper = to_start.upper_bound
            gap: mathopt.LinearBase | float = to_start - from_end
            bound: float = max(
                abs(to_lower - from_upper),
                abs(to_upper - from_lower),
            )
            return DependencyExpression(gap, from_presence, to_presence, bound)
        case AggregateMeasure():
            dates: tuple[date, ...] = grid.dates
            values: dict[date, mathopt.LinearBase | float] = {
                day: mathopt.LinearSum(
                    (
                        1
                        if measure.quantity is AggregateQuantity.COUNT
                        else durations[task_id]
                    )
                    * variable
                    for task_id in measure.task_ids
                    if task_id not in fixed_intervals
                    for start, variable in placements[task_id].items()
                    if grid.date_of(grid.time_at(start)) == day
                )
                + sum(
                    1.0
                    if measure.quantity is AggregateQuantity.COUNT
                    else fixed_intervals[task_id].duration / grid.slot
                    for task_id in measure.task_ids
                    if task_id in fixed_intervals
                    and grid.date_of(fixed_intervals[task_id].start) == day
                )
                for day in dates
            }
            return DailyVectorExpression(values, measure.quantity)
        case _:
            raise ValueError("Unsupported measure")
