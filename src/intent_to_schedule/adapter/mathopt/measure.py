from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta

from ortools.math_opt.python import mathopt

from intent_to_schedule.domain.measure import (
    AggregateMeasure,
    AggregateQuantity,
    DependencyMeasure,
    IntervalMeasure,
    Measure,
    PointMeasure,
)
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.calendar import TimeInterval
from intent_to_schedule.domain.task import Task, TaskId


@dataclass(frozen=True)
class PointExpression:
    """Placement choices for a Task start."""

    placements: Mapping[int, mathopt.Variable]


@dataclass(frozen=True)
class IntervalExpression:
    """Task occupancy for each slot."""

    occupancy: Mapping[int, mathopt.LinearExpression]


@dataclass(frozen=True)
class DependencyExpression:
    """Task gap and the presences that activate it."""

    gap: mathopt.LinearExpression
    from_presence: mathopt.Variable
    to_presence: mathopt.Variable
    from_duration: int


@dataclass(frozen=True)
class DailyVectorExpression:
    """Measured value for each calendar date."""

    values: Mapping[date, mathopt.LinearExpression]
    quantity: AggregateQuantity


type MeasureExpression = PointExpression | IntervalExpression | DependencyExpression | DailyVectorExpression
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
    slot: timedelta = problem.calendar.grid.slot
    horizon: TimeInterval = problem.calendar.grid.horizon
    n: int = (horizon.end - horizon.start) // slot
    tasks: dict[TaskId, Task] = {task.id: task for task in problem.tasks}

    match measure:
        case PointMeasure():
            return PointExpression(placements[measure.task_id])
        case IntervalMeasure():
            duration: int = tasks[measure.task_id].duration // slot
            occupancy: dict[int, mathopt.LinearExpression] = {
                k: mathopt.LinearSum(
                    variable for start, variable in placements[measure.task_id].items() if start <= k < start + duration
                )
                for k in range(n)
            }
            return IntervalExpression(occupancy)
        case DependencyMeasure():
            from_duration: int = tasks[measure.from_task_id].duration // slot
            gap: mathopt.LinearExpression = (
                starts[measure.to_task_id] - starts[measure.from_task_id]
                - from_duration * presences[measure.from_task_id]
            )
            return DependencyExpression(
                gap, presences[measure.from_task_id], presences[measure.to_task_id], from_duration
            )
        case AggregateMeasure():
            dates: set[date] = {(horizon.start + k * slot).date() for k in range(n)}
            values: dict[date, mathopt.LinearExpression] = {
                day: mathopt.LinearSum(
                    (1 if measure.quantity is AggregateQuantity.COUNT else tasks[task_id].duration // slot) * variable
                    for task_id in measure.task_ids
                    for start, variable in placements[task_id].items()
                    if (horizon.start + start * slot).date() == day
                )
                for day in dates
            }
            return DailyVectorExpression(values, measure.quantity)
        case _:
            raise ValueError("Unsupported measure")
