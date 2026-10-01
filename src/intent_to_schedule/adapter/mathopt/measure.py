from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta

from ortools.math_opt.python import mathopt

from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.measure import (
    AggregateMeasure,
    AggregateQuantity,
    DependencyMeasure,
    IntervalMeasure,
    Measure,
    PointMeasure,
)
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import FixedTask, TaskId


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
    bound: float


@dataclass(frozen=True)
class DailyVectorExpression:
    """Measured value for each calendar date."""

    values: Mapping[date, mathopt.LinearExpression]
    quantity: AggregateQuantity


type MeasureExpression = (
    PointExpression | IntervalExpression | DependencyExpression | DailyVectorExpression
)
"""Measure built as MathOpt expressions."""


def fixed_task_slots(task: FixedTask, grid: TimeGrid) -> tuple[int, int]:
    """Round a fixed task outward to its start slot and duration."""
    start: int = (task.start - grid.horizon.start) // grid.slot
    end: int = -((grid.horizon.start - task.start - task.duration) // grid.slot)
    return start, end - start


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
    durations: dict[TaskId, int] = {
        task.id: task.duration // slot for task in problem.tasks
    }
    durations.update(
        (task.id, fixed_task_slots(task, problem.calendar.grid)[1])
        for task in problem.fixed_tasks
    )

    match measure:
        case PointMeasure():
            return PointExpression(placements[measure.task_id])
        case IntervalMeasure():
            duration: int = durations[measure.task_id]
            occupied: dict[int, list[mathopt.Variable]] = {}
            start: int
            variable: mathopt.Variable
            k: int
            for start, variable in placements[measure.task_id].items():
                for k in range(start, start + duration):
                    occupied.setdefault(k, []).append(variable)
            occupancy: dict[int, mathopt.LinearExpression] = {
                k: mathopt.LinearSum(occupied.get(k, ())) for k in range(n)
            }
            return IntervalExpression(occupancy)
        case DependencyMeasure():
            from_duration: int = durations[measure.from_task_id]
            gap: mathopt.LinearExpression = (
                starts[measure.to_task_id]
                - starts[measure.from_task_id]
                - from_duration * presences[measure.from_task_id]
            )
            from_start: mathopt.Variable = starts[measure.from_task_id]
            to_start: mathopt.Variable = starts[measure.to_task_id]
            from_presence: mathopt.Variable = presences[measure.from_task_id]
            bound: float = max(
                abs(
                    to_start.lower_bound
                    - from_start.upper_bound
                    - from_duration * from_presence.upper_bound
                ),
                abs(
                    to_start.upper_bound
                    - from_start.lower_bound
                    - from_duration * from_presence.lower_bound
                ),
            )
            return DependencyExpression(
                gap, from_presence, presences[measure.to_task_id], bound
            )
        case AggregateMeasure():
            dates: set[date] = {(horizon.start + k * slot).date() for k in range(n)}
            values: dict[date, mathopt.LinearExpression] = {
                day: mathopt.LinearSum(
                    (
                        1
                        if measure.quantity is AggregateQuantity.COUNT
                        else durations[task_id]
                    )
                    * variable
                    for task_id in measure.task_ids
                    for start, variable in placements[task_id].items()
                    if (horizon.start + start * slot).date() == day
                )
                for day in dates
            }
            return DailyVectorExpression(values, measure.quantity)
        case _:
            raise ValueError("Unsupported measure")
