from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

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
    """Summed Task occupancy for each slot."""

    occupancy: Mapping[int, mathopt.LinearBase]


@dataclass(frozen=True)
class DependencyExpression:
    """Task gap and the presences that activate it."""

    gap: mathopt.LinearBase
    from_presence: mathopt.Variable
    to_presence: mathopt.Variable
    bound: float


@dataclass(frozen=True)
class DailyVectorExpression:
    """Measured value for each calendar date."""

    values: Mapping[date, mathopt.LinearBase]
    quantity: AggregateQuantity


type MeasureExpression = (
    PointExpression | IntervalExpression | DependencyExpression | DailyVectorExpression
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
        task.id: grid.index_of(grid.horizon.start + task.duration) for task in problem.tasks
    }
    task: FixedTask
    rounded: TimeInterval
    for task in problem.fixed_tasks:
        rounded = grid.round_outward(TimeInterval(task.start, task.start + task.duration))
        durations[task.id] = grid.index_of(rounded.end) - grid.index_of(rounded.start)

    match measure:
        case PointMeasure():
            return PointExpression(placements[measure.task_id])
        case IntervalMeasure():
            occupied: dict[int, list[mathopt.Variable]] = {}
            task_id: TaskId
            duration: int
            start: int
            variable: mathopt.Variable
            slot_index: int
            for task_id in sorted(measure.task_ids, key=lambda item: item.value):
                duration = durations[task_id]
                for start, variable in placements[task_id].items():
                    for slot_index in range(start, start + duration):
                        occupied.setdefault(slot_index, []).append(variable)
            occupancy: dict[int, mathopt.LinearBase] = {
                slot_index: mathopt.LinearSum(occupied.get(slot_index, ()))
                for slot_index in range(grid.slot_count)
            }
            return IntervalExpression(occupancy)
        case DependencyMeasure():
            from_duration: int = durations[measure.from_task_id]
            gap: mathopt.LinearBase = (
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
            dates: set[date] = {grid.time_at(slot_index).date() for slot_index in range(grid.slot_count)}
            values: dict[date, mathopt.LinearBase] = {
                day: mathopt.LinearSum(
                    (
                        1
                        if measure.quantity is AggregateQuantity.COUNT
                        else durations[task_id]
                    )
                    * variable
                    for task_id in measure.task_ids
                    for start, variable in placements[task_id].items()
                    if grid.time_at(start).date() == day
                )
                for day in dates
            }
            return DailyVectorExpression(values, measure.quantity)
        case _:
            raise ValueError("Unsupported measure")
