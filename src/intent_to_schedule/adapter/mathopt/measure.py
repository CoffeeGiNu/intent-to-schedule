from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

from ortools.math_opt.python import mathopt

from intent_to_schedule.domain.measure import AggregateQuantity, Measure
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import TaskId


@dataclass(frozen=True)
class ScalarExpression:
    """Measured value as a single expression."""

    value: mathopt.LinearExpression


@dataclass(frozen=True)
class IntervalExpression:
    """Measured time span as start and end expressions in slots."""

    start: mathopt.LinearExpression
    end: mathopt.LinearExpression


@dataclass(frozen=True)
class DailyVectorExpression:
    """Measured value per day."""

    values: Mapping[date, mathopt.LinearExpression]
    quantity: AggregateQuantity


type MeasureExpression = ScalarExpression | IntervalExpression | DailyVectorExpression
"""Measure built as MathOpt expressions."""


def compile_measure(
    measure: Measure,
    problem: SchedulingProblem,
    model: mathopt.Model,
    starts: Mapping[TaskId, mathopt.Variable],
    presences: Mapping[TaskId, mathopt.Variable],
) -> MeasureExpression:
    """Build a measure expression from a Measure."""
    ...
