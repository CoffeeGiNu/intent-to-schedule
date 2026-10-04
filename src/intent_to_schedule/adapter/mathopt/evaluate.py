from collections.abc import Mapping
from datetime import date, datetime, timedelta

from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.measure import (
    DailyVectorExpression,
    DependencyExpression,
    IntervalExpression,
    MeasureExpression,
    PointExpression,
)
from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.evaluation import (
    Distance,
    Evaluation,
    Excess,
    Intrusion,
    Shortfall,
)
from intent_to_schedule.domain.measure import AggregateQuantity


def compile_evaluation(
    expression: MeasureExpression,
    evaluation: Evaluation,
    model: mathopt.Model,
    grid: TimeGrid,
) -> mathopt.LinearBase:
    """Build an evaluation expression from a measure expression and Evaluation."""
    hours_per_slot: float = grid.slot / timedelta(hours=1)
    placements: Mapping[int, mathopt.Variable]
    target: datetime | timedelta
    offset: timedelta
    occupancy: Mapping[int, mathopt.LinearBase]
    region: tuple[TimeInterval, ...]
    dependency: DependencyExpression
    lower: timedelta
    values: Mapping[date, mathopt.LinearBase]
    value: mathopt.LinearBase
    upper: int | timedelta
    bound: float
    inactive: mathopt.LinearBase
    violation: mathopt.Variable
    violations: list[mathopt.Variable]

    match expression, evaluation:
        case PointExpression(placements=placements, offset=offset), Distance(
            target=datetime() as target
        ):
            return mathopt.LinearSum(
                abs((grid.time_at(start) + offset - target) / timedelta(hours=1))
                * variable
                for start, variable in placements.items()
            )
        case PointExpression(placements=placements, offset=offset), Excess(
            upper=datetime() as target
        ):
            return mathopt.LinearSum(
                max((grid.time_at(start) + offset - target) / timedelta(hours=1), 0.0)
                * variable
                for start, variable in placements.items()
            )
        case PointExpression(placements=placements, offset=offset), Shortfall(
            lower=datetime() as target
        ):
            return mathopt.LinearSum(
                max((target - grid.time_at(start) - offset) / timedelta(hours=1), 0.0)
                * variable
                for start, variable in placements.items()
            )
        case IntervalExpression(occupancy=occupancy), Intrusion(region=region):
            region_slots: set[int] = {
                slot_index
                for interval in region
                for slot_index in grid.slots_within(interval)
            }
            return (
                mathopt.LinearSum(occupancy[slot_index] for slot_index in region_slots)
                * hours_per_slot
            )
        case DependencyExpression() as dependency, Distance(
            target=timedelta() as target
        ):
            target_slots: float = target / grid.slot
            bound = dependency.bound + abs(target_slots)
            inactive = bound * (2 - dependency.from_presence - dependency.to_presence)
            violation = model.add_variable(lb=0.0)
            model.add_linear_constraint(
                violation >= (dependency.gap - target_slots - inactive) * hours_per_slot
            )
            model.add_linear_constraint(
                violation >= (target_slots - dependency.gap - inactive) * hours_per_slot
            )
            return violation
        case DependencyExpression() as dependency, Shortfall(
            lower=timedelta() as lower
        ):
            lower_slots: float = lower / grid.slot
            bound = dependency.bound + abs(lower_slots)
            inactive = bound * (2 - dependency.from_presence - dependency.to_presence)
            violation = model.add_variable(lb=0.0)
            model.add_linear_constraint(
                violation >= (lower_slots - dependency.gap - inactive) * hours_per_slot
            )
            return violation
        case DailyVectorExpression(
            values=values, quantity=AggregateQuantity.COUNT
        ), Excess(upper=int() as upper):
            if isinstance(upper, bool):
                raise ValueError("Unsupported evaluation")
            violations = []
            for value in values.values():
                violation = model.add_variable(lb=0.0)
                model.add_linear_constraint(violation >= value - upper)
                violations.append(violation)
            return mathopt.LinearSum(violations)
        case DailyVectorExpression(
            values=values, quantity=AggregateQuantity.TOTAL_DURATION
        ), Excess(upper=timedelta() as upper):
            upper_hours: float = upper / timedelta(hours=1)
            violations = []
            for value in values.values():
                violation = model.add_variable(lb=0.0)
                model.add_linear_constraint(
                    violation >= value * hours_per_slot - upper_hours
                )
                violations.append(violation)
            return mathopt.LinearSum(violations)
        case _:
            raise ValueError("Unsupported evaluation")
