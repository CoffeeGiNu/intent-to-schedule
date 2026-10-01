from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.measure import MeasureExpression
from intent_to_schedule.domain.calendar import TimeGrid
from intent_to_schedule.domain.evaluation import Evaluation


def compile_evaluation(
    expression: MeasureExpression,
    evaluation: Evaluation,
    model: mathopt.Model,
    grid: TimeGrid,
) -> mathopt.LinearExpression:
    """Build an evaluation expression from a measure expression and Evaluation."""
    ...
