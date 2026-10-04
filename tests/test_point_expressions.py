"""Point expressions preserve boundary penalties and task absence."""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

import pytest
from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt import measure as expressions
from intent_to_schedule.adapter.mathopt.evaluate import compile_evaluation
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import Criterion
from intent_to_schedule.domain.evaluation import Distance, Evaluation, Excess, Shortfall
from intent_to_schedule.domain.measure import Boundary, PointMeasure
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId
from intent_to_schedule.domain.violation import CriterionViolation, measure_criterion

START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
SLOT: timedelta = timedelta(minutes=30)
GRID: TimeGrid = TimeGrid(TimeInterval(START, START + 4 * SLOT), SLOT)
TASK_ID: TaskId = TaskId("task")
EVALUATIONS: tuple[Evaluation, ...] = (
    Distance(START + timedelta(minutes=45)),
    Excess(START + timedelta(minutes=45)),
    Shortfall(START + timedelta(minutes=45)),
)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("boundary", list(Boundary))
def test_compile_point_measure_selects_boundary_variant(
    fixed: bool, boundary: Boundary
) -> None:
    """Compile movable choices and real fixed boundaries separately."""
    model: mathopt.Model = mathopt.Model()
    choices: dict[int, mathopt.Variable] = {0: model.add_binary_variable()}
    placements: Mapping[TaskId, Mapping[int, mathopt.Variable]] = {TASK_ID: choices}
    task: Task = Task(TASK_ID, "Task", SLOT, frozenset(), Importance.LOW, False)
    appointment: FixedTask = FixedTask(
        TASK_ID,
        "Fixed",
        START - timedelta(days=1, minutes=7),
        timedelta(minutes=17),
        frozenset(),
    )
    problem: SchedulingProblem = SchedulingProblem(
        Calendar(GRID, ()),
        (),
        () if fixed else (task,),
        (appointment,) if fixed else (),
        (),
    )
    expression: expressions.MeasureExpression = expressions.compile_measure(
        PointMeasure(TASK_ID, boundary), problem, model, {}, {}, placements
    )
    if fixed:
        assert isinstance(expression, expressions.ConstantPointExpression)
        assert expression.value == (
            appointment.interval.end if boundary is Boundary.END else appointment.start
        )
    else:
        assert isinstance(expression, expressions.PointExpression)
        assert expression.placements == choices
        assert expression.offset == (SLOT if boundary is Boundary.END else timedelta(0))


@pytest.mark.parametrize("evaluation", EVALUATIONS)
@pytest.mark.parametrize("minutes", [-1447, 5, 80, 1507])
def test_constant_point_evaluation_matches_plain_violation(
    evaluation: Evaluation, minutes: int
) -> None:
    """Evaluate real constant points with the plain boundary penalty."""
    at: datetime = START + timedelta(minutes=minutes)
    model: mathopt.Model = mathopt.Model()
    expression: expressions.ConstantPointExpression = (
        expressions.ConstantPointExpression(at)
    )
    penalty: mathopt.LinearBase = compile_evaluation(expression, evaluation, model, GRID)
    model.minimize(penalty)
    result: mathopt.SolveResult = mathopt.solve(model, mathopt.SolverType.GSCIP)
    plain: CriterionViolation = measure_criterion(
        Criterion(PointMeasure(TASK_ID), evaluation),
        {TASK_ID: TimeInterval(at, at + SLOT)},
        GRID,
    )
    assert result.termination.reason is mathopt.TerminationReason.OPTIMAL
    assert result.objective_value() == pytest.approx(plain.amount)
    assert tuple(model.variables()) == ()


@pytest.mark.parametrize("evaluation", EVALUATIONS)
@pytest.mark.parametrize(
    "selected,empty", [(0, False), (2, False), (-1, False), (-1, True)]
)
def test_placement_point_evaluation_matches_plain_violation(
    evaluation: Evaluation, selected: int, empty: bool
) -> None:
    """Evaluate selected end points and omit absent movable tasks."""
    model: mathopt.Model = mathopt.Model()
    start: int
    variable: mathopt.Variable
    choices: dict[int, mathopt.Variable] = {
        start: model.add_binary_variable() for start in (() if empty else (0, 2))
    }
    for start, variable in choices.items():
        model.add_linear_constraint(variable == int(start == selected))
    expression: expressions.PointExpression = expressions.PointExpression(choices, SLOT)
    penalty: mathopt.LinearBase = compile_evaluation(expression, evaluation, model, GRID)
    model.minimize(penalty)
    result: mathopt.SolveResult = mathopt.solve(model, mathopt.SolverType.GSCIP)
    placements: dict[TaskId, TimeInterval] = (
        {TASK_ID: TimeInterval(GRID.time_at(selected), GRID.time_at(selected + 1))}
        if selected >= 0
        else {}
    )
    plain: CriterionViolation = measure_criterion(
        Criterion(PointMeasure(TASK_ID, Boundary.END), evaluation), placements, GRID
    )
    assert result.termination.reason is mathopt.TerminationReason.OPTIMAL
    assert result.objective_value() == pytest.approx(plain.amount)


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("evaluation", [Distance(SLOT), Excess(1), Shortfall(SLOT)])
def test_point_variants_reject_non_datetime_evaluations(
    fixed: bool, evaluation: Evaluation
) -> None:
    """Reject unsupported targets even when placements are empty."""
    expression: expressions.MeasureExpression = (
        expressions.ConstantPointExpression(START)
        if fixed
        else expressions.PointExpression({})
    )
    with pytest.raises(ValueError, match="Unsupported evaluation"):
        compile_evaluation(expression, evaluation, mathopt.Model(), GRID)
