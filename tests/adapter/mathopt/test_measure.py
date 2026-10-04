"""Point measures compile to movable choices or real fixed boundaries."""

from collections.abc import Mapping
from datetime import datetime, timedelta, timezone

import pytest
from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt import measure as expressions
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.measure import Boundary, PointMeasure
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId

START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
SLOT: timedelta = timedelta(minutes=30)
GRID: TimeGrid = TimeGrid(TimeInterval(START, START + 4 * SLOT), SLOT)
TASK_ID: TaskId = TaskId("task")


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
