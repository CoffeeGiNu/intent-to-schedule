"""Fixed-task constants in plain evaluation and solver constraints."""

from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone

import pytest
from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.compile import CompiledProblem, compile_problem
from intent_to_schedule.application.objective import (
    ConstraintEvaluation,
    evaluate_constraints,
)
from intent_to_schedule.application.policy import DEFAULT_POLICY
from intent_to_schedule.domain.calendar import (
    Availability,
    Calendar,
    TimeGrid,
    TimeInterval,
)
from intent_to_schedule.domain.condition import (
    Condition,
    DailyLimitCondition,
    TaskGapCondition,
    TaskGapRelation,
    TimeBoundCondition,
    TimeBoundRelation,
    TimeWindowCondition,
)
from intent_to_schedule.domain.constraint import (
    ConstraintId,
    HardConstraint,
    SoftConstraint,
)
from intent_to_schedule.domain.measure import AggregateQuantity, Boundary
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId
from intent_to_schedule.domain.time_windows import TimeRange, TimeRelation, TimeWindow
from intent_to_schedule.domain.violation import ViolationPart

START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone(timedelta(hours=9)))
SLOT: timedelta = timedelta(minutes=30)
PERSON: PersonId = PersonId("person")
FIXED: FixedTask = FixedTask(
    TaskId("appointment"),
    "Appointment",
    START + 2 * SLOT,
    timedelta(minutes=15),
    frozenset({PERSON}),
)
MOVABLE: Task = Task(
    TaskId("movable"), "Movable", SLOT, frozenset({PERSON}), Importance.LOW, True
)


def problem(condition: Condition, movable: bool = True) -> SchedulingProblem:
    """Build a problem containing a quarter-hour appointment."""
    grid: TimeGrid = TimeGrid(TimeInterval(START, START + 6 * SLOT), SLOT)
    return SchedulingProblem(
        Calendar(grid, (Availability(PERSON, (grid.horizon,)),)),
        (Person(PERSON, "Person"),),
        (MOVABLE,) if movable else (),
        (FIXED,),
        (SoftConstraint(ConstraintId("condition"), condition, Strength.NORMAL),),
    )


def assert_costs(value: SchedulingProblem, expected: float, backend: str) -> Schedule:
    """Check real-time violations and solver costs for selected placements."""
    schedule: Schedule
    objective: float | None = None
    if backend == "solver":
        compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
        result: mathopt.SolveResult = mathopt.solve(
            compiled.model, mathopt.SolverType.GSCIP
        )
        assert result.termination.reason is mathopt.TerminationReason.OPTIMAL
        values: dict[mathopt.Variable, float] = result.variable_values()
        scheduled: list[ScheduledTask] = []
        task: Task
        for task in value.tasks:
            assert values[compiled.presences[task.id]] > 0.5
            start: datetime = value.calendar.grid.time_at(
                next(
                    slot
                    for slot, variable in compiled.placements[task.id].items()
                    if values[variable] > 0.5
                )
            )
            scheduled.append(
                ScheduledTask(task.id, task.name, start, start + task.duration)
            )
        schedule = Schedule(tuple(scheduled), ())
        objective = result.objective_value()
    else:
        schedule = Schedule(
            tuple(
                ScheduledTask(
                    task.id,
                    task.name,
                    FIXED.start + SLOT,
                    FIXED.start + SLOT + task.duration,
                )
                for task in value.tasks
            ),
            (),
        )
    evaluations: tuple[ConstraintEvaluation, ...] = evaluate_constraints(
        value, schedule, DEFAULT_POLICY
    )
    assert len(evaluations) == 1
    evaluation: ConstraintEvaluation = evaluations[0]
    assert evaluation.violation.amount == pytest.approx(expected)
    assert evaluation.coefficient is not None
    assert evaluation.cost == pytest.approx(expected * evaluation.coefficient)
    if objective is not None:
        assert objective == pytest.approx(evaluation.cost)
    return schedule


@pytest.mark.parametrize("backend", ["plain", "solver"])
@pytest.mark.parametrize("boundary", list(Boundary))
@pytest.mark.parametrize("relation", list(TimeBoundRelation))
def test_fixed_time_bound_uses_real_boundary(
    backend: str, boundary: Boundary, relation: TimeBoundRelation
) -> None:
    """Compare fixed starts and ends without rounding."""
    condition: TimeBoundCondition = TimeBoundCondition(
        frozenset({FIXED.id}), boundary, relation, FIXED.start + timedelta(minutes=5)
    )
    difference: float = -1 / 12 if boundary is Boundary.START else 1 / 6
    expected: float = (
        max(difference, 0.0)
        if relation is TimeBoundRelation.AT_OR_BEFORE
        else max(-difference, 0.0)
        if relation is TimeBoundRelation.AT_OR_AFTER
        else abs(difference)
    )
    assert_costs(problem(condition, movable=False), expected, backend)


@pytest.mark.parametrize("backend", ["plain", "solver"])
def test_exact_zero_gap_from_fixed_end_keeps_quarter_hour_violation(
    backend: str,
) -> None:
    """Charge the unavoidable gap between a real end and the next slot."""
    condition: TaskGapCondition = TaskGapCondition(
        FIXED.id, MOVABLE.id, TaskGapRelation.EXACTLY, timedelta(0)
    )
    schedule: Schedule = assert_costs(problem(condition), 0.25, backend)
    assert schedule.scheduled[0].start == FIXED.start + SLOT


@pytest.mark.parametrize("backend", ["plain", "solver"])
def test_daily_duration_includes_fixed_real_duration(backend: str) -> None:
    """Sum movable and real fixed durations on their start date."""
    condition: DailyLimitCondition = DailyLimitCondition(
        frozenset({FIXED.id, MOVABLE.id}), AggregateQuantity.TOTAL_DURATION, SLOT
    )
    assert_costs(problem(condition), 0.25, backend)


@pytest.mark.parametrize("backend", ["plain", "solver"])
def test_avoid_window_counts_fixed_partial_slot_overlap(backend: str) -> None:
    """Charge only the fixed interval's real overlap with region slots."""
    condition: TimeWindowCondition = TimeWindowCondition(
        frozenset({FIXED.id, MOVABLE.id}),
        TimeRelation.AVOID,
        (TimeWindow(None, None, TimeRange(time(10), time(10, 30))),),
    )
    assert_costs(problem(condition), 0.25, backend)


def test_compiled_problem_has_only_movable_task_variables() -> None:
    """Keep fixed tasks out of all task-variable mappings."""
    condition: TaskGapCondition = TaskGapCondition(
        FIXED.id, MOVABLE.id, TaskGapRelation.EXACTLY, timedelta(0)
    )
    compiled: CompiledProblem = compile_problem(problem(condition), DEFAULT_POLICY)
    assert (
        set(compiled.starts)
        == set(compiled.presences)
        == set(compiled.placements)
        == {MOVABLE.id}
    )
    assert all(
        FIXED.id.value not in variable.name for variable in compiled.model.variables()
    )


@pytest.mark.parametrize("backend", ["plain", "solver"])
@pytest.mark.parametrize("quantity", list(AggregateQuantity))
@pytest.mark.parametrize(
    "offset,expected_count", [(-1440, 0), (-20, 1), (20, 1), (1500, 0)]
)
def test_fixed_daily_values_use_real_start_date_in_grid_offset(
    backend: str, quantity: AggregateQuantity, offset: int, expected_count: int
) -> None:
    """Count real start dates only when they occur in the grid's dates."""
    horizon_start: datetime = START.replace(hour=23, minute=50)
    grid: TimeGrid = TimeGrid(
        TimeInterval(horizon_start, horizon_start + 4 * SLOT), SLOT
    )
    fixed: FixedTask = replace(
        FIXED,
        start=(horizon_start + timedelta(minutes=offset)).astimezone(timezone.utc),
    )
    condition: DailyLimitCondition = DailyLimitCondition(
        frozenset({fixed.id}),
        quantity,
        0 if quantity is AggregateQuantity.COUNT else timedelta(0),
    )
    value: SchedulingProblem = replace(
        problem(condition, movable=False),
        calendar=Calendar(grid, ()),
        fixed_tasks=(fixed,),
    )
    expected: float = expected_count * (
        1.0 if quantity is AggregateQuantity.COUNT else 0.25
    )
    schedule: Schedule = assert_costs(value, expected, backend)
    evaluation: ConstraintEvaluation = evaluate_constraints(
        value, schedule, DEFAULT_POLICY
    )[0]
    assert (
        tuple(part.calendar_date for part in evaluation.violation.breakdown) == grid.dates
    )
    if expected_count:
        assert next(
            part.calendar_date
            for part in evaluation.violation.breakdown
            if part.amount
        ) == grid.date_of(fixed.start)


@pytest.mark.parametrize("relation", list(TaskGapRelation))
@pytest.mark.parametrize("fixed_first", [False, True])
def test_outside_fixed_gap_is_inactive_when_movable_drops(
    relation: TaskGapRelation, fixed_first: bool
) -> None:
    """Deactivate gaps with distant fixed constants when the movable task drops."""
    fixed: FixedTask = replace(
        FIXED,
        start=START + timedelta(days=20 if fixed_first else -20, minutes=7),
    )
    condition: TaskGapCondition = TaskGapCondition(
        fixed.id if fixed_first else MOVABLE.id,
        MOVABLE.id if fixed_first else fixed.id,
        relation,
        timedelta(minutes=17),
    )
    value: SchedulingProblem = replace(
        problem(condition),
        tasks=(replace(MOVABLE, required=False),),
        fixed_tasks=(fixed,),
        constraints=(HardConstraint(ConstraintId("gap"), condition),),
    )
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    result: mathopt.SolveResult = mathopt.solve(
        compiled.model, mathopt.SolverType.GSCIP
    )
    assert result.termination.reason is mathopt.TerminationReason.OPTIMAL
    assert result.variable_values()[compiled.presences[MOVABLE.id]] == pytest.approx(0.0)
    assert result.objective_value() == pytest.approx(
        DEFAULT_POLICY.drop_cost(MOVABLE.importance)
    )


@pytest.mark.parametrize("backend", ["plain", "solver"])
@pytest.mark.parametrize("relation", list(TaskGapRelation))
def test_gap_between_two_fixed_tasks_uses_real_constants(
    backend: str, relation: TaskGapRelation
) -> None:
    """Measure a gap between two fixed intervals outside the horizon."""
    first: FixedTask = replace(FIXED, start=START - timedelta(days=2, minutes=7))
    second: FixedTask = replace(
        FIXED,
        id=TaskId("second"),
        start=first.interval.end + timedelta(minutes=10),
    )
    condition: TaskGapCondition = TaskGapCondition(
        first.id, second.id, relation, timedelta(minutes=20)
    )
    value: SchedulingProblem = replace(
        problem(condition, movable=False), fixed_tasks=(first, second)
    )
    assert_costs(value, 1 / 6, backend)


@pytest.mark.parametrize("backend", ["plain", "solver"])
@pytest.mark.parametrize(
    "offset,duration,expected",
    [(-10, 20, 1 / 6), (170, 20, 1 / 6), (-40, 20, 0.0), (190, 20, 0.0), (10, 0, 0.0)],
)
def test_fixed_intrusion_clips_real_overlap_to_horizon(
    backend: str, offset: int, duration: int, expected: float
) -> None:
    """Clip real fixed overlap to region slots inside the horizon."""
    fixed: FixedTask = replace(
        FIXED,
        start=START + timedelta(minutes=offset),
        duration=timedelta(minutes=duration),
    )
    condition: TimeWindowCondition = TimeWindowCondition(
        frozenset({fixed.id}),
        TimeRelation.AVOID,
        (TimeWindow(None, None, None),),
    )
    assert_costs(
        replace(problem(condition, movable=False), fixed_tasks=(fixed,)), expected, backend
    )


@pytest.mark.parametrize("backend", ["plain", "solver"])
@pytest.mark.parametrize("quantity", list(AggregateQuantity))
@pytest.mark.parametrize("duration", [0, 5, 40])
@pytest.mark.parametrize(
    "offset,expected_count", [(-1440, 0), (-5, 1), (15, 1), (45, 1), (1455, 0)]
)
def test_fixed_daily_limit_covers_horizon_dates_without_slot_starts(
    backend: str,
    quantity: AggregateQuantity,
    duration: int,
    offset: int,
    expected_count: int,
) -> None:
    """Count whole fixed intervals on included calendar dates."""
    start: datetime = datetime(2026, 10, 1, 23, 50, tzinfo=timezone.utc)
    grid: TimeGrid = TimeGrid(TimeInterval(start, start + SLOT), SLOT)
    fixed: FixedTask = replace(
        FIXED,
        start=start + timedelta(minutes=offset),
        duration=timedelta(minutes=duration),
    )
    condition: DailyLimitCondition = DailyLimitCondition(
        frozenset({fixed.id}),
        quantity,
        0 if quantity is AggregateQuantity.COUNT else timedelta(0),
    )
    value: SchedulingProblem = replace(
        problem(condition, movable=False),
        calendar=Calendar(grid, ()),
        fixed_tasks=(fixed,),
    )
    expected: float = expected_count * (
        1.0 if quantity is AggregateQuantity.COUNT else duration / 60
    )
    schedule: Schedule = assert_costs(value, expected, backend)
    evaluation: ConstraintEvaluation = evaluate_constraints(
        value, schedule, DEFAULT_POLICY
    )[0]
    part: ViolationPart
    assert tuple(part.calendar_date for part in evaluation.violation.breakdown) == (
        date(2026, 10, 1),
        date(2026, 10, 2),
    )
    assert tuple(part.amount for part in evaluation.violation.breakdown) == pytest.approx(
        (expected, 0.0) if offset == -5 else (0.0, expected)
    )


@pytest.mark.parametrize("quantity", list(AggregateQuantity))
@pytest.mark.parametrize("allowed", [False, True])
def test_hard_daily_limit_checks_date_without_slot_start(
    quantity: AggregateQuantity, allowed: bool
) -> None:
    """Enforce fixed daily limits on a date without decisions."""
    start: datetime = datetime(2026, 10, 1, 23, 50, tzinfo=timezone.utc)
    grid: TimeGrid = TimeGrid(TimeInterval(start, start + SLOT), SLOT)
    fixed: FixedTask = replace(
        FIXED, start=start + timedelta(minutes=15), duration=timedelta(minutes=5)
    )
    maximum: int | timedelta = (
        int(allowed)
        if quantity is AggregateQuantity.COUNT
        else fixed.duration
        if allowed
        else timedelta(0)
    )
    condition: DailyLimitCondition = DailyLimitCondition(
        frozenset({fixed.id}), quantity, maximum
    )
    value: SchedulingProblem = replace(
        problem(condition, movable=False),
        calendar=Calendar(grid, ()),
        fixed_tasks=(fixed,),
        constraints=(HardConstraint(ConstraintId("daily"), condition),),
    )
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    result: mathopt.SolveResult = mathopt.solve(compiled.model, mathopt.SolverType.GSCIP)
    evaluation: ConstraintEvaluation = evaluate_constraints(
        value, Schedule((), ()), DEFAULT_POLICY
    )[0]
    assert (evaluation.violation.amount == 0.0) == allowed
    assert result.termination.reason is (
        mathopt.TerminationReason.OPTIMAL
        if allowed
        else mathopt.TerminationReason.INFEASIBLE
    )


@pytest.mark.parametrize("backend", ["plain", "solver"])
@pytest.mark.parametrize("quantity", list(AggregateQuantity))
def test_daily_breakdown_includes_dates_without_slot_starts(
    backend: str, quantity: AggregateQuantity
) -> None:
    """Include empty dates and fixed facts between multi-day slots."""
    start: datetime = START.replace(hour=23)
    grid: TimeGrid = TimeGrid(
        TimeInterval(start, start + timedelta(days=4)), timedelta(days=2)
    )
    fixed: FixedTask = replace(FIXED, start=start + timedelta(hours=2))
    condition: DailyLimitCondition = DailyLimitCondition(
        frozenset({fixed.id}),
        quantity,
        0 if quantity is AggregateQuantity.COUNT else timedelta(0),
    )
    value: SchedulingProblem = replace(
        problem(condition, movable=False),
        calendar=Calendar(grid, ()),
        fixed_tasks=(fixed,),
    )
    expected: float = 1.0 if quantity is AggregateQuantity.COUNT else 0.25
    schedule: Schedule = assert_costs(value, expected, backend)
    evaluation: ConstraintEvaluation = evaluate_constraints(
        value, schedule, DEFAULT_POLICY
    )[0]
    part: ViolationPart
    assert tuple(part.calendar_date for part in evaluation.violation.breakdown) == (
        date(2026, 10, 1),
        date(2026, 10, 2),
        date(2026, 10, 3),
        date(2026, 10, 4),
        date(2026, 10, 5),
    )
    assert tuple(part.amount for part in evaluation.violation.breakdown) == pytest.approx(
        (0.0, expected, 0.0, 0.0, 0.0)
    )


@pytest.mark.parametrize("relation", list(TimeBoundRelation))
@pytest.mark.parametrize("boundary", list(Boundary))
@pytest.mark.parametrize("allowed", [False, True])
def test_hard_fixed_time_bound_uses_real_outside_boundary(
    relation: TimeBoundRelation, boundary: Boundary, allowed: bool
) -> None:
    """Enforce real fixed boundaries outside the planning horizon."""
    fixed: FixedTask = replace(FIXED, start=START - timedelta(days=1, minutes=7))
    at: datetime = fixed.interval.end if boundary is Boundary.END else fixed.start
    if not allowed:
        at += (
            timedelta(minutes=1)
            if relation is TimeBoundRelation.AT_OR_AFTER
            else -timedelta(minutes=1)
        )
    condition: TimeBoundCondition = TimeBoundCondition(
        frozenset({fixed.id}), boundary, relation, at
    )
    value: SchedulingProblem = replace(
        problem(condition, movable=False),
        fixed_tasks=(fixed,),
        constraints=(HardConstraint(ConstraintId("bound"), condition),),
    )
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    result: mathopt.SolveResult = mathopt.solve(compiled.model, mathopt.SolverType.GSCIP)
    evaluation: ConstraintEvaluation = evaluate_constraints(
        value, Schedule((), ()), DEFAULT_POLICY
    )[0]
    assert evaluation.violation.amount == pytest.approx(0.0 if allowed else 1 / 60)
    assert result.termination.reason is (
        mathopt.TerminationReason.OPTIMAL
        if allowed
        else mathopt.TerminationReason.INFEASIBLE
    )
