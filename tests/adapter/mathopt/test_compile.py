"""Tests for the compiled MathOpt model and its agreement with plain evaluation."""

from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone

import pytest
from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.compile import CompiledProblem, compile_problem
from intent_to_schedule.application.objective import (
    ConstraintEvaluation,
    evaluate_constraints,
)
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.query import (
    Answered,
    AnswerResult,
    AvailableStartsAnswer,
    AvailableStartsQuery,
)
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
from intent_to_schedule.domain.consistency import AlignedToSlots, AllOf, UniqueIds
from intent_to_schedule.domain.constraint import (
    Constraint,
    ConstraintId,
    HardConstraint,
    SoftConstraint,
)
from intent_to_schedule.domain.measure import AggregateQuantity, Boundary
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import DroppedTask, Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId
from intent_to_schedule.domain.time_windows import (
    TimeRange,
    TimeRelation,
    TimeWindow,
)

START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone(timedelta(hours=9)))
SLOT: timedelta = timedelta(minutes=30)
HOUR: timedelta = timedelta(hours=1)
PERSON: PersonId = PersonId("person")
PEOPLE: frozenset[PersonId] = frozenset({PERSON})
FIXED: FixedTask = FixedTask(
    TaskId("appointment"),
    "Appointment",
    START + 2 * SLOT,
    timedelta(minutes=15),
    PEOPLE,
)
MOVABLE: Task = Task(TaskId("movable"), "Movable", SLOT, PEOPLE, Importance.LOW, True)


def task(
    name: str,
    *,
    duration: timedelta = SLOT,
    people: frozenset[PersonId] = frozenset(),
    required: bool = True,
) -> Task:
    return Task(TaskId(name), name, duration, people, Importance.LOW, required)


def problem(
    *tasks: Task,
    constraints: tuple[Constraint, ...] = (),
    fixed_tasks: tuple[FixedTask, ...] = (),
    start: datetime = START,
    end: datetime = START + timedelta(hours=3),
    slot: timedelta = SLOT,
    availabilities: tuple[Availability, ...] | None = None,
) -> SchedulingProblem:
    """Build a valid problem whose participants are available unless told otherwise."""
    people: tuple[Person, ...] = tuple(
        Person(person_id, person_id.value)
        for person_id in sorted(
            {
                person_id
                for item in (*tasks, *fixed_tasks)
                for person_id in item.participant_ids
            },
            key=lambda person_id: person_id.value,
        )
    )
    if availabilities is None:
        availabilities = tuple(
            Availability(person.id, (TimeInterval(start, end),)) for person in people
        )
    value: SchedulingProblem = SchedulingProblem(
        Calendar(TimeGrid(TimeInterval(start, end), slot), availabilities),
        people,
        tasks,
        fixed_tasks,
        constraints,
    )
    assert AllOf(AlignedToSlots(), UniqueIds()).validate(value).is_empty
    return value


def hard(condition: Condition) -> HardConstraint:
    return HardConstraint(ConstraintId.generate(), condition)


def soft(condition: Condition) -> SoftConstraint:
    return SoftConstraint(ConstraintId.generate(), condition, Strength.WEAK)


def bound(
    item: Task | FixedTask,
    boundary: Boundary,
    relation: TimeBoundRelation,
    at: datetime,
) -> TimeBoundCondition:
    return TimeBoundCondition(frozenset({item.id}), boundary, relation, at)


def solve_details(value: SchedulingProblem) -> tuple[float, dict[TaskId, int | None]]:
    """Return the objective and selected start slots."""
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    result: mathopt.SolveResult = mathopt.solve(
        compiled.model, mathopt.SolverType.GSCIP
    )
    assert result.termination.reason is mathopt.TerminationReason.OPTIMAL
    values: dict[mathopt.Variable, float] = result.variable_values()
    placements: dict[TaskId, int | None] = {
        item.id: next(
            (
                start
                for start, variable in compiled.placements[item.id].items()
                if values[variable] > 0.5
            ),
            None,
        )
        for item in value.tasks
    }
    return result.objective_value(), placements


def objective(value: SchedulingProblem) -> float:
    return solve_details(value)[0]


def appointment_problem(
    condition: Condition, movable: bool = True
) -> SchedulingProblem:
    """Build a problem containing a quarter-hour appointment."""
    return problem(
        *((MOVABLE,) if movable else ()),
        fixed_tasks=(FIXED,),
        constraints=(
            SoftConstraint(ConstraintId("condition"), condition, Strength.NORMAL),
        ),
        end=START + 6 * SLOT,
    )


def assert_costs(value: SchedulingProblem, expected: float) -> Schedule:
    """Check that the solver objective equals the real-time violation cost."""
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    result: mathopt.SolveResult = mathopt.solve(
        compiled.model, mathopt.SolverType.GSCIP
    )
    assert result.termination.reason is mathopt.TerminationReason.OPTIMAL
    values: dict[mathopt.Variable, float] = result.variable_values()
    scheduled: list[ScheduledTask] = []
    item: Task
    for item in value.tasks:
        assert values[compiled.presences[item.id]] > 0.5
        start: datetime = value.calendar.grid.time_at(
            next(
                slot
                for slot, variable in compiled.placements[item.id].items()
                if values[variable] > 0.5
            )
        )
        scheduled.append(
            ScheduledTask(item.id, item.name, start, start + item.duration)
        )
    schedule: Schedule = Schedule(tuple(scheduled), ())
    evaluations: tuple[ConstraintEvaluation, ...] = evaluate_constraints(
        value, schedule, DEFAULT_POLICY
    )
    assert len(evaluations) == 1
    evaluation: ConstraintEvaluation = evaluations[0]
    assert evaluation.violation.amount == pytest.approx(expected)
    assert evaluation.coefficient is not None
    assert evaluation.cost == pytest.approx(expected * evaluation.coefficient)
    assert result.objective_value() == pytest.approx(evaluation.cost)
    return schedule


def two_tasks() -> SchedulingProblem:
    """Build two tasks without participants on a short calendar."""
    return problem(
        task("first"), task("second", duration=2 * SLOT), end=START + 6 * SLOT
    )


@pytest.mark.parametrize(
    ("boundary", "relation", "expected"),
    [
        (Boundary.START, TimeBoundRelation.AT_OR_BEFORE, 0.0),
        (Boundary.START, TimeBoundRelation.AT_OR_AFTER, 1 / 12),
        (Boundary.START, TimeBoundRelation.AT, 1 / 12),
        (Boundary.END, TimeBoundRelation.AT_OR_BEFORE, 1 / 6),
        (Boundary.END, TimeBoundRelation.AT_OR_AFTER, 0.0),
        (Boundary.END, TimeBoundRelation.AT, 1 / 6),
    ],
)
def test_fixed_time_bound_uses_real_boundary(
    boundary: Boundary, relation: TimeBoundRelation, expected: float
) -> None:
    """Compare fixed starts and ends without rounding."""
    condition: TimeBoundCondition = TimeBoundCondition(
        frozenset({FIXED.id}), boundary, relation, FIXED.start + timedelta(minutes=5)
    )
    assert_costs(appointment_problem(condition, movable=False), expected)


def test_exact_zero_gap_from_fixed_end_keeps_quarter_hour_violation() -> None:
    """Charge the unavoidable gap between a real end and the next slot."""
    condition: TaskGapCondition = TaskGapCondition(
        FIXED.id, MOVABLE.id, TaskGapRelation.EXACTLY, timedelta(0)
    )
    schedule: Schedule = assert_costs(appointment_problem(condition), 0.25)
    assert schedule.scheduled[0].start == FIXED.start + SLOT


def test_daily_duration_includes_fixed_real_duration() -> None:
    """Sum movable and real fixed durations on their start date."""
    condition: DailyLimitCondition = DailyLimitCondition(
        frozenset({FIXED.id, MOVABLE.id}), AggregateQuantity.TOTAL_DURATION, SLOT
    )
    assert_costs(appointment_problem(condition), 0.25)


def test_avoid_window_counts_fixed_partial_slot_overlap() -> None:
    """Charge only the fixed interval's real overlap with region slots."""
    condition: TimeWindowCondition = TimeWindowCondition(
        frozenset({FIXED.id, MOVABLE.id}),
        TimeRelation.AVOID,
        (TimeWindow(None, None, TimeRange(time(10), time(10, 30))),),
    )
    assert_costs(appointment_problem(condition), 0.25)


def test_compiled_problem_has_only_movable_task_variables() -> None:
    """Keep fixed tasks out of all task-variable mappings."""
    condition: TaskGapCondition = TaskGapCondition(
        FIXED.id, MOVABLE.id, TaskGapRelation.EXACTLY, timedelta(0)
    )
    compiled: CompiledProblem = compile_problem(
        appointment_problem(condition), DEFAULT_POLICY
    )
    assert (
        set(compiled.starts)
        == set(compiled.presences)
        == set(compiled.placements)
        == {MOVABLE.id}
    )
    assert all(
        FIXED.id.value not in variable.name for variable in compiled.model.variables()
    )


@pytest.mark.parametrize(
    ("quantity", "offset", "amounts"),
    [
        (AggregateQuantity.COUNT, -1440, (0.0, 0.0)),
        (AggregateQuantity.COUNT, -20, (1.0, 0.0)),
        (AggregateQuantity.COUNT, 20, (0.0, 1.0)),
        (AggregateQuantity.COUNT, 1500, (0.0, 0.0)),
        (AggregateQuantity.TOTAL_DURATION, -1440, (0.0, 0.0)),
        (AggregateQuantity.TOTAL_DURATION, -20, (0.25, 0.0)),
        (AggregateQuantity.TOTAL_DURATION, 20, (0.0, 0.25)),
        (AggregateQuantity.TOTAL_DURATION, 1500, (0.0, 0.0)),
    ],
)
def test_fixed_daily_values_use_real_start_date_in_grid_offset(
    quantity: AggregateQuantity, offset: int, amounts: tuple[float, float]
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
        appointment_problem(condition, movable=False),
        calendar=Calendar(grid, ()),
        fixed_tasks=(fixed,),
    )
    schedule: Schedule = assert_costs(value, sum(amounts))
    evaluation: ConstraintEvaluation = evaluate_constraints(
        value, schedule, DEFAULT_POLICY
    )[0]
    assert tuple(part.calendar_date for part in evaluation.violation.breakdown) == (
        date(2026, 10, 1),
        date(2026, 10, 2),
    )
    assert tuple(
        part.amount for part in evaluation.violation.breakdown
    ) == pytest.approx(amounts)


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
        appointment_problem(condition),
        tasks=(replace(MOVABLE, required=False),),
        fixed_tasks=(fixed,),
        constraints=(HardConstraint(ConstraintId("gap"), condition),),
    )
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    result: mathopt.SolveResult = mathopt.solve(
        compiled.model, mathopt.SolverType.GSCIP
    )
    assert result.termination.reason is mathopt.TerminationReason.OPTIMAL
    assert result.variable_values()[compiled.presences[MOVABLE.id]] == pytest.approx(
        0.0
    )
    assert result.objective_value() == pytest.approx(
        DEFAULT_POLICY.drop_cost(MOVABLE.importance)
    )


@pytest.mark.parametrize("relation", list(TaskGapRelation))
def test_gap_between_two_fixed_tasks_uses_real_constants(
    relation: TaskGapRelation,
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
        appointment_problem(condition, movable=False), fixed_tasks=(first, second)
    )
    assert_costs(value, 1 / 6)


@pytest.mark.parametrize(
    "offset,duration,expected",
    [(-10, 20, 1 / 6), (170, 20, 1 / 6), (-40, 20, 0.0), (190, 20, 0.0), (10, 0, 0.0)],
)
def test_fixed_intrusion_clips_real_overlap_to_horizon(
    offset: int, duration: int, expected: float
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
        replace(appointment_problem(condition, movable=False), fixed_tasks=(fixed,)),
        expected,
    )


@pytest.mark.parametrize(
    ("quantity", "offset", "duration", "amounts"),
    [
        (AggregateQuantity.COUNT, -1440, 40, (0.0, 0.0)),
        (AggregateQuantity.COUNT, -5, 40, (1.0, 0.0)),
        (AggregateQuantity.COUNT, 15, 0, (0.0, 1.0)),
        (AggregateQuantity.COUNT, 45, 5, (0.0, 1.0)),
        (AggregateQuantity.COUNT, 1455, 40, (0.0, 0.0)),
        (AggregateQuantity.TOTAL_DURATION, -1440, 40, (0.0, 0.0)),
        (AggregateQuantity.TOTAL_DURATION, -5, 40, (2 / 3, 0.0)),
        (AggregateQuantity.TOTAL_DURATION, 15, 0, (0.0, 0.0)),
        (AggregateQuantity.TOTAL_DURATION, 45, 5, (0.0, 1 / 12)),
        (AggregateQuantity.TOTAL_DURATION, 1455, 40, (0.0, 0.0)),
    ],
)
def test_fixed_daily_limit_covers_horizon_dates_without_slot_starts(
    quantity: AggregateQuantity,
    offset: int,
    duration: int,
    amounts: tuple[float, float],
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
        appointment_problem(condition, movable=False),
        calendar=Calendar(grid, ()),
        fixed_tasks=(fixed,),
    )
    schedule: Schedule = assert_costs(value, sum(amounts))
    evaluation: ConstraintEvaluation = evaluate_constraints(
        value, schedule, DEFAULT_POLICY
    )[0]
    assert tuple(part.calendar_date for part in evaluation.violation.breakdown) == (
        date(2026, 10, 1),
        date(2026, 10, 2),
    )
    assert tuple(
        part.amount for part in evaluation.violation.breakdown
    ) == pytest.approx(amounts)


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
        appointment_problem(condition, movable=False),
        calendar=Calendar(grid, ()),
        fixed_tasks=(fixed,),
        constraints=(HardConstraint(ConstraintId("daily"), condition),),
    )
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    result: mathopt.SolveResult = mathopt.solve(
        compiled.model, mathopt.SolverType.GSCIP
    )
    evaluation: ConstraintEvaluation = evaluate_constraints(
        value, Schedule((), ()), DEFAULT_POLICY
    )[0]
    assert (evaluation.violation.amount == 0.0) == allowed
    assert result.termination.reason is (
        mathopt.TerminationReason.OPTIMAL
        if allowed
        else mathopt.TerminationReason.INFEASIBLE
    )


@pytest.mark.parametrize("quantity", list(AggregateQuantity))
def test_daily_breakdown_includes_dates_without_slot_starts(
    quantity: AggregateQuantity,
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
        appointment_problem(condition, movable=False),
        calendar=Calendar(grid, ()),
        fixed_tasks=(fixed,),
    )
    expected: float = 1.0 if quantity is AggregateQuantity.COUNT else 0.25
    schedule: Schedule = assert_costs(value, expected)
    evaluation: ConstraintEvaluation = evaluate_constraints(
        value, schedule, DEFAULT_POLICY
    )[0]
    assert tuple(part.calendar_date for part in evaluation.violation.breakdown) == (
        date(2026, 10, 1),
        date(2026, 10, 2),
        date(2026, 10, 3),
        date(2026, 10, 4),
        date(2026, 10, 5),
    )
    assert tuple(
        part.amount for part in evaluation.violation.breakdown
    ) == pytest.approx((0.0, expected, 0.0, 0.0, 0.0))


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
        appointment_problem(condition, movable=False),
        fixed_tasks=(fixed,),
        constraints=(HardConstraint(ConstraintId("bound"), condition),),
    )
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    result: mathopt.SolveResult = mathopt.solve(
        compiled.model, mathopt.SolverType.GSCIP
    )
    evaluation: ConstraintEvaluation = evaluate_constraints(
        value, Schedule((), ()), DEFAULT_POLICY
    )[0]
    assert evaluation.violation.amount == pytest.approx(0.0 if allowed else 1 / 60)
    assert result.termination.reason is (
        mathopt.TerminationReason.OPTIMAL
        if allowed
        else mathopt.TerminationReason.INFEASIBLE
    )


def test_soft_deadline_prefers_one_hour_late_over_two() -> None:
    item: Task = task("late", duration=HOUR, people=PEOPLE)
    deadline: TimeBoundCondition = bound(
        item, Boundary.START, TimeBoundRelation.AT_OR_BEFORE, START
    )
    value: SchedulingProblem = problem(
        item,
        constraints=(soft(deadline),),
        end=START + 4 * HOUR,
        slot=HOUR,
        availabilities=(
            Availability(PERSON, (TimeInterval(START + HOUR, START + 3 * HOUR),)),
        ),
    )
    cost: float
    placements: dict[TaskId, int | None]
    cost, placements = solve_details(value)
    assert placements == {item.id: 1}
    assert cost == pytest.approx(1.0)
    later: SchedulingProblem = replace(
        value,
        calendar=replace(
            value.calendar,
            availabilities=(
                Availability(
                    PERSON, (TimeInterval(START + 2 * HOUR, START + 3 * HOUR),)
                ),
            ),
        ),
    )
    assert objective(later) == pytest.approx(2.0)


@pytest.mark.parametrize(
    ("boundary", "relation", "target_minutes", "expected"),
    [
        (Boundary.START, TimeBoundRelation.AT_OR_BEFORE, -75, 1.25),
        (Boundary.START, TimeBoundRelation.AT_OR_AFTER, -75, 0.0),
        (Boundary.START, TimeBoundRelation.AT, -75, 1.25),
        (Boundary.END, TimeBoundRelation.AT_OR_BEFORE, -75, 3.25),
        (Boundary.END, TimeBoundRelation.AT_OR_AFTER, -75, 0.0),
        (Boundary.END, TimeBoundRelation.AT, -75, 3.25),
        (Boundary.START, TimeBoundRelation.AT_OR_BEFORE, 75, 0.0),
        (Boundary.START, TimeBoundRelation.AT_OR_AFTER, 75, 1.25),
        (Boundary.START, TimeBoundRelation.AT, 75, 1.25),
        (Boundary.END, TimeBoundRelation.AT_OR_BEFORE, 75, 0.75),
        (Boundary.END, TimeBoundRelation.AT_OR_AFTER, 75, 0.0),
        (Boundary.END, TimeBoundRelation.AT, 75, 0.75),
        (Boundary.START, TimeBoundRelation.AT_OR_BEFORE, 375, 0.0),
        (Boundary.START, TimeBoundRelation.AT_OR_AFTER, 375, 6.25),
        (Boundary.START, TimeBoundRelation.AT, 375, 6.25),
        (Boundary.END, TimeBoundRelation.AT_OR_BEFORE, 375, 0.0),
        (Boundary.END, TimeBoundRelation.AT_OR_AFTER, 375, 4.25),
        (Boundary.END, TimeBoundRelation.AT, 375, 4.25),
    ],
)
def test_time_bound_cost_uses_exact_hours(
    boundary: Boundary,
    relation: TimeBoundRelation,
    target_minutes: int,
    expected: float,
) -> None:
    item: Task = task("cost", duration=2 * HOUR, people=PEOPLE)
    at: datetime = START + timedelta(minutes=target_minutes)
    value: SchedulingProblem = problem(
        item,
        constraints=(soft(bound(item, boundary, relation, at)),),
        end=START + item.duration,
        slot=HOUR,
    )
    assert objective(value) == pytest.approx(expected)


def test_time_bound_criteria_sum_cost_over_tasks() -> None:
    first: Task = task("first", duration=HOUR, people=PEOPLE)
    second: Task = task("second", duration=HOUR, people=PEOPLE)
    condition: TimeBoundCondition = TimeBoundCondition(
        frozenset({first.id, second.id}),
        Boundary.END,
        TimeBoundRelation.AT_OR_BEFORE,
        START,
    )
    value: SchedulingProblem = problem(
        first, second, constraints=(soft(condition),), end=START + 2 * HOUR, slot=HOUR
    )
    assert objective(value) == pytest.approx(3.0)


@pytest.mark.parametrize("relation", list(TimeBoundRelation))
def test_unscheduled_task_has_no_time_bound_violation(
    relation: TimeBoundRelation,
) -> None:
    item: Task = task("optional", duration=HOUR, people=PEOPLE, required=False)
    condition: TimeBoundCondition = bound(
        item, Boundary.END, relation, START + timedelta(minutes=15)
    )
    value: SchedulingProblem = problem(
        item,
        constraints=(hard(condition),),
        end=START + 4 * HOUR,
        slot=HOUR,
        availabilities=(Availability(PERSON, ()),),
    )
    cost: float
    placements: dict[TaskId, int | None]
    cost, placements = solve_details(value)
    assert placements == {item.id: None}
    assert cost == pytest.approx(DEFAULT_POLICY.drop_cost(item.importance))


@pytest.mark.parametrize("relation", list(TaskGapRelation))
def test_unscheduled_task_has_no_gap_violation(relation: TaskGapRelation) -> None:
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Fixed", START, HOUR, PEOPLE)
    item: Task = task("optional", duration=HOUR, people=PEOPLE, required=False)
    condition: TaskGapCondition = TaskGapCondition(
        fixed.id, item.id, relation, 100 * HOUR
    )
    value: SchedulingProblem = problem(
        item,
        constraints=(hard(condition),),
        fixed_tasks=(fixed,),
        end=START + 4 * HOUR,
        slot=HOUR,
        availabilities=(Availability(PERSON, ()),),
    )
    cost: float
    placements: dict[TaskId, int | None]
    cost, placements = solve_details(value)
    assert placements == {item.id: None}
    assert cost == pytest.approx(DEFAULT_POLICY.drop_cost(item.importance))


@pytest.mark.parametrize("relation", list(TaskGapRelation))
def test_unaligned_gap_cost_uses_exact_hours(relation: TaskGapRelation) -> None:
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Fixed", START, HOUR, PEOPLE)
    item: Task = task("following", duration=HOUR, people=PEOPLE)
    condition: TaskGapCondition = TaskGapCondition(
        fixed.id, item.id, relation, timedelta(minutes=15)
    )
    value: SchedulingProblem = problem(
        item,
        constraints=(soft(condition),),
        fixed_tasks=(fixed,),
        end=START + 2 * HOUR,
        slot=HOUR,
    )
    assert objective(value) == pytest.approx(0.25)


@pytest.mark.parametrize(
    "quantity,maximum,expected",
    [
        (AggregateQuantity.COUNT, 0, 2.0),
        (AggregateQuantity.TOTAL_DURATION, timedelta(minutes=30), 3.0),
    ],
)
def test_daily_limit_sums_excess_by_start_date(
    quantity: AggregateQuantity, maximum: int | timedelta, expected: float
) -> None:
    first: FixedTask = FixedTask(
        TaskId("first"), "First", START.replace(hour=23), 2 * HOUR, PEOPLE
    )
    second: FixedTask = replace(
        first, id=TaskId("second"), start=first.start + timedelta(days=1)
    )
    condition: DailyLimitCondition = DailyLimitCondition(
        frozenset({first.id, second.id}), quantity, maximum
    )
    value: SchedulingProblem = problem(
        constraints=(soft(condition),),
        fixed_tasks=(first, second),
        start=START.replace(hour=22),
        end=(START + timedelta(days=2)).replace(hour=2),
        slot=HOUR,
    )
    assert objective(value) == pytest.approx(expected)


def test_available_starts_match_compiled_start_candidates() -> None:
    """Offer queries the same start candidates as the solver."""
    alice: PersonId = PersonId("alice")
    bob: PersonId = PersonId("bob")
    item: Task = task("candidate", duration=HOUR, people=frozenset({alice, bob}))
    meeting: FixedTask = FixedTask(
        TaskId("meeting"), "Team meeting", START + HOUR, SLOT, frozenset({alice, bob})
    )
    late: FixedTask = FixedTask(
        TaskId("late"), "Late meeting", START + 3 * HOUR, SLOT, frozenset({alice})
    )
    given: SchedulingProblem = problem(
        item, fixed_tasks=(meeting, late), end=START + 4 * HOUR
    )
    compiled: CompiledProblem = compile_problem(given, DEFAULT_POLICY)
    result: AnswerResult = AvailableStartsQuery(
        item.participant_ids, item.duration, None, 100
    ).answer(given, None)
    assert isinstance(result, Answered) and isinstance(
        result.answer, AvailableStartsAnswer
    )
    assert result.answer.items == tuple(
        given.calendar.grid.time_at(index)
        for index in sorted(compiled.placements[item.id])
    )


@pytest.mark.parametrize(
    ("strength", "cost"),
    [
        (None, 0.0),
        (Strength.WEAK, 1.5),
        (Strength.NORMAL, 7.5),
        (Strength.STRONG, 30.0),
    ],
)
def test_multi_task_intrusion_matches_separate_constraints(
    strength: Strength | None, cost: float
) -> None:
    """Match separate objectives and placements for every requirement."""
    base: SchedulingProblem = two_tasks()
    task_ids: frozenset[TaskId] = frozenset(item.id for item in base.tasks)
    condition: TimeWindowCondition = TimeWindowCondition(
        task_ids,
        TimeRelation.AVOID,
        (
            TimeWindow(
                None,
                None,
                TimeRange(
                    START.time(), (START + SLOT * (2 if strength is None else 6)).time()
                ),
            ),
        ),
    )
    combined: Constraint = (
        HardConstraint(ConstraintId("combined"), condition)
        if strength is None
        else SoftConstraint(ConstraintId("combined"), condition, strength)
    )
    separate: tuple[Constraint, ...] = tuple(
        replace(
            combined,
            id=ConstraintId(item.id.value),
            condition=replace(combined.condition, task_ids=frozenset({item.id})),
        )
        for item in base.tasks
    )
    targets: tuple[Constraint, ...] = tuple(
        SoftConstraint(
            ConstraintId(f"target-{item.id.value}"),
            TimeBoundCondition(
                frozenset({item.id}),
                Boundary.START,
                TimeBoundRelation.AT,
                START + SLOT * (index + (2 if strength is None else 0)),
            ),
            Strength.WEAK,
        )
        for index, item in enumerate(base.tasks)
    )
    expected: tuple[float, dict[TaskId, int | None]] = (
        cost,
        {
            item.id: index + (2 if strength is None else 0)
            for index, item in enumerate(base.tasks)
        },
    )
    assert solve_details(replace(base, constraints=(combined, *targets))) == expected
    assert solve_details(replace(base, constraints=(*separate, *targets))) == expected


def test_multi_task_soft_intrusion_matches_separate_drops() -> None:
    """Match task dropping under summed soft penalties."""
    base: SchedulingProblem = two_tasks()
    tasks: tuple[Task, ...] = (
        replace(base.tasks[0], required=False),
        replace(base.tasks[1], required=False, duration=SLOT * 3),
    )
    combined: SoftConstraint = SoftConstraint(
        ConstraintId("combined"),
        TimeWindowCondition(
            frozenset(item.id for item in tasks),
            TimeRelation.AVOID,
            (TimeWindow(None, None, TimeRange(time(9), time(12))),),
        ),
        Strength.NORMAL,
    )
    separate: tuple[Constraint, ...] = tuple(
        replace(
            combined,
            id=ConstraintId(item.id.value),
            condition=replace(combined.condition, task_ids=frozenset({item.id})),
        )
        for item in tasks
    )
    targets: tuple[Constraint, ...] = tuple(
        SoftConstraint(
            ConstraintId(f"target-{item.id.value}"),
            TimeBoundCondition(
                frozenset({item.id}), Boundary.START, TimeBoundRelation.AT, START
            ),
            Strength.WEAK,
        )
        for item in tasks
    )
    expected: tuple[float, dict[TaskId, int | None]] = (
        7.5,
        {tasks[0].id: 0, tasks[1].id: None},
    )
    assert (
        solve_details(replace(base, tasks=tasks, constraints=(combined, *targets)))
        == expected
    )
    assert (
        solve_details(replace(base, tasks=tasks, constraints=(*separate, *targets)))
        == expected
    )


def test_multi_task_intrusion_sums_real_fixed_and_movable_overlap() -> None:
    """Count overlapping fixed and movable occupancy separately."""
    base: SchedulingProblem = two_tasks()
    first: Task = base.tasks[0]
    second: Task = base.tasks[1]
    fixed: FixedTask = FixedTask(
        second.id,
        second.name,
        START + timedelta(minutes=10),
        second.duration,
        second.participant_ids,
    )
    combined: SoftConstraint = SoftConstraint(
        ConstraintId("combined"),
        TimeWindowCondition(
            frozenset({first.id, fixed.id}),
            TimeRelation.AVOID,
            (TimeWindow(None, None, TimeRange(time(9), time(12))),),
        ),
        Strength.NORMAL,
    )
    separate: tuple[Constraint, ...] = tuple(
        replace(
            combined,
            id=ConstraintId(task_id.value),
            condition=replace(combined.condition, task_ids=frozenset({task_id})),
        )
        for task_id in (first.id, fixed.id)
    )
    target: SoftConstraint = SoftConstraint(
        ConstraintId("target"),
        TimeBoundCondition(
            frozenset({first.id}), Boundary.START, TimeBoundRelation.AT, START
        ),
        Strength.WEAK,
    )
    given: SchedulingProblem = replace(base, tasks=(first,), fixed_tasks=(fixed,))
    expected: tuple[float, dict[TaskId, int | None]] = (7.5, {first.id: 0})
    assert solve_details(replace(given, constraints=(combined, target))) == expected
    assert solve_details(replace(given, constraints=(*separate, target))) == expected


def test_multi_task_hard_intrusion_matches_separate_infeasibility() -> None:
    """Reject a hard region covering every possible placement."""
    base: SchedulingProblem = two_tasks()
    combined: HardConstraint = HardConstraint(
        ConstraintId("combined"),
        TimeWindowCondition(
            frozenset(item.id for item in base.tasks),
            TimeRelation.AVOID,
            (TimeWindow(None, None, TimeRange(time(9), time(12))),),
        ),
    )
    separate: tuple[Constraint, ...] = tuple(
        replace(
            combined,
            id=ConstraintId(item.id.value),
            condition=replace(combined.condition, task_ids=frozenset({item.id})),
        )
        for item in base.tasks
    )
    constraints: tuple[Constraint, ...]
    for constraints in ((combined,), separate):
        compiled: CompiledProblem = compile_problem(
            replace(base, constraints=constraints), DEFAULT_POLICY
        )
        result: mathopt.SolveResult = mathopt.solve(
            compiled.model, mathopt.SolverType.GSCIP
        )
        assert result.termination.reason is mathopt.TerminationReason.INFEASIBLE


@pytest.mark.parametrize(
    ("participant_count", "duration", "expected"),
    [
        (0, SLOT, (0, 1, 2, 3, 4, 5)),
        (0, 2 * SLOT, (0, 1, 2, 3, 4)),
        (0, 7 * SLOT, ()),
        (1, SLOT, (0, 2, 3, 4, 5)),
        (1, 2 * SLOT, (2, 3, 4)),
        (1, 7 * SLOT, ()),
        (2, SLOT, (0, 2, 3, 4, 5)),
        (2, 2 * SLOT, (2, 3, 4)),
        (2, 7 * SLOT, ()),
    ],
)
def test_start_candidates_skip_slots_blocked_for_participants(
    participant_count: int, duration: timedelta, expected: tuple[int, ...]
) -> None:
    """Offer only starts where every participant is free for the whole duration."""
    participants: frozenset[PersonId] = frozenset(
        PersonId(f"person_{index}") for index in range(participant_count)
    )
    item: Task = task("movable", duration=duration, people=participants, required=False)
    fixed: FixedTask = FixedTask(
        TaskId("fixed"),
        "Existing",
        START + timedelta(minutes=40),
        timedelta(minutes=10),
        participants,
    )
    compiled: CompiledProblem = compile_problem(
        problem(item, fixed_tasks=(fixed,)), DEFAULT_POLICY
    )
    assert tuple(compiled.placements[item.id]) == expected


@pytest.mark.parametrize(
    ("importance", "strength", "hours", "required", "expected"),
    [
        (Importance.LOW, Strength.WEAK, 0.25, False, 3 / 14),
        (Importance.MEDIUM, Strength.NORMAL, -0.25, True, 30 / 29),
        (Importance.HIGH, Strength.STRONG, 0.0, True, 0.0),
        (Importance.LOW, Strength.STRONG, 24.0, False, 160 / 107),
        (Importance.MEDIUM, Strength.NORMAL, -24.0, True, 40 / 7),
        (Importance.HIGH, Strength.STRONG, 24.0, False, 480 / 17),
    ],
)
def test_stability_objective_uses_bounded_cost(
    importance: Importance,
    strength: Strength,
    hours: float,
    required: bool,
    expected: float,
) -> None:
    """Charge bounded stability cost for a task with one available start."""
    item: Task = replace(
        task("movable", required=required), importance=importance, stability=strength
    )
    previous_start: datetime = START - timedelta(hours=hours)
    previous: Schedule = Schedule(
        (
            ScheduledTask(
                item.id, "Previous name", previous_start, previous_start + item.duration
            ),
        ),
        (),
    )
    policy: ObjectivePolicy = replace(DEFAULT_POLICY, stability_drop_cost_ratio=0.3)
    compiled: CompiledProblem = compile_problem(
        problem(item, end=START + SLOT), policy, previous
    )
    result: mathopt.SolveResult = mathopt.solve(
        compiled.model, mathopt.SolverType.GSCIP
    )
    assert result.variable_values()[compiled.presences[item.id]] == pytest.approx(1.0)
    assert result.objective_value() == pytest.approx(expected)


def test_previous_dropped_tasks_have_no_stability_cost() -> None:
    """Schedule previously dropped tasks without charging movement."""
    item: Task = task("movable", required=False)
    previous: Schedule = Schedule((), (DroppedTask(item.id, "Previous name"),))
    compiled: CompiledProblem = compile_problem(
        problem(item, end=START + SLOT), DEFAULT_POLICY, previous
    )
    result: mathopt.SolveResult = mathopt.solve(
        compiled.model, mathopt.SolverType.GSCIP
    )
    assert result.objective_value() == pytest.approx(0.0)
