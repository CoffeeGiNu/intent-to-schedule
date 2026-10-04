from dataclasses import replace
from datetime import datetime, time, timedelta, timezone

import pytest
from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.compile import CompiledProblem, compile_problem
from intent_to_schedule.adapter.mathopt.solve import MathOptSchedulingSolver
from intent_to_schedule.application.policy import DEFAULT_POLICY
from intent_to_schedule.application.solve import Infeasible, Solved
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
from intent_to_schedule.domain.consistency import (
    AlignedToSlots,
    UniqueIds,
)
from intent_to_schedule.domain.constraint import (
    Constraint,
    ConstraintId,
    HardConstraint,
    SoftConstraint,
)
from intent_to_schedule.domain.measure import AggregateQuantity, Boundary
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId
from intent_to_schedule.domain.time_windows import TimeRange, TimeRelation, TimeWindow

START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone(timedelta(hours=9)))
HOUR: timedelta = timedelta(hours=1)
PERSON: PersonId = PersonId("person")
SOLVER: MathOptSchedulingSolver = MathOptSchedulingSolver(DEFAULT_POLICY)


def task(name: str, duration: timedelta = HOUR, required: bool = True) -> Task:
    return Task(
        TaskId(name), name, duration, frozenset({PERSON}), Importance.LOW, required
    )


def problem(
    tasks: tuple[Task, ...],
    requirements: tuple[Condition, ...] = (),
    preferences: tuple[Condition, ...] = (),
    horizon: TimeInterval = TimeInterval(START, START + 4 * HOUR),
    fixed_tasks: tuple[FixedTask, ...] = (),
    availability: tuple[TimeInterval, ...] | None = None,
) -> SchedulingProblem:
    grid: TimeGrid = TimeGrid(horizon, HOUR)
    constraints: list[Constraint] = []
    condition: Condition
    for condition in requirements:
        constraints.append(
            HardConstraint(ConstraintId(str(len(constraints))), condition)
        )
    for condition in preferences:
        constraints.append(
            SoftConstraint(
                ConstraintId(str(len(constraints))), condition, Strength.WEAK
            )
        )
    value: SchedulingProblem = SchedulingProblem(
        Calendar(
            grid,
            (
                Availability(
                    PERSON, availability if availability is not None else (horizon,)
                ),
            ),
        ),
        (Person(PERSON, "Person"),),
        tasks,
        fixed_tasks,
        tuple(constraints),
    )
    assert AlignedToSlots().validate(value).is_empty
    assert UniqueIds().validate(value).is_empty
    return value


def starts(value: SchedulingProblem) -> dict[TaskId, datetime]:
    result: Solved | Infeasible = SOLVER.solve(value)
    assert isinstance(result, Solved)
    return {item.task_id: item.start for item in result.schedule.scheduled}


def objective(value: SchedulingProblem) -> float:
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    result: mathopt.SolveResult = mathopt.solve(
        compiled.model, mathopt.SolverType.GSCIP
    )
    assert result.termination.reason is mathopt.TerminationReason.OPTIMAL
    return result.objective_value()


def bound(
    item: Task | FixedTask,
    boundary: Boundary,
    relation: TimeBoundRelation,
    at: datetime,
) -> TimeBoundCondition:
    return TimeBoundCondition(frozenset({item.id}), boundary, relation, at)


@pytest.mark.parametrize("boundary", list(Boundary))
def test_hard_deadline_moves_task_before_bound(boundary: Boundary) -> None:
    item: Task = task("deadline")
    deadline: datetime = START + timedelta(minutes=150)
    value: SchedulingProblem = problem(
        (item,),
        (bound(item, boundary, TimeBoundRelation.AT_OR_BEFORE, deadline),),
        (bound(item, Boundary.START, TimeBoundRelation.AT, START + 3 * HOUR),),
    )
    assert (
        starts(value)[item.id]
        == START + (2 if boundary is Boundary.START else 1) * HOUR
    )


@pytest.mark.parametrize("boundary", list(Boundary))
def test_at_or_after_moves_task_after_bound(boundary: Boundary) -> None:
    item: Task = task("earliest")
    value: SchedulingProblem = problem(
        (item,),
        (
            bound(
                item,
                boundary,
                TimeBoundRelation.AT_OR_AFTER,
                START + timedelta(minutes=90),
            ),
        ),
        (bound(item, Boundary.START, TimeBoundRelation.AT, START),),
    )
    assert (
        starts(value)[item.id]
        == START + (2 if boundary is Boundary.START else 1) * HOUR
    )


def test_soft_deadline_prefers_one_hour_late_over_two() -> None:
    item: Task = task("late")
    deadline: TimeBoundCondition = bound(
        item, Boundary.START, TimeBoundRelation.AT_OR_BEFORE, START
    )
    value: SchedulingProblem = problem(
        (item,),
        preferences=(deadline,),
        availability=(TimeInterval(START + HOUR, START + 3 * HOUR),),
    )
    assert starts(value)[item.id] == START + HOUR
    assert objective(value) == pytest.approx(1.0)
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


@pytest.mark.parametrize("boundary", list(Boundary))
@pytest.mark.parametrize("relation", list(TimeBoundRelation))
@pytest.mark.parametrize("target_minutes", [-75, 75, 375])
def test_time_bound_cost_uses_exact_hours(
    boundary: Boundary, relation: TimeBoundRelation, target_minutes: int
) -> None:
    item: Task = task("cost", duration=2 * HOUR)
    at: datetime = START + timedelta(minutes=target_minutes)
    value: SchedulingProblem = problem(
        (item,),
        preferences=(bound(item, boundary, relation, at),),
        horizon=TimeInterval(START, START + item.duration),
    )
    difference: float = (
        (START if boundary is Boundary.START else START + item.duration) - at
    ) / HOUR
    expected: float = (
        max(difference, 0.0)
        if relation is TimeBoundRelation.AT_OR_BEFORE
        else max(-difference, 0.0)
        if relation is TimeBoundRelation.AT_OR_AFTER
        else abs(difference)
    )
    assert objective(value) == pytest.approx(expected)


def test_time_bound_criteria_sum_cost_over_tasks() -> None:
    first: Task = task("first")
    second: Task = task("second")
    condition: TimeBoundCondition = TimeBoundCondition(
        frozenset({first.id, second.id}),
        Boundary.END,
        TimeBoundRelation.AT_OR_BEFORE,
        START,
    )
    assert objective(
        problem(
            (first, second),
            preferences=(condition,),
            horizon=TimeInterval(START, START + 2 * HOUR),
        )
    ) == pytest.approx(3.0)


@pytest.mark.parametrize("relation", list(TimeBoundRelation))
def test_unscheduled_task_has_no_time_bound_violation(
    relation: TimeBoundRelation,
) -> None:
    item: Task = task("optional", required=False)
    condition: TimeBoundCondition = bound(
        item, Boundary.END, relation, START + timedelta(minutes=15)
    )
    value: SchedulingProblem = problem(
        (item,), requirements=(condition,), availability=()
    )
    assert starts(value) == {}
    assert objective(value) == pytest.approx(DEFAULT_POLICY.drop_cost(item.importance))


def test_zero_minimum_gap_orders_tasks() -> None:
    first: Task = task("first")
    second: Task = task("second")
    condition: TaskGapCondition = TaskGapCondition(
        first.id, second.id, TaskGapRelation.AT_LEAST, timedelta(0)
    )
    result: dict[TaskId, datetime] = starts(
        problem(
            (first, second), (condition,), horizon=TimeInterval(START, START + 2 * HOUR)
        )
    )
    assert result == {first.id: START, second.id: START + HOUR}


def test_exact_zero_gap_places_task_after_fixed_task() -> None:
    fixed: FixedTask = FixedTask(
        TaskId("fixed"), "Fixed", START, HOUR, frozenset({PERSON})
    )
    item: Task = task("following")
    condition: TaskGapCondition = TaskGapCondition(
        fixed.id, item.id, TaskGapRelation.EXACTLY, timedelta(0)
    )
    assert starts(problem((item,), (condition,), fixed_tasks=(fixed,))) == {
        item.id: START + HOUR
    }


@pytest.mark.parametrize("relation", list(TaskGapRelation))
def test_unscheduled_task_has_no_gap_violation(relation: TaskGapRelation) -> None:
    fixed: FixedTask = FixedTask(
        TaskId("fixed"), "Fixed", START, HOUR, frozenset({PERSON})
    )
    item: Task = task("optional", required=False)
    condition: TaskGapCondition = TaskGapCondition(
        fixed.id, item.id, relation, 100 * HOUR
    )
    value: SchedulingProblem = problem(
        (item,), (condition,), fixed_tasks=(fixed,), availability=()
    )
    assert starts(value) == {}
    assert objective(value) == pytest.approx(DEFAULT_POLICY.drop_cost(item.importance))


@pytest.mark.parametrize("boundary", list(Boundary))
@pytest.mark.parametrize("relation", list(TimeBoundRelation))
def test_fixed_time_bound_cost_uses_existing_real_interval(
    boundary: Boundary, relation: TimeBoundRelation
) -> None:
    fixed: FixedTask = FixedTask(
        TaskId("fixed"),
        "Fixed",
        START + timedelta(minutes=10),
        HOUR,
        frozenset({PERSON}),
    )
    target: datetime = START + timedelta(minutes=45)
    difference: float = (
        (fixed.interval.start if boundary is Boundary.START else fixed.interval.end)
        - target
    ) / HOUR
    expected: float = (
        max(difference, 0.0)
        if relation is TimeBoundRelation.AT_OR_BEFORE
        else max(-difference, 0.0)
        if relation is TimeBoundRelation.AT_OR_AFTER
        else abs(difference)
    )
    value: SchedulingProblem = problem(
        (),
        preferences=(bound(fixed, boundary, relation, target),),
        fixed_tasks=(fixed,),
    )
    assert objective(value) == pytest.approx(expected)


@pytest.mark.parametrize("relation", list(TaskGapRelation))
def test_unaligned_gap_cost_uses_exact_hours(relation: TaskGapRelation) -> None:
    fixed: FixedTask = FixedTask(
        TaskId("fixed"), "Fixed", START, HOUR, frozenset({PERSON})
    )
    item: Task = task("following")
    condition: TaskGapCondition = TaskGapCondition(
        fixed.id, item.id, relation, timedelta(minutes=15)
    )
    value: SchedulingProblem = problem(
        (item,),
        preferences=(condition,),
        fixed_tasks=(fixed,),
        horizon=TimeInterval(START, START + 2 * HOUR),
    )
    assert objective(value) == pytest.approx(0.25)


@pytest.mark.parametrize(
    "quantity,maximum",
    [(AggregateQuantity.COUNT, 1), (AggregateQuantity.TOTAL_DURATION, 2 * HOUR)],
)
def test_daily_limit_distributes_tasks_across_two_dates(
    quantity: AggregateQuantity, maximum: int | timedelta
) -> None:
    first: Task = task("first", duration=2 * HOUR)
    second: Task = task("second", duration=2 * HOUR)
    horizon: TimeInterval = TimeInterval(
        START.replace(hour=22),
        (START + timedelta(days=1)).replace(hour=4).astimezone(timezone.utc),
    )
    condition: DailyLimitCondition = DailyLimitCondition(
        frozenset({first.id, second.id}), quantity, maximum
    )
    result: dict[TaskId, datetime] = starts(
        problem((first, second), (condition,), horizon=horizon)
    )
    assert {at.date() for at in result.values()} == {
        START.date(),
        (START + timedelta(days=1)).date(),
    }


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
        TaskId("first"), "First", START.replace(hour=23), 2 * HOUR, frozenset({PERSON})
    )
    second: FixedTask = replace(
        first, id=TaskId("second"), start=first.start + timedelta(days=1)
    )
    horizon: TimeInterval = TimeInterval(
        START.replace(hour=22), (START + timedelta(days=2)).replace(hour=2)
    )
    condition: DailyLimitCondition = DailyLimitCondition(
        frozenset({first.id, second.id}), quantity, maximum
    )
    assert objective(
        problem(
            (), preferences=(condition,), fixed_tasks=(first, second), horizon=horizon
        )
    ) == pytest.approx(expected)


@pytest.mark.parametrize("relation", list(TimeRelation))
def test_hard_time_window_places_whole_task_in_allowed_region(
    relation: TimeRelation,
) -> None:
    item: Task = task("window", duration=2 * HOUR)
    condition: TimeWindowCondition = TimeWindowCondition(
        frozenset({item.id}),
        relation,
        (TimeWindow(None, None, TimeRange(time(9), time(11))),),
    )
    assert starts(problem((item,), (condition,))) == {
        item.id: START + (0 if relation is TimeRelation.WITHIN else 2) * HOUR
    }
