from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.compile import CompiledProblem, _free_slots, compile_problem
from intent_to_schedule.adapter.mathopt.evaluate import compile_evaluation
from intent_to_schedule.adapter.mathopt.measure import PointExpression
from intent_to_schedule.adapter.mathopt.solve import MathOptSchedulingSolver
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.solve import Infeasible, Solved
from intent_to_schedule.domain.calendar import Availability, BusyInterval, Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint, SoftConstraint
from intent_to_schedule.domain.evaluation import Distance, Excess, Intrusion, Shortfall
from intent_to_schedule.domain.measure import (
    AggregateMeasure,
    AggregateQuantity,
    DependencyMeasure,
    IntervalMeasure,
    PointMeasure,
)
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId


START = datetime(2026, 10, 1, 9)
SLOT = timedelta(minutes=30)
SOLVER = MathOptSchedulingSolver(DEFAULT_POLICY)


def task(name: str, *, duration: timedelta = SLOT, people: frozenset[PersonId] = frozenset(), required: bool = True) -> Task:
    return Task(TaskId(name), name, duration, people, Importance.LOW, required)


def problem(
    *tasks: Task,
    constraints: tuple[HardConstraint | SoftConstraint, ...] = (),
    end: datetime = START + timedelta(hours=3),
    availabilities: tuple[Availability, ...] | None = None,
    busy: tuple[BusyInterval, ...] = (),
) -> SchedulingProblem:
    people: tuple[Person, ...] = tuple(
        Person(person_id, person_id.value)
        for person_id in {person_id for item in tasks for person_id in item.participant_ids}
    )
    if availabilities is None:
        availabilities = tuple(Availability(person.id, (TimeInterval(START, end),)) for person in people)
    calendar: Calendar = Calendar(TimeGrid(TimeInterval(START, end), SLOT), availabilities, busy)
    return SchedulingProblem(calendar, people, tasks, constraints)


def schedule_for(value: SchedulingProblem) -> Schedule:
    result: Solved | Infeasible = SOLVER.solve(value)
    assert isinstance(result, Solved)
    return result.schedule


def starts(schedule: Schedule) -> dict[TaskId, datetime]:
    return {item.task_id: item.start for item in schedule.scheduled}


def test_soft_point_distance_chooses_target_and_can_drop() -> None:
    item: Task = task("point")
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("point"), PointMeasure(item.id), Distance(START + timedelta(hours=1)), Strength.STRONG
    )
    result: Schedule = schedule_for(problem(item, constraints=(preference,)))
    assert starts(result)[item.id] == START + timedelta(hours=1)

    optional: Task = replace(item, required=False)
    distant: SoftConstraint = replace(preference, evaluation=Distance(START + timedelta(hours=4)))
    dropped: Schedule = schedule_for(problem(optional, constraints=(distant,), end=START + SLOT))
    assert dropped.dropped_task_ids == frozenset({item.id})


def test_soft_interval_intrusion_avoids_region() -> None:
    item: Task = task("interval", duration=timedelta(hours=1))
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("intrusion"),
        IntervalMeasure(item.id),
        Intrusion((TimeInterval(START, START + timedelta(hours=1)),)),
        Strength.STRONG,
    )
    result: Schedule = schedule_for(problem(item, constraints=(preference,)))
    assert starts(result)[item.id] >= START + timedelta(hours=1)


def test_soft_dependency_distance_targets_gap() -> None:
    first: Task = task("first")
    second: Task = task("second")
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("distance"),
        DependencyMeasure(first.id, second.id),
        Distance(SLOT),
        Strength.STRONG,
    )
    result: dict[TaskId, datetime] = starts(schedule_for(problem(first, second, constraints=(preference,))))
    assert result[second.id] - result[first.id] - first.duration == SLOT


def test_soft_dependency_shortfall_requires_gap() -> None:
    first: Task = task("first")
    second: Task = task("second")
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("shortfall"),
        DependencyMeasure(first.id, second.id),
        Shortfall(timedelta(hours=1)),
        Strength.STRONG,
    )
    result: dict[TaskId, datetime] = starts(schedule_for(problem(first, second, constraints=(preference,))))
    assert result[second.id] - result[first.id] - first.duration >= timedelta(hours=1)


@pytest.mark.parametrize(
    ("quantity", "upper"),
    [
        (AggregateQuantity.COUNT, 1),
        (AggregateQuantity.TOTAL_DURATION, timedelta(minutes=30)),
    ],
)
def test_soft_aggregate_excess_drops_one_task(quantity: AggregateQuantity, upper: int | timedelta) -> None:
    first: Task = task("first", required=False)
    second: Task = task("second", required=False)
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("excess"),
        AggregateMeasure(frozenset({first.id, second.id}), quantity),
        Excess(upper),
        Strength.STRONG,
    )
    result: Schedule = schedule_for(problem(first, second, constraints=(preference,)))
    assert len(result.scheduled) == 1
    assert len(result.dropped_task_ids) == 1


def test_hard_intrusion_excludes_region() -> None:
    item: Task = task("hard", duration=timedelta(hours=1))
    constraint: HardConstraint = HardConstraint(
        ConstraintId("hard"),
        IntervalMeasure(item.id),
        Intrusion((TimeInterval(START, START + timedelta(hours=1)),)),
    )
    result: Schedule = schedule_for(problem(item, constraints=(constraint,)))
    assert starts(result)[item.id] >= START + timedelta(hours=1)


def test_busy_interval_rounds_outward() -> None:
    person_id: PersonId = PersonId("person")
    item: Task = task("busy", people=frozenset({person_id}))
    busy: BusyInterval = BusyInterval(
        person_id, TimeInterval(START + timedelta(minutes=10), START + timedelta(minutes=20))
    )
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("early"), PointMeasure(item.id), Distance(START), Strength.STRONG
    )
    value: SchedulingProblem = problem(item, constraints=(preference,), end=START + timedelta(hours=1), busy=(busy,))
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    assert set(compiled.placements[item.id]) == {1}
    assert starts(schedule_for(value))[item.id] == START + SLOT


def test_intervals_before_horizon_leave_free_slots_unchanged() -> None:
    person_id: PersonId = PersonId("person")
    item: Task = task("early", people=frozenset({person_id}))
    in_horizon: TimeInterval = TimeInterval(START, START + timedelta(hours=1))
    before_horizon: TimeInterval = TimeInterval(START - timedelta(hours=1), START - SLOT)
    baseline: SchedulingProblem = problem(
        item, end=in_horizon.end, availabilities=(Availability(person_id, (in_horizon,)),)
    )
    availability: SchedulingProblem = problem(
        item, end=in_horizon.end,
        availabilities=(Availability(person_id, (before_horizon, in_horizon)),),
    )
    busy: SchedulingProblem = problem(
        item, end=in_horizon.end,
        availabilities=(Availability(person_id, (in_horizon,)),),
        busy=(BusyInterval(person_id, before_horizon),),
    )
    expected: list[bool] = _free_slots(person_id, baseline, 2)
    assert expected == [True, True]
    assert _free_slots(person_id, availability, 2) == expected
    assert _free_slots(person_id, busy, 2) == expected


def test_count_penalty_scales_soft_objective() -> None:
    item: Task = task("count", required=False)
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("count"),
        AggregateMeasure(frozenset({item.id}), AggregateQuantity.COUNT),
        Excess(0),
        Strength.WEAK,
    )
    policy: ObjectivePolicy = replace(DEFAULT_POLICY, per_count=10.0)
    result: Solved | Infeasible = MathOptSchedulingSolver(policy).solve(problem(item, constraints=(preference,)))
    assert isinstance(result, Solved)
    assert result.schedule.dropped_task_ids == frozenset({item.id})


def test_solver_passes_time_limit() -> None:
    limit: timedelta = timedelta(seconds=1)
    solve_mock: MagicMock
    with patch("intent_to_schedule.adapter.mathopt.solve.mathopt.solve", wraps=mathopt.solve) as solve_mock:
        result: Solved | Infeasible = MathOptSchedulingSolver(DEFAULT_POLICY, time_limit=limit).solve(
            problem(task("timed"))
        )
    assert isinstance(result, Solved)
    assert solve_mock.call_args.kwargs["params"].time_limit == limit


def test_shared_person_tasks_do_not_overlap() -> None:
    person_id: PersonId = PersonId("shared")
    first: Task = task("first", duration=timedelta(hours=1), people=frozenset({person_id}))
    second: Task = task("second", duration=timedelta(hours=1), people=frozenset({person_id}))
    value: SchedulingProblem = problem(first, second, end=START + timedelta(hours=2))
    result: dict[TaskId, datetime] = starts(schedule_for(value))
    assert abs(result[first.id] - result[second.id]) == timedelta(hours=1)


def test_unavailable_optional_drops_and_required_is_infeasible() -> None:
    person_id: PersonId = PersonId("unavailable")
    optional: Task = task("optional", people=frozenset({person_id}), required=False)
    availability: Availability = Availability(person_id, ())
    dropped: Schedule = schedule_for(problem(optional, availabilities=(availability,)))
    assert dropped.dropped_task_ids == frozenset({optional.id})
    required: Task = replace(optional, required=True)
    assert isinstance(SOLVER.solve(problem(required, availabilities=(availability,))), Infeasible)


def test_hard_dependency_is_inactive_when_task_drops() -> None:
    person_id: PersonId = PersonId("unavailable")
    first: Task = task("first")
    second: Task = task("second", people=frozenset({person_id}), required=False)
    constraint: HardConstraint = HardConstraint(
        ConstraintId("dependency"),
        DependencyMeasure(first.id, second.id),
        Shortfall(timedelta(hours=100)),
    )
    result: Schedule = schedule_for(problem(first, second, constraints=(constraint,), availabilities=(Availability(person_id, ()),)))
    assert {item.task_id for item in result.scheduled} == {first.id}
    assert result.dropped_task_ids == frozenset({second.id})


def test_unsupported_evaluation_raises_value_error() -> None:
    grid: TimeGrid = TimeGrid(TimeInterval(START, START + SLOT), SLOT)
    model: mathopt.Model = mathopt.Model()
    expression: PointExpression = PointExpression({0: model.add_binary_variable()})
    with pytest.raises(ValueError, match="Unsupported evaluation"):
        compile_evaluation(expression, Shortfall(SLOT), model, grid)
