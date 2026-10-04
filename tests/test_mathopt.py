from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.compile import CompiledProblem, compile_problem
from intent_to_schedule.adapter.mathopt.evaluate import compile_evaluation
from intent_to_schedule.adapter.mathopt.measure import PointExpression
from intent_to_schedule.adapter.mathopt.solve import MathOptSchedulingSolver
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import Infeasible, Solved
from intent_to_schedule.domain.availability import available_start_slots, free_slots
from intent_to_schedule.domain.calendar import Availability, Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint, SoftConstraint
from intent_to_schedule.domain.consistency import AllOf
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
from intent_to_schedule.domain.schedule import DroppedTask, Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId


START: datetime = datetime(2026, 10, 1, 9)
SLOT: timedelta = timedelta(minutes=30)
SOLVER: MathOptSchedulingSolver = MathOptSchedulingSolver(DEFAULT_POLICY)


def task(name: str, *, duration: timedelta = SLOT, people: frozenset[PersonId] = frozenset(), required: bool = True) -> Task:
    return Task(TaskId(name), name, duration, people, Importance.LOW, required)


def problem(
    *tasks: Task,
    constraints: tuple[HardConstraint | SoftConstraint, ...] = (),
    end: datetime = START + timedelta(hours=3),
    availabilities: tuple[Availability, ...] | None = None,
    fixed_tasks: tuple[FixedTask, ...] = (),
) -> SchedulingProblem:
    people: tuple[Person, ...] = tuple(
        Person(person_id, person_id.value)
        for person_id in {person_id for item in (*tasks, *fixed_tasks) for person_id in item.participant_ids}
    )
    if availabilities is None:
        availabilities = tuple(Availability(person.id, (TimeInterval(START, end),)) for person in people)
    calendar: Calendar = Calendar(TimeGrid(TimeInterval(START, end), SLOT), availabilities)
    return SchedulingProblem(calendar, people, tasks, fixed_tasks, constraints)


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
    assert dropped.dropped == (DroppedTask(item.id, item.name),)


def test_soft_interval_intrusion_avoids_region() -> None:
    item: Task = task("interval", duration=timedelta(hours=1))
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("intrusion"),
        IntervalMeasure(frozenset({item.id})),
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
    assert len(result.dropped) == 1


def test_hard_intrusion_excludes_region() -> None:
    item: Task = task("hard", duration=timedelta(hours=1))
    constraint: HardConstraint = HardConstraint(
        ConstraintId("hard"),
        IntervalMeasure(frozenset({item.id})),
        Intrusion((TimeInterval(START, START + timedelta(hours=1)),)),
    )
    result: Schedule = schedule_for(problem(item, constraints=(constraint,)))
    assert starts(result)[item.id] >= START + timedelta(hours=1)


def test_fixed_task_rounds_outward() -> None:
    person_id: PersonId = PersonId("person")
    item: Task = task("movable", people=frozenset({person_id}))
    fixed: FixedTask = FixedTask(
        TaskId("fixed"), "Existing", START + timedelta(minutes=10), timedelta(minutes=30), frozenset({person_id})
    )
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("early"), PointMeasure(item.id), Distance(START), Strength.STRONG
    )
    value: SchedulingProblem = problem(item, constraints=(preference,), end=START + timedelta(minutes=90), fixed_tasks=(fixed,))
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    assert set(compiled.placements[item.id]) == {2}
    assert starts(schedule_for(value))[item.id] == START + 2 * SLOT


def test_overlapping_fixed_tasks_still_solve() -> None:
    person_id: PersonId = PersonId("person")
    first: FixedTask = FixedTask(TaskId("first"), "First", START, timedelta(hours=1), frozenset({person_id}))
    second: FixedTask = replace(first, id=TaskId("second"), start=START + SLOT)
    movable: Task = task("movable", people=frozenset({person_id}))
    result: Schedule = schedule_for(problem(movable, fixed_tasks=(first, second)))
    assert set(starts(result)) == {movable.id}
    assert starts(result)[movable.id] >= START + timedelta(minutes=90)
    assert result.dropped == ()
    assert schedule_for(problem(fixed_tasks=(first, second), availabilities=(Availability(person_id, ()),))) == Schedule((), ())


def test_dependency_shortfall_schedules_task_after_rounded_fixed_task() -> None:
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Existing", START + timedelta(minutes=10), timedelta(minutes=30), frozenset())
    movable: Task = task("movable")
    constraint: HardConstraint = HardConstraint(ConstraintId("after"), DependencyMeasure(fixed.id, movable.id), Shortfall(SLOT))
    preference: SoftConstraint = SoftConstraint(ConstraintId("early"), PointMeasure(movable.id), Distance(START), Strength.STRONG)
    value: SchedulingProblem = problem(movable, fixed_tasks=(fixed,), constraints=(constraint, preference))
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    assert set(compiled.placements[fixed.id]) == {0}
    assert compiled.placements[fixed.id][0].lower_bound == compiled.placements[fixed.id][0].upper_bound == 1
    assert compiled.presences[fixed.id].lower_bound == compiled.presences[fixed.id].upper_bound == 1
    assert compiled.starts[fixed.id].lower_bound == compiled.starts[fixed.id].upper_bound == 0
    assert starts(schedule_for(value)) == {movable.id: START + 3 * SLOT}


def test_dependency_shortfall_to_fixed_task_schedules_task_before_it() -> None:
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Existing", START + timedelta(minutes=100), SLOT, frozenset())
    movable: Task = task("movable")
    constraint: HardConstraint = HardConstraint(ConstraintId("before"), DependencyMeasure(movable.id, fixed.id), Shortfall(SLOT))
    preference: SoftConstraint = SoftConstraint(ConstraintId("late"), PointMeasure(movable.id), Distance(START + timedelta(hours=2)), Strength.STRONG)
    assert starts(schedule_for(problem(movable, fixed_tasks=(fixed,), constraints=(constraint, preference)))) == {movable.id: START + SLOT}


@pytest.mark.parametrize("quantity, upper", [(AggregateQuantity.COUNT, 0), (AggregateQuantity.TOTAL_DURATION, SLOT)])
def test_fixed_task_aggregates_use_rounded_duration(quantity: AggregateQuantity, upper: int | timedelta) -> None:
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Existing", START + timedelta(minutes=10), SLOT, frozenset())
    constraint: HardConstraint = HardConstraint(ConstraintId("aggregate"), AggregateMeasure(frozenset({fixed.id}), quantity), Excess(upper))
    assert isinstance(SOLVER.solve(problem(fixed_tasks=(fixed,), constraints=(constraint,))), Infeasible)


def test_fixed_task_point_and_interval_measures_use_rounded_slots() -> None:
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Existing", START + timedelta(minutes=10), SLOT, frozenset())
    point: HardConstraint = HardConstraint(ConstraintId("point"), PointMeasure(fixed.id), Distance(START))
    assert schedule_for(problem(fixed_tasks=(fixed,), constraints=(point,))) == Schedule((), ())
    interval: HardConstraint = HardConstraint(ConstraintId("interval"), IntervalMeasure(frozenset({fixed.id})), Intrusion((TimeInterval(START + SLOT, START + 2 * SLOT),)))
    assert isinstance(SOLVER.solve(problem(fixed_tasks=(fixed,), constraints=(interval,))), Infeasible)


def test_dependency_to_fixed_task_outside_horizon_is_inactive_when_task_drops() -> None:
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Past event", START - timedelta(days=1), SLOT, frozenset())
    movable: Task = task("optional", required=False)
    constraint: HardConstraint = HardConstraint(ConstraintId("before"), DependencyMeasure(movable.id, fixed.id), Shortfall(SLOT))
    result: Schedule = schedule_for(problem(movable, fixed_tasks=(fixed,), constraints=(constraint,)))
    assert result == Schedule((), (DroppedTask(movable.id, movable.name),))


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
    assert result.schedule.dropped == (DroppedTask(item.id, item.name),)


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
    assert dropped.dropped == (DroppedTask(optional.id, optional.name),)
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
    assert result.dropped == (DroppedTask(second.id, second.name),)


def test_unsupported_evaluation_raises_value_error() -> None:
    grid: TimeGrid = TimeGrid(TimeInterval(START, START + SLOT), SLOT)
    model: mathopt.Model = mathopt.Model()
    expression: PointExpression = PointExpression({0: model.add_binary_variable()})
    with pytest.raises(ValueError, match="Unsupported evaluation"):
        compile_evaluation(expression, Shortfall(SLOT), model, grid)


@pytest.mark.parametrize("duration", [SLOT, 2 * SLOT, 7 * SLOT])
@pytest.mark.parametrize("participant_count", [0, 1, 2])
def test_solver_start_candidates_match_shared_availability(duration: timedelta, participant_count: int) -> None:
    participants: frozenset[PersonId] = frozenset(PersonId(f"person_{index}") for index in range(participant_count))
    item: Task = task("movable", duration=duration, people=participants, required=False)
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Existing", START + timedelta(minutes=40), timedelta(minutes=10), participants)
    value: SchedulingProblem = problem(item, fixed_tasks=(fixed,))
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    participants_free: tuple[tuple[bool, ...], ...] = tuple(free_slots(value, person_id) for person_id in participants)
    expected: tuple[int, ...] = available_start_slots(value.calendar.grid, participants_free, duration)
    assert tuple(compiled.placements[item.id]) == expected


def test_compile_computes_person_free_slots_once() -> None:
    person_id: PersonId = PersonId("shared")
    value: SchedulingProblem = problem(task("first", people=frozenset({person_id})), task("second", people=frozenset({person_id})))
    free_slots_mock: MagicMock
    starts_mock: MagicMock
    with patch("intent_to_schedule.adapter.mathopt.compile.free_slots", wraps=free_slots) as free_slots_mock:
        with patch("intent_to_schedule.adapter.mathopt.compile.available_start_slots", wraps=available_start_slots) as starts_mock:
            compile_problem(value, DEFAULT_POLICY)
    free_slots_mock.assert_called_once_with(value, person_id)
    assert starts_mock.call_count == 2


@pytest.mark.parametrize("importance", list(Importance))
def test_stability_moves_optional_task_instead_of_dropping(importance: Importance) -> None:
    """Move an optional task when its previous slot becomes occupied."""
    person_id: PersonId = PersonId("person")
    item: Task = replace(
        task("movable", people=frozenset({person_id}), required=False),
        importance=importance,
    )
    distant_start: datetime = START + timedelta(hours=24)
    availability: Availability = Availability(person_id, (
        TimeInterval(START, START + SLOT),
        TimeInterval(distant_start, distant_start + SLOT),
    ))
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("early"), PointMeasure(item.id), Distance(START), Strength.WEAK
    )
    original: SchedulingProblem = problem(
        item, constraints=(preference,), end=distant_start + SLOT,
        availabilities=(availability,),
    )
    previous: Schedule = schedule_for(original)
    assert starts(previous) == {item.id: START}
    fixed: FixedTask = FixedTask(
        TaskId("occupied"), "Occupied", START, SLOT, item.participant_ids
    )
    changed: SchedulingProblem = replace(original, constraints=(), fixed_tasks=(fixed,))
    result: Solved | Infeasible = Scheduling(SOLVER, AllOf()).solve(changed, previous)
    assert isinstance(result, Solved)
    assert starts(result.schedule) == {item.id: distant_start}


def test_stability_keeps_required_task_at_nearest_free_start() -> None:
    """Prefer the nearest free start for a required task."""
    person_id: PersonId = PersonId("person")
    item: Task = replace(task("required", people=frozenset({person_id})), importance=Importance.HIGH)
    target: datetime = START + timedelta(hours=1)
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("target"), PointMeasure(item.id), Distance(target), Strength.NORMAL
    )
    original: SchedulingProblem = problem(item, constraints=(preference,))
    previous: Schedule = schedule_for(original)
    assert starts(previous) == {item.id: target}
    fixed: FixedTask = FixedTask(
        TaskId("occupied"), "Occupied", START, timedelta(hours=1.5), item.participant_ids
    )
    changed: SchedulingProblem = replace(original, constraints=(), fixed_tasks=(fixed,))
    result: Solved | Infeasible = Scheduling(SOLVER, AllOf()).solve(changed, previous)
    assert isinstance(result, Solved)
    assert starts(result.schedule) == {item.id: target + SLOT}


@pytest.mark.parametrize("importance", list(Importance))
@pytest.mark.parametrize("strength", list(Strength))
@pytest.mark.parametrize("hours", [-24.0, -0.25, 0.0, 0.25, 24.0])
@pytest.mark.parametrize("required", [False, True])
def test_stability_objective_uses_capped_cost(
    importance: Importance, strength: Strength, hours: float, required: bool
) -> None:
    """Charge capped stability cost for a task with one available start."""
    item: Task = replace(task("movable", required=required), importance=importance, stability=strength)
    previous_start: datetime = START - timedelta(hours=hours)
    previous: Schedule = Schedule((
        ScheduledTask(item.id, "Previous name", previous_start, previous_start + item.duration),
    ), ())
    policy: ObjectivePolicy = replace(DEFAULT_POLICY, stability_drop_cost_ratio=0.3)
    compiled: CompiledProblem = compile_problem(problem(item, end=START + SLOT), policy, previous)
    result: mathopt.SolveResult = mathopt.solve(compiled.model, mathopt.SolverType.GSCIP)
    assert result.variable_values()[compiled.presences[item.id]] == pytest.approx(1.0)
    expected: float = min(policy.weight(strength) * abs(hours), 0.3 * policy.drop_cost(importance))
    assert result.objective_value() == pytest.approx(expected)


def test_previous_dropped_tasks_have_no_stability_cost() -> None:
    """Schedule previously dropped tasks without charging movement."""
    item: Task = task("movable", required=False)
    previous: Schedule = Schedule((), (DroppedTask(item.id, "Previous name"),))
    compiled: CompiledProblem = compile_problem(problem(item, end=START + SLOT), DEFAULT_POLICY, previous)
    result: mathopt.SolveResult = mathopt.solve(compiled.model, mathopt.SolverType.GSCIP)
    assert result.objective_value() == pytest.approx(0.0)
