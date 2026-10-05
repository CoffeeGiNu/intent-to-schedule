"""Tests for schedules returned by the MathOpt solver."""

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest
from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.compile import CompiledProblem, compile_problem
from intent_to_schedule.adapter.mathopt.solve import MathOptSchedulingSolver
from intent_to_schedule.application.command import AddTask, Executed, Rejected
from intent_to_schedule.application.objective import (
    evaluate_constraints,
    summarize_schedule,
)
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.query import (
    Answered,
    AvailableStartsAnswer,
    AvailableStartsQuery,
)
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import (
    Conflicts,
    DroppedRequiredTask,
    DropReason,
    Infeasible,
    Solved,
    SolveResult,
)
from intent_to_schedule.domain.availability import free_slots
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
    AllOf,
    AvailabilityForEveryone,
    NonemptyTimeWindows,
    ReferencesExist,
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
SOLVER: MathOptSchedulingSolver = MathOptSchedulingSolver(DEFAULT_POLICY)


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


def schedule_for(value: SchedulingProblem) -> Schedule:
    result: Solved | Infeasible = SOLVER.solve(value)
    assert isinstance(result, Solved)
    return result.schedule


def starts(schedule: Schedule) -> dict[TaskId, datetime]:
    return {item.task_id: item.start for item in schedule.scheduled}


def test_soft_point_distance_chooses_target_and_can_drop() -> None:
    item: Task = task("point")
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("point"),
        TimeBoundCondition(
            frozenset({item.id}),
            Boundary.START,
            TimeBoundRelation.AT,
            START + timedelta(hours=1),
        ),
        Strength.STRONG,
    )
    result: Schedule = schedule_for(problem(item, constraints=(preference,)))
    assert starts(result)[item.id] == START + timedelta(hours=1)

    optional: Task = replace(item, required=False)
    distant: SoftConstraint = replace(
        preference,
        condition=replace(preference.condition, at=START + timedelta(hours=4)),
    )
    dropped: Schedule = schedule_for(
        problem(optional, constraints=(distant,), end=START + SLOT)
    )
    assert dropped.dropped == (DroppedTask(item.id, item.name),)


def test_soft_interval_intrusion_avoids_region() -> None:
    item: Task = task("interval", duration=timedelta(hours=1))
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("intrusion"),
        TimeWindowCondition(
            frozenset({item.id}),
            TimeRelation.AVOID,
            (TimeWindow(None, None, TimeRange(time(9), time(10))),),
        ),
        Strength.STRONG,
    )
    result: Schedule = schedule_for(problem(item, constraints=(preference,)))
    assert starts(result)[item.id] >= START + timedelta(hours=1)


def test_soft_dependency_distance_targets_gap() -> None:
    first: Task = task("first")
    second: Task = task("second")
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("distance"),
        TaskGapCondition(first.id, second.id, TaskGapRelation.EXACTLY, SLOT),
        Strength.STRONG,
    )
    result: dict[TaskId, datetime] = starts(
        schedule_for(problem(first, second, constraints=(preference,)))
    )
    assert result[second.id] - result[first.id] - first.duration == SLOT


def test_soft_dependency_shortfall_requires_gap() -> None:
    first: Task = task("first")
    second: Task = task("second")
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("shortfall"),
        TaskGapCondition(
            first.id, second.id, TaskGapRelation.AT_LEAST, timedelta(hours=1)
        ),
        Strength.STRONG,
    )
    result: dict[TaskId, datetime] = starts(
        schedule_for(problem(first, second, constraints=(preference,)))
    )
    assert result[second.id] - result[first.id] - first.duration >= timedelta(hours=1)


@pytest.mark.parametrize(
    ("quantity", "upper"),
    [
        (AggregateQuantity.COUNT, 1),
        (AggregateQuantity.TOTAL_DURATION, timedelta(minutes=30)),
    ],
)
def test_soft_aggregate_excess_drops_one_task(
    quantity: AggregateQuantity, upper: int | timedelta
) -> None:
    first: Task = task("first", required=False)
    second: Task = task("second", required=False)
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("excess"),
        DailyLimitCondition(frozenset({first.id, second.id}), quantity, upper),
        Strength.STRONG,
    )
    result: Schedule = schedule_for(problem(first, second, constraints=(preference,)))
    assert len(result.scheduled) == 1
    assert len(result.dropped) == 1


def test_fixed_task_rounds_outward() -> None:
    person_id: PersonId = PersonId("person")
    item: Task = task("movable", people=frozenset({person_id}))
    fixed: FixedTask = FixedTask(
        TaskId("fixed"),
        "Existing",
        START + timedelta(minutes=10),
        timedelta(minutes=30),
        frozenset({person_id}),
    )
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("early"),
        TimeBoundCondition(
            frozenset({item.id}), Boundary.START, TimeBoundRelation.AT, START
        ),
        Strength.STRONG,
    )
    value: SchedulingProblem = problem(
        item,
        constraints=(preference,),
        end=START + timedelta(minutes=90),
        fixed_tasks=(fixed,),
    )
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    assert set(compiled.placements[item.id]) == {2}
    assert starts(schedule_for(value))[item.id] == START + 2 * SLOT


def test_overlapping_fixed_tasks_still_solve() -> None:
    person_id: PersonId = PersonId("person")
    first: FixedTask = FixedTask(
        TaskId("first"), "First", START, timedelta(hours=1), frozenset({person_id})
    )
    second: FixedTask = replace(first, id=TaskId("second"), start=START + SLOT)
    movable: Task = task("movable", people=frozenset({person_id}))
    result: Schedule = schedule_for(problem(movable, fixed_tasks=(first, second)))
    assert set(starts(result)) == {movable.id}
    assert starts(result)[movable.id] >= START + timedelta(minutes=90)
    assert result.dropped == ()
    assert schedule_for(
        problem(
            fixed_tasks=(first, second), availabilities=(Availability(person_id, ()),)
        )
    ) == Schedule((), ())


def test_dependency_shortfall_schedules_task_after_real_fixed_task() -> None:
    fixed: FixedTask = FixedTask(
        TaskId("fixed"),
        "Existing",
        START + timedelta(minutes=10),
        timedelta(minutes=30),
        frozenset(),
    )
    movable: Task = task("movable")
    constraint: HardConstraint = HardConstraint(
        ConstraintId("after"),
        TaskGapCondition(fixed.id, movable.id, TaskGapRelation.AT_LEAST, SLOT),
    )
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("early"),
        TimeBoundCondition(
            frozenset({movable.id}), Boundary.START, TimeBoundRelation.AT, START
        ),
        Strength.STRONG,
    )
    value: SchedulingProblem = problem(
        movable, fixed_tasks=(fixed,), constraints=(constraint, preference)
    )
    compiled: CompiledProblem = compile_problem(value, DEFAULT_POLICY)
    assert fixed.id not in compiled.placements
    assert fixed.id not in compiled.presences
    assert fixed.id not in compiled.starts
    assert starts(schedule_for(value)) == {movable.id: START + 3 * SLOT}


def test_dependency_shortfall_to_fixed_task_schedules_task_before_it() -> None:
    fixed: FixedTask = FixedTask(
        TaskId("fixed"), "Existing", START + timedelta(minutes=100), SLOT, frozenset()
    )
    movable: Task = task("movable")
    constraint: HardConstraint = HardConstraint(
        ConstraintId("before"),
        TaskGapCondition(movable.id, fixed.id, TaskGapRelation.AT_LEAST, SLOT),
    )
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("late"),
        TimeBoundCondition(
            frozenset({movable.id}),
            Boundary.START,
            TimeBoundRelation.AT,
            START + timedelta(hours=2),
        ),
        Strength.STRONG,
    )
    assert starts(
        schedule_for(
            problem(movable, fixed_tasks=(fixed,), constraints=(constraint, preference))
        )
    ) == {movable.id: START + SLOT}


def test_fixed_task_point_and_interval_measures_use_real_interval() -> None:
    fixed: FixedTask = FixedTask(
        TaskId("fixed"), "Existing", START + timedelta(minutes=10), SLOT, frozenset()
    )
    point: HardConstraint = HardConstraint(
        ConstraintId("point"),
        TimeBoundCondition(
            frozenset({fixed.id}), Boundary.START, TimeBoundRelation.AT, START
        ),
    )
    assert isinstance(
        SOLVER.solve(problem(fixed_tasks=(fixed,), constraints=(point,))), Infeasible
    )
    interval: HardConstraint = HardConstraint(
        ConstraintId("interval"),
        TimeWindowCondition(
            frozenset({fixed.id}),
            TimeRelation.AVOID,
            (TimeWindow(None, None, TimeRange(time(9, 30), time(10))),),
        ),
    )
    assert isinstance(
        SOLVER.solve(problem(fixed_tasks=(fixed,), constraints=(interval,))), Infeasible
    )


def test_count_penalty_scales_soft_objective() -> None:
    item: Task = task("count", required=False)
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("count"),
        DailyLimitCondition(frozenset({item.id}), AggregateQuantity.COUNT, 0),
        Strength.WEAK,
    )
    policy: ObjectivePolicy = replace(DEFAULT_POLICY, per_count=10.0)
    result: Solved | Infeasible = MathOptSchedulingSolver(policy).solve(
        problem(item, constraints=(preference,))
    )
    assert isinstance(result, Solved)
    assert result.schedule.dropped == (DroppedTask(item.id, item.name),)


def test_solver_passes_time_limit() -> None:
    limit: timedelta = timedelta(seconds=1)
    solve_mock: MagicMock
    with patch(
        "intent_to_schedule.adapter.mathopt.solve.mathopt.solve", wraps=mathopt.solve
    ) as solve_mock:
        result: Solved | Infeasible = MathOptSchedulingSolver(
            DEFAULT_POLICY, time_limit=limit
        ).solve(problem(task("timed")))
    assert isinstance(result, Solved)
    assert solve_mock.call_args.kwargs["params"].time_limit == limit


def test_shared_person_tasks_do_not_overlap() -> None:
    person_id: PersonId = PersonId("shared")
    first: Task = task(
        "first", duration=timedelta(hours=1), people=frozenset({person_id})
    )
    second: Task = task(
        "second", duration=timedelta(hours=1), people=frozenset({person_id})
    )
    value: SchedulingProblem = problem(first, second, end=START + timedelta(hours=2))
    result: dict[TaskId, datetime] = starts(schedule_for(value))
    assert abs(result[first.id] - result[second.id]) == timedelta(hours=1)


def test_scheduled_entries_record_participants() -> None:
    """Record each scheduled task's participants at solve time."""
    item: Task = task("recorded", people=PEOPLE)
    assert schedule_for(problem(item)).scheduled[0].participant_ids == PEOPLE


def test_unavailable_optional_drops_and_required_is_infeasible() -> None:
    person_id: PersonId = PersonId("unavailable")
    optional: Task = task("optional", people=frozenset({person_id}), required=False)
    availability: Availability = Availability(person_id, ())
    dropped: Schedule = schedule_for(problem(optional, availabilities=(availability,)))
    assert dropped.dropped == (DroppedTask(optional.id, optional.name),)
    required: Task = replace(optional, required=True)
    assert isinstance(
        SOLVER.solve(problem(required, availabilities=(availability,))), Infeasible
    )


def test_hard_dependency_is_inactive_when_task_drops() -> None:
    person_id: PersonId = PersonId("unavailable")
    first: Task = task("first")
    second: Task = task("second", people=frozenset({person_id}), required=False)
    constraint: HardConstraint = HardConstraint(
        ConstraintId("dependency"),
        TaskGapCondition(
            first.id, second.id, TaskGapRelation.AT_LEAST, timedelta(hours=100)
        ),
    )
    result: Schedule = schedule_for(
        problem(
            first,
            second,
            constraints=(constraint,),
            availabilities=(Availability(person_id, ()),),
        )
    )
    assert {item.task_id for item in result.scheduled} == {first.id}
    assert result.dropped == (DroppedTask(second.id, second.name),)


@pytest.mark.parametrize("importance", list(Importance))
def test_stability_moves_optional_task_instead_of_dropping(
    importance: Importance,
) -> None:
    """Move an optional task when its previous slot becomes occupied."""
    person_id: PersonId = PersonId("person")
    item: Task = replace(
        task("movable", people=frozenset({person_id}), required=False),
        importance=importance,
    )
    distant_start: datetime = START + timedelta(hours=24)
    availability: Availability = Availability(
        person_id,
        (
            TimeInterval(START, START + SLOT),
            TimeInterval(distant_start, distant_start + SLOT),
        ),
    )
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("early"),
        TimeBoundCondition(
            frozenset({item.id}), Boundary.START, TimeBoundRelation.AT, START
        ),
        Strength.WEAK,
    )
    original: SchedulingProblem = problem(
        item,
        constraints=(preference,),
        end=distant_start + SLOT,
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
    item: Task = replace(
        task("required", people=frozenset({person_id})), importance=Importance.HIGH
    )
    target: datetime = START + timedelta(hours=1)
    preference: SoftConstraint = SoftConstraint(
        ConstraintId("target"),
        TimeBoundCondition(
            frozenset({item.id}), Boundary.START, TimeBoundRelation.AT, target
        ),
        Strength.NORMAL,
    )
    original: SchedulingProblem = problem(item, constraints=(preference,))
    previous: Schedule = schedule_for(original)
    assert starts(previous) == {item.id: target}
    fixed: FixedTask = FixedTask(
        TaskId("occupied"),
        "Occupied",
        START,
        timedelta(hours=1.5),
        item.participant_ids,
    )
    changed: SchedulingProblem = replace(original, constraints=(), fixed_tasks=(fixed,))
    result: Solved | Infeasible = Scheduling(SOLVER, AllOf()).solve(changed, previous)
    assert isinstance(result, Solved)
    assert starts(result.schedule) == {item.id: target + SLOT}


def test_stability_prefers_nearer_of_two_distant_starts() -> None:
    """Prefer a twelve-hour move over five days despite a weak later preference."""
    item: Task = replace(task("required", people=PEOPLE), importance=Importance.HIGH)
    near: datetime = START + timedelta(hours=12)
    far: datetime = START + timedelta(days=5)
    availability: Availability = Availability(
        PERSON,
        tuple(TimeInterval(start, start + SLOT) for start in (START, near, far)),
    )
    previous: Schedule = Schedule(
        (ScheduledTask(item.id, item.name, START, START + SLOT),), ()
    )
    later: SoftConstraint = SoftConstraint(
        ConstraintId("later"),
        bound(item, Boundary.START, TimeBoundRelation.AT_OR_AFTER, near + SLOT),
        Strength.WEAK,
    )
    changed: SchedulingProblem = problem(
        item,
        constraints=(later,),
        fixed_tasks=(FixedTask(TaskId("occupied"), "Occupied", START, SLOT, PEOPLE),),
        end=far + SLOT,
        availabilities=(availability,),
    )
    result: Solved | Infeasible = SOLVER.solve(changed, previous)
    assert isinstance(result, Solved)
    assert starts(result.schedule) == {item.id: near}


@pytest.mark.parametrize("boundary", list(Boundary))
def test_hard_deadline_moves_task_before_bound(boundary: Boundary) -> None:
    item: Task = task("deadline", duration=HOUR, people=PEOPLE)
    deadline: datetime = START + timedelta(minutes=150)
    value: SchedulingProblem = problem(
        item,
        constraints=(
            hard(bound(item, boundary, TimeBoundRelation.AT_OR_BEFORE, deadline)),
            soft(bound(item, Boundary.START, TimeBoundRelation.AT, START + 3 * HOUR)),
        ),
        end=START + 4 * HOUR,
        slot=HOUR,
    )
    assert (
        starts(schedule_for(value))[item.id]
        == START + (2 if boundary is Boundary.START else 1) * HOUR
    )


@pytest.mark.parametrize("boundary", list(Boundary))
def test_at_or_after_moves_task_after_bound(boundary: Boundary) -> None:
    item: Task = task("earliest", duration=HOUR, people=PEOPLE)
    value: SchedulingProblem = problem(
        item,
        constraints=(
            hard(
                bound(
                    item,
                    boundary,
                    TimeBoundRelation.AT_OR_AFTER,
                    START + timedelta(minutes=90),
                )
            ),
            soft(bound(item, Boundary.START, TimeBoundRelation.AT, START)),
        ),
        end=START + 4 * HOUR,
        slot=HOUR,
    )
    assert (
        starts(schedule_for(value))[item.id]
        == START + (2 if boundary is Boundary.START else 1) * HOUR
    )


def test_zero_minimum_gap_orders_tasks() -> None:
    first: Task = task("first", duration=HOUR, people=PEOPLE)
    second: Task = task("second", duration=HOUR, people=PEOPLE)
    condition: TaskGapCondition = TaskGapCondition(
        first.id, second.id, TaskGapRelation.AT_LEAST, timedelta(0)
    )
    value: SchedulingProblem = problem(
        first, second, constraints=(hard(condition),), end=START + 2 * HOUR, slot=HOUR
    )
    assert starts(schedule_for(value)) == {first.id: START, second.id: START + HOUR}


def test_exact_zero_gap_places_task_after_fixed_task() -> None:
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Fixed", START, HOUR, PEOPLE)
    item: Task = task("following", duration=HOUR, people=PEOPLE)
    condition: TaskGapCondition = TaskGapCondition(
        fixed.id, item.id, TaskGapRelation.EXACTLY, timedelta(0)
    )
    value: SchedulingProblem = problem(
        item,
        constraints=(hard(condition),),
        fixed_tasks=(fixed,),
        end=START + 4 * HOUR,
        slot=HOUR,
    )
    assert starts(schedule_for(value)) == {item.id: START + HOUR}


@pytest.mark.parametrize(
    "quantity,maximum",
    [(AggregateQuantity.COUNT, 1), (AggregateQuantity.TOTAL_DURATION, 2 * HOUR)],
)
def test_daily_limit_distributes_tasks_across_two_dates(
    quantity: AggregateQuantity, maximum: int | timedelta
) -> None:
    first: Task = task("first", duration=2 * HOUR, people=PEOPLE)
    second: Task = task("second", duration=2 * HOUR, people=PEOPLE)
    condition: DailyLimitCondition = DailyLimitCondition(
        frozenset({first.id, second.id}), quantity, maximum
    )
    value: SchedulingProblem = problem(
        first,
        second,
        constraints=(hard(condition),),
        start=START.replace(hour=22),
        end=(START + timedelta(days=1)).replace(hour=4).astimezone(timezone.utc),
        slot=HOUR,
    )
    assert {at.date() for at in starts(schedule_for(value)).values()} == {
        START.date(),
        (START + timedelta(days=1)).date(),
    }


@pytest.mark.parametrize("relation", list(TimeRelation))
def test_hard_time_window_places_whole_task_in_allowed_region(
    relation: TimeRelation,
) -> None:
    item: Task = task("window", duration=2 * HOUR, people=PEOPLE)
    condition: TimeWindowCondition = TimeWindowCondition(
        frozenset({item.id}),
        relation,
        (TimeWindow(None, None, TimeRange(time(9), time(11))),),
    )
    value: SchedulingProblem = problem(
        item, constraints=(hard(condition),), end=START + 4 * HOUR, slot=HOUR
    )
    assert starts(schedule_for(value)) == {
        item.id: START + (0 if relation is TimeRelation.WITHIN else 2) * HOUR
    }


@pytest.mark.parametrize("offset, duration", [(90, 90), (95, 80)])
def test_added_fixed_task_blocks_available_starts_and_later_tasks(
    offset: int, duration: int
) -> None:
    """Block all touched slots for present and future tasks."""
    review: Task = task("review", duration=HOUR, people=PEOPLE)
    original: SchedulingProblem = problem(review, end=START + 4 * HOUR)
    scheduling: Scheduling = Scheduling(
        SOLVER,
        AllOf(
            UniqueIds(),
            ReferencesExist(),
            AvailabilityForEveryone(),
            AlignedToSlots(),
            NonemptyTimeWindows(),
        ),
    )
    fixed: FixedTask = FixedTask(
        TaskId("health-check"),
        "Health check",
        START + timedelta(minutes=offset),
        timedelta(minutes=duration),
        PEOPLE,
    )
    added: Executed | Rejected = scheduling.execute(original, (AddTask(fixed),))
    assert isinstance(added, Executed)
    assert free_slots(added.problem, PERSON) == (
        True,
        True,
        True,
        False,
        False,
        False,
        True,
        True,
    )
    expected: tuple[datetime, ...] = (START, START + SLOT, START + 3 * HOUR)
    query: AvailableStartsQuery = AvailableStartsQuery(PEOPLE, HOUR, None, 100)
    assert query.answer(added.problem, None) == Answered(
        AvailableStartsAnswer(expected, len(expected))
    )
    first_solution: SolveResult = scheduling.solve(added.problem, None)
    assert isinstance(first_solution, Solved)
    assert first_solution.schedule.scheduled[0].start in expected
    later: Task = replace(review, id=TaskId("later"))
    with_later: Executed | Rejected = scheduling.execute(
        added.problem, (AddTask(later),)
    )
    assert isinstance(with_later, Executed)
    solution: SolveResult = scheduling.solve(with_later.problem, None)
    assert isinstance(solution, Solved)
    assert {item.task_id for item in solution.schedule.scheduled} == {
        review.id,
        later.id,
    }
    item: ScheduledTask
    for item in solution.schedule.scheduled:
        assert item.start in expected


@pytest.mark.parametrize(
    "scenario",
    ["window", "deadline", "gap", "count", "duration", "drop", "move", "stable"],
)
@pytest.mark.parametrize("custom_policy", [False, True])
def test_summary_matches_objective_of_the_same_solve(
    scenario: str, custom_policy: bool
) -> None:
    """Match independently measured costs to the solved MathOpt objective."""
    item: Task = task("task", duration=HOUR, people=PEOPLE, required=scenario != "drop")
    condition: Condition
    previous: Schedule | None = None
    fixed_tasks: tuple[FixedTask, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    policy: ObjectivePolicy = (
        replace(DEFAULT_POLICY, per_count=2.5, stability_drop_cost_ratio=0.25)
        if custom_policy
        else DEFAULT_POLICY
    )
    if scenario == "window":
        condition = TimeWindowCondition(
            frozenset({item.id}),
            TimeRelation.AVOID,
            (TimeWindow(None, None, TimeRange(time(9, 15), time(12, 45))),),
        )
    elif scenario in {"deadline", "drop"}:
        condition = TimeBoundCondition(
            frozenset({item.id}),
            Boundary.END,
            TimeBoundRelation.AT_OR_BEFORE,
            START - HOUR,
        )
    elif scenario == "gap":
        fixed_tasks = (
            FixedTask(
                TaskId("fixed"),
                "Fixed",
                START + timedelta(minutes=10),
                HOUR,
                frozenset({PERSON}),
            ),
        )
        condition = TaskGapCondition(
            fixed_tasks[0].id, item.id, TaskGapRelation.EXACTLY, timedelta(minutes=15)
        )
    elif scenario in {"count", "duration"}:
        condition = DailyLimitCondition(
            frozenset({item.id}),
            AggregateQuantity.COUNT
            if scenario == "count"
            else AggregateQuantity.TOTAL_DURATION,
            0 if scenario == "count" else timedelta(minutes=15),
        )
    else:
        previous = Schedule(
            (ScheduledTask(item.id, item.name, START, START + item.duration),), ()
        )
        constraints = (
            HardConstraint(
                ConstraintId("move"),
                TimeBoundCondition(
                    frozenset({item.id}),
                    Boundary.START,
                    TimeBoundRelation.AT,
                    START + (3 if scenario == "move" else 0) * HOUR,
                ),
            ),
        )
        condition = TimeBoundCondition(
            frozenset({item.id}), Boundary.START, TimeBoundRelation.AT_OR_AFTER, START
        )
    constraints += (
        SoftConstraint(ConstraintId("preference"), condition, Strength.NORMAL),
    )
    value: SchedulingProblem = problem(
        item,
        constraints=constraints,
        fixed_tasks=fixed_tasks,
        end=START + 4 * HOUR,
        slot=HOUR,
    )
    captured: list[mathopt.SolveResult] = []
    original_solve: Callable[..., mathopt.SolveResult] = mathopt.solve

    def capture(
        model: mathopt.Model,
        solver_type: mathopt.SolverType,
        *,
        params: mathopt.SolveParameters,
    ) -> mathopt.SolveResult:
        result: mathopt.SolveResult = original_solve(model, solver_type, params=params)
        captured.append(result)
        return result

    solved: SolveResult
    with patch(
        "intent_to_schedule.adapter.mathopt.solve.mathopt.solve", side_effect=capture
    ):
        solved = MathOptSchedulingSolver(policy).solve(value, previous)
    assert isinstance(solved, Solved)
    assert solved.summary is not None
    assert solved.summary.total_cost == pytest.approx(captured[0].objective_value())
    assert solved.summary.total_cost == pytest.approx(
        solved.summary.dropped_tasks_cost
        + solved.summary.soft_constraints_cost
        + solved.summary.stability_cost
    )
    assert solved.summary.moved_tasks == (1 if scenario == "move" else 0)
    assert solved.summary.dropped_tasks == (1 if scenario == "drop" else 0)
    assert solved.summary.scheduled_tasks == (0 if scenario == "drop" else 1)
    assert all(
        item.violation.amount == 0
        for item in evaluate_constraints(value, solved.schedule, policy)
        if isinstance(item.constraint, HardConstraint)
    )
    if scenario == "move":
        assert solved.summary.stability_cost == pytest.approx(
            15 / 13 if custom_policy else 15 / 7
        )
    assert summarize_schedule(value, solved.schedule, policy).moved_tasks == 0
    assert summarize_schedule(value, solved.schedule, policy).stability_cost == 0


def test_infeasible_reports_contradictory_deadlines() -> None:
    """Report a broken deadline and relate the contradicting one to it."""
    day: datetime = datetime(2026, 10, 2, tzinfo=START.tzinfo)
    item: Task = task("item", duration=HOUR, people=PEOPLE)
    after: HardConstraint = HardConstraint(
        ConstraintId("after"),
        bound(item, Boundary.START, TimeBoundRelation.AT_OR_AFTER, day),
    )
    before: HardConstraint = HardConstraint(
        ConstraintId("before"),
        bound(item, Boundary.END, TimeBoundRelation.AT_OR_BEFORE, day),
    )
    value: SchedulingProblem = problem(
        item,
        constraints=(after, before),
        start=day - timedelta(days=1),
        end=day + timedelta(days=1),
    )
    result: SolveResult = SOLVER.solve(value)
    assert isinstance(result, Infeasible)
    assert result.conflicts.dropped_required_tasks == ()
    assert result.conflicts.constraints
    assert sum(
        conflict.evaluation.violation.amount
        for conflict in result.conflicts.constraints
    ) == pytest.approx(1.0)
    pair: set[ConstraintId] = {after.id, before.id}
    for conflict in result.conflicts.constraints:
        assert conflict.evaluation.constraint.id in pair
        assert conflict.related_constraint_ids == tuple(
            pair - {conflict.evaluation.constraint.id}
        )


def test_infeasible_reports_required_task_without_shared_free_start() -> None:
    """Report a required task whose participants are never free together."""
    first: PersonId = PersonId("first")
    second: PersonId = PersonId("second")
    item: Task = task("meeting", people=frozenset({first, second}))
    value: SchedulingProblem = problem(
        item,
        availabilities=(
            Availability(first, (TimeInterval(START, START + HOUR),)),
            Availability(second, (TimeInterval(START + HOUR, START + 2 * HOUR),)),
        ),
    )
    assert SOLVER.solve(value) == Infeasible(
        Conflicts(
            (), (DroppedRequiredTask(item.id, item.name, DropReason.NO_FREE_START),)
        )
    )


def test_infeasible_reports_required_task_conflicting_with_another() -> None:
    """Report one of two required tasks competing for the only free hour."""
    first: Task = task("first", duration=HOUR, people=PEOPLE)
    second: Task = task("second", duration=HOUR, people=PEOPLE)
    result: SolveResult = SOLVER.solve(problem(first, second, end=START + HOUR))
    assert isinstance(result, Infeasible)
    assert result.conflicts.constraints == ()
    dropped: tuple[DroppedRequiredTask, ...] = result.conflicts.dropped_required_tasks
    assert len(dropped) == 1
    assert (dropped[0].task_id, dropped[0].reason) in {
        (first.id, DropReason.CONFLICT),
        (second.id, DropReason.CONFLICT),
    }


@pytest.mark.parametrize("feasible", [True, False])
def test_solver_relaxes_only_after_infeasible(feasible: bool) -> None:
    """Solve once when feasible, and once more with the same limit when infeasible."""
    item: Task = task("timed", people=PEOPLE)
    limit: timedelta = timedelta(seconds=1)
    availabilities: tuple[Availability, ...] | None = (
        None if feasible else (Availability(PERSON, ()),)
    )
    solve_mock: MagicMock
    with patch(
        "intent_to_schedule.adapter.mathopt.solve.mathopt.solve", wraps=mathopt.solve
    ) as solve_mock:
        result: SolveResult = MathOptSchedulingSolver(
            DEFAULT_POLICY, time_limit=limit
        ).solve(problem(item, availabilities=availabilities))
    assert isinstance(result, Solved if feasible else Infeasible)
    assert [
        call.kwargs["params"].time_limit for call in solve_mock.call_args_list
    ] == [limit] * (1 if feasible else 2)


def test_infeasible_without_relaxed_solution_has_no_conflicts() -> None:
    """Return infeasible without conflicts when the relaxed solve finds nothing."""
    unsolved: MagicMock = MagicMock()
    unsolved.termination.reason = mathopt.TerminationReason.NO_SOLUTION_FOUND
    results: list[mathopt.SolveResult] = []
    original_solve: Callable[..., mathopt.SolveResult] = mathopt.solve

    def first_only(
        model: mathopt.Model,
        solver_type: mathopt.SolverType,
        *,
        params: mathopt.SolveParameters,
    ) -> mathopt.SolveResult:
        result: mathopt.SolveResult = (
            unsolved if results else original_solve(model, solver_type, params=params)
        )
        results.append(result)
        return result

    item: Task = task("item", people=PEOPLE)
    with patch(
        "intent_to_schedule.adapter.mathopt.solve.mathopt.solve", side_effect=first_only
    ):
        result: SolveResult = SOLVER.solve(
            problem(item, availabilities=(Availability(PERSON, ()),))
        )
    assert result == Infeasible(None)
    assert len(results) == 2
