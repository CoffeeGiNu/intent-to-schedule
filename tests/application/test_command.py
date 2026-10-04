from dataclasses import replace
from datetime import datetime, timedelta

from intent_to_schedule.application.command import (
    AddConstraint,
    AddTask,
    Executed,
    Rejected,
    RemoveConstraint,
    RemoveTask,
    ReplaceConstraint,
    ReplaceTask,
    execute_commands,
)
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import (
    DailyLimitCondition,
    TimeBoundCondition,
    TimeBoundRelation,
)
from intent_to_schedule.domain.consistency import Violation, Violations
from intent_to_schedule.domain.constraint import (
    Constraint,
    ConstraintId,
    HardConstraint,
    SoftConstraint,
)
from intent_to_schedule.domain.measure import AggregateQuantity, Boundary
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId

START: datetime = datetime(2026, 10, 1, 9)


def task(value: str, name: str = "Task") -> Task:
    return Task(
        TaskId(value), name, timedelta(hours=1), frozenset(), Importance.HIGH, True
    )


def fixed(value: str, name: str = "Existing") -> FixedTask:
    return FixedTask(TaskId(value), name, START, timedelta(hours=1), frozenset())


def problem(
    *tasks: Task, constraints: tuple[Constraint, ...] = ()
) -> SchedulingProblem:
    calendar: Calendar = Calendar(
        TimeGrid(TimeInterval(START, START + timedelta(days=1)), timedelta(minutes=30)),
        (),
    )
    return SchedulingProblem(calendar, (), tasks, (), constraints)


def test_task_commands_keep_input_and_order() -> None:
    first: Task = task("a")
    second: Task = task("b")
    original: SchedulingProblem = problem(first, second)
    assert isinstance(AddTask(first).execute(original), Rejected)
    assert isinstance(ReplaceTask(task("missing")).execute(original), Rejected)
    assert isinstance(RemoveTask(TaskId("missing")).execute(original), Rejected)

    added: Executed | Rejected = AddTask(task("c")).execute(original)
    assert isinstance(added, Executed)
    assert tuple(item.id.value for item in added.problem.tasks) == ("a", "b", "c")
    replacement: Task = task("a", "Changed")
    replaced: Executed | Rejected = ReplaceTask(replacement).execute(original)
    assert isinstance(replaced, Executed)
    assert replaced.problem.tasks == (replacement, second)
    assert original.tasks == (first, second)


def test_execute_commands_stops_at_first_rejection() -> None:
    original: SchedulingProblem = problem()
    result: Executed | Rejected = execute_commands(
        original, (RemoveTask(TaskId("missing")), RemoveTask(TaskId("later")))
    )
    assert result == Rejected(Violations((Violation("Task missing does not exist"),)))
    assert execute_commands(original, ()) == Executed(original)


def test_remove_task_repairs_aggregate_and_drops_other_references() -> None:
    first: Task = task("a")
    second: Task = task("b")
    shared: HardConstraint = HardConstraint(
        ConstraintId("shared"),
        DailyLimitCondition(
            frozenset({first.id, second.id}), AggregateQuantity.COUNT, 1
        ),
    )
    lone: HardConstraint = replace(
        shared,
        id=ConstraintId("lone"),
        condition=replace(shared.condition, task_ids=frozenset({first.id})),
    )
    point: HardConstraint = HardConstraint(
        ConstraintId("point"),
        TimeBoundCondition(
            frozenset({first.id}), Boundary.START, TimeBoundRelation.AT, START
        ),
    )
    other: HardConstraint = HardConstraint(
        ConstraintId("other"),
        TimeBoundCondition(
            frozenset({second.id}), Boundary.START, TimeBoundRelation.AT, START
        ),
    )
    original: SchedulingProblem = problem(
        first, second, constraints=(shared, lone, point, other)
    )

    result: Executed | Rejected = RemoveTask(first.id).execute(original)
    assert isinstance(result, Executed)
    assert result.problem.tasks == (second,)
    assert result.problem.constraints == (
        replace(
            shared, condition=replace(shared.condition, task_ids=frozenset({second.id}))
        ),
        other,
    )
    assert original.constraints == (shared, lone, point, other)


def test_constraint_commands_check_only_ids() -> None:
    constraint: HardConstraint = HardConstraint(
        ConstraintId("c"),
        TimeBoundCondition(
            frozenset({TaskId("missing")}), Boundary.START, TimeBoundRelation.AT, START
        ),
    )
    original: SchedulingProblem = problem()
    added: Executed | Rejected = AddConstraint(constraint).execute(original)
    assert isinstance(added, Executed)
    assert added.problem.constraints == (constraint,)
    assert isinstance(AddConstraint(constraint).execute(added.problem), Rejected)
    assert isinstance(
        RemoveConstraint(ConstraintId("missing")).execute(added.problem), Rejected
    )
    removed: Executed | Rejected = RemoveConstraint(constraint.id).execute(
        added.problem
    )
    assert isinstance(removed, Executed)
    assert removed.problem == original


def test_remove_fixed_task_repairs_constraints() -> None:
    movable: Task = task("movable")
    fixed: FixedTask = FixedTask(
        TaskId("fixed"), "Existing", START, timedelta(hours=1), frozenset()
    )
    aggregate: HardConstraint = HardConstraint(
        ConstraintId("aggregate"),
        DailyLimitCondition(
            frozenset({movable.id, fixed.id}), AggregateQuantity.COUNT, 1
        ),
    )
    point: HardConstraint = HardConstraint(
        ConstraintId("point"),
        TimeBoundCondition(
            frozenset({fixed.id}), Boundary.START, TimeBoundRelation.AT, START
        ),
    )
    original: SchedulingProblem = replace(
        problem(movable, constraints=(aggregate, point)), fixed_tasks=(fixed,)
    )
    result: Executed | Rejected = RemoveTask(fixed.id).execute(original)
    assert isinstance(result, Executed)
    assert result.problem.tasks == (movable,)
    assert result.problem.fixed_tasks == ()
    assert result.problem.constraints == (
        replace(
            aggregate,
            condition=replace(aggregate.condition, task_ids=frozenset({movable.id})),
        ),
    )
    assert original.fixed_tasks == (fixed,)


def test_replace_fixed_task_makes_it_movable_and_keeps_constraints() -> None:
    movable: Task = task("movable")
    fixed: FixedTask = FixedTask(
        TaskId("fixed"), "Existing", START, timedelta(hours=1), frozenset()
    )
    other: FixedTask = replace(fixed, id=TaskId("other"))
    constraint: HardConstraint = HardConstraint(
        ConstraintId("point"),
        TimeBoundCondition(
            frozenset({fixed.id}), Boundary.START, TimeBoundRelation.AT, START
        ),
    )
    original: SchedulingProblem = replace(
        problem(movable, constraints=(constraint,)), fixed_tasks=(fixed, other)
    )
    replacement: Task = task("fixed", "Movable meeting")
    assert isinstance(AddTask(replacement).execute(original), Rejected)
    result: Executed | Rejected = ReplaceTask(replacement).execute(original)
    assert isinstance(result, Executed)
    assert result.problem.tasks == (movable, replacement)
    assert result.problem.fixed_tasks == (other,)
    assert result.problem.constraints == original.constraints
    assert original.fixed_tasks == (fixed, other)


def test_remove_added_fixed_task_restores_problem() -> None:
    """Add a fixed task beside movable ones and remove it again."""
    original: SchedulingProblem = problem(task("movable"))
    appointment: FixedTask = fixed("appointment")
    added: Executed | Rejected = AddTask(appointment).execute(original)
    assert isinstance(added, Executed)
    assert added.problem.tasks == original.tasks
    assert added.problem.fixed_tasks == (appointment,)
    assert original.fixed_tasks == ()
    assert RemoveTask(appointment.id).execute(added.problem) == Executed(original)


def test_replace_task_fixes_and_releases_a_task_keeping_constraints() -> None:
    """Fix a movable task and release it again without touching constraints."""
    movable: Task = task("review")
    constraint: HardConstraint = HardConstraint(
        ConstraintId("start"),
        TimeBoundCondition(
            frozenset({movable.id}), Boundary.START, TimeBoundRelation.AT, START
        ),
    )
    original: SchedulingProblem = problem(movable, constraints=(constraint,))
    appointment: FixedTask = fixed("review")
    result: Executed | Rejected = ReplaceTask(appointment).execute(original)
    assert result == Executed(replace(original, tasks=(), fixed_tasks=(appointment,)))
    assert isinstance(result, Executed)
    assert ReplaceTask(movable).execute(result.problem) == Executed(original)


def test_replace_fixed_task_preserves_other_tasks_and_order() -> None:
    """Replace one fixed appointment without disturbing its neighbors."""
    appointment: FixedTask = fixed("appointment")
    other: FixedTask = fixed("other")
    original: SchedulingProblem = replace(
        problem(task("movable")), fixed_tasks=(appointment, other)
    )
    replacement: FixedTask = replace(appointment, start=START + timedelta(minutes=15))
    result: Executed | Rejected = ReplaceTask(replacement).execute(original)
    assert result == Executed(replace(original, fixed_tasks=(replacement, other)))
    assert original.fixed_tasks == (appointment, other)


def test_fixed_task_commands_reject_duplicate_and_missing_identifiers() -> None:
    """Check fixed identifiers against both task collections."""
    movable: Task = task("movable")
    appointment: FixedTask = fixed("appointment")
    original: SchedulingProblem = replace(problem(movable), fixed_tasks=(appointment,))
    assert isinstance(AddTask(appointment).execute(original), Rejected)
    assert isinstance(
        AddTask(replace(appointment, id=movable.id)).execute(original), Rejected
    )
    assert isinstance(
        ReplaceTask(replace(appointment, id=TaskId("missing"))).execute(original),
        Rejected,
    )


def test_replace_constraint_keeps_position_and_rejects_missing_identifier() -> None:
    """Replace a constraint's content in place and reject unknown identifiers."""
    first: HardConstraint = HardConstraint(
        ConstraintId("first"),
        TimeBoundCondition(
            frozenset({TaskId("a")}), Boundary.START, TimeBoundRelation.AT, START
        ),
        "Entered request",
    )
    second: HardConstraint = replace(first, id=ConstraintId("second"))
    original: SchedulingProblem = problem(constraints=(first, second))
    replacement: SoftConstraint = SoftConstraint(
        first.id, first.condition, Strength.STRONG, "Changed request"
    )
    assert ReplaceConstraint(replacement).execute(original) == Executed(
        replace(original, constraints=(replacement, second))
    )
    assert ReplaceConstraint(replace(replacement, id=ConstraintId("missing"))).execute(
        original
    ) == Rejected(Violations((Violation("Constraint missing does not exist"),)))
