from dataclasses import replace
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from intent_to_schedule.application.command import (
    AddConstraint,
    AddTask,
    Executed,
    Rejected,
    RemoveConstraint,
    RemoveTask,
    ReplaceTask,
    execute_commands,
)
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.schedule import Scheduling, stability_constraints
from intent_to_schedule.application.solve import Infeasible
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.consistency import AllOf, Violation, Violations
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint, SoftConstraint
from intent_to_schedule.domain.evaluation import Distance
from intent_to_schedule.domain.measure import AggregateMeasure, AggregateQuantity, PointMeasure
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId


START: datetime = datetime(2026, 10, 1, 9)


def task(value: str, name: str = "Task") -> Task:
    return Task(TaskId(value), name, timedelta(hours=1), frozenset(), Importance.HIGH, True)


def problem(*tasks: Task, constraints: tuple[HardConstraint, ...] = ()) -> SchedulingProblem:
    calendar: Calendar = Calendar(
        TimeGrid(TimeInterval(START, START + timedelta(days=1)), timedelta(minutes=30)),
        (),
    )
    return SchedulingProblem(calendar, (), tasks, (), constraints)


def test_policy_uses_mappings() -> None:
    assert DEFAULT_POLICY.drop_cost(Importance.HIGH) == 100.0
    assert DEFAULT_POLICY.weight(Strength.WEAK) == 1.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_policy_rejects_nonpositive_or_nonfinite_coefficients(value: float) -> None:
    with pytest.raises(ValueError):
        replace(DEFAULT_POLICY, per_count=value)
    with pytest.raises(ValueError):
        replace(DEFAULT_POLICY, drop_costs={**DEFAULT_POLICY.drop_costs, Importance.LOW: value})
    with pytest.raises(ValueError):
        replace(DEFAULT_POLICY, weights={**DEFAULT_POLICY.weights, Strength.WEAK: value})


def test_policy_requires_every_importance_and_strength() -> None:
    with pytest.raises(ValueError):
        ObjectivePolicy({Importance.LOW: 1.0}, DEFAULT_POLICY.weights, 1.0)
    with pytest.raises(ValueError):
        ObjectivePolicy(DEFAULT_POLICY.drop_costs, {Strength.WEAK: 1.0}, 1.0)


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
    with patch.object(AddTask, "execute", side_effect=AssertionError("Later command ran")):
        result: Executed | Rejected = execute_commands(
            original, (RemoveTask(TaskId("missing")), AddTask(task("later")))
        )
    assert result == Rejected(Violations((Violation("Task missing does not exist"),)))
    assert execute_commands(original, ()) == Executed(original)


def test_remove_task_repairs_aggregate_and_drops_other_references() -> None:
    first: Task = task("a")
    second: Task = task("b")
    shared: HardConstraint = HardConstraint(
        ConstraintId("shared"),
        AggregateMeasure(frozenset({first.id, second.id}), AggregateQuantity.COUNT),
        Distance(1),
    )
    lone: HardConstraint = replace(shared, id=ConstraintId("lone"), measure=replace(shared.measure, task_ids=frozenset({first.id})))
    point: HardConstraint = HardConstraint(ConstraintId("point"), PointMeasure(first.id), Distance(START))
    other: HardConstraint = HardConstraint(ConstraintId("other"), PointMeasure(second.id), Distance(START))
    original: SchedulingProblem = problem(first, second, constraints=(shared, lone, point, other))

    result: Executed | Rejected = RemoveTask(first.id).execute(original)
    assert isinstance(result, Executed)
    assert result.problem.tasks == (second,)
    assert result.problem.constraints == (
        replace(shared, measure=replace(shared.measure, task_ids=frozenset({second.id}))),
        other,
    )
    assert original.constraints == (shared, lone, point, other)


def test_constraint_commands_check_only_ids() -> None:
    constraint: HardConstraint = HardConstraint(ConstraintId("c"), PointMeasure(TaskId("missing")), Distance(START))
    original: SchedulingProblem = problem()
    added: Executed | Rejected = AddConstraint(constraint).execute(original)
    assert isinstance(added, Executed)
    assert added.problem.constraints == (constraint,)
    assert isinstance(AddConstraint(constraint).execute(added.problem), Rejected)
    assert isinstance(RemoveConstraint(ConstraintId("missing")).execute(added.problem), Rejected)
    removed: Executed | Rejected = RemoveConstraint(constraint.id).execute(added.problem)
    assert isinstance(removed, Executed)
    assert removed.problem == original


def test_remove_fixed_task_repairs_constraints() -> None:
    movable: Task = task("movable")
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Existing", START, timedelta(hours=1), frozenset())
    aggregate: HardConstraint = HardConstraint(
        ConstraintId("aggregate"), AggregateMeasure(frozenset({movable.id, fixed.id}), AggregateQuantity.COUNT), Distance(1)
    )
    point: HardConstraint = HardConstraint(ConstraintId("point"), PointMeasure(fixed.id), Distance(START))
    original: SchedulingProblem = replace(problem(movable, constraints=(aggregate, point)), fixed_tasks=(fixed,))
    result: Executed | Rejected = RemoveTask(fixed.id).execute(original)
    assert isinstance(result, Executed)
    assert result.problem.tasks == (movable,)
    assert result.problem.fixed_tasks == ()
    assert result.problem.constraints == (replace(aggregate, measure=replace(aggregate.measure, task_ids=frozenset({movable.id}))),)
    assert original.fixed_tasks == (fixed,)


def test_replace_fixed_task_makes_it_movable_and_keeps_constraints() -> None:
    movable: Task = task("movable")
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Existing", START, timedelta(hours=1), frozenset())
    other: FixedTask = replace(fixed, id=TaskId("other"))
    constraint: HardConstraint = HardConstraint(ConstraintId("point"), PointMeasure(fixed.id), Distance(START))
    original: SchedulingProblem = replace(problem(movable, constraints=(constraint,)), fixed_tasks=(fixed, other))
    replacement: Task = task("fixed", "Movable meeting")
    assert isinstance(AddTask(replacement).execute(original), Rejected)
    result: Executed | Rejected = ReplaceTask(replacement).execute(original)
    assert isinstance(result, Executed)
    assert result.problem.tasks == (movable, replacement)
    assert result.problem.fixed_tasks == (other,)
    assert result.problem.constraints == original.constraints
    assert original.fixed_tasks == (fixed, other)


def test_fixed_tasks_get_no_stability_constraints() -> None:
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Existing", START, timedelta(hours=1), frozenset())
    previous: Schedule = Schedule((ScheduledTask(fixed.id, START),), frozenset())
    assert stability_constraints(replace(problem(), fixed_tasks=(fixed,)), previous) == ()


def test_stability_constraints_use_previous_starts() -> None:
    first: Task = task("a")
    second: Task = task("b")
    previous: Schedule = Schedule((ScheduledTask(first.id, START),), frozenset({second.id}))
    with patch.object(ConstraintId, "generate", return_value=ConstraintId("new")):
        constraints: tuple[SoftConstraint, ...] = stability_constraints(problem(first, second), previous)
    assert len(constraints) == 1
    assert constraints[0].id == ConstraintId("new")
    assert constraints[0].measure == PointMeasure(first.id)
    assert constraints[0].evaluation == Distance(START)
    assert constraints[0].strength == first.stability


class Solver:
    def __init__(self) -> None:
        self.problem: SchedulingProblem | None = None

    def solve(self, problem: SchedulingProblem) -> Infeasible:
        self.problem = problem
        return Infeasible()


class Validator:
    def __init__(self, message: str | None = None) -> None:
        self.message: str | None = message
        self.problems: list[SchedulingProblem] = []

    def validate(self, problem: SchedulingProblem) -> Violations:
        self.problems.append(problem)
        return Violations((Violation(self.message),)) if self.message else Violations(())


def test_scheduling_execute_returns_command_rejection() -> None:
    original: SchedulingProblem = problem()
    validator: Validator = Validator()
    scheduling: Scheduling = Scheduling(Solver(), validator)

    result: Executed | Rejected = scheduling.execute(
        original, (RemoveTask(TaskId("missing")), AddTask(task("later")))
    )

    assert isinstance(result, Rejected)
    assert result.violations == Violations((Violation("Task missing does not exist"),))
    assert validator.problems == []
    assert original.tasks == ()


def test_scheduling_execute_returns_merged_validator_rejection() -> None:
    original: SchedulingProblem = problem()
    first: Validator = Validator("first")
    second: Validator = Validator("second")
    scheduling: Scheduling = Scheduling(Solver(), AllOf(first, second))

    result: Executed | Rejected = scheduling.execute(original, (AddTask(task("a")),))

    assert isinstance(result, Rejected)
    assert result.violations == Violations((Violation("first"), Violation("second")))
    assert first.problems == second.problems
    assert first.problems[0].tasks == (task("a"),)
    assert original.tasks == ()
