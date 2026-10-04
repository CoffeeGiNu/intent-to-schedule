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
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import Infeasible
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.consistency import AllOf, Violation, Violations
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint, SoftConstraint
from intent_to_schedule.domain.evaluation import Distance
from intent_to_schedule.domain.measure import AggregateMeasure, AggregateQuantity, PointMeasure
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import DroppedTask, Schedule, ScheduledTask
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
    assert DEFAULT_POLICY.stability_drop_cost_ratio == 0.5


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.5, 1.5, float("inf"), float("nan")])
def test_policy_rejects_invalid_stability_drop_cost_ratio(ratio: float) -> None:
    with pytest.raises(ValueError):
        replace(DEFAULT_POLICY, stability_drop_cost_ratio=ratio)


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
        ObjectivePolicy({Importance.LOW: 1.0}, DEFAULT_POLICY.weights, 1.0, 0.5)
    with pytest.raises(ValueError):
        ObjectivePolicy(DEFAULT_POLICY.drop_costs, {Strength.WEAK: 1.0}, 1.0, 0.5)


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


@pytest.mark.parametrize("stability", [False, True])
def test_scheduling_passes_previous_schedule_to_solver(stability: bool) -> None:
    first: Task = task("a")
    second: Task = task("b")
    previous: Schedule | None = Schedule(
        (ScheduledTask(first.id, first.name, START, START + first.duration),),
        (DroppedTask(second.id, second.name),),
    ) if stability else None
    original: SchedulingProblem = problem(first, second)
    solver: Solver = Solver()
    assert Scheduling(solver, Validator()).solve(original, previous) == Infeasible()
    assert solver.problem is original
    assert solver.previous is previous


class Solver:
    def __init__(self) -> None:
        self.problem: SchedulingProblem | None = None
        self.previous: Schedule | None = None

    def solve(self, problem: SchedulingProblem, previous: Schedule | None) -> Infeasible:
        self.problem = problem
        self.previous = previous
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


@pytest.mark.parametrize("relation_value", ["within", "avoid"])
@pytest.mark.parametrize(
    "strength", [None, Strength.WEAK, Strength.NORMAL, Strength.STRONG]
)
def test_add_time_constraint_builds_one_intrusion_and_delegates(
    relation_value: str,
    strength: Strength | None,
) -> None:
    from datetime import time
    from unittest.mock import Mock

    from intent_to_schedule.application.command import AddTimeConstraint
    from intent_to_schedule.application.time_windows import (
        Expansion,
        TimeRange,
        TimeRelation,
        TimeWindow,
    )
    from intent_to_schedule.domain.evaluation import Intrusion
    from intent_to_schedule.domain.measure import IntervalMeasure

    original: SchedulingProblem = problem(replace(task("a"), required=False))
    allowed: TimeInterval = TimeInterval(
        START + timedelta(hours=1), START + timedelta(hours=2)
    )
    windows: tuple[TimeWindow, ...] = (
        TimeWindow(None, None, TimeRange(time(10), time(11))),
    )
    command: AddTimeConstraint = AddTimeConstraint(
        ConstraintId("time"),
        frozenset({TaskId("a")}),
        TimeRelation(relation_value),
        windows,
        strength,
    )
    expansion: Mock
    delegation: Mock
    with patch(
        "intent_to_schedule.application.command.expand",
        return_value=Expansion((allowed,), True),
    ) as expansion:
        with patch.object(
            AddConstraint, "execute", autospec=True, return_value=Executed(original)
        ) as delegation:
            assert command.execute(original) == Executed(original)
    expansion.assert_called_once_with(windows, command.relation, original.calendar.grid)
    added: AddConstraint = delegation.call_args.args[0]
    assert delegation.call_args.args[1] is original
    region: tuple[TimeInterval, ...] = (
        (
            TimeInterval(START, allowed.start),
            TimeInterval(allowed.end, original.calendar.grid.horizon.end),
        )
        if relation_value == "within"
        else (allowed,)
    )
    assert added.constraint.measure == IntervalMeasure(frozenset({TaskId("a")}))
    assert added.constraint.evaluation == Intrusion(region)
    assert added.constraint.id == command.constraint_id
    assert isinstance(
        added.constraint, HardConstraint if strength is None else SoftConstraint
    )
    if isinstance(added.constraint, SoftConstraint):
        assert added.constraint.strength is strength
    assert original.constraints == ()
    assert not original.tasks[0].required


@pytest.mark.parametrize("relation_value", ["within", "avoid"])
def test_add_time_constraint_rejects_empty_windows_with_actionable_message(
    relation_value: str,
) -> None:
    from intent_to_schedule.application.command import AddTimeConstraint
    from intent_to_schedule.application.time_windows import TimeRelation

    command: AddTimeConstraint = AddTimeConstraint(
        ConstraintId("time"), frozenset({TaskId("a")}), TimeRelation(relation_value), (), None
    )
    result: Executed | Rejected = command.execute(problem(task("a")))
    assert isinstance(result, Rejected)
    message: str = result.violations.items[0].message
    assert "task a" in message
    assert relation_value in message
    assert "horizon" in message


def test_add_time_constraint_rejects_windows_removed_by_rounding() -> None:
    from datetime import time

    from intent_to_schedule.application.command import AddTimeConstraint
    from intent_to_schedule.application.time_windows import (
        TimeRange,
        TimeRelation,
        TimeWindow,
    )

    command: AddTimeConstraint = AddTimeConstraint(
        ConstraintId("time"),
        frozenset({TaskId("a")}),
        TimeRelation.WITHIN,
        (TimeWindow(None, None, TimeRange(time(9, 5), time(9, 10))),),
        None,
    )
    with patch.object(TimeGrid, "round_inward", return_value=None):
        result: Executed | Rejected = command.execute(problem(task("a")))
    assert isinstance(result, Rejected)
    message: str = result.violations.items[0].message
    assert "task a" in message
    assert "slot" in message and "within" in message


def test_add_time_constraint_delegates_duplicate_id_rejection() -> None:
    from intent_to_schedule.application.command import AddTimeConstraint
    from intent_to_schedule.application.time_windows import (
        Expansion,
        TimeRelation,
        TimeWindow,
    )

    existing: HardConstraint = HardConstraint(
        ConstraintId("time"), PointMeasure(TaskId("a")), Distance(START)
    )
    original: SchedulingProblem = problem(task("a"), constraints=(existing,))
    command: AddTimeConstraint = AddTimeConstraint(
        existing.id,
        frozenset({TaskId("a")}),
        TimeRelation.AVOID,
        (TimeWindow(None, None, None),),
        None,
    )
    with patch(
        "intent_to_schedule.application.command.expand",
        return_value=Expansion((original.calendar.grid.horizon,), False),
    ):
        result: Executed | Rejected = command.execute(original)
    assert result == AddConstraint(existing).execute(original)
    assert original.constraints == (existing,)


def test_add_time_constraint_accepts_whole_horizon_with_empty_complement() -> None:
    from intent_to_schedule.application.command import AddTimeConstraint
    from intent_to_schedule.application.time_windows import (
        Expansion,
        TimeRelation,
        TimeWindow,
    )
    from intent_to_schedule.domain.evaluation import Intrusion

    original: SchedulingProblem = problem(task("a"))
    command: AddTimeConstraint = AddTimeConstraint(
        ConstraintId("time"),
        frozenset({TaskId("a")}),
        TimeRelation.WITHIN,
        (TimeWindow(None, None, None),),
        None,
    )
    with patch(
        "intent_to_schedule.application.command.expand",
        return_value=Expansion((original.calendar.grid.horizon,), False),
    ):
        result: Executed | Rejected = command.execute(original)
    assert isinstance(result, Executed)
    assert len(result.problem.constraints) == 1
    assert result.problem.constraints[0].evaluation == Intrusion(())


@pytest.mark.parametrize("relation_value", ["within", "avoid"])
def test_add_time_constraint_real_grid_creates_expected_region(
    relation_value: str,
) -> None:
    from datetime import time

    from intent_to_schedule.application.command import AddTimeConstraint
    from intent_to_schedule.application.time_windows import (
        TimeRange,
        TimeRelation,
        TimeWindow,
    )
    from intent_to_schedule.domain.evaluation import Intrusion

    original: SchedulingProblem = problem(task("a"))
    command: AddTimeConstraint = AddTimeConstraint(
        ConstraintId("time"),
        frozenset({TaskId("a")}),
        TimeRelation(relation_value),
        (TimeWindow(None, None, TimeRange(time(9, 10), time(10, 10))),),
        None,
    )
    result: Executed | Rejected = command.execute(original)
    assert isinstance(result, Executed)
    region: tuple[TimeInterval, ...] = (
        (
            TimeInterval(START, START + timedelta(minutes=30)),
            TimeInterval(START + timedelta(hours=1), START + timedelta(days=1)),
        )
        if relation_value == "within"
        else (TimeInterval(START, START + timedelta(minutes=90)),)
    )
    assert result.problem.constraints[0].evaluation == Intrusion(region)
