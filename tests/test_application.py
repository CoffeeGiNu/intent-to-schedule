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
)
from intent_to_schedule.application.converse import Conversation, stability_constraints
from intent_to_schedule.application.policy import DEFAULT_POLICY
from intent_to_schedule.application.solve import Infeasible
from intent_to_schedule.application.translate import Ambiguous, Translated
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.consistency import ConsistencyError, Violation, Violations
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint, SoftConstraint
from intent_to_schedule.domain.evaluation import Distance
from intent_to_schedule.domain.measure import AggregateMeasure, AggregateQuantity, PointMeasure
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule, ScheduledTask
from intent_to_schedule.application.converse import Response
from intent_to_schedule.application.translate import Utterance
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId


START: datetime = datetime(2026, 10, 1, 9)


def task(value: str, name: str = "Task") -> Task:
    return Task(TaskId(value), name, timedelta(hours=1), frozenset(), Importance.HIGH, True)


def problem(*tasks: Task, constraints: tuple[HardConstraint, ...] = ()) -> SchedulingProblem:
    calendar: Calendar = Calendar(
        TimeGrid(TimeInterval(START, START + timedelta(days=1)), timedelta(minutes=30)),
        (),
        (),
    )
    return SchedulingProblem(calendar, (), tasks, constraints)


def test_policy_uses_mappings() -> None:
    assert DEFAULT_POLICY.drop_cost(Importance.HIGH) == 100.0
    assert DEFAULT_POLICY.weight(Strength.WEAK) == 1.0


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


class Translator:
    def __init__(self, result: Translated | Ambiguous) -> None:
        self.result = result

    def translate(
        self, dialogue: tuple[Utterance, ...], problem: SchedulingProblem, previous: Schedule | None
    ) -> Translated | Ambiguous:
        return self.result


class Solver:
    def __init__(self) -> None:
        self.problem: SchedulingProblem | None = None

    def solve(self, problem: SchedulingProblem) -> Infeasible:
        self.problem = problem
        return Infeasible()


class Validator:
    def __init__(self, message: str | None = None) -> None:
        self.message = message
        self.problems: list[SchedulingProblem] = []

    def validate(self, problem: SchedulingProblem) -> Violations:
        self.problems.append(problem)
        return Violations((Violation(self.message),)) if self.message else Violations(())


def test_conversation_applies_commands_and_solves_with_temporary_stability() -> None:
    first: Task = task("a")
    original: SchedulingProblem = problem()
    previous: Schedule = Schedule((ScheduledTask(first.id, START),), frozenset())
    solver: Solver = Solver()
    validator: Validator = Validator()
    conversation: Conversation = Conversation(Translator(Translated((AddTask(first),))), solver, (validator,))
    with patch.object(ConstraintId, "generate", return_value=ConstraintId("stability")):
        response: Response = conversation.respond((), original, previous)
    assert response.problem.tasks == (first,)
    assert response.problem.constraints == ()
    assert len(solver.problem.constraints) == 1
    assert validator.problems == [response.problem]
    assert original.tasks == ()


def test_conversation_ambiguous_and_failures() -> None:
    original: SchedulingProblem = problem()
    solver: Solver = Solver()
    ambiguous: Ambiguous = Ambiguous("Which task?")
    response: Response = Conversation(Translator(ambiguous), solver, ()).respond((), original, None)
    assert response.problem is original
    assert response.outcome is ambiguous
    assert solver.problem is None

    rejected: Conversation = Conversation(Translator(Translated((RemoveTask(TaskId("a")),))), solver, ())
    with pytest.raises(ConsistencyError, match="Task does not exist"):
        rejected.respond((), original, None)

    first: Validator = Validator("first")
    second: Validator = Validator("second")
    invalid: Conversation = Conversation(Translator(Translated(())), solver, (first, second))
    with pytest.raises(ConsistencyError, match="first; second"):
        invalid.respond((), original, None)
    assert len(first.problems) == len(second.problems) == 1
    assert solver.problem is None

    valid: Conversation = Conversation(Translator(Translated(())), solver, ())
    assert valid.respond((), original, None).problem is original
    assert solver.problem is original
