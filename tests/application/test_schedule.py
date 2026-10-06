from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from intent_to_schedule.application.command import (
    AddTask,
    Executed,
    Rejected,
    RemoveTask,
)
from intent_to_schedule.application.objective import ScheduleSummary
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.query import (
    Answered,
    AnswerResult,
    EvaluationAnswer,
    EvaluationQuery,
    ObjectivePolicyAnswer,
    ObjectivePolicyQuery,
)
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import (
    Conflicts,
    FeasibleSolution,
    NoFeasibleSolution,
    OptimalSolution,
    Solution,
    SolutionNotFound,
)
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import (
    DailyLimitCondition,
    TimeBoundCondition,
    TimeBoundRelation,
)
from intent_to_schedule.domain.consistency import AllOf, Violation, Violations
from intent_to_schedule.domain.constraint import (
    Constraint,
    ConstraintId,
    SoftConstraint,
)
from intent_to_schedule.domain.measure import AggregateQuantity, Boundary
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import DroppedTask, Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId

START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)


def task(value: str, name: str = "Task") -> Task:
    return Task(
        TaskId(value), name, timedelta(hours=1), frozenset(), Importance.HIGH, True
    )


def problem(
    *tasks: Task, constraints: tuple[Constraint, ...] = ()
) -> SchedulingProblem:
    calendar: Calendar = Calendar(
        TimeGrid(TimeInterval(START, START + timedelta(days=1)), timedelta(minutes=30)),
        (),
    )
    return SchedulingProblem(calendar, (), tasks, (), constraints)


class Solver:
    def __init__(
        self, result: Solution = NoFeasibleSolution(Conflicts((), ()))
    ) -> None:
        self.result: Solution = result
        self.problem: SchedulingProblem | None = None
        self.policy: ObjectivePolicy | None = None
        self.previous: Schedule | None = None
        self.stability: bool | None = None

    def solve(
        self,
        problem: SchedulingProblem,
        policy: ObjectivePolicy,
        previous: Schedule | None,
        stability: bool,
    ) -> Solution:
        self.problem = problem
        self.policy = policy
        self.previous = previous
        self.stability = stability
        return self.result


class Validator:
    def __init__(self, message: str | None = None) -> None:
        self.message: str | None = message
        self.problems: list[SchedulingProblem] = []

    def validate(self, problem: SchedulingProblem) -> Violations:
        self.problems.append(problem)
        return (
            Violations((Violation(self.message),)) if self.message else Violations(())
        )


@pytest.mark.parametrize("stability", [False, True])
@pytest.mark.parametrize("saved", [False, True])
@pytest.mark.parametrize(
    "result",
    [
        OptimalSolution(Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)),
        FeasibleSolution(Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)),
        NoFeasibleSolution(Conflicts((), ())),
        SolutionNotFound("time_limit"),
    ],
)
def test_scheduling_passes_policy_previous_schedule_and_stability_to_solver(
    saved: bool, stability: bool, result: Solution
) -> None:
    first: Task = task("a")
    second: Task = task("b")
    previous: Schedule | None = (
        Schedule(
            (
                ScheduledTask(
                    first.id,
                    first.name,
                    START,
                    START + first.duration,
                    first.participant_ids,
                ),
            ),
            (DroppedTask(second.id, second.name),),
        )
        if saved
        else None
    )
    original: SchedulingProblem = problem(first, second)
    policy: ObjectivePolicy = replace(DEFAULT_POLICY, per_count=3.0)
    solver: Solver = Solver(result)
    assert Scheduling(solver, Validator(), policy).schedule(
        original, previous, stability
    ) is result
    assert solver.problem is original
    assert solver.policy is policy
    assert solver.previous is previous
    assert solver.stability is stability


@pytest.mark.parametrize("solution_type", [OptimalSolution, FeasibleSolution])
def test_successful_solution_requires_summary(
    solution_type: type[OptimalSolution] | type[FeasibleSolution],
) -> None:
    """Require a summary for every successful solution."""
    with pytest.raises(TypeError):
        solution_type(Schedule((), ()))  # type: ignore[call-arg]


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


def test_scheduling_answers_with_its_objective_policy() -> None:
    """Return the scheduling service's exact objective policy."""
    policy: ObjectivePolicy = replace(DEFAULT_POLICY, per_count=3.0)
    service: Scheduling = Scheduling(Solver(), AllOf(), policy)
    result: AnswerResult = service.answer(ObjectivePolicyQuery(), problem(), None)
    assert isinstance(result, Answered)
    assert isinstance(result.answer, ObjectivePolicyAnswer)
    assert result.answer.policy is policy


def test_scheduling_evaluation_uses_its_objective_policy() -> None:
    """Compute evaluation weights and count costs from the service policy."""
    policy: ObjectivePolicy = replace(
        DEFAULT_POLICY,
        weights={**DEFAULT_POLICY.weights, Strength.WEAK: 7.0},
        per_count=3.0,
    )
    item: Task = task("task")
    constraints: tuple[Constraint, ...] = (
        SoftConstraint(
            ConstraintId("bound"),
            TimeBoundCondition(
                frozenset({item.id}),
                Boundary.START,
                TimeBoundRelation.AT,
                START + timedelta(hours=1),
            ),
            Strength.WEAK,
        ),
        SoftConstraint(
            ConstraintId("count"),
            DailyLimitCondition(frozenset({item.id}), AggregateQuantity.COUNT, 0),
            Strength.WEAK,
        ),
    )
    value: SchedulingProblem = problem(item, constraints=constraints)
    previous: Schedule = Schedule(
        (
            ScheduledTask(
                item.id, item.name, START, START + item.duration, item.participant_ids
            ),
        ),
        (),
    )
    service: Scheduling = Scheduling(Solver(), AllOf(), policy)
    query: EvaluationQuery = EvaluationQuery(False, None, None, 20)
    result: AnswerResult = service.answer(query, value, previous)
    assert isinstance(result, Answered)
    assert isinstance(result.answer, EvaluationAnswer)
    assert {item.constraint.id.value: item.cost for item in result.answer.items} == {
        "bound": 7.0,
        "count": 21.0,
    }
    assert result == query.answer(value, previous, policy)
