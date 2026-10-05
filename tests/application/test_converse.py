"""Tests for the conversation step loop."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from intent_to_schedule.application.command import (
    AddTask,
    Executed,
    Rejected,
    RemoveTask,
)
from intent_to_schedule.application.converse import (
    STEP_LIMIT,
    Conversation,
    Exhausted,
    Response,
)
from intent_to_schedule.application.objective import ScheduleSummary
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.query import (
    Answered,
    AnswerResult,
    ObjectivePolicyAnswer,
    ObjectivePolicyQuery,
    PeopleQuery,
    SchedulingQuery,
    Summary,
    SummaryQuery,
)
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import (
    Conflicts,
    Infeasible,
    Solved,
    SolveResult,
)
from intent_to_schedule.application.translate import (
    ApplyRecord,
    ApplyStep,
    MessageStep,
    QueryRecord,
    QueryStep,
    SolveStep,
    Speaker,
    Step,
    StepRecord,
    Utterance,
)
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.consistency import AllOf, Violation, Violations
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId


class FakeStepTranslator:
    """Return scripted steps and capture their context."""

    def __init__(self, steps: Sequence[Step]) -> None:
        self.steps: tuple[Step, ...] = tuple(steps)
        self.calls: list[
            tuple[tuple[Utterance, ...], Summary, tuple[StepRecord, ...]]
        ] = []

    def translate(
        self,
        dialogue: Sequence[Utterance],
        summary: Summary,
        steps: Sequence[StepRecord],
    ) -> Step:
        self.calls.append((tuple(dialogue), summary, tuple(steps)))
        return self.steps[len(self.calls) - 1]


class FakeSolver:
    """Return a chosen solve result and capture its inputs."""

    def __init__(self, result: SolveResult) -> None:
        self.result: SolveResult = result
        self.calls: list[tuple[SchedulingProblem, Schedule | None]] = []

    def solve(
        self, problem: SchedulingProblem, previous: Schedule | None
    ) -> SolveResult:
        self.calls.append((problem, previous))
        return self.result


def test_conversation_queries_use_scheduling_policy() -> None:
    """Answer conversation queries with the scheduling service policy."""
    policy: ObjectivePolicy = replace(DEFAULT_POLICY, per_count=3.0)
    step: QueryStep = QueryStep(ObjectivePolicyQuery())
    translator: FakeStepTranslator = FakeStepTranslator((step, MessageStep("done")))
    service: Scheduling = Scheduling(
        FakeSolver(Infeasible(Conflicts((), ()))), AllOf(), policy
    )
    Conversation(translator, service).respond((), make_problem(), None)
    assert translator.calls[1][2] == (
        QueryRecord(step, Answered(ObjectivePolicyAnswer(policy))),
    )


def make_problem() -> SchedulingProblem:
    """Build an empty scheduling problem."""
    start: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    return SchedulingProblem(
        Calendar(
            TimeGrid(
                TimeInterval(start, start + timedelta(hours=3)), timedelta(hours=1)
            ),
            (),
        ),
        (),
        (),
        (),
        (),
    )


def make_task() -> Task:
    """Build a task without participants."""
    return Task(
        TaskId("review"),
        "Review",
        timedelta(hours=1),
        frozenset(),
        Importance.HIGH,
        True,
        Strength.NORMAL,
    )


def test_message_ends_without_solving() -> None:
    """Return the user's message without invoking the solver."""
    problem: SchedulingProblem = make_problem()
    translator: FakeStepTranslator = FakeStepTranslator(
        (MessageStep("When works for you?"), SolveStep(True))
    )
    solver: FakeSolver = FakeSolver(Infeasible(Conflicts((), ())))
    dialogue: tuple[Utterance, ...] = (
        Utterance(Speaker.USER, "Can we talk about the schedule?"),
    )
    response: Response = Conversation(translator, Scheduling(solver, AllOf())).respond(
        dialogue, problem, None
    )
    assert response == Response(problem, MessageStep("When works for you?"))
    assert translator.calls == [
        (dialogue, Summary(problem.calendar.grid, 0, 0, 0, 0, False), ())
    ]
    assert solver.calls == []


@pytest.mark.parametrize("stability", [True, False])
@pytest.mark.parametrize(
    "result",
    [
        Solved(Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)),
        Infeasible(Conflicts((), ())),
    ],
)
def test_apply_then_solve_uses_working_problem_and_stability(
    stability: bool, result: SolveResult
) -> None:
    """Pass successful edits and the selected previous schedule to solve."""
    problem: SchedulingProblem = make_problem()
    previous: Schedule = Schedule((), ())
    task: Task = make_task()
    step: ApplyStep = ApplyStep((AddTask(task),))
    translator: FakeStepTranslator = FakeStepTranslator(
        (step, SolveStep(stability), MessageStep("unused"))
    )
    solver: FakeSolver = FakeSolver(result)
    response: Response = Conversation(translator, Scheduling(solver, AllOf())).respond(
        (), problem, previous
    )
    assert response.problem.tasks == (task,)
    assert problem.tasks == ()
    assert response.outcome == result
    assert solver.calls == [(response.problem, previous if stability else None)]
    assert [call[1] for call in translator.calls] == [
        Summary(problem.calendar.grid, 0, 0, 0, 0, True),
        Summary(problem.calendar.grid, 0, 1, 0, 0, True),
    ]
    assert translator.calls[1][2] == (ApplyRecord(step, Executed(response.problem)),)


def test_rejected_batch_is_recorded_and_can_be_corrected() -> None:
    """Continue after rejection without retaining partial batch edits."""
    problem: SchedulingProblem = make_problem()
    task: Task = make_task()
    rejected: ApplyStep = ApplyStep((AddTask(task), RemoveTask(TaskId("missing"))))
    corrected: ApplyStep = ApplyStep((AddTask(task),))
    translator: FakeStepTranslator = FakeStepTranslator(
        (rejected, corrected, SolveStep(True))
    )
    solver: FakeSolver = FakeSolver(Solved(Schedule((), ())))
    response: Response = Conversation(translator, Scheduling(solver, AllOf())).respond(
        (), problem, None
    )
    record: StepRecord = translator.calls[1][2][0]
    assert isinstance(record, ApplyRecord)
    assert isinstance(record.result, Rejected)
    assert record.result.violations.items[0].message == "Task missing does not exist"
    assert translator.calls[1][1].tasks == 0
    assert response.problem.tasks == (task,)
    assert len(translator.calls[2][2]) == 2


@pytest.mark.parametrize(
    ("query", "result"),
    [
        (
            SummaryQuery(),
            Answered(Summary(make_problem().calendar.grid, 0, 1, 0, 0, True)),
        ),
        (
            PeopleQuery(None, None, False, 0),
            Rejected(
                Violations(
                    (Violation("Query limit 0 must be between 1 and 100 inclusive."),)
                )
            ),
        ),
    ],
)
def test_query_uses_working_problem_and_records_result(
    query: SchedulingQuery, result: AnswerResult
) -> None:
    """Feed query answers and rejections into the next step."""
    problem: SchedulingProblem = make_problem()
    query_step: QueryStep = QueryStep(query)
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep((AddTask(make_task()),)), query_step, MessageStep("done"))
    )
    response: Response = Conversation(
        translator,
        Scheduling(FakeSolver(Infeasible(Conflicts((), ()))), AllOf()),
    ).respond((), problem, Schedule((), ()))
    assert response.problem is problem
    assert translator.calls[2][2][1] == QueryRecord(query_step, result)


def test_step_limit_returns_exhausted_after_repeated_rejections() -> None:
    """Stop at the step limit without raising for repeated rejections."""
    problem: SchedulingProblem = make_problem()
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep((RemoveTask(TaskId("missing")),)),) * STEP_LIMIT + (SolveStep(True),)
    )
    solver: FakeSolver = FakeSolver(Infeasible(Conflicts((), ())))
    response: Response = Conversation(translator, Scheduling(solver, AllOf())).respond(
        (), problem, None
    )
    assert response == Response(problem, Exhausted())
    assert len(translator.calls) == STEP_LIMIT
    assert len(translator.calls[-1][2]) == STEP_LIMIT - 1
    assert solver.calls == []


def test_terminal_step_at_limit_is_honored() -> None:
    """Allow the final permitted step to end the turn."""
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep(()),) * (STEP_LIMIT - 1) + (MessageStep("done"),)
    )
    response: Response = Conversation(
        translator,
        Scheduling(FakeSolver(Infeasible(Conflicts((), ()))), AllOf()),
    ).respond((), make_problem(), None)
    assert response.outcome == MessageStep("done")
    assert len(translator.calls) == STEP_LIMIT
