"""Tests for the conversation step loop."""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import pytest

import intent_to_schedule.application.converse as converse
from intent_to_schedule.application.command import (
    AddTask,
    Executed,
    Rejected,
    RemoveTask,
)
from intent_to_schedule.application.converse import (
    Conversation,
    Exhausted,
    Response,
    STEP_LIMIT,
)
from intent_to_schedule.application.query import (
    Answered,
    AnswerResult,
    Summary,
    SummaryQuery,
)
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import Infeasible, Solved, SolveResult
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
from intent_to_schedule.domain.consistency import AllOf
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
    """Return a chosen solve result and capture problems."""

    def __init__(self, result: SolveResult) -> None:
        self.result: SolveResult = result
        self.problems: list[SchedulingProblem] = []

    def solve(self, problem: SchedulingProblem, previous: Schedule | None) -> SolveResult:
        self.problems.append(problem)
        return self.result


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


@pytest.fixture
def summary_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[SchedulingProblem, Schedule | None]]:
    """Replace the unfinished summary query with a recording fake."""
    calls: list[tuple[SchedulingProblem, Schedule | None]] = []

    def summarize(problem: SchedulingProblem, previous: Schedule | None) -> Summary:
        calls.append((problem, previous))
        return Summary(
            problem.calendar.grid,
            len(problem.people),
            len(problem.tasks),
            len(problem.fixed_tasks),
            len(problem.constraints),
            previous is not None,
        )

    monkeypatch.setattr(converse, "summarize", summarize, raising=False)
    return calls


def test_message_ends_without_solving(
    summary_calls: list[tuple[SchedulingProblem, Schedule | None]],
) -> None:
    """Return the user's message without invoking the solver."""
    problem: SchedulingProblem = make_problem()
    translator: FakeStepTranslator = FakeStepTranslator(
        (MessageStep("When works for you?"), SolveStep(True))
    )
    solver: FakeSolver = FakeSolver(Infeasible())
    dialogue: tuple[Utterance, ...] = (Utterance(Speaker.USER, "Can we talk about the schedule?"),)
    response: Response = Conversation(translator, Scheduling(solver, AllOf())).respond(
        dialogue, problem, None
    )
    assert response == Response(problem, MessageStep("When works for you?"))
    assert translator.calls[0][0] == dialogue
    assert translator.calls[0][2] == ()
    assert summary_calls == [(problem, None)]
    assert solver.problems == []


@pytest.mark.parametrize("stability", [True, False])
@pytest.mark.parametrize("result", [Solved(Schedule((), ())), Infeasible()])
def test_apply_then_solve_uses_working_problem_and_stability(
    summary_calls: list[tuple[SchedulingProblem, Schedule | None]],
    monkeypatch: pytest.MonkeyPatch,
    stability: bool,
    result: SolveResult,
) -> None:
    """Pass successful edits and the selected previous schedule to solve."""
    problem: SchedulingProblem = make_problem()
    previous: Schedule = Schedule((), ())
    task: Task = make_task()
    step: ApplyStep = ApplyStep((AddTask(task),))
    translator: FakeStepTranslator = FakeStepTranslator(
        (step, SolveStep(stability), MessageStep("unused"))
    )
    service: Scheduling = Scheduling(FakeSolver(result), AllOf())
    solve_calls: list[tuple[SchedulingProblem, Schedule | None]] = []

    def solve(working: SchedulingProblem, prior: Schedule | None) -> SolveResult:
        solve_calls.append((working, prior))
        return result

    monkeypatch.setattr(service, "solve", solve)
    response: Response = Conversation(translator, service).respond(
        (), problem, previous
    )
    assert response.problem.tasks == (task,)
    assert problem.tasks == ()
    assert response.outcome == result
    assert solve_calls == [(response.problem, previous if stability else None)]
    assert summary_calls == [(problem, previous), (response.problem, previous)]
    assert translator.calls[1][1].tasks == 1
    assert translator.calls[1][2] == (ApplyRecord(step, Executed(response.problem)),)


def test_rejected_batch_is_recorded_and_can_be_corrected(
    summary_calls: list[tuple[SchedulingProblem, Schedule | None]],
) -> None:
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
    assert summary_calls[1][0] == problem
    assert response.problem.tasks == (task,)
    assert len(translator.calls[2][2]) == 2


@pytest.mark.parametrize("rejected", [False, True])
def test_query_uses_working_problem_and_records_result(
    summary_calls: list[tuple[SchedulingProblem, Schedule | None]],
    monkeypatch: pytest.MonkeyPatch,
    rejected: bool,
) -> None:
    """Feed query answers and rejections into the next step."""
    from intent_to_schedule.domain.consistency import Violation, Violations

    problem: SchedulingProblem = make_problem()
    previous: Schedule = Schedule((), ())
    task: Task = make_task()
    query_step: QueryStep = QueryStep(SummaryQuery())
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep((AddTask(task),)), query_step, MessageStep("done"))
    )
    answer_calls: list[tuple[SchedulingProblem, Schedule | None]] = []
    result: AnswerResult = (
        Rejected(Violations((Violation("Try a narrower query."),)))
        if rejected
        else Answered(Summary(problem.calendar.grid, 0, 1, 0, 0, True))
    )

    def answer(
        self: SummaryQuery, working: SchedulingProblem, prior: Schedule | None
    ) -> AnswerResult:
        answer_calls.append((working, prior))
        return result

    monkeypatch.setattr(SummaryQuery, "answer", answer)
    response: Response = Conversation(
        translator, Scheduling(FakeSolver(Infeasible()), AllOf())
    ).respond((), problem, previous)
    assert len(answer_calls) == 1
    assert answer_calls[0][0].tasks == (task,)
    assert answer_calls[0][1] is previous
    assert response.problem is problem
    assert translator.calls[2][2][1] == QueryRecord(query_step, result)


def test_step_limit_returns_exhausted_after_repeated_rejections(
    summary_calls: list[tuple[SchedulingProblem, Schedule | None]],
) -> None:
    """Stop at the step limit without raising for repeated rejections."""
    problem: SchedulingProblem = make_problem()
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep((RemoveTask(TaskId("missing")),)),) * STEP_LIMIT + (SolveStep(True),)
    )
    solver: FakeSolver = FakeSolver(Infeasible())
    response: Response = Conversation(translator, Scheduling(solver, AllOf())).respond(
        (), problem, None
    )
    assert response == Response(problem, Exhausted())
    assert len(translator.calls) == STEP_LIMIT
    assert len(summary_calls) == STEP_LIMIT
    assert len(translator.calls[-1][2]) == STEP_LIMIT - 1
    assert solver.problems == []


def test_terminal_step_at_limit_is_honored(
    summary_calls: list[tuple[SchedulingProblem, Schedule | None]],
) -> None:
    """Allow the final permitted step to end the turn."""
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep(()),) * (STEP_LIMIT - 1) + (MessageStep("done"),)
    )
    response: Response = Conversation(
        translator, Scheduling(FakeSolver(Infeasible()), AllOf())
    ).respond((), make_problem(), None)
    assert response.outcome == MessageStep("done")
    assert len(translator.calls) == STEP_LIMIT
