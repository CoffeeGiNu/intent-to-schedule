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
    FeasibleSolution,
    NoFeasibleSolution,
    OptimalSolution,
    Solution,
    SolutionNotFound,
)
from intent_to_schedule.application.translate import (
    ApplyRecord,
    ApplyStep,
    MessageStep,
    QueryRecord,
    QueryStep,
    ScheduleStep,
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

    def __init__(
        self,
        steps: Sequence[Step],
        events: list[str] | None = None,
        raise_on_call: int | None = None,
    ) -> None:
        self.steps: tuple[Step, ...] = tuple(steps)
        self.events: list[str] | None = events
        self.raise_on_call: int | None = raise_on_call
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
        if self.events is not None:
            self.events.append("translate")
        if self.raise_on_call == len(self.calls):
            raise RuntimeError("translator failed")
        return self.steps[len(self.calls) - 1]


class FakeSolver:
    """Return a chosen solution and capture its inputs."""

    def __init__(self, result: Solution) -> None:
        self.result: Solution = result
        self.calls: list[tuple[SchedulingProblem, Schedule | None, bool]] = []

    def solve(
        self,
        problem: SchedulingProblem,
        policy: ObjectivePolicy,
        previous: Schedule | None,
        stability: bool,
    ) -> Solution:
        self.calls.append((problem, previous, stability))
        return self.result


def ignore_save(problem: SchedulingProblem) -> None:
    """Ignore a saved problem."""


def test_conversation_queries_use_scheduling_policy() -> None:
    """Answer conversation queries with the scheduling service policy."""
    policy: ObjectivePolicy = replace(DEFAULT_POLICY, per_count=3.0)
    step: QueryStep = QueryStep(ObjectivePolicyQuery())
    translator: FakeStepTranslator = FakeStepTranslator((step, MessageStep("done")))
    service: Scheduling = Scheduling(
        FakeSolver(NoFeasibleSolution(Conflicts((), ()))), AllOf(), policy
    )
    Conversation(translator, service).respond((), make_problem(), None, ignore_save)
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


def test_message_ends_without_scheduling() -> None:
    """Return the user's message without invoking the solver."""
    problem: SchedulingProblem = make_problem()
    translator: FakeStepTranslator = FakeStepTranslator(
        (MessageStep("When works for you?"), ScheduleStep(True))
    )
    solver: FakeSolver = FakeSolver(NoFeasibleSolution(Conflicts((), ())))
    dialogue: tuple[Utterance, ...] = (
        Utterance(Speaker.USER, "Can we talk about the schedule?"),
    )
    response: Response = Conversation(translator, Scheduling(solver, AllOf())).respond(
        dialogue, problem, None, ignore_save
    )
    assert response == Response(problem, MessageStep("When works for you?"))
    assert translator.calls == [
        (dialogue, Summary(problem.calendar.grid, 0, 0, 0, 0, False), ())
    ]
    assert solver.calls == []


@pytest.mark.parametrize("saved", [True, False])
@pytest.mark.parametrize("stability", [True, False])
@pytest.mark.parametrize(
    ("result", "outcome"),
    [
        (
            OptimalSolution(
                Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)
            ),
            OptimalSolution(
                Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)
            ),
        ),
        (
            FeasibleSolution(
                Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)
            ),
            FeasibleSolution(
                Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)
            ),
        ),
        (
            NoFeasibleSolution(Conflicts((), ())),
            NoFeasibleSolution(Conflicts((), ())),
        ),
        (SolutionNotFound("time_limit"), SolutionNotFound("time_limit")),
    ],
)
def test_apply_then_schedule_uses_working_problem_and_stability(
    saved: bool, stability: bool, result: Solution, outcome: Solution
) -> None:
    """Pass successful edits, the saved schedule, and stability to scheduling."""
    problem: SchedulingProblem = make_problem()
    previous: Schedule | None = Schedule((), ()) if saved else None
    task: Task = make_task()
    step: ApplyStep = ApplyStep((AddTask(task),))
    translator: FakeStepTranslator = FakeStepTranslator(
        (step, ScheduleStep(stability), MessageStep("unused"))
    )
    solver: FakeSolver = FakeSolver(result)
    response: Response = Conversation(translator, Scheduling(solver, AllOf())).respond(
        (), problem, previous, ignore_save
    )
    assert response.problem.tasks == (task,)
    assert problem.tasks == ()
    assert response.outcome == outcome
    assert solver.calls == [(response.problem, previous, stability)]
    assert [call[1] for call in translator.calls] == [
        Summary(problem.calendar.grid, 0, 0, 0, 0, saved),
        Summary(problem.calendar.grid, 0, 1, 0, 0, saved),
    ]
    assert translator.calls[1][2] == (ApplyRecord(step, Executed(response.problem)),)


def test_rejected_batch_is_recorded_and_can_be_corrected() -> None:
    """Continue after rejection without retaining partial batch edits."""
    problem: SchedulingProblem = make_problem()
    task: Task = make_task()
    rejected: ApplyStep = ApplyStep((AddTask(task), RemoveTask(TaskId("missing"))))
    corrected: ApplyStep = ApplyStep((AddTask(task),))
    translator: FakeStepTranslator = FakeStepTranslator(
        (rejected, corrected, ScheduleStep(True))
    )
    solver: FakeSolver = FakeSolver(
        OptimalSolution(Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0))
    )
    response: Response = Conversation(translator, Scheduling(solver, AllOf())).respond(
        (), problem, None, ignore_save
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
    saved: list[SchedulingProblem] = []

    def save(problem: SchedulingProblem) -> None:
        saved.append(problem)

    response: Response = Conversation(
        translator,
        Scheduling(FakeSolver(NoFeasibleSolution(Conflicts((), ()))), AllOf()),
    ).respond((), problem, Schedule((), ()), save)
    assert response.problem is saved[0]
    assert response.problem.tasks == (make_task(),)
    assert translator.calls[2][2][1] == QueryRecord(query_step, result)


def test_apply_then_message_saves_before_the_next_translation() -> None:
    """Save an accepted apply before translating the next step."""
    problem: SchedulingProblem = make_problem()
    task: Task = make_task()
    events: list[str] = []
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep((AddTask(task),)), MessageStep("done")), events
    )
    saved: list[SchedulingProblem] = []

    def save(updated: SchedulingProblem) -> None:
        events.append("save")
        saved.append(updated)

    response: Response = Conversation(
        translator,
        Scheduling(FakeSolver(NoFeasibleSolution(Conflicts((), ()))), AllOf()),
    ).respond((), problem, None, save)

    assert response.problem is saved[0]
    assert response.problem.tasks == (task,)
    assert events == ["translate", "save", "translate"]
    assert len(saved) == 1


def test_apply_then_translator_error_saves_before_error() -> None:
    """Keep an accepted apply saved when a later translation raises."""
    problem: SchedulingProblem = make_problem()
    task: Task = make_task()
    events: list[str] = []
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep((AddTask(task),)),), events, raise_on_call=2
    )
    saved: list[SchedulingProblem] = []

    def save(updated: SchedulingProblem) -> None:
        events.append("save")
        saved.append(updated)

    conversation: Conversation = Conversation(
        translator,
        Scheduling(FakeSolver(NoFeasibleSolution(Conflicts((), ()))), AllOf()),
    )

    with pytest.raises(RuntimeError, match="translator failed"):
        conversation.respond((), problem, None, save)

    assert len(saved) == 1
    assert saved[0].tasks == (task,)
    assert events == ["translate", "save", "translate"]


def test_rejected_apply_then_message_does_not_save() -> None:
    """Leave the working problem unchanged after a rejected apply."""
    problem: SchedulingProblem = make_problem()
    translator: FakeStepTranslator = FakeStepTranslator(
        (
            ApplyStep((RemoveTask(TaskId("missing")),)),
            MessageStep("Please clarify."),
        )
    )
    saved: list[SchedulingProblem] = []

    def save(updated: SchedulingProblem) -> None:
        saved.append(updated)

    response: Response = Conversation(
        translator,
        Scheduling(FakeSolver(NoFeasibleSolution(Conflicts((), ()))), AllOf()),
    ).respond((), problem, None, save)

    assert response.problem is problem
    assert saved == []


def test_step_limit_returns_exhausted_after_repeated_rejections() -> None:
    """Stop at the step limit without raising for repeated rejections."""
    problem: SchedulingProblem = make_problem()
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep((RemoveTask(TaskId("missing")),)),)
        * STEP_LIMIT
        + (ScheduleStep(True),)
    )
    solver: FakeSolver = FakeSolver(NoFeasibleSolution(Conflicts((), ())))
    response: Response = Conversation(translator, Scheduling(solver, AllOf())).respond(
        (), problem, None, ignore_save
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
        Scheduling(FakeSolver(NoFeasibleSolution(Conflicts((), ()))), AllOf()),
    ).respond((), make_problem(), None, ignore_save)
    assert response.outcome == MessageStep("done")
    assert len(translator.calls) == STEP_LIMIT
