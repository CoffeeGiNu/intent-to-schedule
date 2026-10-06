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


class MemoryStateStore:
    def __init__(
        self,
        problem: SchedulingProblem,
        previous: Schedule | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.problem: SchedulingProblem = problem
        self.previous: Schedule | None = previous
        self.events: list[str] | None = events
        self.saved_problems: list[SchedulingProblem] = []
        self.saved_previous: list[Schedule | None] = []

    def load_problem(self) -> SchedulingProblem:
        return self.problem

    def save_problem(self, problem: SchedulingProblem) -> None:
        self.problem = problem
        self.saved_problems.append(problem)
        if self.events is not None:
            self.events.append("save_problem")

    def load_previous(self) -> Schedule | None:
        return self.previous

    def save_previous(self, previous: Schedule | None) -> None:
        self.previous = previous
        self.saved_previous.append(previous)


class MemoryDialogueStore:
    def __init__(
        self, dialogue: tuple[Utterance, ...] = (), events: list[str] | None = None
    ) -> None:
        self.dialogue: tuple[Utterance, ...] = dialogue
        self.events: list[str] | None = events
        self.saved: list[tuple[Utterance, ...]] = []

    def load(self) -> tuple[Utterance, ...]:
        return self.dialogue

    def save(self, dialogue: Sequence[Utterance]) -> None:
        self.dialogue = tuple(dialogue)
        self.saved.append(self.dialogue)
        if self.events is not None:
            self.events.append("save_dialogue")


class FakeStepTranslator:
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


def make_conversation(
    translator: FakeStepTranslator,
    state_store: MemoryStateStore | None = None,
    dialogue_store: MemoryDialogueStore | None = None,
    solver: FakeSolver | None = None,
) -> tuple[Conversation, MemoryStateStore, MemoryDialogueStore, FakeSolver]:
    problem: SchedulingProblem = make_problem()
    states: MemoryStateStore = state_store or MemoryStateStore(problem)
    dialogues: MemoryDialogueStore = dialogue_store or MemoryDialogueStore()
    scheduling_solver: FakeSolver = solver or FakeSolver(
        NoFeasibleSolution(Conflicts((), ()))
    )
    service: Scheduling = Scheduling(scheduling_solver, AllOf(), states)
    return Conversation(translator, service, dialogues), states, dialogues, scheduling_solver


def test_conversation_queries_use_scheduling_policy() -> None:
    """Answer conversation queries with the scheduling service policy."""
    policy: ObjectivePolicy = replace(DEFAULT_POLICY, per_count=3.0)
    step: QueryStep = QueryStep(ObjectivePolicyQuery())
    translator: FakeStepTranslator = FakeStepTranslator((step, MessageStep("done")))
    problem: SchedulingProblem = make_problem()
    states: MemoryStateStore = MemoryStateStore(problem)
    dialogues: MemoryDialogueStore = MemoryDialogueStore()
    solver: FakeSolver = FakeSolver(NoFeasibleSolution(Conflicts((), ())))
    service: Scheduling = Scheduling(solver, AllOf(), states, policy)

    Conversation(translator, service, dialogues).respond("question")

    assert translator.calls[1][2] == (
        QueryRecord(step, Answered(ObjectivePolicyAnswer(policy))),
    )


def test_each_step_gets_summary_from_current_stored_problem() -> None:
    """Refresh the summary after an accepted apply is stored."""
    problem: SchedulingProblem = make_problem()
    states: MemoryStateStore = MemoryStateStore(problem)
    task: Task = make_task()
    query: QueryStep = QueryStep(SummaryQuery())
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep((AddTask(task),)), query, MessageStep("done"))
    )
    conversation: Conversation = make_conversation(
        translator, state_store=states
    )[0]

    conversation.respond("add a task")

    assert [call[1].tasks for call in translator.calls] == [0, 1, 1]
    assert translator.calls[2][2][1] == QueryRecord(
        query, Answered(Summary(problem.calendar.grid, 0, 1, 0, 0, False))
    )


def test_message_ends_without_scheduling() -> None:
    """End the turn on a message without invoking the solver."""
    problem: SchedulingProblem = make_problem()
    dialogues: MemoryDialogueStore = MemoryDialogueStore()
    translator: FakeStepTranslator = FakeStepTranslator(
        (MessageStep("When works for you?"), ScheduleStep(True))
    )
    solver: FakeSolver = FakeSolver(NoFeasibleSolution(Conflicts((), ())))
    states: MemoryStateStore = MemoryStateStore(problem)
    conversation: Conversation = make_conversation(
        translator, states, dialogues, solver
    )[0]

    outcome: MessageStep | Solution | Exhausted = conversation.respond(
        "Can we talk about the schedule?"
    )

    dialogue: tuple[Utterance, ...] = (
        Utterance(Speaker.USER, "Can we talk about the schedule?"),
    )
    assert outcome == MessageStep("When works for you?")
    assert translator.calls == [
        (dialogue, Summary(problem.calendar.grid, 0, 0, 0, 0, False), ())
    ]
    assert solver.calls == []


@pytest.mark.parametrize("saved", [True, False])
@pytest.mark.parametrize("stability", [True, False])
@pytest.mark.parametrize(
    "result",
    [
        OptimalSolution(
            Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)
        ),
        FeasibleSolution(
            Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)
        ),
        NoFeasibleSolution(Conflicts((), ())),
        SolutionNotFound("time_limit"),
    ],
)
def test_apply_then_schedule_uses_working_problem_and_stability(
    saved: bool, stability: bool, result: Solution
) -> None:
    """Schedule the accepted edit with the stored previous schedule and stability."""
    problem: SchedulingProblem = make_problem()
    previous: Schedule | None = Schedule((), ()) if saved else None
    states: MemoryStateStore = MemoryStateStore(problem, previous)
    dialogues: MemoryDialogueStore = MemoryDialogueStore()
    task: Task = make_task()
    apply_step: ApplyStep = ApplyStep((AddTask(task),))
    translator: FakeStepTranslator = FakeStepTranslator(
        (apply_step, ScheduleStep(stability), MessageStep("unused"))
    )
    solver: FakeSolver = FakeSolver(result)
    conversation: Conversation = make_conversation(
        translator, states, dialogues, solver
    )[0]

    outcome: MessageStep | Solution | Exhausted = conversation.respond("Add a task.")

    assert states.problem.tasks == (task,)
    assert problem.tasks == ()
    assert outcome is result
    assert solver.calls == [(states.problem, previous, stability)]
    assert [call[1] for call in translator.calls] == [
        Summary(problem.calendar.grid, 0, 0, 0, 0, saved),
        Summary(problem.calendar.grid, 0, 1, 0, 0, saved),
    ]
    assert translator.calls[1][2] == (
        ApplyRecord(apply_step, Executed(states.problem)),
    )


def test_rejected_batch_is_recorded_and_can_be_corrected() -> None:
    """Record a rejection, then accept a corrected batch on the stored problem."""
    problem: SchedulingProblem = make_problem()
    states: MemoryStateStore = MemoryStateStore(problem)
    dialogues: MemoryDialogueStore = MemoryDialogueStore()
    task: Task = make_task()
    rejected_step: ApplyStep = ApplyStep(
        (AddTask(task), RemoveTask(TaskId("missing")))
    )
    corrected_step: ApplyStep = ApplyStep((AddTask(task),))
    translator: FakeStepTranslator = FakeStepTranslator(
        (rejected_step, corrected_step, ScheduleStep(True))
    )
    solution: OptimalSolution = OptimalSolution(
        Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)
    )
    solver: FakeSolver = FakeSolver(solution)
    conversation: Conversation = make_conversation(
        translator, states, dialogues, solver
    )[0]

    outcome: MessageStep | Solution | Exhausted = conversation.respond("Add a task.")

    rejected_record: StepRecord = translator.calls[1][2][0]
    assert isinstance(rejected_record, ApplyRecord)
    assert isinstance(rejected_record.result, Rejected)
    assert rejected_record.result.violations.items[0].message == (
        "Task missing does not exist"
    )
    assert states.saved_problems == [states.problem]
    assert translator.calls[1][1].tasks == 0
    assert translator.calls[2][2] == (
        rejected_record,
        ApplyRecord(corrected_step, Executed(states.problem)),
    )
    assert outcome is solution
    assert states.problem.tasks == (task,)


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
    """Answer or reject a query after an accepted apply is stored."""
    problem: SchedulingProblem = make_problem()
    previous: Schedule = Schedule((), ())
    states: MemoryStateStore = MemoryStateStore(problem, previous)
    dialogues: MemoryDialogueStore = MemoryDialogueStore()
    query_step: QueryStep = QueryStep(query)
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep((AddTask(make_task()),)), query_step, MessageStep("done"))
    )
    conversation: Conversation = make_conversation(
        translator, states, dialogues
    )[0]

    outcome: MessageStep | Solution | Exhausted = conversation.respond("Add a task.")

    assert outcome == MessageStep("done")
    assert states.saved_problems == [states.problem]
    assert states.problem.tasks == (make_task(),)
    assert translator.calls[2][1].tasks == 1
    assert translator.calls[2][2][1] == QueryRecord(query_step, result)


def test_accepted_apply_is_saved_before_the_next_translation() -> None:
    """Save an accepted apply before translating the next step."""
    events: list[str] = []
    problem: SchedulingProblem = make_problem()
    states: MemoryStateStore = MemoryStateStore(problem, events=events)
    dialogues: MemoryDialogueStore = MemoryDialogueStore(events=events)
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep((AddTask(make_task()),)), MessageStep("done")), events
    )
    conversation: Conversation = make_conversation(
        translator, states, dialogues
    )[0]

    outcome: MessageStep | Solution | Exhausted = conversation.respond("add a task")

    assert outcome == MessageStep("done")
    assert events == ["translate", "save_problem", "translate", "save_dialogue"]
    assert len(states.saved_problems) == 1


def test_translator_error_keeps_accepted_apply_but_does_not_save_dialogue() -> None:
    """Keep earlier accepted applies when translation raises without saving dialogue."""
    events: list[str] = []
    states: MemoryStateStore = MemoryStateStore(make_problem(), events=events)
    dialogues: MemoryDialogueStore = MemoryDialogueStore(events=events)
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep((AddTask(make_task()),)),), events, raise_on_call=2
    )
    conversation: Conversation = make_conversation(
        translator, states, dialogues
    )[0]

    with pytest.raises(RuntimeError, match="translator failed"):
        conversation.respond("add a task")

    assert len(states.saved_problems) == 1
    assert states.problem.tasks == (make_task(),)
    assert dialogues.saved == []
    assert events == ["translate", "save_problem", "translate"]


def test_rejected_apply_then_message_does_not_save() -> None:
    """Leave the stored problem unchanged after a rejected apply."""
    problem: SchedulingProblem = make_problem()
    states: MemoryStateStore = MemoryStateStore(problem)
    dialogues: MemoryDialogueStore = MemoryDialogueStore()
    translator: FakeStepTranslator = FakeStepTranslator(
        (
            ApplyStep((RemoveTask(TaskId("missing")),)),
            MessageStep("Please clarify."),
        )
    )
    conversation: Conversation = make_conversation(
        translator, states, dialogues
    )[0]

    outcome: MessageStep | Solution | Exhausted = conversation.respond("Clarify.")

    assert outcome == MessageStep("Please clarify.")
    assert states.problem is problem
    assert states.saved_problems == []


def test_step_limit_returns_exhausted_after_repeated_rejections() -> None:
    """Stop after STEP_LIMIT rejected steps without calling the solver."""
    states: MemoryStateStore = MemoryStateStore(make_problem())
    dialogues: MemoryDialogueStore = MemoryDialogueStore()
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep((RemoveTask(TaskId("missing")),)),) * STEP_LIMIT
        + (ScheduleStep(True),)
    )
    solver: FakeSolver = FakeSolver(NoFeasibleSolution(Conflicts((), ())))
    conversation: Conversation = make_conversation(
        translator, states, dialogues, solver
    )[0]

    outcome: MessageStep | Solution | Exhausted = conversation.respond("Keep trying.")

    assert outcome == Exhausted()
    assert len(translator.calls) == STEP_LIMIT
    assert len(translator.calls[-1][2]) == STEP_LIMIT - 1
    assert solver.calls == []


def test_terminal_step_at_limit_is_honored() -> None:
    """Honor a message returned on the final permitted translation step."""
    states: MemoryStateStore = MemoryStateStore(make_problem())
    dialogues: MemoryDialogueStore = MemoryDialogueStore()
    translator: FakeStepTranslator = FakeStepTranslator(
        (ApplyStep(()),) * (STEP_LIMIT - 1) + (MessageStep("done"),)
    )
    solver: FakeSolver = FakeSolver(NoFeasibleSolution(Conflicts((), ())))
    conversation: Conversation = make_conversation(
        translator, states, dialogues, solver
    )[0]

    outcome: MessageStep | Solution | Exhausted = conversation.respond("Continue.")

    assert outcome == MessageStep("done")
    assert len(translator.calls) == STEP_LIMIT
    assert len(translator.calls[-1][2]) == STEP_LIMIT - 1
    assert solver.calls == []


@pytest.mark.parametrize(
    ("steps", "solution", "assistant_text", "expected"),
    [
        ((MessageStep("Hello."),), NoFeasibleSolution(Conflicts((), ())), "Hello.", MessageStep("Hello.")),
        (
            (ScheduleStep(True),),
            OptimalSolution(Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)),
            "Scheduled.",
            OptimalSolution(Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)),
        ),
        (
            (ScheduleStep(False),),
            FeasibleSolution(Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)),
            "Scheduled.",
            FeasibleSolution(Schedule((), ()), ScheduleSummary(0.0, 0.0, 0.0, 0, 0, 0, 0)),
        ),
        (
            (ScheduleStep(True),),
            NoFeasibleSolution(Conflicts((), ())),
            "No feasible solution.",
            NoFeasibleSolution(Conflicts((), ())),
        ),
        (
            (ScheduleStep(True),),
            SolutionNotFound("time_limit"),
            "Solution not found.",
            SolutionNotFound("time_limit"),
        ),
        (
            (ApplyStep((RemoveTask(TaskId("missing")),)),) * STEP_LIMIT,
            NoFeasibleSolution(Conflicts((), ())),
            "Step limit reached.",
            Exhausted(),
        ),
    ],
)
def test_dialogue_saves_user_and_assistant_for_each_outcome(
    steps: tuple[Step, ...],
    solution: Solution,
    assistant_text: str,
    expected: object,
) -> None:
    """Save both turn utterances after every completed outcome."""
    old: tuple[Utterance, ...] = (Utterance(Speaker.ASSISTANT, "Earlier."),)
    dialogues: MemoryDialogueStore = MemoryDialogueStore(old)
    translator: FakeStepTranslator = FakeStepTranslator(steps)
    states: MemoryStateStore = MemoryStateStore(make_problem())
    solver: FakeSolver = FakeSolver(solution)
    conversation: Conversation = make_conversation(
        translator, states, dialogues, solver
    )[0]

    outcome: MessageStep | Solution | Exhausted = conversation.respond(
        "Latest request."
    )

    assert outcome == expected
    assert dialogues.saved == [
        (
            *old,
            Utterance(Speaker.USER, "Latest request."),
            Utterance(Speaker.ASSISTANT, assistant_text),
        )
    ]
