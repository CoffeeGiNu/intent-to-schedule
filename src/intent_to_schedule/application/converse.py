from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from intent_to_schedule.application.command import ExecuteResult
from intent_to_schedule.application.query import (
    AnswerResult,
    Summary,
)
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import (
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
    StepTranslator,
    Utterance,
)

STEP_LIMIT: int = 12


class DialogueStore(Protocol):
    """Port that keeps the conversation history between turns."""

    def load(self) -> tuple[Utterance, ...]: ...

    def save(self, dialogue: Sequence[Utterance]) -> None: ...


@dataclass(frozen=True)
class Exhausted:
    """Outcome of a turn that reached the step limit."""


# TODO: once the conversation outgrows a demo, move converse, translate, and the
# OpenAI adapter into their own feature package.
class Conversation:
    """Use case that handles one turn of a conversation."""

    def __init__(
        self,
        translator: StepTranslator,
        scheduling: Scheduling,
        dialogue_store: DialogueStore,
    ) -> None:
        self._translator: StepTranslator = translator
        self._scheduling: Scheduling = scheduling
        self._dialogue_store: DialogueStore = dialogue_store

    def respond(self, text: str) -> MessageStep | Solution | Exhausted:
        """Run translation steps and save the completed turn."""
        dialogue: tuple[Utterance, ...] = (
            *self._dialogue_store.load(),
            Utterance(Speaker.USER, text),
        )
        steps: list[StepRecord] = []
        outcome: MessageStep | Solution | Exhausted
        for _ in range(STEP_LIMIT):
            summary: Summary = self._scheduling.summarize()
            step: Step = self._translator.translate(dialogue, summary, tuple(steps))
            if isinstance(step, QueryStep):
                answer: AnswerResult = self._scheduling.answer(step.query)
                steps.append(QueryRecord(step, answer))
            elif isinstance(step, ApplyStep):
                result: ExecuteResult = self._scheduling.execute(step.commands)
                steps.append(ApplyRecord(step, result))
            elif isinstance(step, ScheduleStep):
                outcome = self._scheduling.schedule(step.stability)
                break
            else:
                outcome = step
                break
        else:
            outcome = Exhausted()
        assistant_text: str = _assistant_text(outcome)
        self._dialogue_store.save(
            (*dialogue, Utterance(Speaker.ASSISTANT, assistant_text))
        )
        return outcome


def _assistant_text(outcome: MessageStep | Solution | Exhausted) -> str:
    """Return the assistant utterance for a completed outcome."""
    match outcome:
        case MessageStep(text=text):
            return text
        case OptimalSolution() | FeasibleSolution():
            return "Scheduled."
        case NoFeasibleSolution():
            return "No feasible solution."
        case SolutionNotFound():
            return "Solution not found."
        case Exhausted():
            return "Step limit reached."
