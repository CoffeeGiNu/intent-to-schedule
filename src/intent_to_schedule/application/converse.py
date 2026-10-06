from collections.abc import Callable, Sequence
from dataclasses import dataclass

from intent_to_schedule.application.command import Executed, ExecuteResult
from intent_to_schedule.application.query import AnswerResult, Summary, summarize
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import Solution
from intent_to_schedule.application.translate import (
    ApplyRecord,
    ApplyStep,
    MessageStep,
    QueryRecord,
    QueryStep,
    ScheduleStep,
    Step,
    StepRecord,
    StepTranslator,
    Utterance,
)
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule

STEP_LIMIT: int = 12


@dataclass(frozen=True)
class Exhausted:
    """Outcome of a turn that reached the step limit."""


@dataclass(frozen=True)
class Response:
    """Updated problem and outcome of one conversation turn."""

    problem: SchedulingProblem
    outcome: MessageStep | Solution | Exhausted


# TODO: once the conversation outgrows a demo, move converse, translate, and the
# OpenAI adapter into their own feature package.
class Conversation:
    """Use case that handles one turn of a conversation."""

    def __init__(self, translator: StepTranslator, scheduling: Scheduling) -> None:
        self._translator: StepTranslator = translator
        self._scheduling: Scheduling = scheduling

    def respond(
        self,
        dialogue: Sequence[Utterance],
        problem: SchedulingProblem,
        previous: Schedule | None,
        save: Callable[[SchedulingProblem], None],
    ) -> Response:
        """Run translation steps until a reply, schedule, or step limit."""
        working: SchedulingProblem = problem
        steps: list[StepRecord] = []
        for _ in range(STEP_LIMIT):
            summary: Summary = summarize(working, previous)
            step: Step = self._translator.translate(dialogue, summary, tuple(steps))
            if isinstance(step, QueryStep):
                answer: AnswerResult = self._scheduling.answer(
                    step.query, working, previous
                )
                steps.append(QueryRecord(step, answer))
            elif isinstance(step, ApplyStep):
                result: ExecuteResult = self._scheduling.execute(working, step.commands)
                steps.append(ApplyRecord(step, result))
                if isinstance(result, Executed):
                    working = result.problem
                    save(working)
            elif isinstance(step, ScheduleStep):
                return Response(
                    working,
                    self._scheduling.schedule(working, previous, step.stability),
                )
            else:
                return Response(working, step)
        return Response(working, Exhausted())
