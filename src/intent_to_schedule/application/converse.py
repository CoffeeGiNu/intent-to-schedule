from collections.abc import Sequence
from dataclasses import dataclass

from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import SolveResult
from intent_to_schedule.application.translate import (
    StepTranslator,
    MessageStep,
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
    outcome: MessageStep | SolveResult | Exhausted


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
    ) -> Response:
        raise NotImplementedError
