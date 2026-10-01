from collections.abc import Sequence
from dataclasses import dataclass

from intent_to_schedule.application.solve import Infeasible, SchedulingSolver, Solved
from intent_to_schedule.application.translate import Ambiguous, CommandTranslator, Utterance
from intent_to_schedule.domain.consistency import Validator
from intent_to_schedule.domain.constraint import SoftConstraint
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule


@dataclass(frozen=True)
class Response:
    """Updated problem and outcome of one conversation turn."""

    problem: SchedulingProblem
    outcome: Ambiguous | Solved | Infeasible


class Conversation:
    """Use case that handles one turn of a conversation."""

    def __init__(
        self,
        translator: CommandTranslator,
        solver: SchedulingSolver,
        validators: Sequence[Validator],
    ) -> None:
        self._translator = translator
        self._solver = solver
        self._validators = validators

    def respond(
        self,
        dialogue: Sequence[Utterance],
        problem: SchedulingProblem,
        previous: Schedule | None,
    ) -> Response: ...


def stability_constraints(problem: SchedulingProblem, previous: Schedule) -> tuple[SoftConstraint, ...]:
    """Build constraints that keep Tasks near their previous start."""
    ...
