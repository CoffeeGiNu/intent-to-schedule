from collections.abc import Sequence
from dataclasses import dataclass

from intent_to_schedule.application.command import ExecuteResult, Rejected
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import SolveResult
from intent_to_schedule.application.translate import Ambiguous, CommandTranslator, TranslateResult, Utterance
from intent_to_schedule.domain.consistency import ConsistencyError
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule


@dataclass(frozen=True)
class Response:
    """Updated problem and outcome of one conversation turn."""

    problem: SchedulingProblem
    outcome: Ambiguous | SolveResult


class Conversation:
    """Use case that handles one turn of a conversation."""

    def __init__(self, translator: CommandTranslator, scheduling: Scheduling) -> None:
        self._translator: CommandTranslator = translator
        self._scheduling: Scheduling = scheduling

    def respond(
        self,
        dialogue: Sequence[Utterance],
        problem: SchedulingProblem,
        previous: Schedule | None,
    ) -> Response:
        translated: TranslateResult = self._translator.translate(dialogue, problem, previous)
        if isinstance(translated, Ambiguous):
            return Response(problem, translated)

        result: ExecuteResult = self._scheduling.execute(problem, translated.commands)
        if isinstance(result, Rejected):
            raise ConsistencyError(result.violations)
        updated: SchedulingProblem = result.problem
        return Response(updated, self._scheduling.solve(updated, previous))
