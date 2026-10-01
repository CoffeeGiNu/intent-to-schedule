from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from intent_to_schedule.application.command import Rejected
from intent_to_schedule.application.solve import Infeasible, SchedulingSolver, Solved
from intent_to_schedule.application.translate import Ambiguous, CommandTranslator, Utterance
from intent_to_schedule.domain.consistency import ConsistencyError, Validator, Violations
from intent_to_schedule.domain.constraint import ConstraintId, SoftConstraint
from intent_to_schedule.domain.evaluation import Distance
from intent_to_schedule.domain.measure import PointMeasure
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule
from intent_to_schedule.domain.task import TaskId


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
    ) -> Response:
        translated = self._translator.translate(dialogue, problem, previous)
        if isinstance(translated, Ambiguous):
            return Response(problem, translated)

        updated: SchedulingProblem = problem
        for command in translated.commands:
            result = command.execute(updated)
            if isinstance(result, Rejected):
                raise ConsistencyError(result.violations)
            updated = result.problem

        violations: Violations = Violations(())
        for validator in self._validators:
            violations = violations.merge(validator.validate(updated))
        if not violations.is_empty:
            raise ConsistencyError(violations)

        solve_problem: SchedulingProblem = updated
        if previous is not None:
            solve_problem = replace(
                updated,
                constraints=updated.constraints + stability_constraints(updated, previous),
            )
        return Response(updated, self._solver.solve(solve_problem))


def stability_constraints(problem: SchedulingProblem, previous: Schedule) -> tuple[SoftConstraint, ...]:
    """Build constraints that keep Tasks near their previous start."""
    starts: dict[TaskId, datetime] = {scheduled.task_id: scheduled.start for scheduled in previous.scheduled}
    return tuple(
        SoftConstraint(
            ConstraintId.generate(),
            PointMeasure(task.id),
            Distance(starts[task.id]),
            task.stability,
        )
        for task in problem.tasks
        if task.id in starts
    )
