from collections.abc import Sequence

from intent_to_schedule.application.command import (
    ExecuteResult,
    Rejected,
    SchedulingCommand,
    execute_commands,
)
from intent_to_schedule.application.objective import summarize_schedule
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.solve import SchedulingSolver, Solved, SolveResult
from intent_to_schedule.domain.consistency import Validator, Violations
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule


class Scheduling:
    """Use case that applies commands and solves a SchedulingProblem."""

    def __init__(
        self,
        solver: SchedulingSolver,
        validator: Validator,
        policy: ObjectivePolicy = DEFAULT_POLICY,
    ) -> None:
        self._solver: SchedulingSolver = solver
        self._validator: Validator = validator
        self._policy: ObjectivePolicy = policy

    def execute(
        self, problem: SchedulingProblem, commands: Sequence[SchedulingCommand]
    ) -> ExecuteResult:
        """Apply commands in order and validate the result."""
        result: ExecuteResult = execute_commands(problem, commands)
        if isinstance(result, Rejected):
            return result
        violations: Violations = self._validator.validate(result.problem)
        if not violations.is_empty:
            return Rejected(violations)
        return result

    def solve(
        self, problem: SchedulingProblem, previous: Schedule | None
    ) -> SolveResult:
        """Solve a problem, keeping Tasks near their previous start."""
        result: SolveResult = self._solver.solve(problem, previous)
        if isinstance(result, Solved) and result.summary is None:
            return Solved(
                result.schedule,
                summarize_schedule(problem, result.schedule, self._policy, previous),
            )
        return result
