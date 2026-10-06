from collections.abc import Sequence

from intent_to_schedule.application.command import (
    ExecuteResult,
    Rejected,
    SchedulingCommand,
    execute_commands,
)
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.query import (
    AnswerResult,
    SchedulingQuery,
    Summary,
    summarize,
)
from intent_to_schedule.application.solve import (
    FeasibleSolution,
    OptimalSolution,
    SchedulingSolver,
    Solution,
)
from intent_to_schedule.application.store import StateStore
from intent_to_schedule.domain.consistency import Validator, Violations
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule


class Scheduling:
    """Use case that applies commands and schedules a SchedulingProblem."""

    def __init__(
        self,
        solver: SchedulingSolver,
        validator: Validator,
        store: StateStore,
        policy: ObjectivePolicy = DEFAULT_POLICY,
    ) -> None:
        self._solver: SchedulingSolver = solver
        self._validator: Validator = validator
        self._store: StateStore = store
        self._policy: ObjectivePolicy = policy

    def answer(
        self, query: SchedulingQuery
    ) -> AnswerResult:
        """Answer a query using the scheduling objective policy."""
        problem: SchedulingProblem = self._store.load_problem()
        previous: Schedule | None = self._store.load_previous()
        return query.answer(problem, previous, self._policy)

    def summarize(self) -> Summary:
        """Summarize the stored problem and previous schedule."""
        problem: SchedulingProblem = self._store.load_problem()
        previous: Schedule | None = self._store.load_previous()
        return summarize(problem, previous)

    def execute(self, commands: Sequence[SchedulingCommand]) -> ExecuteResult:
        """Apply commands in order and validate the result."""
        problem: SchedulingProblem = self._store.load_problem()
        result: ExecuteResult = execute_commands(problem, commands)
        if isinstance(result, Rejected):
            return result
        violations: Violations = self._validator.validate(result.problem)
        if not violations.is_empty:
            return Rejected(violations)
        self._store.save_problem(result.problem)
        return result

    def schedule(self, stability: bool) -> Solution:
        """Schedule a problem with the service objective policy."""
        problem: SchedulingProblem = self._store.load_problem()
        previous: Schedule | None = self._store.load_previous()
        result: Solution = self._solver.solve(
            problem, self._policy, previous, stability
        )
        if isinstance(result, (OptimalSolution, FeasibleSolution)):
            self._store.save_previous(result.schedule)
        return result
