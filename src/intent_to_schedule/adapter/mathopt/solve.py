from ortools.math_opt.python import mathopt

from intent_to_schedule.application.policy import ObjectivePolicy
from intent_to_schedule.application.solve import SchedulingSolver, SolveResult
from intent_to_schedule.domain.problem import SchedulingProblem


class MathOptSchedulingSolver(SchedulingSolver):
    """SchedulingSolver backed by MathOpt."""

    def __init__(
        self,
        policy: ObjectivePolicy,
        solver_type: mathopt.SolverType = mathopt.SolverType.GSCIP,
    ) -> None: ...

    def solve(self, problem: SchedulingProblem) -> SolveResult:
        """Solve a SchedulingProblem."""
        ...
