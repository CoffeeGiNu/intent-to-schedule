from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.compile import CompiledProblem, compile_problem
from intent_to_schedule.application.policy import ObjectivePolicy
from intent_to_schedule.application.solve import Infeasible, SchedulingSolver, SolveResult, Solved
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule, ScheduledTask
from intent_to_schedule.domain.task import Task, TaskId


class MathOptSchedulingSolver(SchedulingSolver):
    """SchedulingSolver backed by MathOpt."""

    def __init__(
        self,
        policy: ObjectivePolicy,
        solver_type: mathopt.SolverType = mathopt.SolverType.GSCIP,
    ) -> None:
        self.policy: ObjectivePolicy = policy
        self.solver_type: mathopt.SolverType = solver_type

    def solve(self, problem: SchedulingProblem) -> SolveResult:
        """Solve a SchedulingProblem."""
        compiled: CompiledProblem = compile_problem(problem, self.policy)
        result: mathopt.SolveResult = mathopt.solve(compiled.model, self.solver_type)
        match result.termination.reason:
            case mathopt.TerminationReason.INFEASIBLE:
                return Infeasible()
            case mathopt.TerminationReason.OPTIMAL | mathopt.TerminationReason.FEASIBLE:
                values: dict[mathopt.Variable, float] = result.variable_values()
                scheduled: list[ScheduledTask] = []
                dropped: set[TaskId] = set()
                task: Task
                for task in problem.tasks:
                    if values[compiled.presences[task.id]] > 0.5:
                        start: int = next(
                            slot
                            for slot, variable in compiled.placements[task.id].items()
                            if values[variable] > 0.5
                        )
                        scheduled.append(
                            ScheduledTask(task.id, problem.calendar.grid.horizon.start + start * problem.calendar.grid.slot)
                        )
                    else:
                        dropped.add(task.id)
                return Solved(Schedule(tuple(scheduled), frozenset(dropped)))
            case _:
                raise RuntimeError(f"MathOpt solve failed: {result.termination.reason}")
