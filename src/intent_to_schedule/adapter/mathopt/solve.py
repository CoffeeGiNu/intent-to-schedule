from datetime import datetime, timedelta

from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.compile import CompiledProblem, compile_problem
from intent_to_schedule.application.objective import (
    ScheduleSummary,
    summarize_schedule,
)
from intent_to_schedule.application.policy import ObjectivePolicy
from intent_to_schedule.application.solve import (
    ConflictsNotFound,
    FeasibleSolution,
    NoFeasibleSolution,
    OptimalSolution,
    SchedulingSolver,
    Solution,
    SolutionNotFound,
    find_conflicts,
)
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import DroppedTask, Schedule, ScheduledTask
from intent_to_schedule.domain.task import Task


class MathOptSchedulingSolver(SchedulingSolver):
    """SchedulingSolver backed by MathOpt."""

    def __init__(
        self,
        solver_type: mathopt.SolverType = mathopt.SolverType.GSCIP,
        time_limit: timedelta | None = None,
    ) -> None:
        self.solver_type: mathopt.SolverType = solver_type
        self.time_limit: timedelta | None = time_limit

    def solve(
        self,
        problem: SchedulingProblem,
        policy: ObjectivePolicy,
        previous: Schedule | None,
        stability: bool,
    ) -> Solution:
        """Solve a problem and explain proven infeasibility with a relaxed solve."""
        termination: mathopt.Termination
        schedule: Schedule | None
        termination, schedule = self._solve(problem, policy, previous, stability, False)
        if schedule is not None:
            summary: ScheduleSummary = summarize_schedule(
                problem, schedule, policy, previous, stability
            )
            if termination.reason is mathopt.TerminationReason.OPTIMAL:
                return OptimalSolution(schedule, summary)
            return FeasibleSolution(schedule, summary)
        if termination.reason is mathopt.TerminationReason.NO_SOLUTION_FOUND:
            return SolutionNotFound(_termination_reason(termination))
        if termination.reason is not mathopt.TerminationReason.INFEASIBLE:
            raise RuntimeError(f"MathOpt solve failed: {termination.reason}")
        relaxed: Schedule | None
        termination, relaxed = self._solve(problem, policy, previous, stability, True)
        if relaxed is None:
            return NoFeasibleSolution(
                ConflictsNotFound(_termination_reason(termination))
            )
        return NoFeasibleSolution(find_conflicts(problem, relaxed, policy))

    def _solve(
        self,
        problem: SchedulingProblem,
        policy: ObjectivePolicy,
        previous: Schedule | None,
        stability: bool,
        relaxed: bool,
    ) -> tuple[mathopt.Termination, Schedule | None]:
        """Solve the compiled problem and read its termination and any usable schedule."""
        compiled: CompiledProblem = compile_problem(
            problem, policy, previous, stability, relaxed
        )
        parameters: mathopt.SolveParameters = mathopt.SolveParameters(
            time_limit=self.time_limit
        )
        result: mathopt.SolveResult = mathopt.solve(
            compiled.model, self.solver_type, params=parameters
        )
        found: bool = (
            result.has_primal_feasible_solution()
            if relaxed
            else result.termination.reason
            in (mathopt.TerminationReason.OPTIMAL, mathopt.TerminationReason.FEASIBLE)
        )
        if not found:
            return result.termination, None
        values: dict[mathopt.Variable, float] = result.variable_values()
        scheduled: list[ScheduledTask] = []
        dropped: list[DroppedTask] = []
        task: Task
        for task in problem.tasks:
            if values[compiled.presences[task.id]] > 0.5:
                start: int = next(
                    slot
                    for slot, variable in compiled.placements[task.id].items()
                    if values[variable] > 0.5
                )
                start_time: datetime = problem.calendar.grid.time_at(start)
                scheduled.append(
                    ScheduledTask(
                        task.id,
                        task.name,
                        start_time,
                        start_time + task.duration,
                        task.participant_ids,
                    )
                )
            else:
                dropped.append(DroppedTask(task.id, task.name))
        return result.termination, Schedule(tuple(scheduled), tuple(dropped))


def _termination_reason(termination: mathopt.Termination) -> str:
    """Name why a MathOpt solve ended, reporting a time limit stop as time_limit."""
    if (
        termination.reason is mathopt.TerminationReason.NO_SOLUTION_FOUND
        and termination.limit is mathopt.Limit.TIME
    ):
        return "time_limit"
    return termination.reason.name.lower()
