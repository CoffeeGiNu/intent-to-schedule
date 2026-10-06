from datetime import datetime, timedelta

from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.compile import CompiledProblem, compile_problem
from intent_to_schedule.application.objective import summarize_schedule
from intent_to_schedule.application.policy import ObjectivePolicy
from intent_to_schedule.application.solve import (
    ConflictsNotFound,
    Infeasible,
    SchedulingSolver,
    Solved,
    SolveResult,
    find_conflicts,
)
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import DroppedTask, Schedule, ScheduledTask
from intent_to_schedule.domain.task import Task


class MathOptSchedulingSolver(SchedulingSolver):
    """SchedulingSolver backed by MathOpt."""

    def __init__(
        self,
        policy: ObjectivePolicy,
        solver_type: mathopt.SolverType = mathopt.SolverType.GSCIP,
        time_limit: timedelta | None = None,
    ) -> None:
        self.policy: ObjectivePolicy = policy
        self.solver_type: mathopt.SolverType = solver_type
        self.time_limit: timedelta | None = time_limit

    def solve(
        self, problem: SchedulingProblem, previous: Schedule | None = None
    ) -> SolveResult:
        """Solve a SchedulingProblem, explaining infeasibility with a relaxed solve."""
        termination: mathopt.Termination
        schedule: Schedule | None
        termination, schedule = self._schedule(problem, previous, False)
        if schedule is not None:
            return Solved(
                schedule, summarize_schedule(problem, schedule, self.policy, previous)
            )
        if termination.reason is not mathopt.TerminationReason.INFEASIBLE:
            raise RuntimeError(f"MathOpt solve failed: {termination.reason}")
        relaxed: Schedule | None
        termination, relaxed = self._schedule(problem, previous, True)
        if relaxed is None:
            return Infeasible(ConflictsNotFound(_termination_reason(termination)))
        return Infeasible(find_conflicts(problem, relaxed, self.policy))

    def _schedule(
        self, problem: SchedulingProblem, previous: Schedule | None, relaxed: bool
    ) -> tuple[mathopt.Termination, Schedule | None]:
        """Solve the compiled problem and read its termination and any schedule found."""
        compiled: CompiledProblem = compile_problem(
            problem, self.policy, previous, relaxed
        )
        parameters: mathopt.SolveParameters = mathopt.SolveParameters(
            time_limit=self.time_limit
        )
        result: mathopt.SolveResult = mathopt.solve(
            compiled.model, self.solver_type, params=parameters
        )
        match result.termination.reason:
            case mathopt.TerminationReason.OPTIMAL | mathopt.TerminationReason.FEASIBLE:
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
            case _:
                return result.termination, None


def _termination_reason(termination: mathopt.Termination) -> str:
    """Name why a MathOpt solve ended, reporting a time limit stop as time_limit."""
    if (
        termination.reason is mathopt.TerminationReason.NO_SOLUTION_FOUND
        and termination.limit is mathopt.Limit.TIME
    ):
        return "time_limit"
    return termination.reason.name.lower()
