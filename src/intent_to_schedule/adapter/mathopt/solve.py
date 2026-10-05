from datetime import datetime, timedelta

from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.compile import CompiledProblem, compile_problem
from intent_to_schedule.application.objective import summarize_schedule
from intent_to_schedule.application.policy import ObjectivePolicy
from intent_to_schedule.application.solve import (
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
        schedule: Schedule | None = self._schedule(problem, previous, False)
        if schedule is not None:
            return Solved(
                schedule, summarize_schedule(problem, schedule, self.policy, previous)
            )
        relaxed: Schedule | None = self._schedule(problem, previous, True)
        assert relaxed is not None
        return Infeasible(find_conflicts(problem, relaxed, self.policy))

    def _schedule(
        self, problem: SchedulingProblem, previous: Schedule | None, relaxed: bool
    ) -> Schedule | None:
        """Solve the compiled problem and read its schedule, or None if infeasible."""
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
            case mathopt.TerminationReason.INFEASIBLE:
                return None
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
                            )
                        )
                    else:
                        dropped.append(DroppedTask(task.id, task.name))
                return Schedule(tuple(scheduled), tuple(dropped))
            case _:
                raise RuntimeError(f"MathOpt solve failed: {result.termination.reason}")
