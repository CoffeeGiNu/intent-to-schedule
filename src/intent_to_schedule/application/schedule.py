from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime

from intent_to_schedule.application.command import ExecuteResult, Executed, Rejected, SchedulingCommand
from intent_to_schedule.application.solve import SchedulingSolver, SolveResult
from intent_to_schedule.domain.consistency import Validator, Violations
from intent_to_schedule.domain.constraint import ConstraintId, SoftConstraint
from intent_to_schedule.domain.evaluation import Distance
from intent_to_schedule.domain.measure import PointMeasure
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule
from intent_to_schedule.domain.task import TaskId


class Scheduling:
    """Use case that applies commands and solves a SchedulingProblem."""

    def __init__(self, solver: SchedulingSolver, validators: Sequence[Validator]) -> None:
        self._solver: SchedulingSolver = solver
        self._validators: Sequence[Validator] = validators

    def execute(self, problem: SchedulingProblem, commands: Sequence[SchedulingCommand]) -> ExecuteResult:
        """Apply commands in order and validate the result."""
        updated: SchedulingProblem = problem
        for command in commands:
            result: ExecuteResult = command.execute(updated)
            if isinstance(result, Rejected):
                return result
            updated = result.problem

        violations: Violations = Violations(())
        for validator in self._validators:
            violations = violations.merge(validator.validate(updated))
        if not violations.is_empty:
            return Rejected(violations)
        return Executed(updated)

    def solve(self, problem: SchedulingProblem, previous: Schedule | None) -> SolveResult:
        """Solve a problem, keeping Tasks near their previous start."""
        solve_problem: SchedulingProblem = problem
        if previous is not None:
            solve_problem = replace(
                problem,
                constraints=problem.constraints + stability_constraints(problem, previous),
            )
        return self._solver.solve(solve_problem)


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
