from collections.abc import Sequence
from dataclasses import dataclass, replace

from intent_to_schedule.application.time_windows import (
    Expansion,
    TimeRelation,
    TimeWindow,
    complement,
    expand,
)
from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.consistency import Violation, Violations
from intent_to_schedule.domain.constraint import (
    Constraint,
    ConstraintId,
    HardConstraint,
    SoftConstraint,
)
from intent_to_schedule.domain.evaluation import Intrusion
from intent_to_schedule.domain.measure import IntervalMeasure, Measure
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Task, TaskId


@dataclass(frozen=True)
class Executed:
    """Result of a command that was executed."""

    problem: SchedulingProblem


@dataclass(frozen=True)
class Rejected:
    """Result of a command or query that was rejected."""

    violations: Violations


type ExecuteResult = Executed | Rejected
"""Result of executing a command."""


@dataclass(frozen=True)
class AddTask:
    """Command to add a movable or fixed Task."""

    task: Task | FixedTask

    def execute(self, problem: SchedulingProblem) -> ExecuteResult:
        if any(
            task.id == self.task.id for task in (*problem.tasks, *problem.fixed_tasks)
        ):
            return Rejected(
                Violations((Violation(f"Task {self.task.id.value} already exists"),))
            )
        if isinstance(self.task, FixedTask):
            return Executed(
                replace(problem, fixed_tasks=(*problem.fixed_tasks, self.task))
            )
        return Executed(replace(problem, tasks=(*problem.tasks, self.task)))


@dataclass(frozen=True)
class ReplaceTask:
    """Command to replace the Task with the same ID."""

    task: Task | FixedTask

    def execute(self, problem: SchedulingProblem) -> ExecuteResult:
        if not any(
            task.id == self.task.id for task in (*problem.tasks, *problem.fixed_tasks)
        ):
            return Rejected(
                Violations((Violation(f"Task {self.task.id.value} does not exist"),))
            )
        tasks: tuple[Task, ...]
        if isinstance(self.task, FixedTask):
            tasks = tuple(task for task in problem.tasks if task.id != self.task.id)
            fixed_tasks: tuple[FixedTask, ...] = (
                tuple(
                    self.task if task.id == self.task.id else task
                    for task in problem.fixed_tasks
                )
                if any(task.id == self.task.id for task in problem.fixed_tasks)
                else (*problem.fixed_tasks, self.task)
            )
            return Executed(replace(problem, tasks=tasks, fixed_tasks=fixed_tasks))
        if any(task.id == self.task.id for task in problem.fixed_tasks):
            return Executed(
                replace(
                    problem,
                    tasks=(*problem.tasks, self.task),
                    fixed_tasks=tuple(
                        task for task in problem.fixed_tasks if task.id != self.task.id
                    ),
                )
            )
        tasks = tuple(
            self.task if task.id == self.task.id else task for task in problem.tasks
        )
        return Executed(replace(problem, tasks=tasks))


@dataclass(frozen=True)
class RemoveTask:
    """Command to remove a Task."""

    task_id: TaskId

    def execute(self, problem: SchedulingProblem) -> ExecuteResult:
        if not any(
            task.id == self.task_id for task in (*problem.tasks, *problem.fixed_tasks)
        ):
            return Rejected(
                Violations((Violation(f"Task {self.task_id.value} does not exist"),))
            )
        tasks: tuple[Task, ...] = tuple(
            task for task in problem.tasks if task.id != self.task_id
        )
        fixed_tasks: tuple[FixedTask, ...] = tuple(
            task for task in problem.fixed_tasks if task.id != self.task_id
        )
        constraints: list[Constraint] = []
        constraint: Constraint
        for constraint in problem.constraints:
            measure: Measure | None = constraint.measure.without_task(self.task_id)
            if measure is not None:
                constraints.append(replace(constraint, measure=measure))
        return Executed(
            replace(
                problem,
                tasks=tasks,
                fixed_tasks=fixed_tasks,
                constraints=tuple(constraints),
            )
        )


@dataclass(frozen=True)
class AddConstraint:
    """Command to add a constraint."""

    constraint: Constraint

    def execute(self, problem: SchedulingProblem) -> ExecuteResult:
        if any(
            constraint.id == self.constraint.id for constraint in problem.constraints
        ):
            return Rejected(
                Violations(
                    (
                        Violation(
                            f"Constraint {self.constraint.id.value} already exists"
                        ),
                    )
                )
            )
        return Executed(
            replace(problem, constraints=(*problem.constraints, self.constraint))
        )


@dataclass(frozen=True)
class AddTimeConstraint:
    """Command to constrain Tasks within or away from time windows."""

    constraint_id: ConstraintId
    task_ids: frozenset[TaskId]
    relation: TimeRelation
    windows: tuple[TimeWindow, ...]
    strength: Strength | None
    """Strength of a soft constraint, or None for a hard constraint."""

    def execute(self, problem: SchedulingProblem) -> ExecuteResult:
        if not self.task_ids:
            return Rejected(
                Violations(
                    (Violation("Time constraint must reference at least one Task."),)
                )
            )
        grid: TimeGrid = problem.calendar.grid
        expansion: Expansion = expand(self.windows, self.relation, grid)
        if not expansion.intervals:
            covered: str = (
                "whole slot" if self.relation is TimeRelation.WITHIN else "time"
            )
            task_label: str = "task" if len(self.task_ids) == 1 else "tasks"
            task_names: str = ", ".join(
                sorted(task_id.value for task_id in self.task_ids)
            )
            message: str = (
                f"Time constraint on {task_label} {task_names}: the {self.relation.value} windows "
                f"cover no {covered} of the calendar horizon {grid.horizon.start.isoformat()} to "
                f"{grid.horizon.end.isoformat()} (slot {grid.slot}); widen or move the windows."
            )
            return Rejected(Violations((Violation(message),)))
        region: tuple[TimeInterval, ...] = (
            complement(expansion.intervals, grid.horizon)
            if self.relation is TimeRelation.WITHIN
            else expansion.intervals
        )
        measure: IntervalMeasure = IntervalMeasure(self.task_ids)
        evaluation: Intrusion = Intrusion(region)
        constraint: Constraint = (
            HardConstraint(self.constraint_id, measure, evaluation)
            if self.strength is None
            else SoftConstraint(self.constraint_id, measure, evaluation, self.strength)
        )
        return AddConstraint(constraint).execute(problem)


@dataclass(frozen=True)
class RemoveConstraint:
    """Command to remove a constraint."""

    constraint_id: ConstraintId

    def execute(self, problem: SchedulingProblem) -> ExecuteResult:
        if not any(
            constraint.id == self.constraint_id for constraint in problem.constraints
        ):
            return Rejected(
                Violations(
                    (
                        Violation(
                            f"Constraint {self.constraint_id.value} does not exist"
                        ),
                    )
                )
            )
        constraints: tuple[Constraint, ...] = tuple(
            constraint
            for constraint in problem.constraints
            if constraint.id != self.constraint_id
        )
        return Executed(replace(problem, constraints=constraints))


type ElementCommand = AddTask | ReplaceTask | RemoveTask
"""Request to change the elements of a SchedulingProblem."""


# TODO: consider a precedence command (Task B starts at least a gap after Task A ends) built on DependencyMeasure and Shortfall, a common scheduling constraint.
type ConstraintCommand = AddConstraint | AddTimeConstraint | RemoveConstraint
"""Request to change the constraints of a SchedulingProblem."""


type SchedulingCommand = ElementCommand | ConstraintCommand
"""Request to change a SchedulingProblem."""


def execute_commands(
    problem: SchedulingProblem, commands: Sequence[SchedulingCommand]
) -> ExecuteResult:
    """Execute commands in order, stopping at the first rejection."""
    updated: SchedulingProblem = problem
    command: SchedulingCommand
    for command in commands:
        result: ExecuteResult = command.execute(updated)
        if isinstance(result, Rejected):
            return result
        updated = result.problem
    return Executed(updated)
