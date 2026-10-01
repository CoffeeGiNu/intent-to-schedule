from dataclasses import dataclass, replace

from intent_to_schedule.domain.consistency import Violation, Violations
from intent_to_schedule.domain.constraint import Constraint, ConstraintId
from intent_to_schedule.domain.measure import Measure
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import Task, TaskId


@dataclass(frozen=True)
class Executed:
    """Result of a command that was executed."""

    problem: SchedulingProblem


@dataclass(frozen=True)
class Rejected:
    """Result of a command that could not be executed."""

    violations: Violations


type ExecuteResult = Executed | Rejected
"""Result of executing a command."""


@dataclass(frozen=True)
class AddTask:
    """Command to add a Task."""

    task: Task

    def execute(self, problem: SchedulingProblem) -> ExecuteResult:
        if any(task.id == self.task.id for task in problem.tasks):
            return Rejected(Violations((Violation("Task already exists"),)))
        return Executed(replace(problem, tasks=(*problem.tasks, self.task)))


@dataclass(frozen=True)
class ReplaceTask:
    """Command to replace the Task with the same ID."""

    task: Task

    def execute(self, problem: SchedulingProblem) -> ExecuteResult:
        if not any(task.id == self.task.id for task in problem.tasks):
            return Rejected(Violations((Violation("Task does not exist"),)))
        tasks: tuple[Task, ...] = tuple(self.task if task.id == self.task.id else task for task in problem.tasks)
        return Executed(replace(problem, tasks=tasks))


@dataclass(frozen=True)
class RemoveTask:
    """Command to remove a Task."""

    task_id: TaskId

    def execute(self, problem: SchedulingProblem) -> ExecuteResult:
        if not any(task.id == self.task_id for task in problem.tasks):
            return Rejected(Violations((Violation("Task does not exist"),)))
        tasks: tuple[Task, ...] = tuple(task for task in problem.tasks if task.id != self.task_id)
        constraints: list[Constraint] = []
        for constraint in problem.constraints:
            measure: Measure | None = constraint.measure.without_task(self.task_id)
            if measure is not None:
                constraints.append(replace(constraint, measure=measure))
        return Executed(replace(problem, tasks=tasks, constraints=tuple(constraints)))


@dataclass(frozen=True)
class AddConstraint:
    """Command to add a constraint."""

    constraint: Constraint

    def execute(self, problem: SchedulingProblem) -> ExecuteResult:
        if any(constraint.id == self.constraint.id for constraint in problem.constraints):
            return Rejected(Violations((Violation("Constraint already exists"),)))
        return Executed(replace(problem, constraints=(*problem.constraints, self.constraint)))


@dataclass(frozen=True)
class RemoveConstraint:
    """Command to remove a constraint."""

    constraint_id: ConstraintId

    def execute(self, problem: SchedulingProblem) -> ExecuteResult:
        if not any(constraint.id == self.constraint_id for constraint in problem.constraints):
            return Rejected(Violations((Violation("Constraint does not exist"),)))
        constraints: tuple[Constraint, ...] = tuple(
            constraint for constraint in problem.constraints if constraint.id != self.constraint_id
        )
        return Executed(replace(problem, constraints=constraints))


type ElementCommand = AddTask | ReplaceTask | RemoveTask
"""Request to change the elements of a SchedulingProblem."""


type ConstraintCommand = AddConstraint | RemoveConstraint
"""Request to change the constraints of a SchedulingProblem."""


type SchedulingCommand = ElementCommand | ConstraintCommand
"""Request to change a SchedulingProblem."""
