from dataclasses import dataclass

from intent_to_schedule.domain.consistency import Violations
from intent_to_schedule.domain.constraint import Constraint, ConstraintId
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

    def execute(self, problem: SchedulingProblem) -> ExecuteResult: ...


@dataclass(frozen=True)
class ReplaceTask:
    """Command to replace the Task with the same ID."""

    task: Task

    def execute(self, problem: SchedulingProblem) -> ExecuteResult: ...


@dataclass(frozen=True)
class RemoveTask:
    """Command to remove a Task."""

    task_id: TaskId

    def execute(self, problem: SchedulingProblem) -> ExecuteResult: ...


@dataclass(frozen=True)
class AddConstraint:
    """Command to add a constraint."""

    constraint: Constraint

    def execute(self, problem: SchedulingProblem) -> ExecuteResult: ...


@dataclass(frozen=True)
class RemoveConstraint:
    """Command to remove a constraint."""

    constraint_id: ConstraintId

    def execute(self, problem: SchedulingProblem) -> ExecuteResult: ...


type ElementCommand = AddTask | ReplaceTask | RemoveTask
"""Request to change the elements of a SchedulingProblem."""


type ConstraintCommand = AddConstraint | RemoveConstraint
"""Request to change the constraints of a SchedulingProblem."""


type SchedulingCommand = ElementCommand | ConstraintCommand
"""Request to change a SchedulingProblem."""
