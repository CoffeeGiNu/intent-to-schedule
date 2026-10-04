from collections.abc import Sequence
from dataclasses import dataclass, replace

from intent_to_schedule.domain.condition import Condition
from intent_to_schedule.domain.consistency import Violation, Violations
from intent_to_schedule.domain.constraint import Constraint, ConstraintId
from intent_to_schedule.domain.problem import SchedulingProblem
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
            condition: Condition | None = constraint.condition.without_task(
                self.task_id
            )
            if condition is not None:
                constraints.append(replace(constraint, condition=condition))
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
class ReplaceConstraint:
    """Command to replace the constraint with the same ID."""

    constraint: Constraint

    def execute(self, problem: SchedulingProblem) -> ExecuteResult:
        if not any(
            constraint.id == self.constraint.id for constraint in problem.constraints
        ):
            return Rejected(
                Violations(
                    (
                        Violation(
                            f"Constraint {self.constraint.id.value} does not exist"
                        ),
                    )
                )
            )
        constraints: tuple[Constraint, ...] = tuple(
            self.constraint if constraint.id == self.constraint.id else constraint
            for constraint in problem.constraints
        )
        return Executed(replace(problem, constraints=constraints))


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


type ConstraintCommand = AddConstraint | ReplaceConstraint | RemoveConstraint
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
