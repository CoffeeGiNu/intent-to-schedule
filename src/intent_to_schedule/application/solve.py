from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from intent_to_schedule.application.objective import (
    HardConstraintEvaluation,
    ScheduleSummary,
    evaluate_constraints,
)
from intent_to_schedule.application.policy import ObjectivePolicy
from intent_to_schedule.domain.availability import available_start_slots, free_slots
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule
from intent_to_schedule.domain.task import Task, TaskId


@dataclass(frozen=True)
class OptimalSolution:
    """Optimal schedule and objective summary."""

    schedule: Schedule
    summary: ScheduleSummary


class DropReason(Enum):
    """Why a relaxed schedule dropped a required task."""

    NO_FREE_START = "no_free_start"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class ConstraintConflict:
    """Hard constraint broken by the relaxed schedule."""

    evaluation: HardConstraintEvaluation
    related_constraint_ids: tuple[ConstraintId, ...]
    """Other hard constraints referencing any of the same tasks."""


@dataclass(frozen=True)
class DroppedRequiredTask:
    """Required task dropped by the relaxed schedule."""

    task_id: TaskId
    name: str
    reason: DropReason


@dataclass(frozen=True)
class Conflicts:
    """Hard constraints and required tasks given up by the relaxed schedule."""

    constraints: tuple[ConstraintConflict, ...]
    dropped_required_tasks: tuple[DroppedRequiredTask, ...]


@dataclass(frozen=True)
class ConflictsNotFound:
    """Relaxed solve that ended without a schedule to explain infeasibility."""

    reason: str
    """Why the relaxed solve ended, such as time_limit."""


@dataclass(frozen=True)
class FeasibleSolution:
    """Feasible schedule and objective summary."""

    schedule: Schedule
    summary: ScheduleSummary


@dataclass(frozen=True)
class NoFeasibleSolution:
    """Proven absence of a feasible schedule."""

    conflicts: Conflicts | ConflictsNotFound
    """What a relaxed solve gave up, or why it found no schedule."""


@dataclass(frozen=True)
class SolutionNotFound:
    """Solve stopped without a solution or infeasibility proof."""

    reason: str


type Solution = (
    OptimalSolution | FeasibleSolution | NoFeasibleSolution | SolutionNotFound
)
"""Result of solving a SchedulingProblem."""


class SchedulingSolver(Protocol):
    """Port that solves a SchedulingProblem."""

    def solve(
        self,
        problem: SchedulingProblem,
        policy: ObjectivePolicy,
        previous: Schedule | None,
        stability: bool,
    ) -> Solution: ...


def find_conflicts(
    problem: SchedulingProblem, relaxed: Schedule, policy: ObjectivePolicy
) -> Conflicts:
    """Collect the hard constraints and required tasks a relaxed schedule gives up."""
    hard_constraints: tuple[HardConstraint, ...] = tuple(
        constraint
        for constraint in problem.constraints
        if isinstance(constraint, HardConstraint)
    )
    broken: list[HardConstraintEvaluation] = sorted(
        (
            item
            for item in evaluate_constraints(problem, relaxed, policy)
            if isinstance(item, HardConstraintEvaluation)
            and item.violation.amount > 0
        ),
        key=lambda item: (-item.violation.amount, item.constraint.id.value),
    )
    tasks: dict[TaskId, Task] = {task.id: task for task in problem.tasks}
    dropped: list[DroppedRequiredTask] = []
    task: Task
    for task in sorted(
        (tasks[item.task_id] for item in relaxed.dropped),
        key=lambda task: task.id.value,
    ):
        if not task.required:
            continue
        starts: tuple[int, ...] = available_start_slots(
            problem.calendar.grid,
            tuple(free_slots(problem, person_id) for person_id in task.participant_ids),
            task.duration,
        )
        dropped.append(
            DroppedRequiredTask(
                task.id,
                task.name,
                DropReason.CONFLICT if starts else DropReason.NO_FREE_START,
            )
        )
    return Conflicts(
        tuple(
            ConstraintConflict(
                item,
                tuple(
                    sorted(
                        (
                            other.id
                            for other in hard_constraints
                            if other.id != item.constraint.id
                            and other.condition.task_ids
                            & item.constraint.condition.task_ids
                        ),
                        key=lambda constraint_id: constraint_id.value,
                    )
                ),
            )
            for item in broken
        ),
        tuple(dropped),
    )
