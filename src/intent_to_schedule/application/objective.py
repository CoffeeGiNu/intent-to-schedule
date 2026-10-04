"""Schedule evaluation under an objective policy."""

from dataclasses import dataclass
from datetime import datetime, timedelta

from intent_to_schedule.application.policy import ObjectivePolicy
from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import DailyLimitCondition
from intent_to_schedule.domain.constraint import Constraint, SoftConstraint
from intent_to_schedule.domain.measure import AggregateQuantity
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule
from intent_to_schedule.domain.task import Task, TaskId
from intent_to_schedule.domain.violation import CriterionViolation, measure_criterion


@dataclass(frozen=True)
class ConstraintEvaluation:
    """A constraint's measured violation and policy cost."""

    constraint: Constraint
    violation: CriterionViolation
    cost: float | None
    coefficient: float | None


@dataclass(frozen=True)
class ScheduleSummary:
    """Objective costs and counts of a solved schedule."""

    dropped_tasks_cost: float
    soft_constraints_cost: float
    stability_cost: float
    scheduled_tasks: int
    dropped_tasks: int
    violated_soft_constraints: int
    moved_tasks: int

    @property
    def total_cost(self) -> float:
        """Total objective cost of the schedule."""
        return self.dropped_tasks_cost + self.soft_constraints_cost + self.stability_cost


def evaluate_constraints(
    problem: SchedulingProblem, schedule: Schedule, policy: ObjectivePolicy
) -> tuple[ConstraintEvaluation, ...]:
    """Evaluate current constraints against saved task intervals."""
    grid: TimeGrid = problem.calendar.grid
    current_task_ids: set[TaskId] = {task.id for task in problem.tasks}
    placements: dict[TaskId, TimeInterval] = {
        item.task_id: TimeInterval(item.start, item.end)
        for item in schedule.scheduled
        if item.task_id in current_task_ids
    }
    placements.update(
        {
            task.id: grid.round_outward(TimeInterval(task.start, task.start + task.duration))
            for task in problem.fixed_tasks
        }
    )
    evaluations: list[ConstraintEvaluation] = []
    constraint: Constraint
    for constraint in problem.constraints:
        violations: tuple[CriterionViolation, ...] = tuple(
            measure_criterion(criterion, placements, grid)
            for criterion in constraint.condition.criteria(grid)
        )
        violation: CriterionViolation = CriterionViolation(
            sum(item.amount for item in violations),
            violations[0].unit,
            tuple(part for item in violations for part in item.breakdown),
        )
        coefficient: float | None = None
        if isinstance(constraint, SoftConstraint):
            scale: float = (
                policy.per_count
                if isinstance(constraint.condition, DailyLimitCondition)
                and constraint.condition.quantity is AggregateQuantity.COUNT
                else 1.0
            )
            coefficient = policy.weight(constraint.strength) * scale
        evaluations.append(
            ConstraintEvaluation(constraint, violation, coefficient * violation.amount if coefficient is not None else None, coefficient)
        )
    return tuple(evaluations)


def summarize_schedule(
    problem: SchedulingProblem,
    schedule: Schedule,
    policy: ObjectivePolicy,
    previous: Schedule | None = None,
) -> ScheduleSummary:
    """Compute objective costs and counts from a solved schedule."""
    evaluations: tuple[ConstraintEvaluation, ...] = evaluate_constraints(problem, schedule, policy)
    tasks: dict[TaskId, Task] = {task.id: task for task in problem.tasks}
    previous_starts: dict[TaskId, datetime] = (
        {item.task_id: item.start for item in previous.scheduled}
        if previous is not None
        else {}
    )
    stability_cost: float = 0.0
    moved_tasks: int = 0
    for item in schedule.scheduled:
        if item.task_id not in previous_starts:
            continue
        task: Task = tasks[item.task_id]
        moved_hours: float = abs((item.start - previous_starts[item.task_id]) / timedelta(hours=1))
        moved_tasks += int(moved_hours > 0)
        stability_cost += min(policy.weight(task.stability) * moved_hours, policy.stability_drop_cost_ratio * policy.drop_cost(task.importance))
    return ScheduleSummary(
        sum(policy.drop_cost(tasks[item.task_id].importance) for item in schedule.dropped),
        sum(item.cost for item in evaluations if item.cost is not None),
        stability_cost,
        len(schedule.scheduled),
        len(schedule.dropped),
        sum(item.cost is not None and item.violation.amount > 0 for item in evaluations),
        moved_tasks,
    )
