from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from intent_to_schedule.domain.calendar import Availability, TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import Condition, TimeWindowCondition
from intent_to_schedule.domain.constraint import Constraint, ConstraintId
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import FixedTask, Task, TaskId
from intent_to_schedule.domain.time_windows import Expansion, TimeRelation, expand


@dataclass(frozen=True)
class Violation:
    """Reason a problem is inconsistent or a command was rejected."""

    message: str


@dataclass(frozen=True)
class Violations:
    """Collection of Violation."""

    items: tuple[Violation, ...]

    @property
    def is_empty(self) -> bool:
        return not self.items

    def merge(self, other: "Violations") -> "Violations":
        return Violations(self.items + other.items)


class ConsistencyError(Exception):
    """Error carrying the Violations of an inconsistent problem or a rejected command."""

    def __init__(self, violations: Violations) -> None:
        super().__init__("; ".join(item.message for item in violations.items))
        self.violations: Violations = violations


class Validator(Protocol):
    """Consistency check on a SchedulingProblem."""

    def validate(self, problem: SchedulingProblem) -> Violations: ...


class AllOf:
    """Validator that all of the given validators hold."""

    def __init__(self, *validators: Validator) -> None:
        self._validators: tuple[Validator, ...] = validators

    def validate(self, problem: SchedulingProblem) -> Violations:
        violations: Violations = Violations(())
        validator: Validator
        for validator in self._validators:
            violations = violations.merge(validator.validate(problem))
        return violations


class UniqueIds:
    """Validator that IDs are not duplicated."""

    def validate(self, problem: SchedulingProblem) -> Violations:
        violations: list[Violation] = []
        label: str
        values: list[PersonId] | list[TaskId] | list[ConstraintId]
        value: PersonId | TaskId | ConstraintId
        for label, values in (
            ("Person", [person.id for person in problem.people]),
            ("Task", [task.id for task in (*problem.tasks, *problem.fixed_tasks)]),
            ("Constraint", [constraint.id for constraint in problem.constraints]),
        ):
            seen: set[PersonId | TaskId | ConstraintId] = set()
            reported: set[PersonId | TaskId | ConstraintId] = set()
            for value in values:
                if value in seen and value not in reported:
                    violations.append(Violation(f"Duplicate {label} id {value.value}."))
                    reported.add(value)
                seen.add(value)
        return Violations(tuple(violations))


class ReferencesExist:
    """Validator that referenced people and Tasks exist."""

    def validate(self, problem: SchedulingProblem) -> Violations:
        people: set[PersonId] = {person.id for person in problem.people}
        tasks: set[TaskId] = {
            task.id for task in (*problem.tasks, *problem.fixed_tasks)
        }
        violations: list[Violation] = []
        task: Task | FixedTask
        person_id: PersonId
        constraint: Constraint
        task_id: TaskId
        availability: Availability
        for task in (*problem.tasks, *problem.fixed_tasks):
            for person_id in task.participant_ids - people:
                violations.append(
                    Violation(
                        f"Task {task.id.value} references missing person id {person_id.value}."
                    )
                )
        for constraint in problem.constraints:
            for task_id in constraint.condition.task_ids - tasks:
                violations.append(
                    Violation(
                        f"Constraint {constraint.id.value} references missing task id {task_id.value}."
                    )
                )
        for availability in problem.calendar.availabilities:
            if availability.person_id not in people:
                violations.append(
                    Violation(
                        f"Availability references missing person id {availability.person_id.value}."
                    )
                )
        return Violations(tuple(violations))


class AvailabilityForEveryone:
    """Validator that every person has an Availability."""

    def validate(self, problem: SchedulingProblem) -> Violations:
        given: set[PersonId] = {
            availability.person_id for availability in problem.calendar.availabilities
        }
        return Violations(
            tuple(
                Violation(f"Person {person.id.value} has no availability.")
                for person in problem.people
                if person.id not in given
            )
        )


class AlignedToSlots:
    """Validator that times and durations fall on TimeGrid slots."""

    def validate(self, problem: SchedulingProblem) -> Violations:
        grid: TimeGrid = problem.calendar.grid
        violations: list[Violation] = []

        def check_time(value: datetime, label: str) -> None:
            if not grid.is_aligned(value):
                violations.append(
                    Violation(f"{label} is not aligned to the time grid.")
                )

        def check_duration(value: timedelta, label: str) -> None:
            if not grid.is_whole_slots(value):
                violations.append(
                    Violation(f"{label} is not aligned to the time grid.")
                )

        check_time(grid.horizon.end, "Calendar horizon end")
        task: Task
        availability: Availability
        interval: TimeInterval
        for task in problem.tasks:
            if task.duration <= timedelta(0):
                violations.append(
                    Violation(f"Task {task.id.value} duration must be positive.")
                )
            check_duration(task.duration, f"Task {task.id.value} duration")
        fixed_task: FixedTask
        for fixed_task in problem.fixed_tasks:
            if fixed_task.duration < timedelta(0):
                violations.append(
                    Violation(
                        f"Fixed task {fixed_task.id.value} duration must not be negative."
                    )
                )
        for availability in problem.calendar.availabilities:
            for interval in availability.intervals:
                check_time(
                    interval.start,
                    f"Availability for person {availability.person_id.value} start",
                )
                check_time(
                    interval.end,
                    f"Availability for person {availability.person_id.value} end",
                )
        return Violations(tuple(violations))


class NonemptyTimeWindows:
    """Validator that time windows cover the calendar horizon."""

    def validate(self, problem: SchedulingProblem) -> Violations:
        grid: TimeGrid = problem.calendar.grid
        violations: list[Violation] = []
        constraint: Constraint
        for constraint in problem.constraints:
            condition: Condition = constraint.condition
            if not isinstance(condition, TimeWindowCondition):
                continue
            expansion: Expansion = expand(condition.windows, condition.relation, grid)
            if expansion.intervals:
                continue
            covered: str = (
                "whole slot" if condition.relation is TimeRelation.WITHIN else "time"
            )
            task_label: str = "task" if len(condition.task_ids) == 1 else "tasks"
            task_names: str = ", ".join(
                sorted(task_id.value for task_id in condition.task_ids)
            )
            message: str = (
                f"Time constraint on {task_label} {task_names}: the {condition.relation.value} windows "
                f"cover no {covered} of the calendar horizon {grid.horizon.start.isoformat()} to "
                f"{grid.horizon.end.isoformat()} (slot {grid.slot}); widen or move the windows."
            )
            violations.append(Violation(message))
        return Violations(tuple(violations))
