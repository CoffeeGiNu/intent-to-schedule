from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from intent_to_schedule.domain.calendar import Availability, TimeGrid, TimeInterval
from intent_to_schedule.domain.compatibility import is_supported
from intent_to_schedule.domain.constraint import Constraint, ConstraintId
from intent_to_schedule.domain.evaluation import Distance, Excess, Intrusion, Shortfall
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import FixedTask, Task, TaskId


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
            for task_id in constraint.measure.task_ids - tasks:
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
            if (value - grid.horizon.start) % grid.slot != timedelta(0):
                violations.append(
                    Violation(f"{label} is not aligned to the time grid.")
                )

        def check_duration(value: timedelta, label: str) -> None:
            if value % grid.slot != timedelta(0):
                violations.append(
                    Violation(f"{label} is not aligned to the time grid.")
                )

        check_time(grid.horizon.end, "Calendar horizon end")
        task: Task
        availability: Availability
        interval: TimeInterval
        constraint: Constraint
        label: str
        region: tuple[TimeInterval, ...]
        quantity: datetime | timedelta | int
        for task in problem.tasks:
            if task.duration <= timedelta(0):
                violations.append(
                    Violation(f"Task {task.id.value} duration must be positive.")
                )
            check_duration(task.duration, f"Task {task.id.value} duration")
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
        for constraint in problem.constraints:
            label = f"Constraint {constraint.id.value} evaluation"
            match constraint.evaluation:
                case Intrusion(region=region):
                    for interval in region:
                        check_time(interval.start, f"{label} region start")
                        check_time(interval.end, f"{label} region end")
                case (
                    Distance(target=quantity)
                    | Shortfall(lower=quantity)
                    | Excess(upper=quantity)
                ):
                    match quantity:
                        case datetime():
                            check_time(quantity, label)
                        case timedelta():
                            check_duration(quantity, label)
        return Violations(tuple(violations))


class SupportedCombinations:
    """Validator that each constraint's Measure and Evaluation are compatible."""

    def validate(self, problem: SchedulingProblem) -> Violations:
        violations = tuple(
            Violation(
                f"Constraint {constraint.id.value} has an unsupported measure and evaluation combination."
            )
            for constraint in problem.constraints
            if not is_supported(constraint.measure, constraint.evaluation)
        )
        return Violations(violations)
