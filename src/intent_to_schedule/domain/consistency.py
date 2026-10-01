from dataclasses import dataclass
from typing import Protocol

from intent_to_schedule.domain.problem import SchedulingProblem


@dataclass(frozen=True)
class Violation:
    """Reason a problem is inconsistent or a command was rejected."""

    message: str


@dataclass(frozen=True)
class Violations:
    """Collection of Violation."""

    items: tuple[Violation, ...]

    @property
    def is_empty(self) -> bool: ...

    def merge(self, other: "Violations") -> "Violations": ...


class Validator(Protocol):
    """Consistency check on a SchedulingProblem."""

    def validate(self, problem: SchedulingProblem) -> Violations: ...


class UniqueIds:
    """Validator that IDs are not duplicated."""

    def validate(self, problem: SchedulingProblem) -> Violations: ...


class ReferencesExist:
    """Validator that referenced people and Tasks exist."""

    def validate(self, problem: SchedulingProblem) -> Violations: ...


class AlignedToSlots:
    """Validator that times and durations fall on TimeGrid slots."""

    def validate(self, problem: SchedulingProblem) -> Violations: ...


class SupportedCombinations:
    """Validator that each constraint's Measure and Evaluation are compatible."""

    def validate(self, problem: SchedulingProblem) -> Violations: ...
