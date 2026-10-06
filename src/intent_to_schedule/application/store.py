from typing import Protocol

from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule


class StateStore(Protocol):
    """Port that keeps the problem and previous schedule between operations."""

    def load_problem(self) -> SchedulingProblem: ...

    def save_problem(self, problem: SchedulingProblem) -> None: ...

    def load_previous(self) -> Schedule | None: ...

    def save_previous(self, previous: Schedule | None) -> None: ...
