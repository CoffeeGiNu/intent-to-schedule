from dataclasses import dataclass
from typing import Protocol

from intent_to_schedule.application.objective import ScheduleSummary
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule


@dataclass(frozen=True)
class Solved:
    """Result with a schedule and its objective summary."""

    schedule: Schedule
    summary: ScheduleSummary | None = None


@dataclass(frozen=True)
class Infeasible:
    """Result indicating that no feasible schedule exists."""


type SolveResult = Solved | Infeasible
"""Result of solving a SchedulingProblem."""


class SchedulingSolver(Protocol):
    """Port that solves a SchedulingProblem."""

    def solve(
        self, problem: SchedulingProblem, previous: Schedule | None
    ) -> SolveResult: ...
