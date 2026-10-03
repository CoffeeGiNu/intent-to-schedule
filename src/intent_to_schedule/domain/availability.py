from collections.abc import Sequence
from datetime import timedelta

from intent_to_schedule.domain.calendar import TimeGrid
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.problem import SchedulingProblem


def free_slots(problem: SchedulingProblem, person_id: PersonId) -> tuple[bool, ...]:
    """Slots where a person is available and not taken by a fixed task."""
    raise NotImplementedError


def available_start_slots(
    grid: TimeGrid, participants_free: Sequence[Sequence[bool]], duration: timedelta
) -> tuple[int, ...]:
    """Start slots where every participant is free for the duration."""
    raise NotImplementedError
