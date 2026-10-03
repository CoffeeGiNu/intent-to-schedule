from collections.abc import Sequence
from datetime import timedelta

from intent_to_schedule.domain.calendar import Availability, TimeGrid, TimeInterval
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import FixedTask


def free_slots(problem: SchedulingProblem, person_id: PersonId) -> tuple[bool, ...]:
    """Slots where a person is available and not taken by a fixed task."""
    grid: TimeGrid = problem.calendar.grid
    free: list[bool] = [False] * grid.slot_count
    availability: Availability
    interval: TimeInterval
    rounded: TimeInterval | None
    first: int
    last: int
    for availability in problem.calendar.availabilities:
        if availability.person_id != person_id:
            continue
        for interval in availability.intervals:
            rounded = grid.round_inward(interval)
            if rounded is None:
                continue
            first = max(0, grid.index_of(rounded.start))
            last = min(grid.slot_count, grid.index_of(rounded.end))
            if first < last:
                free[first:last] = [True] * (last - first)

    task: FixedTask
    for task in problem.fixed_tasks:
        if person_id not in task.participant_ids or task.duration == timedelta(0):
            continue
        rounded = grid.round_outward(TimeInterval(task.start, task.start + task.duration))
        first = max(0, grid.index_of(rounded.start))
        last = min(grid.slot_count, grid.index_of(rounded.end))
        if first < last:
            free[first:last] = [False] * (last - first)
    return tuple(free)


def available_start_slots(
    grid: TimeGrid, participants_free: Sequence[Sequence[bool]], duration: timedelta
) -> tuple[int, ...]:
    """Start slots where every participant is free for the duration."""
    if duration <= timedelta(0):
        raise ValueError("Task duration must be positive.")
    if duration % grid.slot != timedelta(0):
        raise ValueError("Task duration must be a multiple of the time grid slot.")
    if duration > grid.horizon.end - grid.horizon.start:
        return ()
    duration_slots: int = grid.index_of(grid.horizon.start + duration)
    return tuple(
        start
        for start in range(max(0, grid.slot_count - duration_slots + 1))
        if all(
            all(free[slot] for slot in range(start, start + duration_slots))
            for free in participants_free
        )
    )
