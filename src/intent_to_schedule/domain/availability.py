from collections.abc import Sequence
from datetime import timedelta

from intent_to_schedule.domain.calendar import Availability, TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import TimeWindowCondition
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import FixedTask, Task
from intent_to_schedule.domain.time_windows import TimeRelation, expand


def free_slots(problem: SchedulingProblem, person_id: PersonId) -> tuple[bool, ...]:
    """Slots where a person is available and not taken by a fixed task."""
    grid: TimeGrid = problem.calendar.grid
    free: list[bool] = [False] * grid.slot_count
    availability: Availability
    interval: TimeInterval
    slots: range
    for availability in problem.calendar.availabilities:
        if availability.person_id != person_id:
            continue
        for interval in availability.intervals:
            slots = grid.slots_within(interval)
            free[slots.start:slots.stop] = [True] * len(slots)

    task: FixedTask
    for task in problem.fixed_tasks:
        if person_id not in task.participant_ids:
            continue
        slots = grid.slots_touching(task.interval)
        free[slots.start:slots.stop] = [False] * len(slots)
    return tuple(free)


def available_start_slots(
    grid: TimeGrid, participants_free: Sequence[Sequence[bool]], duration: timedelta
) -> tuple[int, ...]:
    """Start slots where every participant is free for the duration."""
    if duration <= timedelta(0):
        raise ValueError("Task duration must be positive.")
    if not grid.is_whole_slots(duration):
        raise ValueError("Task duration must be a multiple of the time grid slot.")
    if duration > grid.horizon.duration:
        return ()
    duration_slots: int = grid.slots_of(duration)
    return tuple(
        start
        for start in range(max(0, grid.slot_count - duration_slots + 1))
        if all(
            all(free[slot] for slot in range(start, start + duration_slots))
            for free in participants_free
        )
    )


def tasks_without_satisfying_start(
    problem: SchedulingProblem, condition: TimeWindowCondition
) -> tuple[Task, ...]:
    """Movable tasks of a time window condition with no available start satisfying it."""
    grid: TimeGrid = problem.calendar.grid
    within: bool = condition.relation is TimeRelation.WITHIN
    allowed: list[bool] = [not within] * grid.slot_count
    interval: TimeInterval
    slots: range
    for interval in expand(condition.windows, condition.relation, grid).intervals:
        slots = grid.slots_within(interval)
        allowed[slots.start:slots.stop] = [within] * len(slots)
    return tuple(
        task
        for task in problem.tasks
        if task.id in condition.task_ids
        and not available_start_slots(
            grid,
            (
                *(free_slots(problem, person_id) for person_id in task.participant_ids),
                allowed,
            ),
            task.duration,
        )
    )
