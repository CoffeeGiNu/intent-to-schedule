"""Forms and conversions for persisted local state."""

from datetime import timedelta
from typing import Literal

from intent_to_schedule.adapter.data_model import (
    ConstraintData,
    DataModel,
    FixedTaskData,
    PersonData,
    PersonIdField,
    ScheduledTaskData,
    ScheduleEntryData,
    TaskData,
    TimeIntervalData,
    convert_constraint,
    convert_fixed_task,
    convert_task,
    to_constraint_data,
    to_fixed_task_data,
    to_schedule_entry_data,
    to_task_data,
    to_time_interval_data,
)
from intent_to_schedule.application.translate import Speaker, Utterance
from intent_to_schedule.domain.calendar import (
    Availability,
    Calendar,
    TimeGrid,
    TimeInterval,
)
from intent_to_schedule.domain.person import Person
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import DroppedTask, Schedule, ScheduledTask


class AvailabilityState(DataModel):
    """Available intervals for one person."""

    person_id: PersonIdField
    intervals: tuple[TimeIntervalData, ...]


class CalendarState(DataModel):
    """A calendar and its time grid."""

    horizon: TimeIntervalData
    slot: timedelta
    availabilities: tuple[AvailabilityState, ...]


class ProblemState(DataModel):
    """Persisted scheduling problem."""

    calendar: CalendarState
    people: tuple[PersonData, ...]
    tasks: tuple[TaskData, ...]
    fixed_tasks: tuple[FixedTaskData, ...]
    constraints: tuple[ConstraintData, ...]


class ScheduleState(DataModel):
    """A persisted schedule."""

    items: tuple[ScheduleEntryData, ...]


class UtteranceState(DataModel):
    """A persisted dialogue utterance."""

    speaker: Literal["user", "assistant"]
    text: str


class State(DataModel):
    """Persisted problem, previous schedule, and dialogue."""

    problem: ProblemState
    previous: ScheduleState | None
    dialogue: tuple[UtteranceState, ...]


def to_interval(output: TimeIntervalData) -> TimeInterval:
    """Convert an interval form to the domain."""
    return TimeInterval(output.start, output.end)


def to_problem(form: ProblemState) -> SchedulingProblem:
    """Convert a persisted problem to the domain."""
    calendar: Calendar = Calendar(
        TimeGrid(to_interval(form.calendar.horizon), form.calendar.slot),
        tuple(
            Availability(
                item.person_id,
                tuple(to_interval(interval) for interval in item.intervals),
            )
            for item in form.calendar.availabilities
        ),
    )
    return SchedulingProblem(
        calendar,
        tuple(Person(item.id, item.name) for item in form.people),
        tuple(convert_task(item.id, item) for item in form.tasks),
        tuple(convert_fixed_task(item.id, item) for item in form.fixed_tasks),
        tuple(convert_constraint(item.id, item) for item in form.constraints),
    )


def to_problem_state(problem: SchedulingProblem) -> ProblemState:
    """Convert a domain problem to its persisted form."""
    calendar: Calendar = problem.calendar
    return ProblemState(
        calendar=CalendarState(
            horizon=to_time_interval_data(calendar.grid.horizon),
            slot=calendar.grid.slot,
            availabilities=tuple(
                AvailabilityState(
                    person_id=item.person_id,
                    intervals=tuple(
                        to_time_interval_data(interval) for interval in item.intervals
                    ),
                )
                for item in calendar.availabilities
            ),
        ),
        people=tuple(PersonData(id=item.id, name=item.name) for item in problem.people),
        tasks=tuple(to_task_data(item) for item in problem.tasks),
        fixed_tasks=tuple(to_fixed_task_data(item) for item in problem.fixed_tasks),
        constraints=tuple(to_constraint_data(item) for item in problem.constraints),
    )


def to_schedule(form: ScheduleState | None) -> Schedule | None:
    """Convert a persisted schedule to the domain."""
    if form is None:
        return None
    return Schedule(
        tuple(
            ScheduledTask(
                item.task_id,
                item.name,
                item.start,
                item.end,
                frozenset(item.participant_ids),
            )
            for item in form.items
            if isinstance(item, ScheduledTaskData)
        ),
        tuple(
            DroppedTask(item.task_id, item.name)
            for item in form.items
            if not isinstance(item, ScheduledTaskData)
        ),
    )


def to_schedule_state(schedule: Schedule | None) -> ScheduleState | None:
    """Convert a domain schedule to its persisted form."""
    if schedule is None:
        return None
    return ScheduleState(
        items=tuple(
            to_schedule_entry_data(item)
            for item in (
                *sorted(
                    schedule.scheduled,
                    key=lambda item: (item.start, item.task_id.value),
                ),
                *sorted(schedule.dropped, key=lambda item: item.task_id.value),
            )
        ),
    )


def to_dialogue(form: tuple[UtteranceState, ...]) -> tuple[Utterance, ...]:
    """Convert persisted dialogue to utterances."""
    return tuple(Utterance(Speaker(item.speaker), item.text) for item in form)

