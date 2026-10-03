"""Forms and conversions for the persisted command line state."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from intent_to_schedule.adapter.data_model import (
    DataModel,
    FixedTaskData,
    HardConstraintData,
    NewFixedTaskData,
    PersonData,
    PersonIdField,
    SoftConstraintData,
    TaskData,
    TaskIdField,
    TimeIntervalData,
    convert_constraint,
    convert_fixed_task,
    convert_task,
    to_constraint_data,
    to_fixed_task_data,
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
from intent_to_schedule.domain.schedule import Schedule, ScheduledTask


class AvailabilityState(DataModel):
    """Available intervals for one person."""

    person_id: PersonIdField
    intervals: tuple[TimeIntervalData, ...]


class CalendarState(DataModel):
    """A calendar and its time grid."""

    horizon: TimeIntervalData
    slot: timedelta
    availabilities: tuple[AvailabilityState, ...]


class CalendarInput(CalendarState):
    """Calendar file accepted by init."""

    people: tuple[PersonData, ...]
    fixed_tasks: tuple[NewFixedTaskData, ...]


class ProblemState(DataModel):
    """Persisted scheduling problem."""

    calendar: CalendarState
    people: tuple[PersonData, ...]
    tasks: tuple[TaskData, ...]
    fixed_tasks: tuple[FixedTaskData, ...]
    constraints: tuple[
        Annotated[HardConstraintData | SoftConstraintData, Field(discriminator="kind")],
        ...,
    ]


class ScheduledTaskState(DataModel):
    """A persisted scheduled task."""

    task_id: TaskIdField
    start: datetime


class ScheduleState(DataModel):
    """A persisted schedule."""

    scheduled: tuple[ScheduledTaskState, ...]
    dropped_task_ids: tuple[TaskIdField, ...]


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
        tuple(ScheduledTask(item.task_id, item.start) for item in form.scheduled),
        frozenset(form.dropped_task_ids),
    )


def to_schedule_state(schedule: Schedule | None) -> ScheduleState | None:
    """Convert a domain schedule to its persisted form."""
    if schedule is None:
        return None
    return ScheduleState(
        scheduled=tuple(
            ScheduledTaskState(task_id=item.task_id, start=item.start)
            for item in schedule.scheduled
        ),
        dropped_task_ids=tuple(
            sorted(schedule.dropped_task_ids, key=lambda identifier: identifier.value)
        ),
    )


def to_dialogue(form: tuple[UtteranceState, ...]) -> tuple[Utterance, ...]:
    """Convert persisted dialogue to utterances."""
    return tuple(Utterance(Speaker(item.speaker), item.text) for item in form)


def load_state(path: Path) -> State:
    """Load a state file."""
    return State.model_validate_json(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: State) -> None:
    """Write a state file, creating its parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
