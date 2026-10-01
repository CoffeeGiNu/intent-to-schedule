"""Forms and conversions for the persisted command line state."""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from intent_to_schedule.adapter.data_model import (
    ConstraintIdField,
    EvaluationOutput,
    HardConstraintOutput,
    MeasureOutput,
    OutputModel,
    PersonIdField,
    SoftConstraintOutput,
    TaskIdField,
    TaskOutput,
    TimeIntervalOutput,
    convert_evaluation,
    convert_measure,
    convert_task,
    to_evaluation_output,
    to_measure_output,
    to_task_output,
    to_time_interval_output,
)
from intent_to_schedule.application.translate import Speaker, Utterance
from intent_to_schedule.domain.calendar import Availability, BusyInterval, Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.constraint import Constraint, HardConstraint, SoftConstraint
from intent_to_schedule.domain.person import Person
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength


class PersonState(OutputModel):
    """A person with a stable external ID."""

    id: PersonIdField
    name: str


class AvailabilityState(OutputModel):
    """Available intervals for one person."""

    person_id: PersonIdField
    intervals: tuple[TimeIntervalOutput, ...]


class BusyIntervalState(OutputModel):
    """A busy interval for one person."""

    person_id: PersonIdField
    interval: TimeIntervalOutput


class CalendarState(OutputModel):
    """A calendar and its time grid."""

    horizon: TimeIntervalOutput
    slot: timedelta
    availabilities: tuple[AvailabilityState, ...]
    busy_intervals: tuple[BusyIntervalState, ...]


class CalendarInput(CalendarState):
    """Calendar file accepted by init."""

    people: tuple[PersonState, ...]


class TaskState(TaskOutput):
    """A task with its persisted ID."""

    id: TaskIdField


class HardConstraintState(HardConstraintOutput):
    """A persisted hard constraint with an ID."""

    id: ConstraintIdField


class SoftConstraintState(SoftConstraintOutput):
    """A persisted soft constraint with an ID."""

    id: ConstraintIdField


type ConstraintState = Annotated[HardConstraintState | SoftConstraintState, Field(discriminator="kind")]


class ProblemState(OutputModel):
    """Persisted scheduling problem."""

    calendar: CalendarState
    people: tuple[PersonState, ...]
    tasks: tuple[TaskState, ...]
    constraints: tuple[ConstraintState, ...]


class ScheduledTaskState(OutputModel):
    """A persisted scheduled task."""

    task_id: TaskIdField
    start: datetime


class ScheduleState(OutputModel):
    """A persisted schedule."""

    scheduled: tuple[ScheduledTaskState, ...]
    dropped_task_ids: tuple[TaskIdField, ...]


class UtteranceState(OutputModel):
    """A persisted dialogue utterance."""

    speaker: Literal["user", "assistant"]
    text: str


class State(OutputModel):
    """Persisted problem, previous schedule, and dialogue."""

    problem: ProblemState
    previous: ScheduleState | None
    dialogue: tuple[UtteranceState, ...]


def to_interval(output: TimeIntervalOutput) -> TimeInterval:
    """Convert an interval form to the domain."""
    return TimeInterval(output.start, output.end)


def to_problem(form: ProblemState) -> SchedulingProblem:
    """Convert a persisted problem to the domain."""
    calendar: Calendar = Calendar(
        TimeGrid(to_interval(form.calendar.horizon), form.calendar.slot),
        tuple(
            Availability(item.person_id, tuple(to_interval(interval) for interval in item.intervals))
            for item in form.calendar.availabilities
        ),
        tuple(BusyInterval(item.person_id, to_interval(item.interval)) for item in form.calendar.busy_intervals),
    )
    constraints: tuple[Constraint, ...] = tuple(to_constraint(item) for item in form.constraints)
    return SchedulingProblem(
        calendar,
        tuple(Person(item.id, item.name) for item in form.people),
        tuple(convert_task(item.id, item) for item in form.tasks),
        constraints,
    )


def to_constraint(form: HardConstraintState | SoftConstraintState) -> Constraint:
    """Convert a persisted constraint to the domain."""
    match form:
        case HardConstraintState(id=identifier, measure=measure, evaluation=evaluation):
            return HardConstraint(identifier, convert_measure(measure), convert_evaluation(evaluation))
        case SoftConstraintState(id=identifier, measure=measure, evaluation=evaluation, strength=strength):
            return SoftConstraint(
                identifier, convert_measure(measure), convert_evaluation(evaluation), Strength(strength)
            )


def to_problem_state(problem: SchedulingProblem) -> ProblemState:
    """Convert a domain problem to its persisted form."""
    calendar: Calendar = problem.calendar
    return ProblemState(
        calendar=CalendarState(
            horizon=to_time_interval_output(calendar.grid.horizon),
            slot=calendar.grid.slot,
            availabilities=tuple(
                AvailabilityState(
                    person_id=item.person_id,
                    intervals=tuple(to_time_interval_output(interval) for interval in item.intervals),
                )
                for item in calendar.availabilities
            ),
            busy_intervals=tuple(
                BusyIntervalState(person_id=item.person_id, interval=to_time_interval_output(item.interval))
                for item in calendar.busy_intervals
            ),
        ),
        people=tuple(PersonState(id=item.id, name=item.name) for item in problem.people),
        tasks=tuple(TaskState.model_validate({"id": item.id.value, **to_task_output(item).model_dump()}) for item in problem.tasks),
        constraints=tuple(to_constraint_state(item) for item in problem.constraints),
    )


def to_constraint_state(constraint: Constraint) -> HardConstraintState | SoftConstraintState:
    """Convert a domain constraint to its persisted form."""
    measure: MeasureOutput = to_measure_output(constraint.measure)
    evaluation: EvaluationOutput = to_evaluation_output(constraint.evaluation)
    match constraint:
        case HardConstraint(id=identifier):
            return HardConstraintState(id=identifier, kind="hard", measure=measure, evaluation=evaluation)
        case SoftConstraint(id=identifier, strength=strength):
            return SoftConstraintState(
                id=identifier, kind="soft", measure=measure, evaluation=evaluation, strength=strength.value
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
        scheduled=tuple(ScheduledTaskState(task_id=item.task_id, start=item.start) for item in schedule.scheduled),
        dropped_task_ids=tuple(sorted(schedule.dropped_task_ids, key=lambda identifier: identifier.value)),
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
