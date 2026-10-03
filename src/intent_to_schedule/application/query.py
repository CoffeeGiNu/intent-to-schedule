from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from intent_to_schedule.application.command import Rejected
from intent_to_schedule.application.time_windows import TimeWindow, window_times
from intent_to_schedule.domain.availability import available_start_slots, free_slots
from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.consistency import Violation, Violations
from intent_to_schedule.domain.constraint import Constraint, ConstraintId
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule, ScheduledTask
from intent_to_schedule.domain.task import FixedTask, Task, TaskId


@dataclass(frozen=True)
class Summary:
    """Time grid and element counts of a problem."""

    grid: TimeGrid
    people: int
    tasks: int
    fixed_tasks: int
    constraints: int
    has_previous: bool
    """Whether a previous schedule exists."""


@dataclass(frozen=True)
class Listing[Item]:
    """Items matching a query, cut to a limit."""

    items: tuple[Item, ...]
    total: int
    """Number of matching items before the limit."""

    @property
    def truncated(self) -> bool:
        return len(self.items) < self.total


@dataclass(frozen=True)
class PeopleAnswer(Listing[Person]):
    """People matching a PeopleQuery."""


@dataclass(frozen=True)
class TasksAnswer(Listing[Task | FixedTask]):
    """Tasks and FixedTasks matching a TasksQuery."""


@dataclass(frozen=True)
class ConstraintsAnswer(Listing[Constraint]):
    """Constraints matching a ConstraintsQuery."""


@dataclass(frozen=True)
class PreviousScheduleAnswer(Listing[ScheduledTask | TaskId]):
    """Scheduled and dropped Tasks of the previous schedule."""

    has_previous: bool
    """Whether a previous schedule exists."""


@dataclass(frozen=True)
class AvailableStartsAnswer(Listing[datetime]):
    """Start times where every participant is free."""


type Answer = (
    Summary
    | PeopleAnswer
    | TasksAnswer
    | ConstraintsAnswer
    | PreviousScheduleAnswer
    | AvailableStartsAnswer
)
"""Answer to a query."""


@dataclass(frozen=True)
class Answered:
    """Result of a query that was answered."""

    answer: Answer


type AnswerResult = Answered | Rejected
"""Result of answering a query."""


class TaskType(Enum):
    """Whether a Task is movable or fixed."""

    TASK = "task"
    FIXED = "fixed"


@dataclass(frozen=True)
class SummaryQuery:
    """Query for the Summary of a problem."""

    def answer(
        self, problem: SchedulingProblem, previous: Schedule | None
    ) -> AnswerResult:
        """Answer with the problem summary."""
        return Answered(summarize(problem, previous))


@dataclass(frozen=True)
class PeopleQuery:
    """Query for people."""

    person_ids: frozenset[PersonId] | None
    name_equals: str | None
    one: bool
    """Whether exactly one person must match."""
    limit: int

    def answer(
        self, problem: SchedulingProblem, previous: Schedule | None
    ) -> AnswerResult:
        """Answer with matching people."""
        invalid_limit: Rejected | None = _check_limit(self.limit)
        if invalid_limit is not None:
            return invalid_limit
        people: tuple[Person, ...] = tuple(
            sorted(
                (
                    person
                    for person in problem.people
                    if (self.person_ids is None or person.id in self.person_ids)
                    and (self.name_equals is None or person.name == self.name_equals)
                ),
                key=lambda person: (person.name, person.id.value),
            )
        )
        if self.one and len(people) != 1:
            return _reject_selection(
                "people",
                len(people),
                tuple(
                    f"name={person.name!r}, ID={person.id.value}"
                    for person in people[:5]
                ),
            )
        return Answered(PeopleAnswer(people[: self.limit], len(people)))


@dataclass(frozen=True)
class TasksQuery:
    """Query for Tasks and FixedTasks."""

    task_ids: frozenset[TaskId] | None
    type: TaskType | None
    name_equals: str | None
    participant_ids_all: frozenset[PersonId] | None
    start_range: TimeInterval | None
    """Range of FixedTask starts."""
    one: bool
    """Whether exactly one Task must match."""
    limit: int

    def answer(
        self, problem: SchedulingProblem, previous: Schedule | None
    ) -> AnswerResult:
        """Answer with matching movable and fixed tasks."""
        invalid_limit: Rejected | None = _check_limit(self.limit)
        if invalid_limit is not None:
            return invalid_limit
        tasks: tuple[Task | FixedTask, ...] = tuple(
            sorted(
                (
                    task
                    for task in (*problem.tasks, *problem.fixed_tasks)
                    if (self.task_ids is None or task.id in self.task_ids)
                    and (
                        self.type is None
                        or (self.type is TaskType.TASK and isinstance(task, Task))
                        or (self.type is TaskType.FIXED and isinstance(task, FixedTask))
                    )
                    and (self.name_equals is None or task.name == self.name_equals)
                    and (
                        self.participant_ids_all is None
                        or self.participant_ids_all <= task.participant_ids
                    )
                    and (
                        self.start_range is None
                        or (
                            isinstance(task, FixedTask)
                            and self.start_range.start
                            <= task.start
                            < self.start_range.end
                        )
                    )
                ),
                key=lambda task: (
                    isinstance(task, Task),
                    task.start
                    if isinstance(task, FixedTask)
                    else problem.calendar.grid.horizon.end,
                    task.name,
                    task.id.value,
                ),
            )
        )
        if self.one and len(tasks) != 1:
            return _reject_selection(
                "tasks", len(tasks), tuple(_task_candidate(task) for task in tasks[:5])
            )
        return Answered(TasksAnswer(tasks[: self.limit], len(tasks)))


@dataclass(frozen=True)
class ConstraintsQuery:
    """Query for constraints."""

    constraint_ids: frozenset[ConstraintId] | None
    task_ids: frozenset[TaskId] | None
    """Tasks of which the constraint references any."""
    limit: int

    def answer(
        self, problem: SchedulingProblem, previous: Schedule | None
    ) -> AnswerResult:
        """Answer with constraints matching identifiers and referenced tasks."""
        invalid_limit: Rejected | None = _check_limit(self.limit)
        if invalid_limit is not None:
            return invalid_limit
        constraints: tuple[Constraint, ...] = tuple(
            sorted(
                (
                    constraint
                    for constraint in problem.constraints
                    if (
                        self.constraint_ids is None
                        or constraint.id in self.constraint_ids
                    )
                    and (
                        self.task_ids is None
                        or bool(self.task_ids & constraint.measure.task_ids)
                    )
                ),
                key=lambda constraint: constraint.id.value,
            )
        )
        return Answered(ConstraintsAnswer(constraints[: self.limit], len(constraints)))


@dataclass(frozen=True)
class PreviousScheduleQuery:
    """Query for the scheduled and dropped Tasks of the previous schedule."""

    task_ids: frozenset[TaskId] | None
    start_range: TimeInterval | None
    limit: int

    def answer(
        self, problem: SchedulingProblem, previous: Schedule | None
    ) -> AnswerResult:
        """Answer with matching entries from the previous schedule."""
        invalid_limit: Rejected | None = _check_limit(self.limit)
        if invalid_limit is not None:
            return invalid_limit
        if previous is None:
            return Answered(PreviousScheduleAnswer((), 0, False))
        scheduled: tuple[ScheduledTask, ...] = tuple(
            sorted(
                (
                    item
                    for item in previous.scheduled
                    if (self.task_ids is None or item.task_id in self.task_ids)
                    and (
                        self.start_range is None
                        or self.start_range.start <= item.start < self.start_range.end
                    )
                ),
                key=lambda item: (item.start, item.task_id.value),
            )
        )
        dropped: tuple[TaskId, ...] = (
            tuple(
                sorted(
                    (
                        task_id
                        for task_id in previous.dropped_task_ids
                        if self.task_ids is None or task_id in self.task_ids
                    ),
                    key=lambda task_id: task_id.value,
                )
            )
            if self.start_range is None
            else ()
        )
        items: tuple[ScheduledTask | TaskId, ...] = (*scheduled, *dropped)
        return Answered(PreviousScheduleAnswer(items[: self.limit], len(items), True))


@dataclass(frozen=True)
class AvailableStartsQuery:
    """Query for start times where every participant is free."""

    participant_ids: frozenset[PersonId]
    duration: timedelta
    windows: tuple[TimeWindow, ...] | None
    """Windows the whole Task must fall within, or None for the whole horizon."""
    limit: int

    def answer(
        self, problem: SchedulingProblem, previous: Schedule | None
    ) -> AnswerResult:
        """Answer with shared free starts contained in the requested windows."""
        invalid_limit: Rejected | None = _check_limit(self.limit)
        if invalid_limit is not None:
            return invalid_limit
        grid: TimeGrid = problem.calendar.grid
        violations: list[Violation] = []
        known_people: frozenset[PersonId] = frozenset(
            person.id for person in problem.people
        )
        person_id: PersonId
        for person_id in sorted(
            self.participant_ids - known_people, key=lambda person: person.value
        ):
            violations.append(
                Violation(
                    f"available_starts participant ID {person_id.value} does not exist."
                )
            )
        if self.duration <= timedelta(0):
            violations.append(
                Violation(
                    f"available_starts duration {self.duration} must be positive."
                )
            )
        elif self.duration % grid.slot != timedelta(0):
            violations.append(
                Violation(
                    f"available_starts duration {self.duration} must be a multiple of slot {grid.slot}."
                )
            )
        if violations:
            return Rejected(Violations(tuple(violations)))
        participants_free: tuple[tuple[bool, ...], ...] = tuple(
            free_slots(problem, person_id)
            for person_id in sorted(
                self.participant_ids, key=lambda person: person.value
            )
        )
        starts: tuple[int, ...] = available_start_slots(
            grid, participants_free, self.duration
        )
        intervals: tuple[TimeInterval, ...] = (
            (grid.horizon,)
            if self.windows is None
            else window_times(self.windows, grid.horizon)
        )
        items: list[datetime] = []
        start: int
        for start in sorted(starts):
            at: datetime = grid.time_at(start)
            if any(
                interval.start <= at and at + self.duration <= interval.end
                for interval in intervals
            ):
                items.append(at)
        return Answered(AvailableStartsAnswer(tuple(items[: self.limit]), len(items)))


# TODO: consider a query for the requests as given (e.g. time windows before expansion), separate from ConstraintsQuery which returns expanded constraints; the requests are not stored yet.
type SchedulingQuery = (
    SummaryQuery
    | PeopleQuery
    | TasksQuery
    | ConstraintsQuery
    | PreviousScheduleQuery
    | AvailableStartsQuery
)
"""Request to read a SchedulingProblem and its previous schedule."""


def summarize(problem: SchedulingProblem, previous: Schedule | None) -> Summary:
    """Build the Summary of a problem."""
    return Summary(
        problem.calendar.grid,
        len(problem.people),
        len(problem.tasks),
        len(problem.fixed_tasks),
        len(problem.constraints),
        previous is not None,
    )


def _check_limit(limit: int) -> Rejected | None:
    """Reject a listing limit outside the supported range."""
    if 1 <= limit <= 100:
        return None
    return Rejected(
        Violations(
            (Violation(f"Query limit {limit} must be between 1 and 100 inclusive."),)
        )
    )


def _reject_selection(kind: str, total: int, candidates: tuple[str, ...]) -> Rejected:
    """Describe a selection that does not have exactly one match."""
    message: str = f"select one requires exactly one match for {kind}; found {total}."
    if candidates:
        message += " Candidates: " + "; ".join(candidates) + "."
    if total > 1:
        message += f" {total - len(candidates)} remaining."
    return Rejected(Violations((Violation(message),)))


def _task_candidate(task: Task | FixedTask) -> str:
    """Describe a task candidate for an ambiguous selection."""
    start: str = task.start.isoformat() if isinstance(task, FixedTask) else "not fixed"
    participants: str = ", ".join(
        sorted(person_id.value for person_id in task.participant_ids)
    )
    return f"start={start}, name={task.name!r}, participants=[{participants}], ID={task.id.value}"
