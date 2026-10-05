from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum

from intent_to_schedule.application.command import Rejected
from intent_to_schedule.application.objective import ConstraintEvaluation, evaluate_constraints
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.domain.time_windows import (
    DateRange,
    TimeWindow,
    complement,
    window_times,
)
from intent_to_schedule.domain.availability import available_start_slots, free_slots
from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.consistency import Violation, Violations
from intent_to_schedule.domain.constraint import Constraint, ConstraintId
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import DroppedTask, Schedule, ScheduledTask
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
class PreviousScheduleAnswer(Listing[ScheduledTask | DroppedTask]):
    """Scheduled and dropped Tasks of the previous schedule."""

    has_previous: bool
    """Whether a previous schedule exists."""


@dataclass(frozen=True)
class AvailableStartsAnswer(Listing[datetime]):
    """Start times where every participant is free."""


@dataclass(frozen=True)
class EvaluationAnswer(Listing[ConstraintEvaluation]):
    """Current constraint evaluations against the saved schedule."""

    has_previous: bool


@dataclass(frozen=True)
class AgendaItem:
    """Fixed task or previously scheduled task taking a person's time."""

    fixed: bool
    """Whether the item is a fixed task rather than a previously scheduled task."""
    task_id: TaskId
    name: str
    interval: TimeInterval
    participant_ids: frozenset[PersonId]


@dataclass(frozen=True)
class AgendaDay:
    """A person's working time, items, and free time on one date."""

    date: date
    working: tuple[TimeInterval, ...]
    items: tuple[AgendaItem, ...]
    """Items overlapping the date in start order."""
    free: tuple[TimeInterval, ...]
    """Working time not covered by any item."""


@dataclass(frozen=True)
class AgendaAnswer:
    """A person's agenda for each requested date."""

    person_id: PersonId
    has_previous: bool
    """Whether a previous schedule exists."""
    days: tuple[AgendaDay, ...]


@dataclass(frozen=True)
class ObjectivePolicyAnswer:
    """The objective coefficients used by the default solver."""

    policy: ObjectivePolicy


type Answer = (
    Summary
    | PeopleAnswer
    | TasksAnswer
    | ConstraintsAnswer
    | PreviousScheduleAnswer
    | AvailableStartsAnswer
    | EvaluationAnswer
    | ObjectivePolicyAnswer
    | AgendaAnswer
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
        self,
        problem: SchedulingProblem,
        previous: Schedule | None,
        policy: ObjectivePolicy = DEFAULT_POLICY,
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
        self,
        problem: SchedulingProblem,
        previous: Schedule | None,
        policy: ObjectivePolicy = DEFAULT_POLICY,
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
        self,
        problem: SchedulingProblem,
        previous: Schedule | None,
        policy: ObjectivePolicy = DEFAULT_POLICY,
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
        self,
        problem: SchedulingProblem,
        previous: Schedule | None,
        policy: ObjectivePolicy = DEFAULT_POLICY,
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
                        or bool(self.task_ids & constraint.condition.task_ids)
                    )
                ),
                key=lambda constraint: constraint.id.value,
            )
        )
        return Answered(ConstraintsAnswer(constraints[: self.limit], len(constraints)))


@dataclass(frozen=True)
class EvaluationQuery:
    """Query for current constraints evaluated against the saved schedule."""

    violated_only: bool
    constraint_ids: frozenset[ConstraintId] | None
    task_ids: frozenset[TaskId] | None
    limit: int

    def answer(
        self,
        problem: SchedulingProblem,
        previous: Schedule | None,
        policy: ObjectivePolicy = DEFAULT_POLICY,
    ) -> AnswerResult:
        """Answer with filtered constraint evaluations in cost order."""
        invalid_limit: Rejected | None = _check_limit(self.limit)
        if invalid_limit is not None:
            return invalid_limit
        if previous is None:
            return Answered(EvaluationAnswer((), 0, False))
        evaluations: tuple[ConstraintEvaluation, ...] = tuple(
            sorted(
                (
                    item
                    for item in evaluate_constraints(problem, previous, policy)
                    if (not self.violated_only or item.violation.amount > 0)
                    and (
                        self.constraint_ids is None
                        or item.constraint.id in self.constraint_ids
                    )
                    and (
                        self.task_ids is None
                        or bool(self.task_ids & item.constraint.condition.task_ids)
                    )
                ),
                key=lambda item: (
                    0
                    if item.cost is None and item.violation.amount > 0
                    else 1
                    if item.cost is not None
                    else 2,
                    -(item.cost if item.cost is not None else item.violation.amount),
                    item.constraint.id.value,
                ),
            )
        )
        return Answered(
            EvaluationAnswer(evaluations[: self.limit], len(evaluations), True)
        )


@dataclass(frozen=True)
class ObjectivePolicyQuery:
    """Query for current objective coefficients."""

    def answer(
        self,
        problem: SchedulingProblem,
        previous: Schedule | None,
        policy: ObjectivePolicy = DEFAULT_POLICY,
    ) -> AnswerResult:
        """Answer with the scheduling objective policy."""
        return Answered(ObjectivePolicyAnswer(policy))


@dataclass(frozen=True)
class PreviousScheduleQuery:
    """Query for the scheduled and dropped Tasks of the previous schedule."""

    task_ids: frozenset[TaskId] | None
    start_range: TimeInterval | None
    limit: int

    def answer(
        self,
        problem: SchedulingProblem,
        previous: Schedule | None,
        policy: ObjectivePolicy = DEFAULT_POLICY,
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
                        or self.start_range.includes(item.start)
                    )
                ),
                key=lambda item: (item.start, item.task_id.value),
            )
        )
        dropped: tuple[DroppedTask, ...] = (
            tuple(
                sorted(
                    (
                        item
                        for item in previous.dropped
                        if self.task_ids is None or item.task_id in self.task_ids
                    ),
                    key=lambda item: item.task_id.value,
                )
            )
            if self.start_range is None
            else ()
        )
        items: tuple[ScheduledTask | DroppedTask, ...] = (*scheduled, *dropped)
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
        self,
        problem: SchedulingProblem,
        previous: Schedule | None,
        policy: ObjectivePolicy = DEFAULT_POLICY,
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
        elif not grid.is_whole_slots(self.duration):
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
            candidate: TimeInterval = TimeInterval(at, at + self.duration)
            if any(interval.contains(candidate) for interval in intervals):
                items.append(at)
        return Answered(AvailableStartsAnswer(tuple(items[: self.limit]), len(items)))


@dataclass(frozen=True)
class AgendaQuery:
    """Query for a person's working time, items, and free time on each date."""

    person_id: PersonId
    date_range: DateRange | None
    """Dates to include, or None for every date of the horizon."""

    def answer(
        self,
        problem: SchedulingProblem,
        previous: Schedule | None,
        policy: ObjectivePolicy = DEFAULT_POLICY,
    ) -> AnswerResult:
        """Answer with the person's agenda for each date of the horizon in range."""
        if all(person.id != self.person_id for person in problem.people):
            return Rejected(
                Violations(
                    (
                        Violation(
                            f"agenda person ID {self.person_id.value} does not exist."
                        ),
                    )
                )
            )
        grid: TimeGrid = problem.calendar.grid
        working: tuple[TimeInterval, ...] = tuple(
            interval
            for availability in problem.calendar.availabilities
            if availability.person_id == self.person_id
            for interval in availability.intervals
        )
        items: tuple[AgendaItem, ...] = (
            *(
                AgendaItem(True, task.id, task.name, task.interval, task.participant_ids)
                for task in problem.fixed_tasks
                if self.person_id in task.participant_ids
            ),
            *(
                AgendaItem(
                    False,
                    item.task_id,
                    item.name,
                    TimeInterval(item.start, item.end),
                    item.participant_ids,
                )
                for item in (previous.scheduled if previous is not None else ())
                if self.person_id in item.participant_ids
            ),
        )
        days: list[AgendaDay] = []
        day: date
        for day in grid.dates:
            if self.date_range is not None and not (
                self.date_range.start <= day < self.date_range.end
            ):
                continue
            midnight: datetime = datetime.combine(
                day, time.min, grid.horizon.start.tzinfo
            )
            span: TimeInterval = TimeInterval(midnight, midnight + timedelta(days=1))
            day_working: tuple[TimeInterval, ...] = tuple(
                sorted(
                    (
                        TimeInterval(
                            max(interval.start, span.start), min(interval.end, span.end)
                        )
                        for interval in working
                        if interval.overlap(span) > timedelta(0)
                    ),
                    key=lambda interval: interval.start,
                )
            )
            day_items: tuple[AgendaItem, ...] = tuple(
                sorted(
                    (
                        item
                        for item in items
                        if span.includes(item.interval.start)
                        or item.interval.start < span.start < item.interval.end
                    ),
                    key=lambda item: (item.interval.start, item.task_id.value),
                )
            )
            free: tuple[TimeInterval, ...] = tuple(
                gap
                for interval in day_working
                for gap in complement(
                    tuple(item.interval for item in day_items), interval
                )
            )
            days.append(AgendaDay(day, day_working, day_items, free))
        return Answered(AgendaAnswer(self.person_id, previous is not None, tuple(days)))


type SchedulingQuery = (
    SummaryQuery
    | PeopleQuery
    | TasksQuery
    | ConstraintsQuery
    | PreviousScheduleQuery
    | AvailableStartsQuery
    | EvaluationQuery
    | ObjectivePolicyQuery
    | AgendaQuery
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
