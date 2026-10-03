from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from intent_to_schedule.application.command import Rejected
from intent_to_schedule.application.time_windows import TimeWindow
from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
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
    """Scheduled and dropped Tasks of the previous schedule; a TaskId item is a dropped Task."""

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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


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
        raise NotImplementedError


@dataclass(frozen=True)
class PreviousScheduleQuery:
    """Query for the scheduled and dropped Tasks of the previous schedule."""

    task_ids: frozenset[TaskId] | None
    start_range: TimeInterval | None
    limit: int

    def answer(
        self, problem: SchedulingProblem, previous: Schedule | None
    ) -> AnswerResult:
        raise NotImplementedError


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
        raise NotImplementedError


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
    raise NotImplementedError
