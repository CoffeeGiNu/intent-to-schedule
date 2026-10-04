"""Tests for scheduling queries."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone

import pytest

import intent_to_schedule.application.query as query_module
from intent_to_schedule.application.command import Rejected
from intent_to_schedule.application.query import (
    Answered,
    AnswerResult,
    AvailableStartsAnswer,
    AvailableStartsQuery,
    ConstraintsAnswer,
    ConstraintsQuery,
    EvaluationQuery,
    PeopleAnswer,
    PeopleQuery,
    PreviousScheduleAnswer,
    PreviousScheduleQuery,
    SchedulingQuery,
    Summary,
    SummaryQuery,
    TasksAnswer,
    TasksQuery,
    TaskType,
    summarize,
)
from intent_to_schedule.domain.calendar import (
    Availability,
    Calendar,
    TimeGrid,
    TimeInterval,
)
from intent_to_schedule.domain.condition import (
    DailyLimitCondition,
    TaskGapCondition,
    TaskGapRelation,
    TimeBoundCondition,
    TimeBoundRelation,
    TimeWindowCondition,
)
from intent_to_schedule.domain.constraint import (
    Constraint,
    ConstraintId,
    HardConstraint,
    SoftConstraint,
)
from intent_to_schedule.domain.measure import AggregateQuantity, Boundary
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import DroppedTask, Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId
from intent_to_schedule.domain.time_windows import (
    DateRange,
    TimeRange,
    TimeRelation,
    TimeWindow,
)

LIMITED_QUERIES: tuple[SchedulingQuery, ...] = (
    PeopleQuery(None, None, False, 20),
    TasksQuery(None, None, None, None, None, False, 20),
    ConstraintsQuery(None, None, 20),
    PreviousScheduleQuery(None, None, 20),
    AvailableStartsQuery(frozenset(), timedelta(minutes=30), None, 20),
    EvaluationQuery(False, None, None, 20),
)


def at(hour: int, minute: int = 0) -> datetime:
    """Build a time on the example calendar day."""
    return datetime(2026, 10, 3, hour, minute, tzinfo=timezone(timedelta(hours=9)))


def movable(identifier: str, name: str = "Review") -> Task:
    """Build a movable task."""
    return Task(
        TaskId(identifier),
        name,
        timedelta(hours=1),
        frozenset({PersonId("alice")}),
        Importance.HIGH,
        True,
    )


def fixed(
    identifier: str,
    hour: int = 10,
    name: str = "Team meeting",
    participants: tuple[str, ...] = ("alice", "bob"),
) -> FixedTask:
    """Build a fixed task."""
    return FixedTask(
        TaskId(identifier),
        name,
        at(hour),
        timedelta(minutes=30),
        frozenset(PersonId(person) for person in participants),
    )


@pytest.fixture
def problem() -> SchedulingProblem:
    """Build a problem with unordered elements."""
    grid: TimeGrid = TimeGrid(TimeInterval(at(9), at(13)), timedelta(minutes=30))
    people: tuple[Person, ...] = (
        Person(PersonId("bob"), "Bob"),
        Person(PersonId("alice-two"), "Alice"),
        Person(PersonId("alice"), "Alice"),
    )
    availabilities: tuple[Availability, ...] = tuple(
        Availability(person.id, (grid.horizon,)) for person in people
    )
    return SchedulingProblem(
        Calendar(grid, availabilities),
        people,
        (movable("task-two", "Zulu"), movable("task-one", "Alpha")),
        (fixed("fixed-late", 12), fixed("fixed-second", 10), fixed("fixed-first", 10)),
        (),
    )


def rejection(result: AnswerResult) -> str:
    """Read the messages from a rejected answer."""
    assert isinstance(result, Rejected)
    return " ".join(violation.message for violation in result.violations.items)


def test_summary_counts_elements_and_previous(problem: SchedulingProblem) -> None:
    """Return compact counts and the grid."""
    previous: Schedule = Schedule((), ())
    summary: Summary = summarize(problem, previous)
    assert summary == Summary(problem.calendar.grid, 3, 2, 3, 0, True)
    assert SummaryQuery().answer(problem, previous) == Answered(summary)
    assert not summarize(problem, None).has_previous


@pytest.mark.parametrize(
    "query",
    LIMITED_QUERIES,
    ids=[
        "people",
        "tasks",
        "constraints",
        "previous_schedule",
        "available_starts",
        "evaluation",
    ],
)
@pytest.mark.parametrize("limit", [0, 101])
def test_invalid_limits_are_explanatory_rejections(
    problem: SchedulingProblem, query: SchedulingQuery, limit: int
) -> None:
    """Reject limits outside the supported range."""
    message: str = rejection(replace(query, limit=limit).answer(problem, None))
    assert "limit" in message.lower()
    assert str(limit) in message
    assert "1" in message and "100" in message


def test_people_filters_order_and_zero_matches(problem: SchedulingProblem) -> None:
    """Combine person filters and sort names before identifiers."""
    query: PeopleQuery = PeopleQuery(
        frozenset({PersonId("bob"), PersonId("alice-two")}), "Alice", False, 20
    )
    assert query.answer(problem, None) == Answered(
        PeopleAnswer((problem.people[1],), 1)
    )
    assert PeopleQuery(None, None, False, 2).answer(problem, None) == Answered(
        PeopleAnswer((problem.people[2], problem.people[1]), 3)
    )
    assert PeopleQuery(frozenset(), None, False, 20).answer(problem, None) == Answered(
        PeopleAnswer((), 0)
    )


@pytest.mark.parametrize("limit", [20, 100])
def test_limit_cuts_items_after_counting_all_matches(
    problem: SchedulingProblem, limit: int
) -> None:
    """Count every match before applying the limit."""
    people: tuple[Person, ...] = tuple(
        Person(PersonId(f"person-{index:03}"), "Person") for index in range(105)
    )
    assert PeopleQuery(None, None, False, limit).answer(
        replace(problem, people=people), None
    ) == Answered(PeopleAnswer(people[:limit], 105))


@pytest.mark.parametrize(
    ("query", "candidate", "total"),
    [
        (PeopleQuery(None, None, True, 1), "alice", "3"),
        (TasksQuery(None, None, None, None, None, True, 1), "fixed-first", "5"),
    ],
    ids=["people", "tasks"],
)
def test_select_one_uses_total_before_limit(
    problem: SchedulingProblem,
    query: PeopleQuery | TasksQuery,
    candidate: str,
    total: str,
) -> None:
    """Reject ambiguous selections even when the limit is one."""
    message: str = rejection(query.answer(problem, None))
    assert candidate in message
    assert total in message
    empty: str = rejection(replace(query, name_equals="Absent").answer(problem, None))
    assert "0" in empty or "no " in empty.lower()


def test_select_one_candidates_are_bounded_and_descriptive(
    problem: SchedulingProblem,
) -> None:
    """List five ambiguous candidates and the remaining count."""
    tasks: tuple[FixedTask, ...] = tuple(
        fixed(f"candidate-{index}") for index in range(8)
    )
    message: str = rejection(
        TasksQuery(None, None, None, None, None, True, 1).answer(
            replace(problem, tasks=(), fixed_tasks=tuple(reversed(tasks))), None
        )
    )
    assert "8" in message and "3 remaining" in message.lower()
    assert "alice" in message and "bob" in message and at(10).isoformat() in message
    assert all(f"candidate-{index}" in message for index in range(5))
    assert all(f"candidate-{index}" not in message for index in range(5, 8))


def test_select_one_success(problem: SchedulingProblem) -> None:
    """Answer unique selections as ordinary listings."""
    assert PeopleQuery(frozenset({PersonId("alice")}), None, True, 20).answer(
        problem, None
    ) == Answered(PeopleAnswer((problem.people[2],), 1))
    assert TasksQuery(
        frozenset({TaskId("task-one")}), None, None, None, None, True, 20
    ).answer(problem, None) == Answered(TasksAnswer((problem.tasks[1],), 1))


def test_task_filters_are_anded_and_range_is_half_open(
    problem: SchedulingProblem,
) -> None:
    """Match all participants and include only starts in the range."""
    tasks: tuple[FixedTask, ...] = (
        fixed("start", 10),
        fixed("end", 12),
        fixed("one-person", 10, participants=("alice",)),
        fixed("wrong-name", 10, "Other"),
    )
    query: TasksQuery = TasksQuery(
        frozenset(item.id for item in tasks),
        TaskType.FIXED,
        "Team meeting",
        frozenset({PersonId("alice"), PersonId("bob")}),
        TimeInterval(at(10), at(12)),
        False,
        20,
    )
    assert query.answer(replace(problem, fixed_tasks=tasks), None) == Answered(
        TasksAnswer((tasks[0],), 1)
    )
    movable_range: TasksQuery = TasksQuery(
        None, TaskType.TASK, None, None, TimeInterval(at(9), at(13)), False, 20
    )
    assert movable_range.answer(problem, None) == Answered(TasksAnswer((), 0))


def test_tasks_sort_fixed_starts_before_movable_tasks(
    problem: SchedulingProblem,
) -> None:
    """Sort dated tasks first and movable tasks by name."""
    result: AnswerResult = TasksQuery(None, None, None, None, None, False, 20).answer(
        problem, None
    )
    assert isinstance(result, Answered) and isinstance(result.answer, TasksAnswer)
    assert [item.id.value for item in result.answer.items] == [
        "fixed-first",
        "fixed-second",
        "fixed-late",
        "task-one",
        "task-two",
    ]
    assert result.answer.total == 5
    assert TasksQuery(None, TaskType.TASK, None, None, None, False, 20).answer(
        problem, None
    ) == Answered(TasksAnswer(tuple(reversed(problem.tasks)), 2))


def test_constraints_filter_identifiers_and_referenced_tasks(
    problem: SchedulingProblem,
) -> None:
    """Filter referenced tasks before limiting."""
    region: tuple[TimeInterval, ...] = tuple(
        TimeInterval(
            at(9) + timedelta(minutes=30 * index),
            at(9) + timedelta(minutes=30 * (index + 1)),
        )
        for index in range(5)
    )
    constraints: tuple[Constraint, ...] = (
        SoftConstraint(
            ConstraintId("constraint-b"),
            TaskGapCondition(
                TaskId("task-one"),
                TaskId("task-two"),
                TaskGapRelation.AT_LEAST,
                timedelta(minutes=30),
            ),
            Strength.STRONG,
        ),
        HardConstraint(
            ConstraintId("constraint-a"),
            TimeWindowCondition(
                frozenset({TaskId("task-one")}),
                TimeRelation.AVOID,
                tuple(
                    TimeWindow(
                        DateRange(
                            interval.start.date(),
                            interval.start.date() + timedelta(days=1),
                        ),
                        None,
                        TimeRange(interval.start.time(), interval.end.time()),
                    )
                    for interval in region
                ),
            ),
        ),
        HardConstraint(
            ConstraintId("constraint-c"),
            DailyLimitCondition(
                frozenset({TaskId("task-two")}), AggregateQuantity.COUNT, 2
            ),
        ),
    )
    given: SchedulingProblem = replace(problem, constraints=constraints)
    query: ConstraintsQuery = ConstraintsQuery(
        frozenset(constraint.id for constraint in constraints),
        frozenset({TaskId("task-one")}),
        1,
    )
    assert query.answer(given, None) == Answered(
        ConstraintsAnswer((constraints[1],), 2)
    )
    assert ConstraintsQuery(None, frozenset(), 20).answer(given, None) == Answered(
        ConstraintsAnswer((), 0)
    )


def test_previous_schedule_keeps_history_and_sorts_starts(
    problem: SchedulingProblem,
) -> None:
    """Return scheduled and dropped identifiers even after tasks disappear."""
    previous: Schedule = Schedule(
        (
            ScheduledTask(TaskId("old-late"), "Old late", at(12), at(13)),
            ScheduledTask(TaskId("old-b"), "Old B", at(10), at(11)),
            ScheduledTask(TaskId("old-a"), "Old A", at(10), at(11)),
        ),
        (
            DroppedTask(TaskId("dropped-b"), "Dropped B"),
            DroppedTask(TaskId("dropped-a"), "Dropped A"),
        ),
    )
    assert PreviousScheduleQuery(None, None, 4).answer(problem, previous) == Answered(
        PreviousScheduleAnswer(
            (
                previous.scheduled[2],
                previous.scheduled[1],
                previous.scheduled[0],
                DroppedTask(TaskId("dropped-a"), "Dropped A"),
            ),
            5,
            True,
        )
    )
    assert PreviousScheduleQuery(
        frozenset({TaskId("old-a"), TaskId("old-late"), TaskId("dropped-a")}),
        TimeInterval(at(10), at(12)),
        20,
    ).answer(problem, previous) == Answered(
        PreviousScheduleAnswer((previous.scheduled[2],), 1, True)
    )
    assert PreviousScheduleQuery(frozenset({TaskId("dropped-a")}), None, 20).answer(
        problem, previous
    ) == Answered(
        PreviousScheduleAnswer(
            (DroppedTask(TaskId("dropped-a"), "Dropped A"),), 1, True
        )
    )


def test_previous_schedule_without_a_previous_solution(
    problem: SchedulingProblem,
) -> None:
    """Distinguish a missing previous schedule from an empty one."""
    query: PreviousScheduleQuery = PreviousScheduleQuery(None, None, 20)
    assert query.answer(problem, None) == Answered(PreviousScheduleAnswer((), 0, False))
    assert query.answer(problem, Schedule((), ())) == Answered(
        PreviousScheduleAnswer((), 0, True)
    )


@pytest.mark.parametrize(
    "duration, participants, message",
    [
        (timedelta(0), (), "positive"),
        (timedelta(minutes=-30), (), "positive"),
        (timedelta(minutes=45), (), "multiple"),
        (timedelta(minutes=30), ("missing",), "missing"),
    ],
)
def test_available_starts_rejects_invalid_inputs(
    problem: SchedulingProblem,
    duration: timedelta,
    participants: tuple[str, ...],
    message: str,
) -> None:
    """Reject unknown participants and invalid durations."""
    query: AvailableStartsQuery = AvailableStartsQuery(
        frozenset(PersonId(person) for person in participants), duration, None, 20
    )
    assert message in rejection(query.answer(problem, None)).lower()


def stub_start_candidates(
    monkeypatch: pytest.MonkeyPatch, candidates: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
) -> None:
    """Supply shared start candidates without invoking unfinished dependencies."""

    def available_starts(
        grid: TimeGrid, participants_free: Sequence[Sequence[bool]], duration: timedelta
    ) -> tuple[int, ...]:
        return candidates

    def time_at(grid: TimeGrid, index: int) -> datetime:
        return (
            at(9),
            at(9, 30),
            at(10),
            at(10, 30),
            at(11),
            at(11, 30),
            at(12),
            at(12, 30),
        )[index]

    monkeypatch.setattr(query_module, "available_start_slots", available_starts)
    monkeypatch.setattr(TimeGrid, "time_at", time_at)


def test_available_starts_reuses_shared_functions_for_requested_people(
    problem: SchedulingProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Build requested people only and limit shared candidates afterward."""
    calls: list[PersonId] = []
    participants: list[tuple[bool, ...]] = []

    def free_slots(
        given_problem: SchedulingProblem, person_id: PersonId
    ) -> tuple[bool, ...]:
        assert given_problem is problem
        calls.append(person_id)
        return (person_id == PersonId("alice"),) * 8

    def available_starts(
        grid: TimeGrid, participants_free: Sequence[Sequence[bool]], duration: timedelta
    ) -> tuple[int, ...]:
        assert grid is problem.calendar.grid and duration == timedelta(hours=1)
        participants.extend(tuple(slots) for slots in participants_free)
        return (2, 3, 4)

    stub_start_candidates(monkeypatch)
    monkeypatch.setattr(query_module, "free_slots", free_slots)
    monkeypatch.setattr(query_module, "available_start_slots", available_starts)
    query: AvailableStartsQuery = AvailableStartsQuery(
        frozenset({PersonId("bob"), PersonId("alice")}), timedelta(hours=1), None, 1
    )
    result: AnswerResult = query.answer(problem, Schedule((), ()))
    assert result == Answered(AvailableStartsAnswer((at(10),), 3))
    assert calls == [PersonId("alice"), PersonId("bob")]
    assert participants == [(True,) * 8, (False,) * 8]


def test_available_starts_window_filter_uses_whole_duration(
    problem: SchedulingProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Require the whole task to fit a merged window."""
    stub_start_candidates(monkeypatch)
    windows: tuple[TimeWindow, ...] = (
        TimeWindow(None, None, TimeRange(time(10, 15), time(12))),
    )

    def window_times(
        given_windows: Sequence[TimeWindow], horizon: TimeInterval
    ) -> tuple[TimeInterval, ...]:
        assert (
            tuple(given_windows) == windows and horizon == problem.calendar.grid.horizon
        )
        return (TimeInterval(at(10, 15), at(12)),)

    monkeypatch.setattr(query_module, "window_times", window_times)
    result: AnswerResult = AvailableStartsQuery(
        frozenset(), timedelta(hours=1), windows, 1
    ).answer(problem, None)
    assert result == Answered(AvailableStartsAnswer((at(10, 30),), 2))


@pytest.mark.parametrize("windows, total", [(None, 7), ((), 0)])
def test_available_starts_omitted_and_empty_windows(
    problem: SchedulingProblem,
    monkeypatch: pytest.MonkeyPatch,
    windows: tuple[TimeWindow, ...] | None,
    total: int,
) -> None:
    """Distinguish the whole horizon from an empty set of windows."""
    stub_start_candidates(monkeypatch)

    def window_times(
        given_windows: Sequence[TimeWindow], horizon: TimeInterval
    ) -> tuple[TimeInterval, ...]:
        assert tuple(given_windows) == ()
        return ()

    monkeypatch.setattr(query_module, "window_times", window_times)
    result: AnswerResult = AvailableStartsQuery(
        frozenset(), timedelta(hours=1), windows, 20
    ).answer(problem, None)
    assert isinstance(result, Answered) and isinstance(
        result.answer, AvailableStartsAnswer
    )
    assert result.answer.total == total and not result.answer.truncated


def test_available_starts_shared_availability_intersection(
    problem: SchedulingProblem,
) -> None:
    """Intersect participant availability and subtract rounded fixed tasks."""
    availabilities: tuple[Availability, ...] = (
        Availability(PersonId("alice"), (TimeInterval(at(9), at(13)),)),
        Availability(PersonId("bob"), (TimeInterval(at(9, 30), at(12, 30)),)),
    )
    occupied: FixedTask = replace(
        fixed("busy"), start=at(10, 15), duration=timedelta(minutes=30)
    )
    unrelated: FixedTask = fixed("unrelated", 11, participants=("alice-two",))
    given: SchedulingProblem = replace(
        problem,
        calendar=Calendar(problem.calendar.grid, availabilities),
        fixed_tasks=(occupied, unrelated),
    )
    query: AvailableStartsQuery = AvailableStartsQuery(
        frozenset({PersonId("alice"), PersonId("bob")}), timedelta(hours=1), None, 20
    )
    assert query.answer(given, None) == Answered(
        AvailableStartsAnswer((at(11), at(11, 30)), 2)
    )


@pytest.mark.parametrize(
    "duration, expected",
    [
        (
            timedelta(hours=1),
            (at(9), at(9, 30), at(10), at(10, 30), at(11), at(11, 30), at(12)),
        ),
        (timedelta(hours=5), ()),
    ],
)
def test_available_starts_no_participants_shared_horizon(
    problem: SchedulingProblem, duration: timedelta, expected: tuple[datetime, ...]
) -> None:
    """Return overlapping starts without participants or none for excess duration."""
    assert AvailableStartsQuery(frozenset(), duration, None, 20).answer(
        problem, None
    ) == Answered(AvailableStartsAnswer(expected, len(expected)))


def test_available_starts_shared_windows_merge_before_containment(
    problem: SchedulingProblem, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fit across adjacent windows before checking the whole duration."""
    stub_start_candidates(monkeypatch)
    windows: tuple[TimeWindow, ...] = (
        TimeWindow(None, None, TimeRange(time(10, 15), time(11))),
        TimeWindow(None, None, TimeRange(time(11), time(12))),
    )
    assert AvailableStartsQuery(frozenset(), timedelta(hours=1), windows, 20).answer(
        problem, None
    ) == Answered(AvailableStartsAnswer((at(10, 30), at(11)), 2))


def test_task_order_uses_time_then_name_then_identifier(
    problem: SchedulingProblem,
) -> None:
    """Order task matches by start time, name, and identifier."""
    tasks: tuple[FixedTask, ...] = (
        fixed("early", 9, "Zulu"),
        fixed("same-b", 10, "Alpha"),
        fixed("same-a", 10, "Alpha"),
        fixed("later-name", 10, "Zulu"),
    )
    result: AnswerResult = TasksQuery(None, None, None, None, None, False, 3).answer(
        replace(problem, tasks=(), fixed_tasks=tuple(reversed(tasks))), None
    )
    assert result == Answered(TasksAnswer((tasks[0], tasks[2], tasks[1]), 4))


def test_constraints_combine_identifier_and_task_filters(
    problem: SchedulingProblem,
) -> None:
    """Require both the constraint and referenced task filters."""
    constraints: tuple[Constraint, ...] = (
        HardConstraint(
            ConstraintId("first"),
            TimeBoundCondition(
                frozenset({TaskId("task-one")}),
                Boundary.START,
                TimeBoundRelation.AT,
                at(10),
            ),
        ),
        HardConstraint(
            ConstraintId("second"),
            TimeBoundCondition(
                frozenset({TaskId("task-two")}),
                Boundary.START,
                TimeBoundRelation.AT,
                at(11),
            ),
        ),
    )
    query: ConstraintsQuery = ConstraintsQuery(
        frozenset({ConstraintId("first")}), frozenset({TaskId("task-two")}), 20
    )
    assert query.answer(replace(problem, constraints=constraints), None) == Answered(
        ConstraintsAnswer((), 0)
    )


def test_people_select_one_candidates_are_bounded(problem: SchedulingProblem) -> None:
    """Bound ambiguous people candidates before returning a rejection."""
    people: tuple[Person, ...] = tuple(
        Person(PersonId(f"person-{index}"), "Same name") for index in range(7)
    )
    message: str = rejection(
        PeopleQuery(None, None, True, 20).answer(replace(problem, people=people), None)
    )
    assert "7" in message and "2 remaining" in message
    assert all(f"person-{index}" in message for index in range(5))
    assert all(f"person-{index}" not in message for index in range(5, 7))


def test_select_one_reports_no_remaining_candidates(problem: SchedulingProblem) -> None:
    """Report the remaining count when all ambiguous candidates fit."""
    message: str = rejection(PeopleQuery(None, None, True, 20).answer(problem, None))
    assert "0 remaining" in message


def test_available_starts_ignores_movable_tasks_constraints_and_previous(
    problem: SchedulingProblem,
) -> None:
    """Compute free starts without movable tasks, constraints, or previous placements."""
    task: Task = replace(movable("blocking"), duration=timedelta(hours=4))
    constraint: Constraint = HardConstraint(
        ConstraintId("blocking"),
        TimeWindowCondition(
            frozenset({task.id}),
            TimeRelation.AVOID,
            tuple(
                TimeWindow(
                    DateRange(
                        interval.start.date(), interval.start.date() + timedelta(days=1)
                    ),
                    None,
                    TimeRange(interval.start.time(), interval.end.time()),
                )
                for interval in (problem.calendar.grid.horizon,)
            ),
        ),
    )
    previous: Schedule = Schedule(
        (ScheduledTask(task.id, task.name, at(9), at(9) + task.duration),), ()
    )
    given: SchedulingProblem = replace(
        problem, tasks=(task,), fixed_tasks=(), constraints=(constraint,)
    )
    result: AnswerResult = AvailableStartsQuery(
        frozenset({PersonId("alice")}), timedelta(hours=1), None, 2
    ).answer(given, previous)
    assert result == Answered(AvailableStartsAnswer((at(9), at(9, 30)), 7))
