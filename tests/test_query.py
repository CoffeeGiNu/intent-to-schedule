"""Tests for scheduling queries and their JSON boundary."""

from collections.abc import Sequence
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from typing import Annotated

import pytest
from pydantic import Field, TypeAdapter

import intent_to_schedule.application.query as query_module
from intent_to_schedule.adapter.data_model import (
    QueryData,
    TimeWindowData,
    answer_record,
    convert_query,
    to_constraint_data,
    to_fixed_task_data,
    to_task_data,
    to_time_interval_data,
)
from intent_to_schedule.adapter.mathopt.compile import CompiledProblem, compile_problem
from intent_to_schedule.application.command import Rejected
from intent_to_schedule.application.policy import DEFAULT_POLICY
from intent_to_schedule.application.query import (
    Answered,
    AnswerResult,
    AvailableStartsAnswer,
    AvailableStartsQuery,
    ConstraintsAnswer,
    ConstraintsQuery,
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
from intent_to_schedule.application.time_windows import TimeRange, TimeWindow
from intent_to_schedule.domain.calendar import (
    Availability,
    Calendar,
    TimeGrid,
    TimeInterval,
)
from intent_to_schedule.domain.constraint import (
    Constraint,
    ConstraintId,
    HardConstraint,
    SoftConstraint,
)
from intent_to_schedule.domain.evaluation import Distance, Excess, Intrusion, Shortfall
from intent_to_schedule.domain.measure import (
    AggregateMeasure,
    AggregateQuantity,
    DependencyMeasure,
    IntervalMeasure,
    PointMeasure,
)
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId


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


def parse_query(value: dict[str, object]) -> SchedulingQuery:
    """Parse and convert a JSON query."""
    adapter: TypeAdapter[QueryData] = TypeAdapter(
        Annotated[QueryData, Field(discriminator="kind")]
    )
    return convert_query(adapter.validate_python(value))


def rejection(result: AnswerResult) -> str:
    """Read the messages from a rejected answer."""
    assert isinstance(result, Rejected)
    return " ".join(violation.message for violation in result.violations.items)


def test_summary_counts_and_json(problem: SchedulingProblem) -> None:
    """Return compact counts and the grid."""
    previous: Schedule = Schedule((), frozenset())
    summary: Summary = summarize(problem, previous)
    assert summary == Summary(problem.calendar.grid, 3, 2, 3, 0, True)
    result: AnswerResult = SummaryQuery().answer(problem, previous)
    assert result == Answered(summary)
    assert answer_record(summary) == {
        "kind": "summary",
        "grid": {
            "horizon": to_time_interval_data(summary.grid.horizon).model_dump(
                mode="json"
            ),
            "slot": "PT30M",
        },
        "counts": {"people": 3, "tasks": 2, "fixed_tasks": 3, "constraints": 0},
        "has_previous": True,
    }
    assert not summarize(problem, None).has_previous


@pytest.mark.parametrize(
    "kind", ["people", "tasks", "constraints", "previous_schedule"]
)
def test_filter_and_limit_defaults(kind: str) -> None:
    """Allow omitted filters and default the limit to twenty."""
    query: SchedulingQuery = parse_query({"kind": kind})
    assert getattr(query, "limit") == 20


@pytest.mark.parametrize(
    "kind", ["people", "tasks", "constraints", "previous_schedule", "available_starts"]
)
@pytest.mark.parametrize("limit", [0, -1, 101])
def test_invalid_limits_are_explanatory_rejections(
    problem: SchedulingProblem, kind: str, limit: int
) -> None:
    """Reject limits outside the supported range."""
    value: dict[str, object] = {"kind": kind, "limit": limit}
    if kind == "available_starts":
        value.update(participant_ids=[], duration="PT30M")
    query: SchedulingQuery = parse_query(value)
    message: str = rejection(query.answer(problem, None))
    assert "limit" in message.lower()
    assert str(limit) in message
    assert "1" in message and "100" in message


def test_people_filters_order_and_zero_matches(problem: SchedulingProblem) -> None:
    """Combine person filters and sort names before identifiers."""
    query: SchedulingQuery = parse_query(
        {
            "kind": "people",
            "filter": {"person_ids": ["bob", "alice-two"], "name_equals": "Alice"},
        }
    )
    assert query == PeopleQuery(
        frozenset({PersonId("bob"), PersonId("alice-two")}), "Alice", False, 20
    )
    assert query.answer(problem, None) == Answered(
        PeopleAnswer((problem.people[1],), 1)
    )
    result: AnswerResult = parse_query({"kind": "people", "limit": 2}).answer(
        problem, None
    )
    assert isinstance(result, Answered) and isinstance(result.answer, PeopleAnswer)
    assert [item.id.value for item in result.answer.items] == ["alice", "alice-two"]
    assert result.answer.total == 3 and result.answer.truncated
    assert answer_record(result.answer) == {
        "kind": "people",
        "items": [
            {"id": "alice", "name": "Alice"},
            {"id": "alice-two", "name": "Alice"},
        ],
        "total": 3,
        "truncated": True,
    }
    assert parse_query({"kind": "people", "filter": {"person_ids": []}}).answer(
        problem, None
    ) == Answered(PeopleAnswer((), 0))


def test_default_limit_counts_all_matches(problem: SchedulingProblem) -> None:
    """Count every match before applying the default limit."""
    people: tuple[Person, ...] = tuple(
        Person(PersonId(f"person-{index:03}"), "Person") for index in range(105)
    )
    result: AnswerResult = parse_query({"kind": "people"}).answer(
        replace(problem, people=people), None
    )
    assert isinstance(result, Answered) and isinstance(result.answer, PeopleAnswer)
    assert len(result.answer.items) == 20
    assert result.answer.total == 105 and result.answer.truncated
    maximum: AnswerResult = parse_query({"kind": "people", "limit": 100}).answer(
        replace(problem, people=people), None
    )
    assert isinstance(maximum, Answered) and isinstance(maximum.answer, PeopleAnswer)
    assert len(maximum.answer.items) == 100 and maximum.answer.total == 105


@pytest.mark.parametrize("kind", ["people", "tasks"])
def test_select_one_uses_total_before_limit(
    problem: SchedulingProblem, kind: str
) -> None:
    """Reject ambiguous selections even when the limit is one."""
    message: str = rejection(
        parse_query({"kind": kind, "limit": 1, "select": "one"}).answer(problem, None)
    )
    assert "alice" in message if kind == "people" else "fixed-first" in message
    assert "3" in message if kind == "people" else "5" in message
    empty: str = rejection(
        parse_query(
            {"kind": kind, "filter": {"name_equals": "Absent"}, "select": "one"}
        ).answer(problem, None)
    )
    assert "0" in empty or "no " in empty.lower()


def test_select_one_candidates_are_bounded_and_descriptive(
    problem: SchedulingProblem,
) -> None:
    """List five ambiguous candidates and the remaining count."""
    tasks: tuple[FixedTask, ...] = tuple(
        fixed(f"candidate-{index}") for index in range(8)
    )
    message: str = rejection(
        parse_query({"kind": "tasks", "select": "one", "limit": 1}).answer(
            replace(problem, tasks=(), fixed_tasks=tuple(reversed(tasks))), None
        )
    )
    assert "8" in message and "3 remaining" in message.lower()
    assert "alice" in message and "bob" in message and at(10).isoformat() in message
    assert all(f"candidate-{index}" in message for index in range(5))
    assert all(f"candidate-{index}" not in message for index in range(5, 8))


def test_select_one_success(problem: SchedulingProblem) -> None:
    """Answer unique selections as ordinary listings."""
    assert parse_query(
        {"kind": "people", "filter": {"person_ids": ["alice"]}, "select": "one"}
    ).answer(problem, None) == Answered(PeopleAnswer((problem.people[2],), 1))
    assert parse_query(
        {"kind": "tasks", "filter": {"task_ids": ["task-one"]}, "select": "one"}
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
    query: SchedulingQuery = parse_query(
        {
            "kind": "tasks",
            "filter": {
                "task_ids": ["start", "end", "one-person", "wrong-name"],
                "type": "fixed",
                "name_equals": "Team meeting",
                "participant_ids_all": ["alice", "bob"],
                "start_range": {"start": at(10), "end": at(12)},
            },
        }
    )
    assert isinstance(query, TasksQuery) and query.type is TaskType.FIXED
    assert query.start_range == TimeInterval(at(10), at(12))
    assert query.answer(replace(problem, fixed_tasks=tasks), None) == Answered(
        TasksAnswer((tasks[0],), 1)
    )
    movable_range: SchedulingQuery = parse_query(
        {
            "kind": "tasks",
            "filter": {"type": "task", "start_range": {"start": at(9), "end": at(13)}},
        }
    )
    assert movable_range.answer(problem, None) == Answered(TasksAnswer((), 0))


def test_tasks_order_and_json_preserve_element_shapes(
    problem: SchedulingProblem,
) -> None:
    """Sort dated tasks first and serialize the existing element shapes."""
    result: AnswerResult = parse_query({"kind": "tasks"}).answer(problem, None)
    assert isinstance(result, Answered) and isinstance(result.answer, TasksAnswer)
    assert [item.id.value for item in result.answer.items] == [
        "fixed-first",
        "fixed-second",
        "fixed-late",
        "task-one",
        "task-two",
    ]
    record: dict[str, object] = answer_record(result.answer)
    assert record == {
        "kind": "tasks",
        "items": [
            to_fixed_task_data(item).model_dump(mode="json")
            if isinstance(item, FixedTask)
            else to_task_data(item).model_dump(mode="json")
            for item in result.answer.items
        ],
        "total": 5,
        "truncated": False,
    }
    assert parse_query({"kind": "tasks", "filter": {"type": "task"}}).answer(
        problem, None
    ) == Answered(TasksAnswer(tuple(reversed(problem.tasks)), 2))


def test_constraints_filters_references_and_compression(
    problem: SchedulingProblem,
) -> None:
    """Filter referenced tasks and shorten intrusion regions."""
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
            DependencyMeasure(TaskId("task-one"), TaskId("task-two")),
            Shortfall(timedelta(minutes=30)),
            Strength.STRONG,
        ),
        HardConstraint(
            ConstraintId("constraint-a"),
            IntervalMeasure(frozenset({TaskId("task-one")})),
            Intrusion(region),
        ),
        HardConstraint(
            ConstraintId("constraint-c"),
            AggregateMeasure(frozenset({TaskId("task-two")}), AggregateQuantity.COUNT),
            Excess(2),
        ),
    )
    query: SchedulingQuery = parse_query(
        {
            "kind": "constraints",
            "filter": {
                "constraint_ids": ["constraint-a", "constraint-b", "constraint-c"],
                "task_ids": ["task-one"],
            },
            "limit": 1,
        }
    )
    result: AnswerResult = query.answer(replace(problem, constraints=constraints), None)
    assert result == Answered(ConstraintsAnswer((constraints[1],), 2))
    assert isinstance(result, Answered)
    record: dict[str, object] = answer_record(result.answer)
    expected: dict[str, object] = to_constraint_data(constraints[1]).model_dump(
        mode="json"
    )
    expected["evaluation"] = {
        "kind": "intrusion",
        "region": {
            "items": [
                to_time_interval_data(item).model_dump(mode="json")
                for item in region[:3]
            ],
            "total": 5,
            "truncated": True,
        },
    }
    assert record == {
        "kind": "constraints",
        "items": [expected],
        "total": 2,
        "truncated": True,
    }
    assert constraints[1].evaluation == Intrusion(region)
    assert parse_query({"kind": "constraints", "filter": {"task_ids": []}}).answer(
        replace(problem, constraints=constraints), None
    ) == Answered(ConstraintsAnswer((), 0))
    assert answer_record(ConstraintsAnswer((constraints[0], constraints[2]), 2))[
        "items"
    ] == [
        to_constraint_data(item).model_dump(mode="json")
        for item in (constraints[0], constraints[2])
    ]


@pytest.mark.parametrize("count", [0, 3])
def test_short_intrusion_regions_are_not_truncated(
    problem: SchedulingProblem, count: int
) -> None:
    """Keep short intrusion regions complete."""
    region: tuple[TimeInterval, ...] = tuple(
        TimeInterval(at(9), at(10)) for _ in range(count)
    )
    constraint: Constraint = HardConstraint(
        ConstraintId("constraint"),
        IntervalMeasure(frozenset({TaskId("task-one")})),
        Intrusion(region),
    )
    record: dict[str, object] = answer_record(ConstraintsAnswer((constraint,), 1))
    assert record["items"][0]["evaluation"]["region"]["total"] == count
    assert record["items"][0]["evaluation"]["region"]["truncated"] is False


def test_previous_schedule_keeps_history_and_sorts_starts(
    problem: SchedulingProblem,
) -> None:
    """Return scheduled and dropped identifiers even after tasks disappear."""
    previous: Schedule = Schedule(
        (
            ScheduledTask(TaskId("old-late"), at(12)),
            ScheduledTask(TaskId("old-b"), at(10)),
            ScheduledTask(TaskId("old-a"), at(10)),
        ),
        frozenset({TaskId("dropped-b"), TaskId("dropped-a")}),
    )
    result: AnswerResult = parse_query(
        {"kind": "previous_schedule", "limit": 4}
    ).answer(problem, previous)
    expected: PreviousScheduleAnswer = PreviousScheduleAnswer(
        (
            previous.scheduled[2],
            previous.scheduled[1],
            previous.scheduled[0],
            TaskId("dropped-a"),
        ),
        5,
        True,
    )
    assert result == Answered(expected)
    assert answer_record(expected) == {
        "kind": "previous_schedule",
        "items": [
            {"status": "scheduled", "task_id": "old-a", "start": at(10).isoformat()},
            {"status": "scheduled", "task_id": "old-b", "start": at(10).isoformat()},
            {"status": "scheduled", "task_id": "old-late", "start": at(12).isoformat()},
            {"status": "dropped", "task_id": "dropped-a"},
        ],
        "total": 5,
        "truncated": True,
        "has_previous": True,
    }
    query: SchedulingQuery = parse_query(
        {
            "kind": "previous_schedule",
            "filter": {
                "task_ids": ["old-a", "old-late", "dropped-a"],
                "start_range": {"start": at(10), "end": at(12)},
            },
        }
    )
    assert query.answer(problem, previous) == Answered(
        PreviousScheduleAnswer((previous.scheduled[2],), 1, True)
    )
    assert parse_query(
        {"kind": "previous_schedule", "filter": {"task_ids": ["dropped-a"]}}
    ).answer(problem, previous) == Answered(
        PreviousScheduleAnswer((TaskId("dropped-a"),), 1, True)
    )


def test_previous_schedule_without_a_previous_solution(
    problem: SchedulingProblem,
) -> None:
    """Answer normally when no previous schedule exists."""
    result: AnswerResult = parse_query({"kind": "previous_schedule"}).answer(
        problem, None
    )
    assert result == Answered(PreviousScheduleAnswer((), 0, False))
    assert isinstance(result, Answered)
    assert answer_record(result.answer) == {
        "kind": "previous_schedule",
        "items": [],
        "total": 0,
        "truncated": False,
        "has_previous": False,
    }
    assert parse_query({"kind": "previous_schedule"}).answer(
        problem, Schedule((), frozenset())
    ) == Answered(PreviousScheduleAnswer((), 0, True))


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
    query: SchedulingQuery = parse_query(
        {
            "kind": "available_starts",
            "duration": duration,
            "participant_ids": participants,
        }
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
    query: SchedulingQuery = parse_query(
        {
            "kind": "available_starts",
            "participant_ids": ["bob", "alice", "alice"],
            "duration": "PT1H",
            "limit": 1,
        }
    )
    assert isinstance(query, AvailableStartsQuery) and query.windows is None
    result: AnswerResult = query.answer(problem, Schedule((), frozenset()))
    assert result == Answered(AvailableStartsAnswer((at(10),), 3))
    assert calls == [PersonId("alice"), PersonId("bob")]
    assert participants == [(True,) * 8, (False,) * 8]
    assert isinstance(result, Answered)
    record: dict[str, object] = answer_record(result.answer)
    assert {key: value for key, value in record.items() if key != "note"} == {
        "kind": "available_starts",
        "items": [at(10).isoformat()],
        "total": 3,
        "truncated": True,
    }
    assert (
        "movable" in record["note"].lower() and "constraints" in record["note"].lower()
    )


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


def test_convert_query_calls_shared_time_window_conversion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Delegate the available starts window conversion."""
    import intent_to_schedule.adapter.data_model as data_model

    converted: TimeWindow = TimeWindow(None, None, TimeRange(time(10), time(12)))
    calls: list[TimeWindowData] = []

    def convert_window(data: TimeWindowData) -> TimeWindow:
        calls.append(data)
        return converted

    monkeypatch.setattr(data_model, "convert_time_window", convert_window)
    query: SchedulingQuery = parse_query(
        {
            "kind": "available_starts",
            "participant_ids": [],
            "duration": "PT1H",
            "windows": [{"time_range": {"start": "10:00", "end": "12:00"}}],
        }
    )
    assert query == AvailableStartsQuery(
        frozenset(), timedelta(hours=1), (converted,), 20
    )
    assert len(calls) == 1


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


def test_available_starts_matches_solver_candidates(problem: SchedulingProblem) -> None:
    """Match the shared start candidates used by the solver."""
    task: Task = replace(
        movable("candidate"),
        participant_ids=frozenset({PersonId("alice"), PersonId("bob")}),
    )
    given: SchedulingProblem = replace(problem, tasks=(task,))
    compiled: CompiledProblem = compile_problem(given, DEFAULT_POLICY)
    result: AnswerResult = AvailableStartsQuery(
        task.participant_ids, task.duration, None, 100
    ).answer(given, None)
    assert isinstance(result, Answered) and isinstance(
        result.answer, AvailableStartsAnswer
    )
    assert result.answer.items == tuple(
        given.calendar.grid.time_at(index)
        for index in sorted(compiled.placements[task.id])
    )


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


def test_convert_query_shared_time_window() -> None:
    """Convert windows through the shared time window adapter."""
    assert parse_query(
        {
            "kind": "available_starts",
            "participant_ids": [],
            "duration": "PT1H",
            "windows": [{"time_range": {"start": "10:00", "end": "12:00"}}],
        }
    ) == AvailableStartsQuery(
        frozenset(),
        timedelta(hours=1),
        (TimeWindow(None, None, TimeRange(time(10), time(12))),),
        20,
    )


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
    result: AnswerResult = parse_query({"kind": "tasks", "limit": 3}).answer(
        replace(problem, tasks=(), fixed_tasks=tuple(reversed(tasks))), None
    )
    assert result == Answered(TasksAnswer((tasks[0], tasks[2], tasks[1]), 4))


def test_constraints_combine_identifier_and_task_filters(
    problem: SchedulingProblem,
) -> None:
    """Require both the constraint and referenced task filters."""
    constraints: tuple[Constraint, ...] = (
        HardConstraint(
            ConstraintId("first"), PointMeasure(TaskId("task-one")), Distance(at(10))
        ),
        HardConstraint(
            ConstraintId("second"), PointMeasure(TaskId("task-two")), Distance(at(11))
        ),
    )
    query: SchedulingQuery = parse_query(
        {
            "kind": "constraints",
            "filter": {"constraint_ids": ["first"], "task_ids": ["task-two"]},
        }
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
        parse_query({"kind": "people", "select": "one"}).answer(
            replace(problem, people=people), None
        )
    )
    assert "7" in message and "2 remaining" in message
    assert all(f"person-{index}" in message for index in range(5))
    assert all(f"person-{index}" not in message for index in range(5, 7))


def test_select_one_reports_no_remaining_candidates(problem: SchedulingProblem) -> None:
    """Report the remaining count when all ambiguous candidates fit."""
    message: str = rejection(
        parse_query({"kind": "people", "select": "one"}).answer(problem, None)
    )
    assert "0 remaining" in message


def test_available_starts_ignores_movable_tasks_constraints_and_previous(
    problem: SchedulingProblem,
) -> None:
    """Compute free starts without movable tasks, constraints, or previous placements."""
    task: Task = replace(movable("blocking"), duration=timedelta(hours=4))
    constraint: Constraint = HardConstraint(
        ConstraintId("blocking"),
        IntervalMeasure(frozenset({task.id})),
        Intrusion((problem.calendar.grid.horizon,)),
    )
    previous: Schedule = Schedule((ScheduledTask(task.id, at(9)),), frozenset())
    given: SchedulingProblem = replace(
        problem, tasks=(task,), fixed_tasks=(), constraints=(constraint,)
    )
    result: AnswerResult = AvailableStartsQuery(
        frozenset({PersonId("alice")}), timedelta(hours=1), None, 2
    ).answer(given, previous)
    assert result == Answered(AvailableStartsAnswer((at(9), at(9, 30)), 7))
