"""Tests for schedule costs and constraint evaluation."""

import json
import re
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Annotated, cast
from unittest.mock import patch

import pytest
from ortools.math_opt.python import mathopt
from pydantic import Field, TypeAdapter

from intent_to_schedule.adapter.command_line_interface.main import main
from intent_to_schedule.adapter.command_line_interface.state import (
    State,
    load_state,
    save_state,
    to_problem_state,
    to_schedule_state,
)
from intent_to_schedule.adapter.data_model import (
    QueryData,
    answer_record,
    convert_query,
)
from intent_to_schedule.adapter.mathopt.solve import MathOptSchedulingSolver
from intent_to_schedule.application.objective import (
    evaluate_constraints,
    summarize_schedule,
)
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.query import Answered, AnswerResult
from intent_to_schedule.application.solve import Solved, SolveResult
from intent_to_schedule.domain.calendar import (
    Availability,
    Calendar,
    TimeGrid,
    TimeInterval,
)
from intent_to_schedule.domain.condition import (
    Condition,
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
from intent_to_schedule.domain.time_windows import TimeRange, TimeRelation, TimeWindow
from intent_to_schedule.domain.violation import CriterionViolation, measure_criterion

START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone(timedelta(hours=9)))
HOUR: timedelta = timedelta(hours=1)
PERSON: PersonId = PersonId("person")
QUERY_ADAPTER: TypeAdapter[QueryData] = TypeAdapter(
    Annotated[QueryData, Field(discriminator="kind")]
)


def task(identifier: str, required: bool = True) -> Task:
    """Build a movable task."""
    return Task(
        TaskId(identifier),
        identifier,
        HOUR,
        frozenset({PERSON}),
        Importance.LOW,
        required,
    )


def problem(
    tasks: tuple[Task, ...],
    constraints: tuple[Constraint, ...] = (),
    fixed_tasks: tuple[FixedTask, ...] = (),
) -> SchedulingProblem:
    """Build a small scheduling problem."""
    grid: TimeGrid = TimeGrid(TimeInterval(START, START + 4 * HOUR), HOUR)
    return SchedulingProblem(
        Calendar(grid, (Availability(PERSON, (grid.horizon,)),)),
        (Person(PERSON, "Person"),),
        tasks,
        fixed_tasks,
        constraints,
    )


def previous_task(item: Task, start: datetime) -> ScheduledTask:
    """Build a saved placement."""
    return ScheduledTask(item.id, item.name, start, start + item.duration)


def query_record(
    source: dict[str, object], value: SchedulingProblem, previous: Schedule | None
) -> dict[str, object]:
    """Answer a query through its JSON forms."""
    result: AnswerResult = convert_query(QUERY_ADAPTER.validate_python(source)).answer(
        value, previous
    )
    assert isinstance(result, Answered)
    return answer_record(result.answer)


@pytest.mark.parametrize(
    "scenario",
    ["window", "deadline", "gap", "count", "duration", "drop", "move", "stable"],
)
@pytest.mark.parametrize("custom_policy", [False, True])
def test_summary_matches_objective_of_the_same_solve(
    scenario: str, custom_policy: bool
) -> None:
    """Match independently measured costs to the solved MathOpt objective."""
    item: Task = task("task", required=scenario != "drop")
    condition: Condition
    previous: Schedule | None = None
    fixed_tasks: tuple[FixedTask, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    policy: ObjectivePolicy = (
        replace(DEFAULT_POLICY, per_count=2.5, stability_drop_cost_ratio=0.25)
        if custom_policy
        else DEFAULT_POLICY
    )
    if scenario == "window":
        condition = TimeWindowCondition(
            frozenset({item.id}),
            TimeRelation.AVOID,
            (TimeWindow(None, None, TimeRange(time(9, 15), time(12, 45))),),
        )
    elif scenario in {"deadline", "drop"}:
        condition = TimeBoundCondition(
            frozenset({item.id}),
            Boundary.END,
            TimeBoundRelation.AT_OR_BEFORE,
            START - HOUR,
        )
    elif scenario == "gap":
        fixed_tasks = (
            FixedTask(
                TaskId("fixed"),
                "Fixed",
                START + timedelta(minutes=10),
                HOUR,
                frozenset({PERSON}),
            ),
        )
        condition = TaskGapCondition(
            fixed_tasks[0].id, item.id, TaskGapRelation.EXACTLY, timedelta(minutes=15)
        )
    elif scenario in {"count", "duration"}:
        condition = DailyLimitCondition(
            frozenset({item.id}),
            AggregateQuantity.COUNT
            if scenario == "count"
            else AggregateQuantity.TOTAL_DURATION,
            0 if scenario == "count" else timedelta(minutes=15),
        )
    else:
        previous = Schedule((previous_task(item, START),), ())
        constraints = (
            HardConstraint(
                ConstraintId("move"),
                TimeBoundCondition(
                    frozenset({item.id}),
                    Boundary.START,
                    TimeBoundRelation.AT,
                    START + (3 if scenario == "move" else 0) * HOUR,
                ),
            ),
        )
        condition = TimeBoundCondition(
            frozenset({item.id}), Boundary.START, TimeBoundRelation.AT_OR_AFTER, START
        )
    constraints += (
        SoftConstraint(ConstraintId("preference"), condition, Strength.NORMAL),
    )
    value: SchedulingProblem = problem((item,), constraints, fixed_tasks)
    captured: list[mathopt.SolveResult] = []
    original_solve: Callable[..., mathopt.SolveResult] = mathopt.solve

    def capture(
        model: mathopt.Model,
        solver_type: mathopt.SolverType,
        *,
        params: mathopt.SolveParameters,
    ) -> mathopt.SolveResult:
        result: mathopt.SolveResult = original_solve(model, solver_type, params=params)
        captured.append(result)
        return result

    solved: SolveResult
    with patch(
        "intent_to_schedule.adapter.mathopt.solve.mathopt.solve", side_effect=capture
    ):
        solved = MathOptSchedulingSolver(policy).solve(value, previous)
    assert isinstance(solved, Solved)
    assert solved.summary is not None
    assert solved.summary.total_cost == pytest.approx(captured[0].objective_value())
    assert solved.summary.total_cost == pytest.approx(
        solved.summary.dropped_tasks_cost
        + solved.summary.soft_constraints_cost
        + solved.summary.stability_cost
    )
    assert solved.summary.moved_tasks == (1 if scenario == "move" else 0)
    assert solved.summary.dropped_tasks == (1 if scenario == "drop" else 0)
    assert solved.summary.scheduled_tasks == (0 if scenario == "drop" else 1)
    assert all(
        item.violation.amount == 0
        for item in evaluate_constraints(value, solved.schedule, policy)
        if isinstance(item.constraint, HardConstraint)
    )
    if scenario == "move":
        assert (
            solved.summary.stability_cost
            == policy.drop_cost(item.importance) * policy.stability_drop_cost_ratio
        )
    assert summarize_schedule(value, solved.schedule, policy).moved_tasks == 0
    assert summarize_schedule(value, solved.schedule, policy).stability_cost == 0


def evaluation_problem() -> tuple[SchedulingProblem, Schedule]:
    """Build current constraints and an older schedule."""
    first: Task = task("first")
    missing: Task = task("missing")
    dropped: Task = task("dropped", False)
    fixed: FixedTask = FixedTask(
        TaskId("fixed"), "Fixed", START + timedelta(minutes=10), HOUR, frozenset()
    )
    identifiers: frozenset[TaskId] = frozenset(
        {first.id, missing.id, dropped.id, fixed.id}
    )
    deadline: TimeBoundCondition = TimeBoundCondition(
        identifiers,
        Boundary.END,
        TimeBoundRelation.AT_OR_BEFORE,
        START + timedelta(minutes=30),
    )
    constraints: tuple[Constraint, ...] = (
        SoftConstraint(ConstraintId("weak"), deadline, Strength.WEAK, "Deadline"),
        SoftConstraint(
            ConstraintId("strong"),
            DailyLimitCondition(identifiers, AggregateQuantity.COUNT, 1),
            Strength.STRONG,
        ),
        HardConstraint(ConstraintId("hard"), deadline, "Required deadline"),
        SoftConstraint(
            ConstraintId("duration"),
            DailyLimitCondition(
                identifiers, AggregateQuantity.TOTAL_DURATION, timedelta(minutes=30)
            ),
            Strength.NORMAL,
        ),
        SoftConstraint(
            ConstraintId("window"),
            TimeWindowCondition(
                identifiers,
                TimeRelation.WITHIN,
                (TimeWindow(None, None, TimeRange(time(9, 15), time(10, 45))),),
            ),
            Strength.NORMAL,
        ),
        SoftConstraint(
            ConstraintId("gap"),
            TaskGapCondition(fixed.id, first.id, TaskGapRelation.AT_LEAST, HOUR),
            Strength.NORMAL,
        ),
        HardConstraint(
            ConstraintId("satisfied"),
            TimeBoundCondition(
                frozenset({missing.id}), Boundary.START, TimeBoundRelation.AT, START
            ),
        ),
    )
    previous: Schedule = Schedule(
        (
            previous_task(first, START + HOUR),
            ScheduledTask(TaskId("removed"), "Removed", START, START + HOUR),
        ),
        (DroppedTask(dropped.id, dropped.name),),
    )
    return problem((first, missing, dropped), constraints, (fixed,)), previous


def test_evaluation_amounts_units_costs_breakdowns_and_ordering() -> None:
    """Evaluate current constraints against recorded and fixed placements."""
    value: SchedulingProblem
    previous: Schedule
    value, previous = evaluation_problem()
    record: dict[str, object] = query_record({"kind": "evaluation"}, value, previous)
    assert record["has_previous"] is True
    assert record["total"] == 7 and record["truncated"] is False
    items: list[dict[str, object]] = cast(list[dict[str, object]], record["items"])
    assert [item["constraint_id"] for item in items] == [
        "hard",
        "strong",
        "window",
        "duration",
        "gap",
        "weak",
        "satisfied",
    ]
    by_identifier: dict[str, dict[str, object]] = {
        str(item["constraint_id"]): item for item in items
    }
    assert by_identifier["hard"]["cost"] is None
    assert by_identifier["hard"]["requirement"] == {"kind": "hard"}
    hard_breakdown: list[dict[str, object]] = cast(
        list[dict[str, object]], by_identifier["hard"]["breakdown"]
    )
    assert all(part["cost"] is None for part in hard_breakdown)
    assert by_identifier["weak"]["label"] == "Deadline"
    assert by_identifier["weak"]["violation"] == {"amount": 3.0, "unit": "hours"}
    assert by_identifier["weak"]["cost"] == 3.0
    assert by_identifier["weak"]["requirement"] == {"kind": "soft", "strength": "weak"}
    assert by_identifier["weak"]["breakdown"] == [
        {
            "task_id": "dropped",
            "violation": {"amount": 0.0, "unit": "hours"},
            "cost": 0.0,
        },
        {
            "task_id": "first",
            "violation": {"amount": 1.5, "unit": "hours"},
            "cost": 1.5,
        },
        {
            "task_id": "fixed",
            "violation": {"amount": 1.5, "unit": "hours"},
            "cost": 1.5,
        },
        {
            "task_id": "missing",
            "violation": {"amount": 0.0, "unit": "hours"},
            "cost": 0.0,
        },
    ]
    assert by_identifier["strong"]["violation"] == {"amount": 1.0, "unit": "count"}
    assert by_identifier["strong"]["cost"] == 20.0
    assert by_identifier["strong"]["breakdown"] == [
        {
            "date": "2026-10-01",
            "violation": {"amount": 1.0, "unit": "count"},
            "cost": 20.0,
        }
    ]
    assert by_identifier["duration"]["violation"] == {"amount": 2.5, "unit": "hours"}
    assert by_identifier["duration"]["cost"] == 12.5
    assert by_identifier["window"]["violation"] == {"amount": 3.0, "unit": "hours"}
    assert by_identifier["gap"]["violation"] == {"amount": 2.0, "unit": "hours"}
    assert by_identifier["gap"]["breakdown"] == []


@pytest.mark.parametrize(
    "filters,limit,expected,total",
    [
        ({"violated_only": True}, 2, ["hard", "strong"], 6),
        ({"constraint_ids": ["weak", "hard"]}, 20, ["hard", "weak"], 2),
        (
            {"task_ids": ["first"], "constraint_ids": ["weak", "satisfied"]},
            20,
            ["weak"],
            1,
        ),
        ({"task_ids": []}, 20, [], 0),
        ({"constraint_ids": []}, 20, [], 0),
    ],
)
def test_evaluation_filters_and_limit(
    filters: dict[str, object], limit: int, expected: list[str], total: int
) -> None:
    """Filter complete constraint evaluations before limiting."""
    value: SchedulingProblem
    previous: Schedule
    value, previous = evaluation_problem()
    record: dict[str, object] = query_record(
        {"kind": "evaluation", "filter": filters, "limit": limit}, value, previous
    )
    items: list[dict[str, object]] = cast(list[dict[str, object]], record["items"])
    assert [item["constraint_id"] for item in items] == expected
    assert record["total"] == total
    assert record["truncated"] == (total > limit)
    if items and items[-1]["constraint_id"] == "weak":
        assert items[-1]["violation"] == {"amount": 3.0, "unit": "hours"}


def test_evaluation_without_previous_and_with_empty_previous() -> None:
    """Distinguish a missing saved solution from an empty one."""
    value: SchedulingProblem
    previous: Schedule
    value, previous = evaluation_problem()
    assert query_record({"kind": "evaluation"}, value, None) == {
        "kind": "evaluation",
        "has_previous": False,
        "items": [],
        "total": 0,
        "truncated": False,
    }
    record: dict[str, object] = query_record(
        {"kind": "evaluation", "filter": {"violated_only": True}},
        value,
        Schedule((), ()),
    )
    assert record["has_previous"] is True
    assert record["total"] == 4


def test_objective_policy_returns_default_policy() -> None:
    """Expose every coefficient used by the default solver."""
    record: dict[str, object] = query_record(
        {"kind": "objective_policy"}, problem(()), None
    )
    assert record == {
        "kind": "objective_policy",
        "drop_costs": {
            key.value: value for key, value in DEFAULT_POLICY.drop_costs.items()
        },
        "weights": {key.value: value for key, value in DEFAULT_POLICY.weights.items()},
        "per_count": DEFAULT_POLICY.per_count,
        "stability_drop_cost_ratio": DEFAULT_POLICY.stability_drop_cost_ratio,
    }


def test_readme_new_query_examples_match_schema() -> None:
    """Validate the new README queries against their input schema."""
    readme: str = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    kinds: set[str] = set()
    block: str
    for block in re.findall(
        r"^```json\s*\n(.*?)^```\s*$", readme, re.MULTILINE | re.DOTALL
    ):
        source: dict[str, object] = json.loads(block)
        if "kind" in source:
            QUERY_ADAPTER.validate_python(source)
        if source.get("kind") in {"evaluation", "objective_policy"}:
            kinds.add(str(source["kind"]))
    assert kinds == {"evaluation", "objective_policy"}


@pytest.mark.parametrize("boundary", list(Boundary))
@pytest.mark.parametrize("relation", list(TimeBoundRelation))
def test_criterion_time_bounds_use_exact_hours(
    boundary: Boundary, relation: TimeBoundRelation
) -> None:
    """Measure every boundary relation without rounding the bound."""
    item: Task = task("task")
    grid: TimeGrid = problem((item,)).calendar.grid
    interval: TimeInterval = TimeInterval(START, START + HOUR)
    target: datetime = START + timedelta(minutes=15)
    condition: TimeBoundCondition = TimeBoundCondition(
        frozenset({item.id}), boundary, relation, target
    )
    difference: float = (
        (interval.start if boundary is Boundary.START else interval.end) - target
    ) / HOUR
    expected: float = (
        max(difference, 0.0)
        if relation is TimeBoundRelation.AT_OR_BEFORE
        else max(-difference, 0.0)
        if relation is TimeBoundRelation.AT_OR_AFTER
        else abs(difference)
    )
    measured: CriterionViolation = measure_criterion(
        condition.criteria(grid)[0], {item.id: interval}, grid
    )
    assert measured.amount == expected
    assert measured.unit == "hours"
    assert measured.breakdown[0].task_id == item.id
    assert measure_criterion(condition.criteria(grid)[0], {}, grid).amount == 0.0


@pytest.mark.parametrize("relation", list(TaskGapRelation))
def test_gap_missing_task_contributes_nothing(relation: TaskGapRelation) -> None:
    """Deactivate a task gap when either endpoint is unscheduled."""
    first: Task = task("first")
    second: Task = task("second")
    grid: TimeGrid = problem((first, second)).calendar.grid
    condition: TaskGapCondition = TaskGapCondition(
        first.id, second.id, relation, timedelta(minutes=15)
    )
    assert (
        measure_criterion(
            condition.criteria(grid)[0],
            {first.id: TimeInterval(START, START + HOUR)},
            grid,
        ).amount
        == 0.0
    )
    assert (
        measure_criterion(
            condition.criteria(grid)[0],
            {second.id: TimeInterval(START, START + HOUR)},
            grid,
        ).amount
        == 0.0
    )


@pytest.mark.parametrize("quantity", list(AggregateQuantity))
def test_daily_breakdown_uses_start_date_offset_and_rounded_fixed_duration(
    quantity: AggregateQuantity,
) -> None:
    """Assign entire rounded durations to their local start dates."""
    first: FixedTask = FixedTask(
        TaskId("first"),
        "First",
        START.replace(hour=23, minute=40).astimezone(timezone.utc),
        HOUR,
        frozenset(),
    )
    second: FixedTask = replace(
        first, id=TaskId("second"), start=first.start + timedelta(days=1)
    )
    horizon: TimeInterval = TimeInterval(
        START.replace(hour=22),
        (START + timedelta(days=3)).replace(hour=2).astimezone(timezone.utc),
    )
    condition: DailyLimitCondition = DailyLimitCondition(
        frozenset({first.id, second.id}),
        quantity,
        0 if quantity is AggregateQuantity.COUNT else timedelta(minutes=30),
    )
    value: SchedulingProblem = replace(
        problem(
            (),
            (SoftConstraint(ConstraintId("daily"), condition, Strength.NORMAL),),
            (first, second),
        ),
        calendar=Calendar(TimeGrid(horizon, HOUR), ()),
    )
    record: dict[str, object] = query_record(
        {"kind": "evaluation"}, value, Schedule((), ())
    )
    items: list[dict[str, object]] = cast(list[dict[str, object]], record["items"])
    expected_amount: float = 1.0 if quantity is AggregateQuantity.COUNT else 1.5
    unit: str = "count" if quantity is AggregateQuantity.COUNT else "hours"
    assert items[0]["violation"] == {"amount": 2 * expected_amount, "unit": unit}
    assert items[0]["breakdown"] == [
        {
            "date": "2026-10-01",
            "violation": {"amount": expected_amount, "unit": unit},
            "cost": expected_amount * 5.0,
        },
        {
            "date": "2026-10-02",
            "violation": {"amount": expected_amount, "unit": unit},
            "cost": expected_amount * 5.0,
        },
        {"date": "2026-10-03", "violation": {"amount": 0.0, "unit": unit}, "cost": 0.0},
        {"date": "2026-10-04", "violation": {"amount": 0.0, "unit": unit}, "cost": 0.0},
    ]
    solved: SolveResult = MathOptSchedulingSolver(DEFAULT_POLICY).solve(value)
    assert isinstance(solved, Solved) and solved.summary is not None
    assert solved.summary.soft_constraints_cost == items[0]["cost"]


def test_saved_duration_is_used_after_current_task_edits() -> None:
    """Evaluate the recorded interval after a movable task changes."""
    item: Task = task("task")
    previous: Schedule = Schedule((previous_task(item, START),), ())
    condition: TimeBoundCondition = TimeBoundCondition(
        frozenset({item.id}), Boundary.END, TimeBoundRelation.AT_OR_BEFORE, START
    )
    value: SchedulingProblem = problem(
        (replace(item, duration=3 * HOUR),),
        (SoftConstraint(ConstraintId("deadline"), condition, Strength.NORMAL),),
    )
    assert (
        evaluate_constraints(value, previous, DEFAULT_POLICY)[0].violation.amount == 1.0
    )


@pytest.mark.parametrize("stability", [False, True])
def test_command_line_summary_and_queries_use_saved_solution(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], stability: bool
) -> None:
    """Expose measured summaries and queries through the command line."""
    item: Task = task("task")
    previous: Schedule = Schedule((previous_task(item, START),), ())
    condition: TimeBoundCondition = TimeBoundCondition(
        frozenset({item.id}), Boundary.START, TimeBoundRelation.AT, START + 3 * HOUR
    )
    value: SchedulingProblem = problem(
        (item,), (HardConstraint(ConstraintId("moved"), condition),)
    )
    path: Path = tmp_path / "state.json"
    save_state(
        path,
        State(
            problem=to_problem_state(value),
            previous=to_schedule_state(previous),
            dialogue=(),
        ),
    )
    arguments: list[str] = ["--state", str(path), "solve"] + (
        [] if stability else ["--no-stability"]
    )
    assert main(arguments) == 0
    output: dict[str, object] = json.loads(capsys.readouterr().out)
    assert output["summary"] == {
        "total_cost": 2.5 if stability else 0.0,
        "costs": {
            "dropped_tasks": 0.0,
            "soft_constraints": 0.0,
            "stability": 2.5 if stability else 0.0,
        },
        "counts": {
            "scheduled_tasks": 1,
            "dropped_tasks": 0,
            "violated_soft_constraints": 0,
            "moved_tasks": 1 if stability else 0,
        },
    }
    saved: State = load_state(path)
    assert saved.previous is not None
    assert saved.previous.model_dump(mode="json") == {"items": output["items"]}
    query_path: Path = tmp_path / "query.json"
    query_path.write_text('{"kind":"evaluation","filter":{"violated_only":true}}')
    assert main(["--state", str(path), "query", "--file", str(query_path)]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "kind": "evaluation",
        "has_previous": True,
        "items": [],
        "total": 0,
        "truncated": False,
    }
    query_path.write_text('{"kind":"objective_policy"}')
    assert main(["--state", str(path), "query", "--file", str(query_path)]) == 0
    assert json.loads(capsys.readouterr().out)["per_count"] == DEFAULT_POLICY.per_count


@pytest.mark.parametrize("limit", [0, 101])
def test_evaluation_rejects_invalid_limit(limit: int) -> None:
    """Reject unsupported evaluation listing limits."""
    result: AnswerResult = convert_query(
        QUERY_ADAPTER.validate_python({"kind": "evaluation", "limit": limit})
    ).answer(problem(()), None)
    assert not isinstance(result, Answered)
