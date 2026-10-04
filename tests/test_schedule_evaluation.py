"""Tests for schedule costs and constraint evaluation."""

import json
import re
from dataclasses import replace
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Annotated, cast
from unittest.mock import patch

import pytest
from ortools.math_opt.python import mathopt
from pydantic import Field, TypeAdapter

from intent_to_schedule.adapter.data_model import QueryData, answer_record, convert_query
from intent_to_schedule.adapter.mathopt.solve import MathOptSchedulingSolver
from intent_to_schedule.application.objective import summarize_schedule
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.query import Answered, AnswerResult
from intent_to_schedule.application.solve import Solved
from intent_to_schedule.domain.calendar import Availability, Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import (
    Condition,
    DailyLimitCondition,
    TaskGapCondition,
    TaskGapRelation,
    TimeBoundCondition,
    TimeBoundRelation,
    TimeWindowCondition,
)
from intent_to_schedule.domain.constraint import Constraint, ConstraintId, HardConstraint, SoftConstraint
from intent_to_schedule.domain.measure import AggregateQuantity, Boundary
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import DroppedTask, Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId
from intent_to_schedule.domain.time_windows import TimeRange, TimeRelation, TimeWindow

START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone(timedelta(hours=9)))
HOUR: timedelta = timedelta(hours=1)
PERSON: PersonId = PersonId("person")
QUERY_ADAPTER: TypeAdapter[QueryData] = TypeAdapter(Annotated[QueryData, Field(discriminator="kind")])


def task(identifier: str, required: bool = True) -> Task:
    """Build a movable task."""
    return Task(TaskId(identifier), identifier, HOUR, frozenset({PERSON}), Importance.LOW, required)


def problem(tasks: tuple[Task, ...], constraints: tuple[Constraint, ...] = (), fixed_tasks: tuple[FixedTask, ...] = ()) -> SchedulingProblem:
    """Build a small scheduling problem."""
    grid: TimeGrid = TimeGrid(TimeInterval(START, START + 4 * HOUR), HOUR)
    return SchedulingProblem(Calendar(grid, (Availability(PERSON, (grid.horizon,)),)), (Person(PERSON, "Person"),), tasks, fixed_tasks, constraints)


def previous_task(item: Task, start: datetime) -> ScheduledTask:
    """Build a saved placement."""
    return ScheduledTask(item.id, item.name, start, start + item.duration)


def query_record(source: dict[str, object], value: SchedulingProblem, previous: Schedule | None) -> dict[str, object]:
    """Answer a query through its JSON forms."""
    result: AnswerResult = convert_query(QUERY_ADAPTER.validate_python(source)).answer(value, previous)
    assert isinstance(result, Answered)
    return answer_record(result.answer)


@pytest.mark.parametrize("scenario", ["window", "deadline", "gap", "count", "duration", "drop", "move", "stable"])
@pytest.mark.parametrize("custom_policy", [False, True])
def test_summary_matches_objective_of_the_same_solve(scenario: str, custom_policy: bool) -> None:
    """Match independently measured costs to the solved MathOpt objective."""
    item: Task = task("task", required=scenario != "drop")
    condition: Condition
    previous: Schedule | None = None
    fixed_tasks: tuple[FixedTask, ...] = ()
    constraints: tuple[Constraint, ...] = ()
    policy: ObjectivePolicy = replace(DEFAULT_POLICY, per_count=2.5, stability_drop_cost_ratio=0.25) if custom_policy else DEFAULT_POLICY
    if scenario == "window":
        condition = TimeWindowCondition(frozenset({item.id}), TimeRelation.AVOID, (TimeWindow(None, None, TimeRange(time(9, 15), time(12, 45))),))
    elif scenario in {"deadline", "drop"}:
        condition = TimeBoundCondition(frozenset({item.id}), Boundary.END, TimeBoundRelation.AT_OR_BEFORE, START - HOUR)
    elif scenario == "gap":
        fixed_tasks = (FixedTask(TaskId("fixed"), "Fixed", START + timedelta(minutes=10), HOUR, frozenset({PERSON})),)
        condition = TaskGapCondition(fixed_tasks[0].id, item.id, TaskGapRelation.EXACTLY, timedelta(minutes=15))
    elif scenario in {"count", "duration"}:
        condition = DailyLimitCondition(frozenset({item.id}), AggregateQuantity.COUNT if scenario == "count" else AggregateQuantity.TOTAL_DURATION, 0 if scenario == "count" else timedelta(minutes=15))
    else:
        previous = Schedule((previous_task(item, START),), ())
        constraints = (HardConstraint(ConstraintId("move"), TimeBoundCondition(frozenset({item.id}), Boundary.START, TimeBoundRelation.AT, START + (3 if scenario == "move" else 0) * HOUR)),)
        condition = TimeBoundCondition(frozenset({item.id}), Boundary.START, TimeBoundRelation.AT_OR_AFTER, START)
    constraints += (SoftConstraint(ConstraintId("preference"), condition, Strength.NORMAL),)
    value: SchedulingProblem = problem((item,), constraints, fixed_tasks)
    captured: list[mathopt.SolveResult] = []
    original_solve = mathopt.solve

    def capture(model: mathopt.Model, solver_type: mathopt.SolverType, *, params: mathopt.SolveParameters) -> mathopt.SolveResult:
        result: mathopt.SolveResult = original_solve(model, solver_type, params=params)
        captured.append(result)
        return result

    with patch("intent_to_schedule.adapter.mathopt.solve.mathopt.solve", side_effect=capture):
        solved = MathOptSchedulingSolver(policy).solve(value, previous)
    assert isinstance(solved, Solved)
    assert solved.summary is not None
    assert solved.summary.total_cost == pytest.approx(captured[0].objective_value())
    assert solved.summary.total_cost == pytest.approx(solved.summary.dropped_tasks_cost + solved.summary.soft_constraints_cost + solved.summary.stability_cost)
    assert solved.summary.moved_tasks == (1 if scenario == "move" else 0)
    assert solved.summary.dropped_tasks == (1 if scenario == "drop" else 0)
    assert solved.summary.scheduled_tasks == (0 if scenario == "drop" else 1)
    if scenario == "move":
        assert solved.summary.stability_cost == policy.drop_cost(item.importance) * policy.stability_drop_cost_ratio
    assert summarize_schedule(value, solved.schedule, policy).moved_tasks == 0
    assert summarize_schedule(value, solved.schedule, policy).stability_cost == 0


def evaluation_problem() -> tuple[SchedulingProblem, Schedule]:
    """Build current constraints and an older schedule."""
    first: Task = task("first")
    missing: Task = task("missing")
    dropped: Task = task("dropped", False)
    fixed: FixedTask = FixedTask(TaskId("fixed"), "Fixed", START + timedelta(minutes=10), HOUR, frozenset())
    identifiers: frozenset[TaskId] = frozenset({first.id, missing.id, dropped.id, fixed.id})
    deadline: TimeBoundCondition = TimeBoundCondition(identifiers, Boundary.END, TimeBoundRelation.AT_OR_BEFORE, START + timedelta(minutes=30))
    constraints: tuple[Constraint, ...] = (
        SoftConstraint(ConstraintId("weak"), deadline, Strength.WEAK, "Deadline"),
        SoftConstraint(ConstraintId("strong"), DailyLimitCondition(identifiers, AggregateQuantity.COUNT, 1), Strength.STRONG),
        HardConstraint(ConstraintId("hard"), deadline, "Required deadline"),
        SoftConstraint(ConstraintId("duration"), DailyLimitCondition(identifiers, AggregateQuantity.TOTAL_DURATION, timedelta(minutes=30)), Strength.NORMAL),
        SoftConstraint(ConstraintId("window"), TimeWindowCondition(identifiers, TimeRelation.WITHIN, (TimeWindow(None, None, TimeRange(time(9, 15), time(10, 45))),)), Strength.NORMAL),
        SoftConstraint(ConstraintId("gap"), TaskGapCondition(fixed.id, first.id, TaskGapRelation.AT_LEAST, HOUR), Strength.NORMAL),
        HardConstraint(ConstraintId("satisfied"), TimeBoundCondition(frozenset({missing.id}), Boundary.START, TimeBoundRelation.AT, START)),
    )
    previous: Schedule = Schedule((previous_task(first, START + HOUR), ScheduledTask(TaskId("removed"), "Removed", START, START + HOUR)), (DroppedTask(dropped.id, dropped.name),))
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
    assert [item["constraint_id"] for item in items] == ["hard", "strong", "duration", "window", "gap", "weak", "satisfied"]
    by_identifier: dict[str, dict[str, object]] = {str(item["constraint_id"]): item for item in items}
    assert by_identifier["hard"]["cost"] is None
    assert by_identifier["hard"]["requirement"] == {"kind": "hard"}
    assert by_identifier["weak"]["label"] == "Deadline"
    assert by_identifier["weak"]["violation"] == {"amount": 3.0, "unit": "hours"}
    assert by_identifier["weak"]["cost"] == 3.0
    assert by_identifier["weak"]["requirement"] == {"kind": "soft", "strength": "weak"}
    assert by_identifier["weak"]["breakdown"] == [
        {"task_id": "dropped", "violation": {"amount": 0.0, "unit": "hours"}, "cost": 0.0},
        {"task_id": "first", "violation": {"amount": 1.5, "unit": "hours"}, "cost": 1.5},
        {"task_id": "fixed", "violation": {"amount": 1.5, "unit": "hours"}, "cost": 1.5},
        {"task_id": "missing", "violation": {"amount": 0.0, "unit": "hours"}, "cost": 0.0},
    ]
    assert by_identifier["strong"]["violation"] == {"amount": 1.0, "unit": "count"}
    assert by_identifier["strong"]["cost"] == 20.0
    assert by_identifier["strong"]["breakdown"] == [{"date": "2026-10-01", "violation": {"amount": 1.0, "unit": "count"}, "cost": 20.0}]
    assert by_identifier["duration"]["violation"] == {"amount": 2.5, "unit": "hours"}
    assert by_identifier["duration"]["cost"] == 12.5
    assert by_identifier["window"]["violation"] == {"amount": 3.0, "unit": "hours"}
    assert by_identifier["gap"]["violation"] == {"amount": 2.0, "unit": "hours"}
    assert by_identifier["gap"]["breakdown"] == []


@pytest.mark.parametrize("filters,limit,expected,total", [
    ({"violated_only": True}, 2, ["hard", "strong"], 6),
    ({"constraint_ids": ["weak", "hard"]}, 20, ["hard", "weak"], 2),
    ({"task_ids": ["first"], "constraint_ids": ["weak", "satisfied"]}, 20, ["weak"], 1),
    ({"task_ids": []}, 20, [], 0),
    ({"constraint_ids": []}, 20, [], 0),
])
def test_evaluation_filters_and_limit(filters: dict[str, object], limit: int, expected: list[str], total: int) -> None:
    """Filter complete constraint evaluations before limiting."""
    value: SchedulingProblem
    previous: Schedule
    value, previous = evaluation_problem()
    record: dict[str, object] = query_record({"kind": "evaluation", "filter": filters, "limit": limit}, value, previous)
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
    assert query_record({"kind": "evaluation"}, value, None) == {"kind": "evaluation", "has_previous": False, "items": [], "total": 0, "truncated": False}
    record: dict[str, object] = query_record({"kind": "evaluation", "filter": {"violated_only": True}}, value, Schedule((), ()))
    assert record["has_previous"] is True
    assert record["total"] == 4


def test_objective_policy_returns_default_policy() -> None:
    """Expose every coefficient used by the default solver."""
    record: dict[str, object] = query_record({"kind": "objective_policy"}, problem(()), None)
    assert record == {"kind": "objective_policy", "drop_costs": {key.value: value for key, value in DEFAULT_POLICY.drop_costs.items()}, "weights": {key.value: value for key, value in DEFAULT_POLICY.weights.items()}, "per_count": DEFAULT_POLICY.per_count, "stability_drop_cost_ratio": DEFAULT_POLICY.stability_drop_cost_ratio}


def test_readme_new_query_examples_match_schema() -> None:
    """Validate the new README queries against their input schema."""
    readme: str = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    kinds: set[str] = set()
    block: str
    for block in re.findall(r"^```json\s*\n(.*?)^```\s*$", readme, re.MULTILINE | re.DOTALL):
        source: dict[str, object] = json.loads(block)
        if source.get("kind") in {"evaluation", "objective_policy"}:
            QUERY_ADAPTER.validate_python(source)
            kinds.add(str(source["kind"]))
    assert kinds == {"evaluation", "objective_policy"}
