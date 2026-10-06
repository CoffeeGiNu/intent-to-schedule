from datetime import date, datetime, timedelta, timezone

import pytest

from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import (
    DailyLimitCondition,
    TaskGapCondition,
    TaskGapRelation,
    TimeBoundCondition,
    TimeBoundRelation,
)
from intent_to_schedule.domain.measure import AggregateQuantity, Boundary
from intent_to_schedule.domain.task import TaskId
from intent_to_schedule.domain.violation import (
    CriterionViolation,
    DateViolationPart,
    TaskViolationPart,
    measure_criterion,
)

START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone(timedelta(hours=9)))
HOUR: timedelta = timedelta(hours=1)
GRID: TimeGrid = TimeGrid(TimeInterval(START, START + 4 * HOUR), HOUR)
FIRST: TaskId = TaskId("first")
SECOND: TaskId = TaskId("second")


@pytest.mark.parametrize(
    ("boundary", "relation", "expected"),
    [
        (Boundary.START, TimeBoundRelation.AT_OR_BEFORE, 0.0),
        (Boundary.START, TimeBoundRelation.AT_OR_AFTER, 0.25),
        (Boundary.START, TimeBoundRelation.AT, 0.25),
        (Boundary.END, TimeBoundRelation.AT_OR_BEFORE, 0.75),
        (Boundary.END, TimeBoundRelation.AT_OR_AFTER, 0.0),
        (Boundary.END, TimeBoundRelation.AT, 0.75),
    ],
)
def test_criterion_time_bounds_use_exact_hours(
    boundary: Boundary, relation: TimeBoundRelation, expected: float
) -> None:
    """Measure every boundary relation without rounding the bound."""
    interval: TimeInterval = TimeInterval(START, START + HOUR)
    condition: TimeBoundCondition = TimeBoundCondition(
        frozenset({FIRST}), boundary, relation, START + timedelta(minutes=15)
    )
    measured: CriterionViolation = measure_criterion(
        condition.criteria(GRID)[0], {FIRST: interval}, GRID
    )
    assert measured.amount == expected
    assert measured.unit == "hours"
    assert measured.breakdown == (TaskViolationPart(expected, FIRST),)
    assert measure_criterion(condition.criteria(GRID)[0], {}, GRID).amount == 0.0


@pytest.mark.parametrize("relation", list(TaskGapRelation))
def test_gap_missing_task_contributes_nothing(relation: TaskGapRelation) -> None:
    """Deactivate a task gap when either endpoint is unscheduled."""
    condition: TaskGapCondition = TaskGapCondition(
        FIRST, SECOND, relation, timedelta(minutes=15)
    )
    interval: TimeInterval = TimeInterval(START, START + HOUR)
    assert (
        measure_criterion(condition.criteria(GRID)[0], {FIRST: interval}, GRID).amount
        == 0.0
    )
    assert (
        measure_criterion(condition.criteria(GRID)[0], {SECOND: interval}, GRID).amount
        == 0.0
    )


def test_daily_limit_breaks_down_by_date() -> None:
    """Attribute daily limit violations to calendar dates."""
    condition: DailyLimitCondition = DailyLimitCondition(
        frozenset({FIRST, SECOND}), AggregateQuantity.COUNT, 1
    )
    measured: CriterionViolation = measure_criterion(
        condition.criteria(GRID)[0],
        {
            FIRST: TimeInterval(START, START + HOUR),
            SECOND: TimeInterval(START + HOUR, START + 2 * HOUR),
        },
        GRID,
    )
    assert measured.breakdown == (DateViolationPart(1.0, date(2026, 10, 1)),)
