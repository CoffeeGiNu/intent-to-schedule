from datetime import datetime, timedelta, timezone

import pytest

from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import (
    TaskGapCondition,
    TaskGapRelation,
    TimeBoundCondition,
    TimeBoundRelation,
)
from intent_to_schedule.domain.measure import Boundary
from intent_to_schedule.domain.task import TaskId
from intent_to_schedule.domain.violation import CriterionViolation, measure_criterion

START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone(timedelta(hours=9)))
HOUR: timedelta = timedelta(hours=1)
GRID: TimeGrid = TimeGrid(TimeInterval(START, START + 4 * HOUR), HOUR)
FIRST: TaskId = TaskId("first")
SECOND: TaskId = TaskId("second")


@pytest.mark.parametrize("boundary", list(Boundary))
@pytest.mark.parametrize("relation", list(TimeBoundRelation))
def test_criterion_time_bounds_use_exact_hours(
    boundary: Boundary, relation: TimeBoundRelation
) -> None:
    """Measure every boundary relation without rounding the bound."""
    interval: TimeInterval = TimeInterval(START, START + HOUR)
    target: datetime = START + timedelta(minutes=15)
    condition: TimeBoundCondition = TimeBoundCondition(
        frozenset({FIRST}), boundary, relation, target
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
        condition.criteria(GRID)[0], {FIRST: interval}, GRID
    )
    assert measured.amount == expected
    assert measured.unit == "hours"
    assert measured.breakdown[0].task_id == FIRST
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
