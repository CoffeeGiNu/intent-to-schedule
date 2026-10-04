from dataclasses import replace
from datetime import datetime, time, timedelta

import pytest

from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.condition import (
    Condition,
    Criterion,
    DailyLimitCondition,
    TaskGapCondition,
    TaskGapRelation,
    TimeBoundCondition,
    TimeBoundRelation,
    TimeWindowCondition,
)
from intent_to_schedule.domain.evaluation import (
    Distance,
    Evaluation,
    Excess,
    Intrusion,
    Shortfall,
)
from intent_to_schedule.domain.measure import (
    AggregateMeasure,
    AggregateQuantity,
    Boundary,
    DependencyMeasure,
    IntervalMeasure,
    PointMeasure,
)
from intent_to_schedule.domain.task import TaskId
from intent_to_schedule.domain.time_windows import TimeRange, TimeRelation, TimeWindow

START: datetime = datetime(2026, 10, 1, 9)
GRID: TimeGrid = TimeGrid(
    TimeInterval(START, START + timedelta(hours=4)), timedelta(hours=1)
)
FIRST: TaskId = TaskId("first")
SECOND: TaskId = TaskId("second")
TASK_IDS: frozenset[TaskId] = frozenset({FIRST, SECOND})


@pytest.mark.parametrize(
    "relation,region",
    [
        (
            TimeRelation.WITHIN,
            (
                TimeInterval(START, START + timedelta(hours=1)),
                TimeInterval(START + timedelta(hours=2), GRID.horizon.end),
            ),
        ),
        (TimeRelation.AVOID, (TimeInterval(START, START + timedelta(hours=3)),)),
    ],
)
def test_time_window_criteria_round_inward_or_outward(
    relation: TimeRelation, region: tuple[TimeInterval, ...]
) -> None:
    condition: TimeWindowCondition = TimeWindowCondition(
        TASK_IDS,
        relation,
        (TimeWindow(None, None, TimeRange(time(9, 10), time(11, 10))),),
    )
    assert condition.criteria(GRID) == (
        Criterion(IntervalMeasure(TASK_IDS), Intrusion(region)),
    )


@pytest.mark.parametrize("relation", list(TimeRelation))
def test_empty_window_criteria(relation: TimeRelation) -> None:
    condition: TimeWindowCondition = TimeWindowCondition(TASK_IDS, relation, ())
    assert condition.criteria(GRID) == (
        Criterion(
            IntervalMeasure(TASK_IDS),
            Intrusion((GRID.horizon,) if relation is TimeRelation.WITHIN else ()),
        ),
    )


@pytest.mark.parametrize("boundary", list(Boundary))
@pytest.mark.parametrize(
    "relation,evaluation",
    [
        (TimeBoundRelation.AT_OR_BEFORE, Excess(START + timedelta(minutes=75))),
        (TimeBoundRelation.AT_OR_AFTER, Shortfall(START + timedelta(minutes=75))),
        (TimeBoundRelation.AT, Distance(START + timedelta(minutes=75))),
    ],
)
def test_time_bound_criteria(
    boundary: Boundary, relation: TimeBoundRelation, evaluation: Evaluation
) -> None:
    condition: TimeBoundCondition = TimeBoundCondition(
        TASK_IDS, boundary, relation, START + timedelta(minutes=75)
    )
    criteria: tuple[Criterion, ...] = condition.criteria(GRID)
    assert criteria == tuple(
        Criterion(PointMeasure(task_id, boundary), evaluation)
        for task_id in (FIRST, SECOND)
    )


def test_point_boundary_defaults_to_start() -> None:
    assert PointMeasure(FIRST).boundary is Boundary.START


@pytest.mark.parametrize(
    "relation,evaluation",
    [
        (TaskGapRelation.AT_LEAST, Shortfall(timedelta(minutes=75))),
        (TaskGapRelation.EXACTLY, Distance(timedelta(minutes=75))),
    ],
)
def test_task_gap_criteria(relation: TaskGapRelation, evaluation: Evaluation) -> None:
    condition: TaskGapCondition = TaskGapCondition(
        FIRST, SECOND, relation, timedelta(minutes=75)
    )
    assert condition.task_ids == TASK_IDS
    assert condition.criteria(GRID) == (
        Criterion(DependencyMeasure(FIRST, SECOND), evaluation),
    )


@pytest.mark.parametrize(
    "quantity,maximum",
    [
        (AggregateQuantity.COUNT, 2),
        (AggregateQuantity.TOTAL_DURATION, timedelta(minutes=75)),
    ],
)
def test_daily_limit_criteria(
    quantity: AggregateQuantity, maximum: int | timedelta
) -> None:
    condition: DailyLimitCondition = DailyLimitCondition(TASK_IDS, quantity, maximum)
    assert condition.criteria(GRID) == (
        Criterion(AggregateMeasure(TASK_IDS, quantity), Excess(maximum)),
    )


@pytest.mark.parametrize(
    "condition",
    [
        TimeWindowCondition(TASK_IDS, TimeRelation.WITHIN, ()),
        TimeBoundCondition(TASK_IDS, Boundary.END, TimeBoundRelation.AT, START),
        DailyLimitCondition(TASK_IDS, AggregateQuantity.COUNT, 0),
        DailyLimitCondition(TASK_IDS, AggregateQuantity.TOTAL_DURATION, timedelta(0)),
    ],
)
def test_without_task_preserves_fields_and_removes_empty_conditions(
    condition: Condition,
) -> None:
    remaining: Condition | None = condition.without_task(FIRST)
    assert remaining == replace(condition, task_ids=frozenset({SECOND}))
    assert remaining is not None
    assert remaining.without_task(SECOND) is None
    assert condition.without_task(TaskId("other")) == condition
    assert condition.task_ids == TASK_IDS


@pytest.mark.parametrize("relation", list(TaskGapRelation))
def test_task_gap_without_task(relation: TaskGapRelation) -> None:
    condition: TaskGapCondition = TaskGapCondition(
        FIRST, SECOND, relation, timedelta(0)
    )
    assert condition.without_task(FIRST) is None
    assert condition.without_task(SECOND) is None
    assert condition.without_task(TaskId("other")) == condition


def test_conditions_reject_empty_task_sets() -> None:
    with pytest.raises(ValueError, match="at least one"):
        TimeWindowCondition(frozenset(), TimeRelation.AVOID, ())
    with pytest.raises(ValueError, match="at least one"):
        TimeBoundCondition(frozenset(), Boundary.START, TimeBoundRelation.AT, START)
    with pytest.raises(ValueError, match="at least one"):
        DailyLimitCondition(frozenset(), AggregateQuantity.COUNT, 0)


def test_task_gap_rejects_negative_duration() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TaskGapCondition(
            FIRST, SECOND, TaskGapRelation.AT_LEAST, -timedelta(microseconds=1)
        )


@pytest.mark.parametrize("maximum", [-1, True, 1.5, timedelta(0)])
def test_daily_count_rejects_invalid_maximum(maximum: int | float | timedelta) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        DailyLimitCondition(TASK_IDS, AggregateQuantity.COUNT, maximum)  # type: ignore[arg-type]


@pytest.mark.parametrize("maximum", [-timedelta(microseconds=1), 0])
def test_daily_duration_rejects_invalid_maximum(maximum: int | timedelta) -> None:
    with pytest.raises(ValueError, match="non-negative duration"):
        DailyLimitCondition(TASK_IDS, AggregateQuantity.TOTAL_DURATION, maximum)
