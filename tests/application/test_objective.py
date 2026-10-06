from dataclasses import replace
from datetime import datetime, timedelta, timezone

from intent_to_schedule.application.objective import (
    ConstraintEvaluation,
    HardConstraintEvaluation,
    ScheduleSummary,
    SoftConstraintEvaluation,
    evaluate_constraints,
    summarize_schedule,
)
from intent_to_schedule.application.policy import DEFAULT_POLICY
from intent_to_schedule.domain.calendar import (
    Availability,
    Calendar,
    TimeGrid,
    TimeInterval,
)
from intent_to_schedule.domain.condition import TimeBoundCondition, TimeBoundRelation
from intent_to_schedule.domain.constraint import (
    ConstraintId,
    HardConstraint,
    SoftConstraint,
)
from intent_to_schedule.domain.measure import Boundary
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId

START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone(timedelta(hours=9)))
HOUR: timedelta = timedelta(hours=1)
PERSON: PersonId = PersonId("person")


def test_saved_duration_is_used_after_current_task_edits() -> None:
    """Evaluate the recorded interval after a movable task changes."""
    item: Task = Task(
        TaskId("task"), "task", HOUR, frozenset({PERSON}), Importance.LOW, True
    )
    previous: Schedule = Schedule(
        (
            ScheduledTask(
                item.id, item.name, START, START + item.duration, item.participant_ids
            ),
        ),
        (),
    )
    condition: TimeBoundCondition = TimeBoundCondition(
        frozenset({item.id}), Boundary.END, TimeBoundRelation.AT_OR_BEFORE, START
    )
    grid: TimeGrid = TimeGrid(TimeInterval(START, START + 4 * HOUR), HOUR)
    value: SchedulingProblem = SchedulingProblem(
        Calendar(grid, (Availability(PERSON, (grid.horizon,)),)),
        (Person(PERSON, "Person"),),
        (replace(item, duration=3 * HOUR),),
        (),
        (SoftConstraint(ConstraintId("deadline"), condition, Strength.NORMAL),),
    )
    assert (
        evaluate_constraints(value, previous, DEFAULT_POLICY)[0].violation.amount == 1.0
    )


def test_hard_evaluations_have_no_cost_and_do_not_change_soft_totals() -> None:
    """Separate hard evaluations from costed soft evaluations."""
    item: Task = Task(
        TaskId("task"), "task", HOUR, frozenset({PERSON}), Importance.LOW, True
    )
    previous: Schedule = Schedule(
        (ScheduledTask(item.id, item.name, START, START + HOUR, item.participant_ids),),
        (),
    )
    late: TimeBoundCondition = TimeBoundCondition(
        frozenset({item.id}), Boundary.END, TimeBoundRelation.AT_OR_BEFORE, START
    )
    early: TimeBoundCondition = TimeBoundCondition(
        frozenset({item.id}),
        Boundary.END,
        TimeBoundRelation.AT_OR_BEFORE,
        START + 2 * HOUR,
    )
    grid: TimeGrid = TimeGrid(TimeInterval(START, START + 4 * HOUR), HOUR)
    value: SchedulingProblem = SchedulingProblem(
        Calendar(grid, (Availability(PERSON, (grid.horizon,)),)),
        (Person(PERSON, "Person"),),
        (item,),
        (),
        (
            HardConstraint(ConstraintId("hard"), late),
            SoftConstraint(ConstraintId("soft"), late, Strength.STRONG),
            SoftConstraint(ConstraintId("satisfied"), early, Strength.NORMAL),
        ),
    )
    hard: ConstraintEvaluation
    soft: ConstraintEvaluation
    satisfied: ConstraintEvaluation
    hard, soft, satisfied = evaluate_constraints(value, previous, DEFAULT_POLICY)
    assert isinstance(hard, HardConstraintEvaluation)
    assert hard.violation.amount == 1.0
    assert not hasattr(hard, "cost") and not hasattr(hard, "coefficient")
    assert isinstance(soft, SoftConstraintEvaluation)
    assert soft.coefficient == DEFAULT_POLICY.weight(Strength.STRONG)
    assert soft.cost == DEFAULT_POLICY.weight(Strength.STRONG)
    assert isinstance(satisfied, SoftConstraintEvaluation)
    assert satisfied.cost == 0.0
    summary: ScheduleSummary = summarize_schedule(value, previous, DEFAULT_POLICY)
    assert summary.soft_constraints_cost == DEFAULT_POLICY.weight(Strength.STRONG)
    assert summary.violated_soft_constraints == 1
