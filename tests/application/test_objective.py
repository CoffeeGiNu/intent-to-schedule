from dataclasses import replace
from datetime import datetime, timedelta, timezone

from intent_to_schedule.application.objective import evaluate_constraints
from intent_to_schedule.application.policy import DEFAULT_POLICY
from intent_to_schedule.domain.calendar import (
    Availability,
    Calendar,
    TimeGrid,
    TimeInterval,
)
from intent_to_schedule.domain.condition import TimeBoundCondition, TimeBoundRelation
from intent_to_schedule.domain.constraint import ConstraintId, SoftConstraint
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
