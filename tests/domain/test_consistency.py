from dataclasses import replace
from datetime import datetime, timedelta

from intent_to_schedule.domain.calendar import (
    Availability,
    Calendar,
    TimeGrid,
    TimeInterval,
)
from intent_to_schedule.domain.condition import (
    TimeBoundCondition,
    TimeBoundRelation,
    TimeWindowCondition,
)
from intent_to_schedule.domain.consistency import (
    AlignedToSlots,
    AllOf,
    AvailabilityForEveryone,
    ReferencesExist,
    UniqueIds,
    Violation,
    Violations,
)
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint
from intent_to_schedule.domain.measure import Boundary
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId
from intent_to_schedule.domain.time_windows import TimeRelation, TimeWindow


def make_problem() -> SchedulingProblem:
    start: datetime = datetime(2026, 1, 1)
    person_id: PersonId = PersonId("p1")
    task_id: TaskId = TaskId("t1")
    task: Task = Task(
        task_id,
        "Task",
        timedelta(minutes=30),
        frozenset({person_id}),
        Importance.MEDIUM,
        True,
    )
    return SchedulingProblem(
        Calendar(
            TimeGrid(
                TimeInterval(start, start + timedelta(hours=4)), timedelta(minutes=30)
            ),
            (),
        ),
        (Person(person_id, "Person"),),
        (task,),
        (),
        (),
    )


def test_validators_report_duplicate_missing_and_unaligned_values() -> None:
    problem: SchedulingProblem = make_problem()
    task: Task = problem.tasks[0]
    duplicate: Task = Task(
        task.id,
        "Duplicate",
        task.duration,
        task.participant_ids,
        task.importance,
        task.required,
    )
    missing_task_constraint: HardConstraint = HardConstraint(
        ConstraintId("c1"),
        TimeBoundCondition(
            frozenset({TaskId("missing")}),
            Boundary.START,
            TimeBoundRelation.AT,
            datetime(2026, 1, 1),
        ),
    )
    unaligned_task: Task = replace(duplicate, duration=timedelta(minutes=15))
    invalid: SchedulingProblem = SchedulingProblem(
        problem.calendar,
        problem.people,
        (task, unaligned_task),
        (),
        (missing_task_constraint,),
    )
    assert not UniqueIds().validate(invalid).is_empty
    references: Violations = ReferencesExist().validate(invalid)
    assert len(references.items) == 1
    assert "missing" in references.items[0].message
    aligned: Violations = AlignedToSlots().validate(invalid)
    assert len(aligned.items) == 1
    assert "t1" in aligned.items[0].message
    assert Violations(()).merge(references).items == references.items


def test_availability_for_everyone_reports_person_without_availability() -> None:
    problem: SchedulingProblem = make_problem()
    violations: Violations = AvailabilityForEveryone().validate(problem)
    assert len(violations.items) == 1
    assert "p1" in violations.items[0].message


def test_all_of_merges_violations_and_nests() -> None:
    original: SchedulingProblem = make_problem()
    problem: SchedulingProblem = SchedulingProblem(
        original.calendar,
        original.people,
        (*original.tasks, original.tasks[0]),
        original.fixed_tasks,
        original.constraints,
    )
    validator: AllOf = AllOf(
        ReferencesExist(), AllOf(AvailabilityForEveryone(), UniqueIds())
    )
    assert validator.validate(problem) == Violations(
        (
            Violation("Person p1 has no availability."),
            Violation("Duplicate Task id t1."),
        )
    )


def test_unique_ids_checks_tasks_and_fixed_tasks_together() -> None:
    problem: SchedulingProblem = make_problem()
    task: Task = problem.tasks[0]
    fixed: FixedTask = FixedTask(
        task.id,
        "Existing",
        problem.calendar.grid.horizon.start,
        task.duration,
        task.participant_ids,
    )
    assert UniqueIds().validate(replace(problem, fixed_tasks=(fixed,))) == Violations(
        (Violation("Duplicate Task id t1."),)
    )
    assert (
        not UniqueIds()
        .validate(replace(problem, tasks=(), fixed_tasks=(fixed, fixed)))
        .is_empty
    )


def test_fixed_task_references_and_off_grid_times() -> None:
    problem: SchedulingProblem = make_problem()
    fixed: FixedTask = FixedTask(
        TaskId("fixed"),
        "Existing",
        datetime(2026, 1, 1, 0, 15),
        timedelta(minutes=10),
        frozenset({PersonId("p1")}),
    )
    constraint: HardConstraint = HardConstraint(
        ConstraintId("fixed"),
        TimeBoundCondition(
            frozenset({fixed.id}),
            Boundary.START,
            TimeBoundRelation.AT,
            datetime(2026, 1, 1),
        ),
    )
    problem = replace(problem, fixed_tasks=(fixed,), constraints=(constraint,))
    assert ReferencesExist().validate(problem).is_empty
    assert AlignedToSlots().validate(problem).is_empty
    missing: FixedTask = replace(
        fixed, participant_ids=frozenset({PersonId("missing")})
    )
    assert ReferencesExist().validate(
        replace(problem, fixed_tasks=(missing,))
    ) == Violations((Violation("Task fixed references missing person id missing."),))


def test_alignment_uses_grid_origin_for_availability_and_durations() -> None:
    original: SchedulingProblem = make_problem()
    origin: datetime = original.calendar.grid.horizon.start + timedelta(minutes=10)
    grid: TimeGrid = TimeGrid(
        TimeInterval(origin, origin + timedelta(hours=4)), timedelta(minutes=30)
    )
    calendar: Calendar = Calendar(
        grid, (Availability(original.people[0].id, (grid.horizon,)),)
    )
    aligned: SchedulingProblem = replace(original, calendar=calendar)
    assert AlignedToSlots().validate(aligned).is_empty
    availability: Availability = Availability(
        original.people[0].id,
        (TimeInterval(origin + timedelta(minutes=1), grid.horizon.end),),
    )
    unaligned: SchedulingProblem = replace(
        aligned,
        calendar=replace(calendar, availabilities=(availability,)),
        tasks=(replace(original.tasks[0], duration=timedelta(minutes=15)),),
    )
    assert AlignedToSlots().validate(unaligned) == Violations(
        (
            Violation("Task t1 duration is not aligned to the time grid."),
            Violation(
                "Availability for person p1 start is not aligned to the time grid."
            ),
        )
    )


def test_references_exist_reports_every_missing_task_of_a_condition() -> None:
    """Validate every Task referenced by one condition."""
    problem: SchedulingProblem = make_problem()
    constraint: HardConstraint = HardConstraint(
        ConstraintId("combined"),
        TimeWindowCondition(
            frozenset(
                {problem.tasks[0].id, TaskId("missing-one"), TaskId("missing-two")}
            ),
            TimeRelation.WITHIN,
            (TimeWindow(None, None, None),),
        ),
    )
    given: SchedulingProblem = replace(problem, constraints=(constraint,))
    assert {item.message for item in ReferencesExist().validate(given).items} == {
        "Constraint combined references missing task id missing-one.",
        "Constraint combined references missing task id missing-two.",
    }


def test_aligned_to_slots_rejects_negative_fixed_duration() -> None:
    """Reject negative fixed durations and accept zero."""
    problem: SchedulingProblem = make_problem()
    fixed: FixedTask = FixedTask(
        TaskId("fixed"),
        "Existing",
        problem.calendar.grid.horizon.start,
        timedelta(minutes=-1),
        frozenset(),
    )
    assert AlignedToSlots().validate(
        replace(problem, fixed_tasks=(fixed,))
    ) == Violations((Violation("Fixed task fixed duration must not be negative."),))
    assert (
        AlignedToSlots()
        .validate(
            replace(problem, fixed_tasks=(replace(fixed, duration=timedelta(0)),))
        )
        .is_empty
    )
