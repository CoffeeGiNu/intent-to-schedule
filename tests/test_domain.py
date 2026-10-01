from datetime import datetime, timedelta

from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.compatibility import is_supported
from intent_to_schedule.domain.consistency import (
    AlignedToSlots,
    AllOf,
    AvailabilityForEveryone,
    ReferencesExist,
    SupportedCombinations,
    UniqueIds,
    Violation,
    Violations,
)
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint
from intent_to_schedule.domain.evaluation import Distance, Excess, Intrusion
from intent_to_schedule.domain.measure import (
    AggregateMeasure,
    AggregateQuantity,
    DependencyMeasure,
    PointMeasure,
)
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import Importance, Task, TaskId


def make_problem() -> SchedulingProblem:
    start: datetime = datetime(2026, 1, 1)
    person_id: PersonId = PersonId("p1")
    task_id: TaskId = TaskId("t1")
    task: Task = Task(task_id, "Task", timedelta(minutes=30), frozenset({person_id}), Importance.MEDIUM, True)
    return SchedulingProblem(
        Calendar(TimeGrid(TimeInterval(start, start + timedelta(hours=4)), timedelta(minutes=30)), (), ()),
        (Person(person_id, "Person"),),
        (task,),
        (),
    )


def test_supported_combinations_check_types_and_quantity() -> None:
    task_id: TaskId = TaskId("t")
    assert is_supported(PointMeasure(task_id), Distance(datetime(2026, 1, 1)))
    assert is_supported(DependencyMeasure(task_id, TaskId("u")), Distance(timedelta(hours=1)))
    assert is_supported(AggregateMeasure(frozenset({task_id}), AggregateQuantity.COUNT), Excess(upper=3))
    assert not is_supported(AggregateMeasure(frozenset({task_id}), AggregateQuantity.COUNT), Excess(upper=True))
    assert not is_supported(PointMeasure(task_id), Intrusion(()))


def test_validators_report_duplicate_missing_and_unaligned_values() -> None:
    problem: SchedulingProblem = make_problem()
    task: Task = problem.tasks[0]
    duplicate: Task = Task(task.id, "Duplicate", task.duration, task.participant_ids, task.importance, task.required)
    missing_task_constraint: HardConstraint = HardConstraint(
        ConstraintId("c1"), PointMeasure(TaskId("missing")), Distance(datetime(2026, 1, 1))
    )
    bad_constraint: HardConstraint = HardConstraint(
        ConstraintId("c2"), PointMeasure(task.id), Excess(timedelta(minutes=15))
    )
    invalid: SchedulingProblem = SchedulingProblem(
        problem.calendar,
        problem.people,
        (task, duplicate),
        (missing_task_constraint, bad_constraint),
    )
    assert not UniqueIds().validate(invalid).is_empty
    references: Violations = ReferencesExist().validate(invalid)
    assert len(references.items) == 1
    assert "missing" in references.items[0].message
    aligned: Violations = AlignedToSlots().validate(invalid)
    assert len(aligned.items) == 1
    assert "c2" in aligned.items[0].message
    assert len(SupportedCombinations().validate(invalid).items) == 1
    assert Violations(()).merge(references).items == references.items


def test_availability_for_everyone_reports_person_without_availability() -> None:
    problem: SchedulingProblem = make_problem()
    violations: Violations = AvailabilityForEveryone().validate(problem)
    assert len(violations.items) == 1
    assert "p1" in violations.items[0].message


def test_all_of_merges_violations_and_nests() -> None:
    original: SchedulingProblem = make_problem()
    problem: SchedulingProblem = SchedulingProblem(
        original.calendar, original.people, (*original.tasks, original.tasks[0]), original.constraints
    )
    validator: AllOf = AllOf(ReferencesExist(), AllOf(AvailabilityForEveryone(), UniqueIds()))
    assert validator.validate(problem) == Violations((
        Violation("Person p1 has no availability."),
        Violation("Duplicate Task id t1."),
    ))
