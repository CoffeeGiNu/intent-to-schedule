from dataclasses import dataclass

from intent_to_schedule.domain.calendar import Calendar
from intent_to_schedule.domain.constraint import Constraint
from intent_to_schedule.domain.person import Person
from intent_to_schedule.domain.task import FixedTask, Task


@dataclass(frozen=True)
class SchedulingProblem:
    """Scheduling problem with calendar, people, tasks, and constraints."""

    calendar: Calendar
    people: tuple[Person, ...]
    tasks: tuple[Task, ...]
    fixed_tasks: tuple[FixedTask, ...]
    constraints: tuple[Constraint, ...]
