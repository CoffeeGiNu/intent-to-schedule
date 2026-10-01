from collections.abc import Mapping
from dataclasses import dataclass

from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.evaluate import compile_evaluation
from intent_to_schedule.adapter.mathopt.measure import (
    MeasureExpression,
    compile_measure,
)
from intent_to_schedule.application.policy import ObjectivePolicy
from intent_to_schedule.domain.calendar import (
    Availability,
    BusyInterval,
    TimeGrid,
    TimeInterval,
)
from intent_to_schedule.domain.constraint import HardConstraint, SoftConstraint
from intent_to_schedule.domain.measure import AggregateMeasure, AggregateQuantity
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Task, TaskId


@dataclass(frozen=True)
class CompiledProblem:
    """MathOpt model with the start and presence variables of each Task."""

    model: mathopt.Model
    starts: Mapping[TaskId, mathopt.Variable]
    presences: Mapping[TaskId, mathopt.Variable]
    placements: Mapping[TaskId, Mapping[int, mathopt.Variable]]


def _free_slots(person_id: PersonId, problem: SchedulingProblem, n: int) -> list[bool]:
    """Find slots available to a person."""
    grid: TimeGrid = problem.calendar.grid
    availability: list[Availability] = [
        entry
        for entry in problem.calendar.availabilities
        if entry.person_id == person_id
    ]
    free: list[bool] = [False] * n
    entry: Availability
    period: TimeInterval
    for entry in availability:
        for period in entry.intervals:
            first: int = min(
                n, max(0, (period.start - grid.horizon.start) // grid.slot)
            )
            last: int = min(n, max(0, (period.end - grid.horizon.start) // grid.slot))
            if first < last:
                free[first:last] = [True] * (last - first)

    busy: BusyInterval
    for busy in problem.calendar.busy_intervals:
        if busy.person_id != person_id:
            continue
        first = min(n, max(0, (busy.interval.start - grid.horizon.start) // grid.slot))
        last = min(n, max(0, -((grid.horizon.start - busy.interval.end) // grid.slot)))
        if first < last:
            free[first:last] = [False] * (last - first)
    return free


def compile_problem(
    problem: SchedulingProblem, policy: ObjectivePolicy
) -> CompiledProblem:
    """Build a MathOpt model from a SchedulingProblem."""
    grid: TimeGrid = problem.calendar.grid
    # TODO: decide whether to handle daylight saving time; slot arithmetic uses wall-clock time.
    n: int = (grid.horizon.end - grid.horizon.start) // grid.slot
    model: mathopt.Model = mathopt.Model()
    starts: dict[TaskId, mathopt.Variable] = {}
    presences: dict[TaskId, mathopt.Variable] = {}
    placements: dict[TaskId, dict[int, mathopt.Variable]] = {}
    durations: dict[TaskId, int] = {
        task.id: task.duration // grid.slot for task in problem.tasks
    }
    participant_ids: set[PersonId] = {
        person_id for task in problem.tasks for person_id in task.participant_ids
    }
    free_slots: dict[PersonId, list[bool]] = {
        person_id: _free_slots(person_id, problem, n) for person_id in participant_ids
    }
    objective_terms: list[mathopt.LinearExpression] = []
    incidence: dict[PersonId, dict[int, list[mathopt.Variable]]] = {
        person_id: {} for person_id in participant_ids
    }

    task: Task
    for task in problem.tasks:
        duration: int = durations[task.id]
        allowed_starts: list[int] = [
            start
            for start in range(max(0, n - duration + 1))
            if all(
                all(
                    free_slots[person_id][slot]
                    for slot in range(start, start + duration)
                )
                for person_id in task.participant_ids
            )
        ]
        choices: dict[int, mathopt.Variable] = {
            start: model.add_binary_variable(name=f"place_{task.id.value}_{start}")
            for start in allowed_starts
        }
        placements[task.id] = choices
        start: int
        variable: mathopt.Variable
        for start, variable in choices.items():
            for person_id in task.participant_ids:
                for slot in range(start, start + duration):
                    incidence[person_id].setdefault(slot, []).append(variable)
        presence: mathopt.Variable = model.add_binary_variable(
            name=f"presence_{task.id.value}"
        )
        start_variable: mathopt.Variable = model.add_variable(
            lb=0.0, ub=float(n), name=f"start_{task.id.value}"
        )
        presences[task.id] = presence
        starts[task.id] = start_variable
        model.add_linear_constraint(presence == mathopt.LinearSum(choices.values()))
        model.add_linear_constraint(
            start_variable
            == mathopt.LinearSum(
                start * variable for start, variable in choices.items()
            )
        )
        if task.required:
            model.add_linear_constraint(presence == 1)
        objective_terms.append(policy.drop_cost(task.importance) * (1 - presence))

    occupied: dict[int, list[mathopt.Variable]]
    competing: list[mathopt.Variable]
    for occupied in incidence.values():
        for competing in occupied.values():
            if len(competing) >= 2:
                model.add_linear_constraint(mathopt.LinearSum(competing) <= 1)

    constraint: HardConstraint | SoftConstraint
    strength: Strength
    for constraint in problem.constraints:
        expression: MeasureExpression = compile_measure(
            constraint.measure, problem, model, starts, presences, placements
        )
        violation: mathopt.LinearExpression = compile_evaluation(
            expression, constraint.evaluation, model, grid
        )
        match constraint:
            case HardConstraint():
                model.add_linear_constraint(violation <= 0)
            case SoftConstraint(strength=strength):
                scale: float = (
                    policy.per_count
                    if isinstance(constraint.measure, AggregateMeasure)
                    and constraint.measure.quantity is AggregateQuantity.COUNT
                    else 1.0
                )
                objective_terms.append(policy.weight(strength) * scale * violation)

    model.minimize(mathopt.LinearSum(objective_terms))
    return CompiledProblem(model, starts, presences, placements)
