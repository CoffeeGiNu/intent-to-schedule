from collections.abc import Mapping
from dataclasses import dataclass

from ortools.math_opt.python import mathopt

from intent_to_schedule.adapter.mathopt.evaluate import compile_evaluation
from intent_to_schedule.adapter.mathopt.measure import (
    MeasureExpression,
    compile_measure,
)
from intent_to_schedule.application.policy import ObjectivePolicy
from intent_to_schedule.domain.availability import available_start_slots, free_slots
from intent_to_schedule.domain.calendar import TimeGrid, TimeInterval
from intent_to_schedule.domain.constraint import HardConstraint, SoftConstraint
from intent_to_schedule.domain.measure import AggregateMeasure, AggregateQuantity
from intent_to_schedule.domain.person import PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Task, TaskId


@dataclass(frozen=True)
class CompiledProblem:
    """MathOpt model with the start and presence variables of each Task."""

    model: mathopt.Model
    starts: Mapping[TaskId, mathopt.Variable]
    presences: Mapping[TaskId, mathopt.Variable]
    placements: Mapping[TaskId, Mapping[int, mathopt.Variable]]


def compile_problem(
    problem: SchedulingProblem, policy: ObjectivePolicy
) -> CompiledProblem:
    """Build a MathOpt model from a SchedulingProblem."""
    grid: TimeGrid = problem.calendar.grid
    slot_count: int = grid.slot_count
    model: mathopt.Model = mathopt.Model()
    starts: dict[TaskId, mathopt.Variable] = {}
    presences: dict[TaskId, mathopt.Variable] = {}
    placements: dict[TaskId, dict[int, mathopt.Variable]] = {}
    durations: dict[TaskId, int] = {
        task.id: grid.index_of(grid.horizon.start + task.duration) for task in problem.tasks
    }
    participant_ids: set[PersonId] = {
        person_id for task in problem.tasks for person_id in task.participant_ids
    }
    people_free: dict[PersonId, tuple[bool, ...]] = {
        person_id: free_slots(problem, person_id) for person_id in participant_ids
    }
    objective_terms: list[mathopt.LinearBase] = []
    incidence: dict[PersonId, dict[int, list[mathopt.Variable]]] = {
        person_id: {} for person_id in participant_ids
    }

    task: Task
    person_id: PersonId
    slot: int
    for task in problem.tasks:
        duration: int = durations[task.id]
        allowed_starts: tuple[int, ...] = available_start_slots(
            grid,
            tuple(people_free[person_id] for person_id in task.participant_ids),
            task.duration,
        )
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
            lb=0.0, ub=float(slot_count), name=f"start_{task.id.value}"
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

    fixed_task: FixedTask
    for fixed_task in problem.fixed_tasks:
        rounded: TimeInterval = grid.round_outward(
            TimeInterval(fixed_task.start, fixed_task.start + fixed_task.duration)
        )
        start = grid.index_of(rounded.start)
        variable = model.add_variable(
            lb=1.0, ub=1.0, is_integer=True, name=f"place_{fixed_task.id.value}_{start}"
        )
        placements[fixed_task.id] = {start: variable}
        presences[fixed_task.id] = model.add_variable(
            lb=1.0, ub=1.0, is_integer=True, name=f"presence_{fixed_task.id.value}"
        )
        starts[fixed_task.id] = model.add_variable(
            lb=float(start), ub=float(start), name=f"start_{fixed_task.id.value}"
        )

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
        violation: mathopt.LinearBase = compile_evaluation(
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
