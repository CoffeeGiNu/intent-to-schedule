from collections.abc import Mapping
from dataclasses import dataclass

from ortools.math_opt.python import mathopt

from intent_to_schedule.application.policy import ObjectivePolicy
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.task import TaskId


@dataclass(frozen=True)
class CompiledProblem:
    """MathOpt model with the start and presence variables of each Task."""

    model: mathopt.Model
    starts: Mapping[TaskId, mathopt.Variable]
    presences: Mapping[TaskId, mathopt.Variable]


def compile_problem(problem: SchedulingProblem, policy: ObjectivePolicy) -> CompiledProblem:
    """Build a MathOpt model from a SchedulingProblem."""
    ...
