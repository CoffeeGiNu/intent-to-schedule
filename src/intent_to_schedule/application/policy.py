from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timedelta
from math import isfinite
from types import MappingProxyType

from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task


@dataclass(frozen=True)
class ObjectivePolicy:
    """Maps importance and strength to objective coefficients."""

    drop_costs: Mapping[Importance, float]
    weights: Mapping[Strength, float]
    per_count: float
    """Penalty of one Task counted, relative to one hour."""
    stability_drop_cost_ratio: float
    """Share of a task's drop cost that its stability cost stays below."""
    hard_violation_weight: float
    """Cost of one hour or count of hard violation when explaining infeasibility."""
    required_drop_cost: float
    """Cost of dropping a required task when explaining infeasibility."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "drop_costs", MappingProxyType(dict(self.drop_costs)))
        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))
        if (
            set(self.drop_costs) != set(Importance)
            or set(self.weights) != set(Strength)
            or any(
                not isfinite(value) or value <= 0 for value in self.drop_costs.values()
            )
            or any(not isfinite(value) or value <= 0 for value in self.weights.values())
            or not isfinite(self.per_count)
            or self.per_count <= 0
        ):
            raise ValueError(
                "Objective policy coefficients must be finite and positive for every member"
            )
        if not 0 < self.stability_drop_cost_ratio < 1:
            raise ValueError(
                "Stability drop cost ratio must be greater than 0 and less than 1"
            )
        soft_maximum: float = max(
            *self.drop_costs.values(),
            *(weight * max(1.0, self.per_count) for weight in self.weights.values()),
        )
        if not all(
            isfinite(value) and value > soft_maximum
            for value in (self.hard_violation_weight, self.required_drop_cost)
        ):
            raise ValueError(
                "Relaxation costs must be finite and greater than every soft cost"
            )

    def drop_cost(self, importance: Importance) -> float:
        return self.drop_costs[importance]

    def weight(self, strength: Strength) -> float:
        return self.weights[strength]

    def stability_cost(self, task: Task, moved: timedelta) -> float:
        """Cost of moving a task from its previous start by the given time."""
        weighted: float = self.weight(task.stability) * abs(moved / timedelta(hours=1))
        limit: float = self.stability_drop_cost_ratio * self.drop_cost(task.importance)
        return weighted / (1 + weighted / limit)


# TODO: tune all values by running the solver.
DEFAULT_POLICY = ObjectivePolicy(
    drop_costs={Importance.LOW: 5.0, Importance.MEDIUM: 20.0, Importance.HIGH: 100.0},
    weights={Strength.WEAK: 1.0, Strength.NORMAL: 5.0, Strength.STRONG: 20.0},
    per_count=1.0,
    stability_drop_cost_ratio=0.5,
    hard_violation_weight=1000.0,
    required_drop_cost=1000000.0,
)
