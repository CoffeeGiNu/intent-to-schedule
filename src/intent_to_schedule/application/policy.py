from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance


@dataclass(frozen=True)
class ObjectivePolicy:
    """Maps importance and strength to objective coefficients."""

    drop_costs: Mapping[Importance, float]
    weights: Mapping[Strength, float]
    per_count: float
    """Penalty of one Task counted, relative to one hour."""
    stability_drop_cost_ratio: float
    """Share of a task's drop cost used as its stability cap."""

    def __post_init__(self) -> None:
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
            raise ValueError("Stability drop cost ratio must be greater than 0 and less than 1")

    def drop_cost(self, importance: Importance) -> float:
        return self.drop_costs[importance]

    def weight(self, strength: Strength) -> float:
        return self.weights[strength]


# TODO: tune all values by running the solver.
DEFAULT_POLICY = ObjectivePolicy(
    drop_costs={Importance.LOW: 5.0, Importance.MEDIUM: 20.0, Importance.HIGH: 100.0},
    weights={Strength.WEAK: 1.0, Strength.NORMAL: 5.0, Strength.STRONG: 20.0},
    per_count=1.0,
    stability_drop_cost_ratio=0.5,
)
