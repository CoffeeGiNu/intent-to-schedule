from collections.abc import Mapping
from dataclasses import dataclass

from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance


@dataclass(frozen=True)
class ObjectivePolicy:
    """Maps importance and strength to objective coefficients."""

    drop_costs: Mapping[Importance, float]
    weights: Mapping[Strength, float]

    def drop_cost(self, importance: Importance) -> float: ...

    def weight(self, strength: Strength) -> float: ...


DEFAULT_POLICY = ObjectivePolicy(
    drop_costs={Importance.LOW: 5.0, Importance.MEDIUM: 20.0, Importance.HIGH: 100.0},
    weights={Strength.WEAK: 1.0, Strength.NORMAL: 5.0, Strength.STRONG: 20.0},
)
