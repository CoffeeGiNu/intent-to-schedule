from dataclasses import dataclass
from typing import Self
import uuid

from intent_to_schedule.domain.evaluation import Evaluation
from intent_to_schedule.domain.measure import Measure
from intent_to_schedule.domain.strength import Strength


@dataclass(frozen=True)
class ConstraintId:
    """Identifier of a constraint."""

    value: str

    @classmethod
    def generate(cls) -> Self:
        """Create a new unique ConstraintId."""
        return cls(uuid.uuid4().hex)


@dataclass(frozen=True)
class HardConstraint:
    """Constraint that must be satisfied."""

    id: ConstraintId
    measure: Measure
    evaluation: Evaluation


@dataclass(frozen=True)
class SoftConstraint:
    """Constraint penalized in the objective."""

    id: ConstraintId
    measure: Measure
    evaluation: Evaluation
    strength: Strength


type Constraint = HardConstraint | SoftConstraint
"""Hard or soft constraint."""
