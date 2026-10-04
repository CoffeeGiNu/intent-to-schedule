import uuid
from dataclasses import dataclass
from typing import Self

from intent_to_schedule.domain.condition import Condition
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
    condition: Condition
    label: str | None = None


@dataclass(frozen=True)
class SoftConstraint:
    """Constraint penalized in the objective."""

    id: ConstraintId
    condition: Condition
    strength: Strength
    label: str | None = None


type Constraint = HardConstraint | SoftConstraint
"""Hard or soft constraint."""
