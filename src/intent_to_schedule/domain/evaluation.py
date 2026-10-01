from dataclasses import dataclass
from datetime import datetime, timedelta

from intent_to_schedule.domain.calendar import TimeInterval

type Quantity = datetime | timedelta | int
"""Value a Measure takes."""


@dataclass(frozen=True)
class Distance:
    """Absolute difference from a target value."""

    target: Quantity


@dataclass(frozen=True)
class Intrusion:
    """Length of overlap with a region."""

    region: tuple[TimeInterval, ...]


@dataclass(frozen=True)
class Shortfall:
    """Amount below a lower bound."""

    lower: Quantity


@dataclass(frozen=True)
class Excess:
    """Amount above an upper bound."""

    upper: Quantity


type Evaluation = Distance | Intrusion | Shortfall | Excess
"""Rule that turns a measured value into an evaluation amount."""
