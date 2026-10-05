from collections.abc import MutableMapping
from dataclasses import replace
from datetime import timedelta
from itertools import pairwise
from typing import cast

import pytest

from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId


def test_policy_copies_input_mappings() -> None:
    """Preserve validated coefficients when the original dictionaries change."""
    drop_costs: dict[Importance, float] = dict(DEFAULT_POLICY.drop_costs)
    weights: dict[Strength, float] = dict(DEFAULT_POLICY.weights)
    policy: ObjectivePolicy = replace(
        DEFAULT_POLICY, drop_costs=drop_costs, weights=weights
    )
    drop_costs[Importance.HIGH] = -1.0
    weights[Strength.WEAK] = -1.0
    assert policy == DEFAULT_POLICY
    assert replace(policy) == policy


@pytest.mark.parametrize("use_default", [False, True])
def test_policy_mappings_are_read_only(use_default: bool) -> None:
    """Reject direct coefficient mutations in default and replaced policies."""
    policy: ObjectivePolicy = DEFAULT_POLICY if use_default else replace(DEFAULT_POLICY)
    drop_costs: MutableMapping[Importance, float] = cast(
        MutableMapping[Importance, float], policy.drop_costs
    )
    weights: MutableMapping[Strength, float] = cast(
        MutableMapping[Strength, float], policy.weights
    )
    with pytest.raises(TypeError):
        drop_costs[Importance.HIGH] = 100.0
    with pytest.raises(TypeError):
        weights[Strength.WEAK] = 1.0


@pytest.mark.parametrize("ratio", [0.0, 1.0, -0.5, 1.5, float("inf"), float("nan")])
def test_policy_rejects_invalid_stability_drop_cost_ratio(ratio: float) -> None:
    with pytest.raises(ValueError):
        replace(DEFAULT_POLICY, stability_drop_cost_ratio=ratio)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_policy_rejects_nonpositive_or_nonfinite_coefficients(value: float) -> None:
    with pytest.raises(ValueError):
        replace(DEFAULT_POLICY, per_count=value)
    with pytest.raises(ValueError):
        replace(
            DEFAULT_POLICY,
            drop_costs={**DEFAULT_POLICY.drop_costs, Importance.LOW: value},
        )
    with pytest.raises(ValueError):
        replace(
            DEFAULT_POLICY, weights={**DEFAULT_POLICY.weights, Strength.WEAK: value}
        )


def test_policy_requires_every_importance_and_strength() -> None:
    with pytest.raises(ValueError):
        ObjectivePolicy({Importance.LOW: 1.0}, DEFAULT_POLICY.weights, 1.0, 0.5)
    with pytest.raises(ValueError):
        ObjectivePolicy(DEFAULT_POLICY.drop_costs, {Strength.WEAK: 1.0}, 1.0, 0.5)


def movable(importance: Importance, stability: Strength) -> Task:
    """Build an optional task with the given importance and stability."""
    return Task(
        TaskId("task"),
        "Task",
        timedelta(hours=1),
        frozenset(),
        importance,
        False,
        stability,
    )


@pytest.mark.parametrize(
    ("hours", "expected"),
    [(0.0, 0.0), (1.0, 50 / 11), (10.0, 25.0), (24.0, 600 / 17), (120.0, 600 / 13)],
)
def test_stability_cost_matches_examples(hours: float, expected: float) -> None:
    """Charge a normal-stability high-importance task by hours moved, either way."""
    item: Task = movable(Importance.HIGH, Strength.NORMAL)
    moved: timedelta = timedelta(hours=hours)
    assert DEFAULT_POLICY.stability_cost(item, moved) == pytest.approx(expected)
    assert DEFAULT_POLICY.stability_cost(item, -moved) == pytest.approx(expected)


@pytest.mark.parametrize("importance", list(Importance))
@pytest.mark.parametrize("stability", list(Strength))
def test_stability_cost_increases_below_its_limit(
    importance: Importance, stability: Strength
) -> None:
    """Grow strictly with hours moved while staying below the limit."""
    item: Task = movable(importance, stability)
    limit: float = DEFAULT_POLICY.stability_drop_cost_ratio * DEFAULT_POLICY.drop_cost(
        importance
    )
    costs: list[float] = [
        DEFAULT_POLICY.stability_cost(item, timedelta(hours=hours))
        for hours in (0.0, 0.5, 1.0, 12.0, 120.0, 1_000.0, 100_000.0)
    ]
    assert all(earlier < later for earlier, later in pairwise(costs))
    assert costs[-1] < limit


@pytest.mark.parametrize("field", ["hard_violation_weight", "required_drop_cost"])
@pytest.mark.parametrize("value", [100.0, float("inf"), float("nan")])
def test_policy_requires_relaxation_costs_above_every_soft_cost(
    field: str, value: float
) -> None:
    """Reject relaxation costs that do not exceed every soft coefficient."""
    with pytest.raises(ValueError):
        replace(DEFAULT_POLICY, **{field: value})


def test_policy_compares_relaxation_costs_with_scaled_count_weights() -> None:
    """Reject a per count scale that makes a soft count cost exceed relaxation."""
    with pytest.raises(ValueError):
        replace(DEFAULT_POLICY, per_count=60.0)
