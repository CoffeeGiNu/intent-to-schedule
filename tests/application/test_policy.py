from collections.abc import MutableMapping
from dataclasses import replace
from typing import cast

import pytest

from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance


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
