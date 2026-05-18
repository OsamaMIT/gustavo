"""Bayesian belief update utilities."""

from __future__ import annotations

import math
from typing import Mapping


Distribution = dict[str, float]
Likelihoods = Mapping[str, Mapping[str, Mapping[str, float]]]


def normalize_distribution(dist: Mapping[str, float]) -> Distribution:
    """Normalize a nonnegative distribution so probabilities sum to one."""

    if not dist:
        raise ValueError("Cannot normalize an empty distribution")

    cleaned = {key: float(value) for key, value in dist.items()}
    negative = {key: value for key, value in cleaned.items() if value < 0.0}
    if negative:
        raise ValueError(f"Distribution contains negative probabilities: {negative}")

    total = sum(cleaned.values())
    if total <= 0.0:
        raise ValueError("Distribution total must be positive")
    return {key: value / total for key, value in cleaned.items()}


def entropy(belief: Mapping[str, float]) -> float:
    """Return Shannon entropy in bits."""

    normalized = normalize_distribution(belief)
    return -sum(p * math.log2(p) for p in normalized.values() if p > 0.0)


def bayesian_update(
    belief: Mapping[str, float],
    test_id: str,
    outcome: str,
    likelihoods: Likelihoods,
) -> Distribution:
    """Update hypothesis probabilities after observing a test outcome."""

    if test_id not in likelihoods:
        raise KeyError(f"No likelihood table found for test {test_id}")

    updated: dict[str, float] = {}
    for hypothesis, prior in normalize_distribution(belief).items():
        try:
            outcome_likelihood = likelihoods[test_id][hypothesis][outcome]
        except KeyError as exc:
            raise KeyError(
                f"Missing likelihood for test={test_id}, "
                f"hypothesis={hypothesis}, outcome={outcome}"
            ) from exc
        updated[hypothesis] = prior * float(outcome_likelihood)

    return normalize_distribution(updated)

