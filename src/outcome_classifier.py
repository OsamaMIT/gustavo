"""Manual, simulated, and simple feature-delta outcome classifiers."""

from __future__ import annotations

import random
from typing import Any, Mapping

from .bayes import Likelihoods


def sample_outcome(
    test_id: str,
    true_hypothesis: str,
    likelihoods: Likelihoods,
    rng: random.Random | None = None,
) -> str:
    """Sample a simulated test outcome from P(outcome | true hypothesis, test)."""

    random_source = rng or random.Random()
    try:
        distribution = likelihoods[test_id][true_hypothesis]
    except KeyError as exc:
        raise KeyError(
            f"Cannot sample outcome for test={test_id}, true_hypothesis={true_hypothesis}"
        ) from exc

    draw = random_source.random()
    cumulative = 0.0
    last_outcome = ""
    for outcome, probability in distribution.items():
        last_outcome = outcome
        cumulative += float(probability)
        if draw <= cumulative:
            return outcome
    return last_outcome


def classify_aero_balance_sensitivity_test(
    baseline_features: Mapping[str, Any],
    test_features: Mapping[str, Any],
) -> str:
    """Classify T1 from baseline and aero-sweep telemetry summaries."""

    baseline_index = float(baseline_features.get("understeer_index") or 0.0)
    test_index = float(test_features.get("understeer_index") or 0.0)
    baseline_temp = baseline_features.get("front_tire_temp_avg")
    test_temp = test_features.get("front_tire_temp_avg")
    improved = baseline_index > 0.0 and test_index <= baseline_index * 0.90
    temp_penalty = (
        isinstance(baseline_temp, (int, float))
        and isinstance(test_temp, (int, float))
        and test_temp >= baseline_temp + 2.0
    )
    if improved and temp_penalty:
        return "improves_balance_but_worsens_front_temps"
    if improved:
        return "improves_balance_without_temp_penalty"
    return "no_meaningful_change"


def classify_long_run_thermal_test(features: Mapping[str, Any]) -> str:
    """Classify T2 from one long-run telemetry summary."""

    trend = features.get("front_tire_temp_trend_category")
    performance_loss = features.get("performance_loss_sec_per_lap")
    if trend == "rising" and isinstance(performance_loss, (int, float)) and performance_loss >= 0.35:
        return "degradation_confirms_tire_limitation"
    if trend == "rising":
        return "temps_runaway"
    return "temps_stabilize"


def classify_platform_sensitivity_test(
    baseline_features: Mapping[str, Any],
    test_features: Mapping[str, Any],
) -> str:
    """Classify T3 from baseline and platform-perturbed telemetry summaries."""

    baseline_index = float(baseline_features.get("understeer_index") or 0.0)
    test_index = float(test_features.get("understeer_index") or 0.0)
    if baseline_index > 0.0 and abs(test_index - baseline_index) / baseline_index >= 0.12:
        return "sensitive_to_platform_change"
    return "not_sensitive_to_platform_change"


def classify_mechanical_balance_sweep(
    baseline_features: Mapping[str, Any],
    test_features: Mapping[str, Any],
) -> str:
    """Classify T4 from baseline and mechanical-sweep telemetry summaries."""

    baseline_index = float(baseline_features.get("understeer_index") or 0.0)
    test_index = float(test_features.get("understeer_index") or 0.0)
    if baseline_index > 0.0 and test_index <= baseline_index * 0.88:
        return "mechanical_change_improves_balance"
    return "no_meaningful_change"


def classify_driver_input_normalization_test(
    baseline_features: Mapping[str, Any],
    smooth_input_features: Mapping[str, Any],
) -> str:
    """Classify T5 from aggressive/baseline and smoother-input summaries."""

    baseline_index = float(baseline_features.get("understeer_index") or 0.0)
    smooth_index = float(smooth_input_features.get("understeer_index") or 0.0)
    baseline_noise = baseline_features.get("steering_noise_index")
    smooth_noise = smooth_input_features.get("steering_noise_index")

    improved = baseline_index > 0.0 and smooth_index <= baseline_index * 0.90
    smoother = (
        isinstance(baseline_noise, (int, float))
        and isinstance(smooth_noise, (int, float))
        and smooth_noise <= baseline_noise * 0.75
    )
    if improved or smoother:
        return "issue_reduced_with_smoother_inputs"
    return "issue_unchanged"
