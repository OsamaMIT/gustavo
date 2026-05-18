"""Initial rule-based hypothesis scoring from telemetry features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .bayes import normalize_distribution


SUPPORTED_HYPOTHESES = (
    "front_aero_load_limitation",
    "rear_aero_load_limitation",
    "aero_balance_forward_shift",
    "aero_balance_rearward_shift",
    "excess_drag_configuration",
    "platform_aero_sensitivity",
    "ride_height_sensitivity",
    "pitch_sensitivity",
    "roll_sensitivity",
    "bottoming_induced_aero_loss",
    "kerb_platform_disturbance",
    "front_tire_thermal_saturation",
    "rear_tire_thermal_saturation",
    "front_tire_wear_limitation",
    "rear_tire_wear_limitation",
    "tire_pressure_window_issue",
    "compound_window_mismatch",
    "mechanical_balance_limitation",
    "front_mechanical_grip_limitation",
    "rear_mechanical_grip_limitation",
    "arb_stiffness_mismatch",
    "spring_damper_mismatch",
    "camber_toe_limitation",
    "brake_balance_too_forward",
    "brake_balance_too_rearward",
    "brake_migration_issue",
    "brake_temperature_issue",
    "engine_braking_instability",
    "diff_entry_instability",
    "diff_mid_corner_rotation_limitation",
    "diff_exit_traction_limitation",
    "power_delivery_too_aggressive",
    "traction_limitation",
    "power_unit_deployment_limitation",
    "ers_deployment_strategy_issue",
    "gear_ratio_mismatch",
    "drag_limited_straight_speed",
    "driver_input_contribution",
    "excessive_steering_energy",
    "late_brake_release",
    "aggressive_throttle_application",
    "inconsistent_line_selection",
    "track_evolution_effect",
    "fuel_load_effect",
    "wind_sensitivity",
    "sensor_noise_or_data_quality_issue",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRIOR_WEIGHTS_PATH = PROJECT_ROOT / "config" / "prior_weights.json"

DEFAULT_PRIOR_WEIGHTS: dict[str, Any] = {
    "candidate_rank_weights": {"1": 8.0, "2": 3.0, "3": 1.4},
    "candidate_position_weights": {
        "1": 1.25,
        "2": 0.9,
        "3": 0.7,
        "4": 0.55,
        "5": 0.45,
        "6": 0.35,
        "7": 0.28,
        "8": 0.22,
    },
    "background_base_weight": 0.005,
    "default_candidate_weight": 3.0,
    "generic_feature_boost_scale": 0.55,
}


def load_prior_weights(path: Path | None = None) -> dict[str, Any]:
    """Load configurable prior weights used by symptom-conditioned belief scoring."""

    weights = dict(DEFAULT_PRIOR_WEIGHTS)
    weight_path = path or PRIOR_WEIGHTS_PATH
    if weight_path.exists():
        with weight_path.open("r", encoding="utf-8") as file:
            loaded = json.load(file)
        for key, value in loaded.items():
            weights[key] = value
    return weights


def _hypothesis_ids(hypotheses: Iterable[str] | Mapping[str, Any] | None) -> list[str]:
    if hypotheses is None:
        return list(SUPPORTED_HYPOTHESES)
    if isinstance(hypotheses, Mapping):
        return [str(key) for key in hypotheses.keys()]
    return [str(item) for item in hypotheses]


def uniform_belief(hypotheses: Iterable[str] | Mapping[str, Any] | None = None) -> dict[str, float]:
    """Return a uniform belief distribution over hypotheses."""

    ids = _hypothesis_ids(hypotheses)
    return normalize_distribution({hypothesis: 1.0 for hypothesis in ids})


def score_initial_belief(
    features: Mapping[str, Any],
    hypotheses: Iterable[str] | Mapping[str, Any] | None = None,
    candidate_hypotheses: Iterable[str] | None = None,
    candidate_hypothesis_weights: Mapping[str, float] | None = None,
    prior_weights: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Map telemetry features to a prior belief distribution."""

    ids = _hypothesis_ids(hypotheses)
    weights = dict(prior_weights or load_prior_weights())
    background_base = float(weights.get("background_base_weight", 0.005))
    default_candidate_weight = float(weights.get("default_candidate_weight", 3.0))
    feature_boost_scale = float(weights.get("generic_feature_boost_scale", 0.55))
    candidates = set(str(item) for item in candidate_hypotheses or ())
    weighted_candidates = {
        str(hypothesis): float(weight)
        for hypothesis, weight in (candidate_hypothesis_weights or {}).items()
    }
    if weighted_candidates:
        scores = {
            hypothesis: max(background_base, weighted_candidates.get(hypothesis, background_base))
            for hypothesis in ids
        }
    else:
        scores = {
            hypothesis: (
                1.0
                if not candidates
                else default_candidate_weight
                if hypothesis in candidates
                else background_base
            )
            for hypothesis in ids
        }

    def add(hypothesis: str, value: float) -> None:
        if hypothesis in scores:
            scores[hypothesis] += value * feature_boost_scale

    if features.get("steering_demand") == "high":
        add("front_aero_load_limitation", 1.5)
        add("mechanical_balance_limitation", 1.2)
        add("front_mechanical_grip_limitation", 1.0)
        add("excessive_steering_energy", 0.8)

    understeer_index = float(features.get("understeer_index") or 0.0)
    if understeer_index >= 14.0:
        add("front_aero_load_limitation", 0.8)
        add("mechanical_balance_limitation", 0.5)
        add("front_mechanical_grip_limitation", 0.6)
        add("aero_balance_rearward_shift", 0.5)

    oversteer_index = float(features.get("oversteer_index") or 0.0)
    if oversteer_index >= 1.35:
        add("rear_aero_load_limitation", 1.0)
        add("rear_mechanical_grip_limitation", 1.0)
        add("aero_balance_forward_shift", 0.8)
        add("driver_input_contribution", 0.4)

    if features.get("front_tire_temp_trend_category") == "rising":
        add("front_tire_thermal_saturation", 2.5)

    if features.get("rear_tire_temp_trend_category") == "rising":
        add("rear_tire_thermal_saturation", 2.5)

    front_tire_temp_avg = features.get("front_tire_temp_avg")
    if isinstance(front_tire_temp_avg, (int, float)) and front_tire_temp_avg >= 100.0:
        add("front_tire_thermal_saturation", 1.0)

    rear_tire_temp_avg = features.get("rear_tire_temp_avg")
    if isinstance(rear_tire_temp_avg, (int, float)) and rear_tire_temp_avg >= 100.0:
        add("rear_tire_thermal_saturation", 1.0)

    if features.get("issue_worsens_over_stint") is True:
        add("front_tire_thermal_saturation", 1.4)
        add("rear_tire_thermal_saturation", 1.0)
        add("front_tire_wear_limitation", 0.8)
        add("rear_tire_wear_limitation", 0.8)
        add("driver_input_contribution", 0.8)

    suspension_variation = features.get("suspension_variation")
    if (
        features.get("platform_signal_available") is True
        and isinstance(suspension_variation, (int, float))
        and suspension_variation >= 0.003
    ):
        add("platform_aero_sensitivity", 2.0)
        add("spring_damper_mismatch", 0.8)

    platform_variation = features.get("platform_variation_index")
    if isinstance(platform_variation, (int, float)) and platform_variation >= 0.006:
        add("platform_aero_sensitivity", 1.4)
        add("ride_height_sensitivity", 0.8)
        add("pitch_sensitivity", 0.8)
        add("roll_sensitivity", 0.8)

    for feature, hypothesis, boost in (
        ("bottoming_index", "bottoming_induced_aero_loss", 2.2),
        ("kerb_strike_index", "kerb_platform_disturbance", 2.0),
        ("pitch_variation_index", "pitch_sensitivity", 1.6),
        ("roll_variation_index", "roll_sensitivity", 1.6),
        ("ride_height_variation_index", "ride_height_sensitivity", 1.6),
        ("front_locking_index", "brake_balance_too_forward", 1.8),
        ("rear_locking_index", "brake_balance_too_rearward", 1.8),
        ("brake_instability_index", "brake_migration_issue", 1.4),
        ("brake_temp_avg", "brake_temperature_issue", 0.8),
        ("wheelspin_index", "diff_exit_traction_limitation", 1.3),
        ("traction_loss_index", "traction_limitation", 1.4),
        ("acceleration_deficit_index", "power_unit_deployment_limitation", 1.0),
        ("drag_index", "drag_limited_straight_speed", 1.5),
        ("straight_line_speed_deficit_index", "excess_drag_configuration", 1.2),
    ):
        value = features.get(feature)
        if isinstance(value, (int, float)) and value > 0.0:
            add(hypothesis, boost * min(float(value), 1.0))

    ers_deploy_avg = features.get("ers_deploy_avg")
    if isinstance(ers_deploy_avg, (int, float)) and ers_deploy_avg <= 0.30:
        add("ers_deployment_strategy_issue", 1.5)

    throttle_aggression = features.get("throttle_aggression")
    if isinstance(throttle_aggression, (int, float)) and throttle_aggression >= 0.16:
        add("aggressive_throttle_application", 1.5)
        add("power_delivery_too_aggressive", 0.8)

    brake_release_aggression = features.get("brake_release_aggression")
    if isinstance(brake_release_aggression, (int, float)) and brake_release_aggression >= 0.18:
        add("late_brake_release", 1.5)
        add("driver_input_contribution", 0.6)

    steering_noise_index = features.get("steering_noise_index")
    if isinstance(steering_noise_index, (int, float)) and steering_noise_index >= 2.2:
        add("driver_input_contribution", 2.2)
        add("excessive_steering_energy", 1.8)

    tire_wear_avg = features.get("tire_wear_avg")
    if isinstance(tire_wear_avg, (int, float)) and tire_wear_avg >= 8.0:
        add("front_tire_wear_limitation", 0.9)
        add("rear_tire_wear_limitation", 0.9)

    if features.get("data_quality_flag") is True:
        add("sensor_noise_or_data_quality_issue", 3.0)

    return normalize_distribution(scores)


def candidate_weights_from_detections(
    detections: Iterable[Any],
    prior_weights: Mapping[str, Any] | None = None,
    max_symptoms: int = 3,
) -> dict[str, float]:
    """Build hypothesis weights from ranked symptom detections."""

    weights = dict(prior_weights or load_prior_weights())
    rank_weights = {
        int(rank): float(weight)
        for rank, weight in dict(weights.get("candidate_rank_weights", {})).items()
    }
    position_weights = {
        int(position): float(weight)
        for position, weight in dict(weights.get("candidate_position_weights", {})).items()
    }
    default_rank_weight = float(weights.get("default_candidate_weight", 3.0))
    default_position_weight = min(position_weights.values()) if position_weights else 0.25
    hypothesis_weights: dict[str, float] = {}
    for rank, detection in enumerate(detections, start=1):
        if rank > max_symptoms:
            break
        rank_weight = rank_weights.get(rank, default_rank_weight / rank)
        confidence = float(getattr(detection, "confidence", 1.0) or 1.0)
        for position, hypothesis in enumerate(
            getattr(detection, "candidate_hypotheses", ()),
            start=1,
        ):
            position_weight = position_weights.get(position, default_position_weight)
            contribution = rank_weight * position_weight * max(0.25, confidence)
            hypothesis_weights[str(hypothesis)] = max(
                hypothesis_weights.get(str(hypothesis), 0.0),
                contribution,
            )
    return hypothesis_weights


def top_hypothesis(belief: Mapping[str, float]) -> tuple[str, float]:
    """Return the most probable hypothesis and its probability."""

    if not belief:
        raise ValueError("Cannot select a top hypothesis from an empty belief")
    hypothesis, probability = max(belief.items(), key=lambda item: item[1])
    return hypothesis, float(probability)
