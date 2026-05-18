"""Synthetic telemetry scenarios used as controlled validation proxies."""

from __future__ import annotations

import csv
import hashlib
import math
import random
from pathlib import Path
from typing import Any

from .test_library import load_hypotheses, load_symptom_hypothesis_map


SYMPTOM_SCENARIOS = (
    "medium_speed_entry_to_apex_understeer",
    "entry_understeer",
    "entry_oversteer",
    "brake_entry_instability",
    "front_locking",
    "rear_locking",
    "mid_corner_understeer",
    "mid_corner_oversteer",
    "minimum_speed_loss",
    "poor_rotation",
    "exit_understeer",
    "exit_oversteer",
    "traction_loss",
    "wheelspin_on_exit",
    "poor_power_down",
    "high_speed_understeer",
    "high_speed_oversteer",
    "high_speed_instability",
    "aero_balance_shift",
    "low_speed_rotation_deficit",
    "low_speed_snap_oversteer",
    "traction_limited_hairpin_exit",
    "long_braking_distance",
    "brake_instability",
    "brake_locking",
    "brake_fade",
    "brake_balance_sensitivity",
    "front_tire_overheating",
    "rear_tire_overheating",
    "thermal_degradation",
    "excessive_wear",
    "pace_falloff_over_stint",
    "bottoming",
    "ride_height_sensitivity",
    "kerb_instability",
    "pitch_sensitivity",
    "roll_sensitivity",
    "straight_line_speed_deficit",
    "poor_acceleration",
    "drag_sensitivity",
    "ers_deployment_deficit",
    "aggressive_steering_inputs",
    "brake_release_instability",
    "throttle_application_instability",
    "inconsistent_cornering",
)

LEGACY_HYPOTHESIS_SCENARIOS = (
    "front_aero_load_limitation",
    "front_tire_thermal_saturation",
    "platform_aero_sensitivity",
    "mechanical_balance_limitation",
    "driver_input_contribution",
)

SCENARIOS = SYMPTOM_SCENARIOS + LEGACY_HYPOTHESIS_SCENARIOS

HYPOTHESIS_SCENARIO_MAP = {
    "front_aero_load_limitation": "entry_understeer",
    "rear_aero_load_limitation": "high_speed_oversteer",
    "aero_balance_forward_shift": "aero_balance_shift",
    "aero_balance_rearward_shift": "high_speed_understeer",
    "excess_drag_configuration": "straight_line_speed_deficit",
    "platform_aero_sensitivity": "high_speed_instability",
    "ride_height_sensitivity": "ride_height_sensitivity",
    "pitch_sensitivity": "pitch_sensitivity",
    "roll_sensitivity": "roll_sensitivity",
    "bottoming_induced_aero_loss": "bottoming",
    "kerb_platform_disturbance": "kerb_instability",
    "front_tire_thermal_saturation": "front_tire_overheating",
    "rear_tire_thermal_saturation": "rear_tire_overheating",
    "front_tire_wear_limitation": "excessive_wear",
    "rear_tire_wear_limitation": "excessive_wear",
    "tire_pressure_window_issue": "thermal_degradation",
    "compound_window_mismatch": "pace_falloff_over_stint",
    "mechanical_balance_limitation": "minimum_speed_loss",
    "front_mechanical_grip_limitation": "poor_rotation",
    "rear_mechanical_grip_limitation": "traction_loss",
    "arb_stiffness_mismatch": "roll_sensitivity",
    "spring_damper_mismatch": "kerb_instability",
    "camber_toe_limitation": "excessive_wear",
    "brake_balance_too_forward": "front_locking",
    "brake_balance_too_rearward": "rear_locking",
    "brake_migration_issue": "brake_balance_sensitivity",
    "brake_temperature_issue": "brake_fade",
    "engine_braking_instability": "brake_entry_instability",
    "diff_entry_instability": "entry_oversteer",
    "diff_mid_corner_rotation_limitation": "poor_rotation",
    "diff_exit_traction_limitation": "wheelspin_on_exit",
    "power_delivery_too_aggressive": "throttle_application_instability",
    "traction_limitation": "poor_power_down",
    "power_unit_deployment_limitation": "poor_acceleration",
    "ers_deployment_strategy_issue": "ers_deployment_deficit",
    "gear_ratio_mismatch": "poor_acceleration",
    "drag_limited_straight_speed": "drag_sensitivity",
    "driver_input_contribution": "inconsistent_cornering",
    "excessive_steering_energy": "aggressive_steering_inputs",
    "late_brake_release": "brake_release_instability",
    "aggressive_throttle_application": "throttle_application_instability",
    "inconsistent_line_selection": "inconsistent_cornering",
    "track_evolution_effect": "pace_falloff_over_stint",
    "fuel_load_effect": "pace_falloff_over_stint",
    "wind_sensitivity": "high_speed_instability",
    "sensor_noise_or_data_quality_issue": "inconsistent_cornering",
}

CSV_COLUMNS = (
    "timestamp",
    "lap_number",
    "distance",
    "speed",
    "steering_angle",
    "throttle",
    "brake",
    "lateral_accel",
    "longitudinal_accel",
    "yaw_rate",
    "front_left_temp",
    "front_right_temp",
    "rear_left_temp",
    "rear_right_temp",
    "tire_wear_fl",
    "tire_wear_fr",
    "tire_wear_rl",
    "tire_wear_rr",
    "brake_temp_fl",
    "brake_temp_fr",
    "brake_temp_rl",
    "brake_temp_rr",
    "wheel_slip_fl",
    "wheel_slip_fr",
    "wheel_slip_rl",
    "wheel_slip_rr",
    "suspension_fl",
    "suspension_fr",
    "suspension_rl",
    "suspension_rr",
    "gear",
    "rpm",
    "ers_deploy",
    "ers_energy",
    "fuel_load",
    "drs",
    "track_grip",
    "wind_speed",
    "brake_balance",
    "setup_id",
    "corner_id",
    "segment_id",
)


def scenario_for_hypothesis(hypothesis: str) -> str:
    """Return a representative symptom scenario for a hypothesis."""

    if hypothesis in HYPOTHESIS_SCENARIO_MAP:
        return HYPOTHESIS_SCENARIO_MAP[hypothesis]
    mapping = load_symptom_hypothesis_map()
    for symptom, hypotheses in mapping.items():
        if hypothesis in hypotheses:
            return symptom
    return "medium_speed_entry_to_apex_understeer"


def true_hypothesis_for_scenario(scenario: str) -> str:
    """Return the simulated true hypothesis for a symptom or hypothesis scenario."""

    hypotheses = load_hypotheses()
    if scenario in hypotheses:
        return scenario
    mapping = load_symptom_hypothesis_map()
    if scenario in mapping and mapping[scenario]:
        return mapping[scenario][0]
    raise ValueError(f"Unknown synthetic scenario: {scenario}")


def _canonical_symptom(scenario: str) -> str:
    if scenario in SYMPTOM_SCENARIOS:
        return scenario
    if scenario in load_hypotheses():
        return scenario_for_hypothesis(scenario)
    raise ValueError(
        f"Unknown synthetic scenario {scenario!r}. Expected one of {len(SCENARIOS)} scenarios."
    )


def _profile(symptom: str) -> dict[str, float]:
    profile = {
        "avg_speed": 145.0,
        "min_speed": 105.0,
        "steering": 14.0,
        "lateral": 1.20,
        "yaw": 1.8,
        "throttle": 0.45,
        "brake": 0.35,
        "front_temp": 92.0,
        "rear_temp": 91.0,
        "front_temp_gain": 0.20,
        "rear_temp_gain": 0.15,
        "wear_gain": 0.35,
        "brake_temp": 560.0,
        "brake_temp_gain": 0.12,
        "front_slip": 0.04,
        "rear_slip": 0.04,
        "suspension_amp": 0.006,
        "suspension_front_base": 0.056,
        "suspension_rear_base": 0.062,
        "platform_mode": "default",
        "duration": 8.2,
        "duration_gain": 0.04,
        "steering_noise": 0.8,
        "throttle_noise": 0.03,
        "brake_noise": 0.04,
        "ers_deploy": 0.55,
        "brake_balance": 56.0,
        "brake_balance_noise": 0.3,
        "longitudinal_scale": 1.0,
    }

    if "high_speed" in symptom or "aero_balance" in symptom:
        profile.update({"avg_speed": 270.0, "min_speed": 225.0, "throttle": 0.70})
    if (
        "straight_line" in symptom
        or "drag" in symptom
        or "ers_deployment" in symptom
        or "poor_acceleration" in symptom
    ):
        profile.update({"avg_speed": 250.0, "min_speed": 242.0, "throttle": 0.94, "brake": 0.02, "lateral": 0.04, "steering": 1.6, "yaw": 0.05})
    if "low_speed" in symptom or "hairpin" in symptom:
        profile.update({"avg_speed": 86.0, "min_speed": 55.0, "throttle": 0.55, "steering": 18.0, "lateral": 0.85})
    if "brake" in symptom or "locking" in symptom:
        profile.update({"brake": 0.82, "throttle": 0.08, "brake_temp": 610.0, "brake_temp_gain": 0.18, "steering": 7.0, "lateral": 0.60})
    if "exit" in symptom or "traction" in symptom or "wheelspin" in symptom or "power_down" in symptom:
        profile.update({"throttle": 0.82, "brake": 0.08})

    if "understeer" in symptom or "rotation_deficit" in symptom or symptom == "poor_rotation":
        profile.update({"steering": 19.0, "lateral": 0.82, "yaw": 0.9})
    if "oversteer" in symptom or "instability" in symptom:
        profile.update({"steering": 11.0, "lateral": 0.90, "yaw": 6.0, "rear_slip": 0.16})
    if "front_locking" in symptom:
        profile.update({"front_slip": 0.44, "rear_slip": 0.04, "brake_balance": 63.5, "yaw": 0.5, "front_temp": 98.0})
    if "rear_locking" in symptom:
        profile.update({"front_slip": 0.05, "rear_slip": 0.44, "brake_balance": 47.0, "yaw": 5.8, "rear_temp": 98.0})
    if symptom == "brake_locking":
        profile.update({"front_slip": 0.40, "rear_slip": 0.34, "brake_balance": 56.0})
    if "wheelspin" in symptom:
        profile.update({"rear_slip": 0.55, "throttle_noise": 0.12, "rear_temp_gain": 1.0, "yaw": 0.7})
    elif "traction" in symptom:
        profile.update({"rear_slip": 0.34, "throttle_noise": 0.08, "rear_temp_gain": 0.8, "yaw": 0.7})
    if "poor_power_down" in symptom:
        profile.update({"throttle": 0.92, "ers_deploy": 0.25, "longitudinal_scale": 0.35})
    if "poor_acceleration" in symptom:
        profile.update({"throttle": 0.94, "ers_deploy": 0.55, "longitudinal_scale": 0.35})
    if "front_tire" in symptom:
        profile.update({"front_temp": 104.0, "front_temp_gain": 1.35})
    elif "understeer" in symptom:
        profile.update({"front_temp": 96.0, "front_temp_gain": 0.45})
    if "rear_tire" in symptom:
        profile.update({"rear_temp": 104.0, "rear_temp_gain": 1.35})
    elif "oversteer" in symptom or "traction" in symptom:
        profile.update({"rear_temp": 96.0, "rear_temp_gain": 0.45})
    if "thermal" in symptom or "pace_falloff" in symptom:
        profile.update({"front_temp_gain": 1.5, "rear_temp_gain": 1.4, "duration_gain": 0.22})
    if "wear" in symptom:
        profile.update({"wear_gain": 3.0, "duration_gain": 0.18})
    if "fade" in symptom:
        profile.update({"brake_temp": 720.0, "brake_temp_gain": 25.0, "duration_gain": 0.16})
    if "bottoming" in symptom:
        profile.update({"suspension_amp": 0.035, "platform_mode": "bottoming"})
    if "ride_height" in symptom:
        profile.update({"suspension_amp": 0.016, "platform_mode": "ride", "suspension_front_base": 0.080, "suspension_rear_base": 0.086})
    if "pitch" in symptom:
        profile.update({"suspension_amp": 0.014, "platform_mode": "pitch", "suspension_front_base": 0.080, "suspension_rear_base": 0.086})
    if "roll" in symptom:
        profile.update({"suspension_amp": 0.014, "platform_mode": "roll", "suspension_front_base": 0.080, "suspension_rear_base": 0.086})
    if "aero_balance" in symptom:
        profile.update({"suspension_amp": 0.018, "platform_mode": "ride", "suspension_front_base": 0.080, "suspension_rear_base": 0.086})
    if "kerb" in symptom:
        profile.update({"suspension_amp": 0.020, "platform_mode": "roll", "suspension_front_base": 0.080, "suspension_rear_base": 0.086, "yaw": 4.5})
    if "drag" in symptom or "straight_line_speed_deficit" in symptom:
        profile.update({"avg_speed": 188.0, "min_speed": 184.0, "ers_deploy": 0.60})
    if "ers_deployment" in symptom:
        profile.update({"ers_deploy": 0.12, "avg_speed": 226.0, "min_speed": 220.0})
    if "aggressive_steering" in symptom:
        profile.update({"steering_noise": 3.2, "steering": 17.0})
    if "brake_release" in symptom:
        profile.update({"brake": 0.65, "brake_noise": 0.65, "yaw": 4.0})
    if "throttle_application" in symptom:
        profile.update({"throttle": 0.72, "throttle_noise": 0.26, "rear_slip": 0.25})
    if "inconsistent" in symptom:
        profile.update({"duration_gain": 1.45, "steering_noise": 1.6, "throttle_noise": 0.10})
    if symptom == "long_braking_distance":
        profile.update({"brake": 0.76, "throttle": 0.08, "longitudinal_scale": 0.20, "steering": 6.0, "lateral": 0.45})
    if symptom == "brake_balance_sensitivity":
        profile.update({"brake": 0.80, "throttle": 0.08, "brake_balance_noise": 1.4, "yaw": 2.8})
    if symptom == "minimum_speed_loss":
        profile.update({"steering": 8.0, "lateral": 1.0, "min_speed": 68.0})
    if symptom == "high_speed_instability":
        profile.update({"suspension_amp": 0.018, "platform_mode": "ride", "suspension_front_base": 0.080, "suspension_rear_base": 0.086, "yaw": 2.5})

    return profile


def _dominant_segment(symptom: str) -> str:
    if symptom in {
        "aggressive_steering_inputs",
        "brake_release_instability",
        "throttle_application_instability",
        "inconsistent_cornering",
    }:
        return "driver"
    if symptom in {"front_locking", "rear_locking", "brake_entry_instability"}:
        return "braking"
    if symptom == "medium_speed_entry_to_apex_understeer":
        return "entry_to_apex"
    if symptom.startswith("entry_"):
        return "entry"
    if symptom.startswith("mid_corner") or symptom in {"minimum_speed_loss", "poor_rotation"}:
        return "mid_corner"
    if symptom.startswith("low_speed") or symptom == "traction_limited_hairpin_exit":
        return "low_speed_exit" if symptom == "traction_limited_hairpin_exit" else "low_speed"
    if "exit" in symptom or "traction" in symptom or "wheelspin" in symptom or "power_down" in symptom:
        return "exit"
    if symptom.startswith("high_speed") or symptom == "aero_balance_shift":
        return "high_speed"
    if symptom.startswith("brake_") or symptom == "long_braking_distance":
        return "braking"
    if "tire" in symptom or "thermal" in symptom or "wear" in symptom or "pace_falloff" in symptom:
        return "stint"
    if symptom == "kerb_instability":
        return "kerb"
    if symptom in {"bottoming", "ride_height_sensitivity", "pitch_sensitivity", "roll_sensitivity"}:
        return "platform"
    if "straight" in symptom or "drag" in symptom or "ers_deployment" in symptom or symptom == "poor_acceleration":
        return "straight"
    return "entry"


def generate_synthetic_rows(
    scenario: str,
    laps: int = 8,
    samples_per_lap: int = 120,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Generate deterministic telemetry for one symptom or hypothesis scenario."""

    symptom = _canonical_symptom(scenario)
    scenario_seed = int(hashlib.sha256(symptom.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(seed + scenario_seed % 10000)
    profile = _profile(symptom)
    dominant_segment = _dominant_segment(symptom)
    rows: list[dict[str, Any]] = []
    global_time = 0.0

    for lap in range(1, laps + 1):
        lap_progress = (lap - 1) / max(1, laps - 1)
        duration = profile["duration"] + profile["duration_gain"] * (lap - 1)
        front_temp = profile["front_temp"] + profile["front_temp_gain"] * (lap - 1)
        rear_temp = profile["rear_temp"] + profile["rear_temp_gain"] * (lap - 1)
        brake_temp = profile["brake_temp"] + profile["brake_temp_gain"] * (lap - 1)
        tire_wear = profile["wear_gain"] * (lap - 1)

        for sample in range(samples_per_lap):
            phase = sample / max(1, samples_per_lap - 1)
            apex_shape = math.sin(math.pi * phase)
            exit_shape = max(0.0, (phase - 0.45) / 0.55)
            entry_shape = max(0.0, (0.55 - phase) / 0.55)
            timestamp = global_time + phase * duration

            if dominant_segment == "straight":
                speed = profile["avg_speed"] - 2.5 * phase - 1.0 * lap_progress + rng.gauss(0.0, 0.9)
                throttle_shape = 0.92 + 0.08 * phase
                brake_shape = 0.05
                lateral_shape = 0.20
                steering_shape = 0.75
                yaw_shape = 0.25
            elif dominant_segment == "braking":
                speed = (
                    profile["avg_speed"]
                    - (profile["avg_speed"] - profile["min_speed"]) * max(phase, apex_shape * 0.45)
                    - 1.2 * lap_progress
                    + rng.gauss(0.0, 1.1)
                )
                throttle_shape = 0.12
                brake_shape = 0.72 + 0.28 * entry_shape
                lateral_shape = max(0.25, entry_shape)
                steering_shape = 0.65 + 0.20 * entry_shape
                yaw_shape = 0.55 + 0.45 * entry_shape
            elif dominant_segment == "exit":
                speed = (
                    profile["avg_speed"]
                    - (profile["avg_speed"] - profile["min_speed"]) * (1.0 - exit_shape) * 0.75
                    - 1.0 * lap_progress
                    + rng.gauss(0.0, 1.2)
                )
                throttle_shape = 0.65 + 0.35 * exit_shape
                brake_shape = 0.08
                lateral_shape = 0.35 + 0.65 * exit_shape
                steering_shape = 0.55 + 0.35 * exit_shape
                yaw_shape = 0.40 + 0.60 * exit_shape
            else:
                speed = (
                    profile["avg_speed"]
                    - (profile["avg_speed"] - profile["min_speed"]) * apex_shape
                    - 1.5 * lap_progress
                    + rng.gauss(0.0, 1.2)
                )
                throttle_shape = 0.35 + 0.65 * exit_shape
                brake_shape = entry_shape
                lateral_shape = apex_shape
                steering_shape = 0.65 + 0.35 * apex_shape
                yaw_shape = apex_shape

            steering = (
                profile["steering"] * steering_shape
                + rng.gauss(0.0, profile["steering_noise"])
            )
            throttle = max(
                0.0,
                min(
                    1.0,
                    profile["throttle"] * throttle_shape
                    + rng.gauss(0.0, profile["throttle_noise"]),
                ),
            )
            brake = max(
                0.0,
                min(
                    1.0,
                    profile["brake"] * brake_shape
                    + rng.gauss(0.0, profile["brake_noise"]),
                ),
            )
            lateral = (
                profile["lateral"] * lateral_shape
                - 0.03 * lap_progress
                + rng.gauss(0.0, 0.035)
            )
            longitudinal = (
                throttle * 0.9 * profile["longitudinal_scale"]
                - brake * 1.2 * max(0.25, profile["longitudinal_scale"])
                + rng.gauss(0.0, 0.04)
            )
            yaw = profile["yaw"] * yaw_shape + rng.gauss(0.0, 0.25)
            rear_slip_demand = brake if dominant_segment == "braking" else throttle
            rear_slip = max(0.0, profile["rear_slip"] * (0.4 + rear_slip_demand) + rng.gauss(0.0, 0.015))
            front_slip = max(0.0, profile["front_slip"] * (0.4 + brake) + rng.gauss(0.0, 0.012))

            suspension_wave = profile["suspension_amp"] * math.sin(2.0 * math.pi * phase)
            if "bottoming" in symptom and sample % 19 == 0:
                suspension_wave -= 0.020
            platform_jitter = rng.gauss(0.0, profile["suspension_amp"] / 3.0)
            front_base = profile["suspension_front_base"]
            rear_base = profile["suspension_rear_base"]
            platform_mode = str(profile["platform_mode"])
            if platform_mode == "ride":
                suspension_fl = front_base + suspension_wave + platform_jitter * 0.15
                suspension_fr = front_base + suspension_wave - platform_jitter * 0.15
                suspension_rl = rear_base + suspension_wave
                suspension_rr = rear_base + suspension_wave
            elif platform_mode in {"pitch", "bottoming"}:
                suspension_fl = front_base + suspension_wave + platform_jitter * 0.20
                suspension_fr = front_base + suspension_wave - platform_jitter * 0.20
                suspension_rl = rear_base - suspension_wave
                suspension_rr = rear_base - suspension_wave
            elif platform_mode == "roll":
                suspension_fl = front_base + suspension_wave + platform_jitter * 0.10
                suspension_fr = front_base - suspension_wave - platform_jitter * 0.10
                suspension_rl = rear_base + suspension_wave * 0.80
                suspension_rr = rear_base - suspension_wave * 0.80
            else:
                suspension_fl = front_base + suspension_wave + platform_jitter
                suspension_fr = front_base + suspension_wave - platform_jitter
                suspension_rl = rear_base - suspension_wave * 0.45
                suspension_rr = rear_base - suspension_wave * 0.45

            segment = dominant_segment
            rows.append(
                {
                    "timestamp": round(timestamp, 4),
                    "lap_number": lap,
                    "distance": round(1000.0 + 520.0 * phase, 3),
                    "speed": round(speed, 3),
                    "steering_angle": round(steering, 3),
                    "throttle": round(throttle, 4),
                    "brake": round(brake, 4),
                    "lateral_accel": round(lateral, 4),
                    "longitudinal_accel": round(longitudinal, 4),
                    "yaw_rate": round(yaw, 4),
                    "front_left_temp": round(front_temp + 0.9 * apex_shape + rng.gauss(0, 0.35), 3),
                    "front_right_temp": round(front_temp + 1.0 * apex_shape + rng.gauss(0, 0.35), 3),
                    "rear_left_temp": round(rear_temp + 0.8 * exit_shape + rng.gauss(0, 0.35), 3),
                    "rear_right_temp": round(rear_temp + 0.8 * exit_shape + rng.gauss(0, 0.35), 3),
                    "tire_wear_fl": round(tire_wear + 0.06 * phase, 4),
                    "tire_wear_fr": round(tire_wear + 0.06 * phase, 4),
                    "tire_wear_rl": round(tire_wear + 0.05 * phase, 4),
                    "tire_wear_rr": round(tire_wear + 0.05 * phase, 4),
                    "brake_temp_fl": round(brake_temp + 12.0 * brake + rng.gauss(0, 4.0), 2),
                    "brake_temp_fr": round(brake_temp + 12.0 * brake + rng.gauss(0, 4.0), 2),
                    "brake_temp_rl": round(brake_temp + 8.0 * brake + rng.gauss(0, 4.0), 2),
                    "brake_temp_rr": round(brake_temp + 8.0 * brake + rng.gauss(0, 4.0), 2),
                    "wheel_slip_fl": round(front_slip, 4),
                    "wheel_slip_fr": round(front_slip * rng.uniform(0.90, 1.10), 4),
                    "wheel_slip_rl": round(rear_slip, 4),
                    "wheel_slip_rr": round(rear_slip * rng.uniform(0.90, 1.10), 4),
                    "suspension_fl": round(suspension_fl, 5),
                    "suspension_fr": round(suspension_fr, 5),
                    "suspension_rl": round(suspension_rl, 5),
                    "suspension_rr": round(suspension_rr, 5),
                    "gear": int(max(1, min(8, speed // 38 + 1))),
                    "rpm": round(8500 + speed * 35 + rng.gauss(0, 120), 1),
                    "ers_deploy": round(max(0.0, min(1.0, profile["ers_deploy"] + rng.gauss(0, 0.04))), 4),
                    "ers_energy": round(max(0.0, min(1.0, 0.8 - 0.05 * lap_progress)), 4),
                    "fuel_load": round(max(5.0, 30.0 - 1.5 * (lap - 1)), 3),
                    "drs": 1 if segment == "straight" and throttle > 0.8 else 0,
                    "track_grip": round(0.95 + 0.01 * lap_progress, 4),
                    "wind_speed": round(8.0 + (4.0 if "wind" in symptom else 0.0), 3),
                    "brake_balance": round(profile["brake_balance"] + rng.gauss(0, profile["brake_balance_noise"]), 3),
                    "setup_id": f"synthetic_{scenario}",
                    "corner_id": f"synthetic_{symptom}",
                    "segment_id": segment,
                }
            )

        global_time += duration + 80.0

    return rows


def write_synthetic_csv(
    scenario: str,
    output_dir: str | Path = "data/synthetic",
    laps: int = 8,
    samples_per_lap: int = 120,
    seed: int = 42,
) -> Path:
    """Write one synthetic scenario CSV and return its path."""

    rows = generate_synthetic_rows(
        scenario=scenario,
        laps=laps,
        samples_per_lap=samples_per_lap,
        seed=seed,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / f"{scenario}.csv"

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


def generate_all_synthetic(
    output_dir: str | Path = "data/synthetic",
    laps: int = 8,
    samples_per_lap: int = 120,
    seed: int = 42,
) -> list[Path]:
    """Write all supported symptom and legacy hypothesis scenario CSVs."""

    return [
        write_synthetic_csv(
            scenario=scenario,
            output_dir=output_dir,
            laps=laps,
            samples_per_lap=samples_per_lap,
            seed=seed,
        )
        for scenario in SCENARIOS
    ]
