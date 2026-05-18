"""Telemetry feature extraction for medium-speed entry-to-apex understeer."""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping

from .telemetry_loader import TelemetryDataset


EPSILON = 1e-6


def _numeric_values(rows: Iterable[Mapping[str, Any]], column: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(column)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) >= 2 else None


def _linear_slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    if denominator <= EPSILON:
        return None
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return numerator / denominator


def _category_from_steering(avg_abs_steering: float) -> str:
    if avg_abs_steering >= 14.0:
        return "high"
    if avg_abs_steering >= 8.0:
        return "medium"
    return "low"


def _category_from_speed(avg_speed: float) -> str:
    if avg_speed < 95.0:
        return "low"
    if avg_speed <= 210.0:
        return "medium"
    return "high"


def _category_from_temp_slope(slope: float | None) -> str:
    if slope is None:
        return "unknown"
    if slope > 0.4:
        return "rising"
    if slope < -0.4:
        return "falling"
    return "stable"


def _average_row_many(rows: list[Mapping[str, Any]], columns: tuple[str, ...]) -> list[float]:
    values: list[float] = []
    for row in rows:
        row_values = [row.get(column) for column in columns]
        numeric = [float(value) for value in row_values if isinstance(value, (int, float))]
        if numeric:
            values.append(statistics.fmean(numeric))
    return values


def _lap_durations(rows: list[Mapping[str, Any]]) -> dict[int, float]:
    timestamps_by_lap: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        lap = row.get("lap_number")
        timestamp = row.get("timestamp")
        if isinstance(lap, int) and isinstance(timestamp, (int, float)):
            timestamps_by_lap[lap].append(float(timestamp))
    durations: dict[int, float] = {}
    for lap, timestamps in timestamps_by_lap.items():
        if len(timestamps) >= 2:
            durations[lap] = max(timestamps) - min(timestamps)
    return dict(sorted(durations.items()))


def _average_temp_by_lap(
    rows: list[Mapping[str, Any]], columns: tuple[str, str]
) -> list[tuple[float, float]]:
    values_by_lap: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        lap = row.get("lap_number")
        left = row.get(columns[0])
        right = row.get(columns[1])
        if (
            isinstance(lap, int)
            and isinstance(left, (int, float))
            and isinstance(right, (int, float))
        ):
            values_by_lap[lap].append((float(left) + float(right)) / 2.0)
    return [
        (float(lap), statistics.fmean(values))
        for lap, values in sorted(values_by_lap.items())
        if values
    ]


def _average_many_by_lap(
    rows: list[Mapping[str, Any]], columns: tuple[str, ...]
) -> list[tuple[float, float]]:
    values_by_lap: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        lap = row.get("lap_number")
        if not isinstance(lap, int):
            continue
        numeric = [
            float(row[column])
            for column in columns
            if isinstance(row.get(column), (int, float))
        ]
        if numeric:
            values_by_lap[lap].append(statistics.fmean(numeric))
    return [
        (float(lap), statistics.fmean(values))
        for lap, values in sorted(values_by_lap.items())
        if values
    ]


def _average_row_pair(
    rows: list[Mapping[str, Any]], columns: tuple[str, str]
) -> list[float]:
    values: list[float] = []
    for row in rows:
        left = row.get(columns[0])
        right = row.get(columns[1])
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            values.append((float(left) + float(right)) / 2.0)
    return values


def _steering_noise(steering_values: list[float]) -> float | None:
    if len(steering_values) < 2:
        return None
    deltas = [
        abs(current - previous)
        for previous, current in zip(steering_values, steering_values[1:])
    ]
    return statistics.fmean(deltas)


def _mean_abs_delta(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.fmean(
        abs(current - previous)
        for previous, current in zip(values, values[1:])
    )


def _mean_positive_drop(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    drops = [
        max(0.0, previous - current)
        for previous, current in zip(values, values[1:])
    ]
    return statistics.fmean(drops)


def _suspension_variation(rows: list[Mapping[str, Any]]) -> float | None:
    columns = (
        "suspension_fl",
        "suspension_fr",
        "suspension_rl",
        "suspension_rr",
    )
    platform_values: list[float] = []
    for row in rows:
        values = [row.get(column) for column in columns]
        if all(isinstance(value, (int, float)) for value in values):
            platform_values.append(statistics.fmean(float(value) for value in values))
    return _std(platform_values)


def _suspension_derived(rows: list[Mapping[str, Any]]) -> dict[str, float | None]:
    pitch_values: list[float] = []
    roll_values: list[float] = []
    ride_values: list[float] = []
    bottoming_hits = 0
    valid = 0
    for row in rows:
        fl = row.get("suspension_fl")
        fr = row.get("suspension_fr")
        rl = row.get("suspension_rl")
        rr = row.get("suspension_rr")
        if all(isinstance(value, (int, float)) for value in (fl, fr, rl, rr)):
            front = (float(fl) + float(fr)) / 2.0
            rear = (float(rl) + float(rr)) / 2.0
            left = (float(fl) + float(rl)) / 2.0
            right = (float(fr) + float(rr)) / 2.0
            all_values = [float(fl), float(fr), float(rl), float(rr)]
            pitch_values.append(front - rear)
            roll_values.append(left - right)
            ride_values.append(statistics.fmean(all_values))
            if min(all_values) < 0.040:
                bottoming_hits += 1
            valid += 1

    if valid == 0:
        return {
            "pitch_variation_index": None,
            "roll_variation_index": None,
            "ride_height_variation_index": None,
            "bottoming_index": None,
            "kerb_strike_index": None,
        }

    pitch_variation = (_std(pitch_values) or 0.0) * 20.0
    roll_variation = (_std(roll_values) or 0.0) * 20.0
    ride_variation = (_std(ride_values) or 0.0) * 20.0
    return {
        "pitch_variation_index": pitch_variation,
        "roll_variation_index": roll_variation,
        "ride_height_variation_index": ride_variation,
        "bottoming_index": bottoming_hits / valid,
        "kerb_strike_index": min(1.0, max(pitch_variation, roll_variation, ride_variation)),
    }


def _slip_index(
    rows: list[Mapping[str, Any]], columns: tuple[str, str], brake_or_throttle: str
) -> float | None:
    values: list[float] = []
    for row in rows:
        left = row.get(columns[0])
        right = row.get(columns[1])
        demand = row.get(brake_or_throttle)
        if (
            isinstance(left, (int, float))
            and isinstance(right, (int, float))
            and isinstance(demand, (int, float))
        ):
            values.append(abs((float(left) + float(right)) / 2.0) * max(0.0, float(demand)))
    return _mean(values)


def extract_features(dataset: TelemetryDataset | list[dict[str, Any]]) -> dict[str, Any]:
    """Extract diagnostic features from a telemetry dataset or row list."""

    if isinstance(dataset, TelemetryDataset):
        rows = dataset.rows
        columns = dataset.columns
        loader_warnings = dataset.warnings
    else:
        rows = dataset
        columns = set(rows[0].keys()) if rows else set()
        loader_warnings = []

    if not rows:
        raise ValueError("Cannot extract features from an empty telemetry dataset")

    speed_values = _numeric_values(rows, "speed")
    steering_values = _numeric_values(rows, "steering_angle")
    throttle_values = _numeric_values(rows, "throttle")
    brake_values = _numeric_values(rows, "brake")
    lateral_values = _numeric_values(rows, "lateral_accel")
    longitudinal_values = _numeric_values(rows, "longitudinal_accel")
    yaw_values = _numeric_values(rows, "yaw_rate")
    abs_lateral_values = [abs(value) for value in lateral_values]
    abs_steering_values = [abs(value) for value in steering_values]
    abs_yaw_values = [abs(value) for value in yaw_values]

    avg_speed = _mean(speed_values) or 0.0
    min_speed = min(speed_values) if speed_values else 0.0
    avg_abs_steering = _mean(abs_steering_values) or 0.0
    avg_lateral_accel = _mean(lateral_values) or 0.0
    avg_abs_lateral_accel = _mean(abs_lateral_values) or 0.0
    avg_longitudinal_accel = _mean(longitudinal_values) or 0.0
    avg_throttle = _mean(throttle_values) or 0.0
    avg_brake = _mean(brake_values) or 0.0
    max_brake = max(brake_values) if brake_values else 0.0
    understeer_index = avg_abs_steering / max(avg_abs_lateral_accel, EPSILON)
    yaw_rate_abs_avg = _mean(abs_yaw_values) or 0.0
    oversteer_index = yaw_rate_abs_avg / max(avg_abs_steering / 12.0, 0.2)
    expected_lateral_from_steering = avg_abs_steering / 18.0
    rotation_deficit_index = max(
        0.0,
        min(1.0, (expected_lateral_from_steering - avg_abs_lateral_accel) / max(expected_lateral_from_steering, EPSILON)),
    )
    min_speed_loss_ratio = (
        max(0.0, avg_speed - min_speed) / max(avg_speed, EPSILON)
        if avg_speed > 0.0
        else 0.0
    )

    front_temp_values = _average_row_pair(
        rows, ("front_left_temp", "front_right_temp")
    )
    rear_temp_values = _average_row_pair(rows, ("rear_left_temp", "rear_right_temp"))
    front_tire_temp_avg = _mean(front_temp_values)
    rear_tire_temp_avg = _mean(rear_temp_values)

    front_temp_slope = None
    rear_temp_slope = None
    if {"front_left_temp", "front_right_temp", "lap_number"} <= columns:
        front_temp_slope = _linear_slope(
            _average_temp_by_lap(rows, ("front_left_temp", "front_right_temp"))
        )
    if {"rear_left_temp", "rear_right_temp", "lap_number"} <= columns:
        rear_temp_slope = _linear_slope(
            _average_temp_by_lap(rows, ("rear_left_temp", "rear_right_temp"))
        )

    front_wear_values = _average_row_pair(rows, ("tire_wear_fl", "tire_wear_fr"))
    rear_wear_values = _average_row_pair(rows, ("tire_wear_rl", "tire_wear_rr"))
    tire_wear_values = _average_row_many(
        rows, ("tire_wear_fl", "tire_wear_fr", "tire_wear_rl", "tire_wear_rr")
    )
    tire_wear_slope = _linear_slope(
        _average_many_by_lap(
            rows, ("tire_wear_fl", "tire_wear_fr", "tire_wear_rl", "tire_wear_rr")
        )
    )

    brake_temp_values = _average_row_many(
        rows, ("brake_temp_fl", "brake_temp_fr", "brake_temp_rl", "brake_temp_rr")
    )
    brake_temp_slope = _linear_slope(
        _average_many_by_lap(
            rows, ("brake_temp_fl", "brake_temp_fr", "brake_temp_rl", "brake_temp_rr")
        )
    )

    lap_durations = _lap_durations(rows)
    lap_time = _mean(list(lap_durations.values()))
    performance_loss = None
    if len(lap_durations) >= 2:
        durations = list(lap_durations.values())
        best = min(durations)
        half = max(1, len(durations) // 2)
        late_average = statistics.fmean(durations[-half:])
        performance_loss = max(0.0, late_average - best)

    front_temp_category = _category_from_temp_slope(front_temp_slope)
    rear_temp_category = _category_from_temp_slope(rear_temp_slope)
    brake_temp_category = _category_from_temp_slope(brake_temp_slope)
    steering_noise_index = _steering_noise(steering_values)
    throttle_aggression = _mean_abs_delta(throttle_values)
    brake_release_aggression = _mean_positive_drop(brake_values)
    suspension_variation = _suspension_variation(rows)
    suspension_features = _suspension_derived(rows)
    platform_variation_index = (
        suspension_variation if suspension_variation is not None else 0.0
    )
    front_locking_index = _slip_index(
        rows, ("wheel_slip_fl", "wheel_slip_fr"), "brake"
    )
    rear_locking_index = _slip_index(rows, ("wheel_slip_rl", "wheel_slip_rr"), "brake")
    wheelspin_index = _slip_index(
        rows, ("wheel_slip_rl", "wheel_slip_rr"), "throttle"
    )
    acceleration_deficit_index = max(
        0.0,
        min(1.0, (avg_throttle * 0.70 - avg_longitudinal_accel) / 0.70),
    )
    traction_loss_index = min(
        1.0,
        max(wheelspin_index or 0.0, acceleration_deficit_index * max(avg_throttle, 0.0)),
    )
    brake_instability_index = min(
        1.0,
        max(
            (rear_locking_index or 0.0),
            (front_locking_index or 0.0) * 0.7,
            avg_brake * min(1.0, yaw_rate_abs_avg / 10.0),
        ),
    )
    braking_distance_index = max(
        0.0,
        min(1.0, avg_brake * 0.55 - abs(min(avg_longitudinal_accel, 0.0)) / 2.0),
    )
    brake_balance_values = _numeric_values(rows, "brake_balance")
    brake_balance_sensitivity_index = min(
        1.0, (_std(brake_balance_values) or 0.0) * 0.35 + brake_instability_index * 0.12
    )
    straight_line_speed_deficit_index = max(0.0, min(1.0, (235.0 - avg_speed) / 235.0))
    if avg_throttle < 0.55:
        straight_line_speed_deficit_index *= 0.4
    drag_index = max(
        0.0,
        min(1.0, straight_line_speed_deficit_index + max(0.0, avg_throttle - 0.75) * 0.25),
    )
    aero_balance_shift_index = min(
        1.0, platform_variation_index * 20.0 + abs(oversteer_index - 1.0) * 0.08
    )
    ers_deploy_avg = _mean(_numeric_values(rows, "ers_deploy"))
    rpm_avg = _mean(_numeric_values(rows, "rpm"))
    lap_time_variability = None
    if lap_time and len(lap_durations) >= 2:
        lap_time_variability = (_std(list(lap_durations.values())) or 0.0) / max(
            lap_time, EPSILON
        )
    issue_worsens = (
        (performance_loss is not None and performance_loss >= 0.25)
        or front_temp_category == "rising"
        or rear_temp_category == "rising"
    )

    affected_segments = sorted(
        {
            str(row.get("segment_id") or row.get("corner_id"))
            for row in rows
            if row.get("segment_id") or row.get("corner_id")
        }
    )

    return {
        "avg_speed": avg_speed,
        "min_speed": min_speed,
        "min_speed_loss_ratio": min_speed_loss_ratio,
        "avg_throttle": avg_throttle,
        "avg_brake": avg_brake,
        "max_brake": max_brake,
        "avg_abs_steering": avg_abs_steering,
        "avg_lateral_accel": avg_lateral_accel,
        "avg_abs_lateral_accel": avg_abs_lateral_accel,
        "avg_longitudinal_accel": avg_longitudinal_accel,
        "understeer_index": understeer_index,
        "oversteer_index": oversteer_index,
        "rotation_deficit_index": rotation_deficit_index,
        "yaw_rate_abs_avg": yaw_rate_abs_avg,
        "front_tire_temp_avg": front_tire_temp_avg,
        "rear_tire_temp_avg": rear_tire_temp_avg,
        "front_tire_temp_trend": front_temp_slope,
        "rear_tire_temp_trend": rear_temp_slope,
        "lap_time": lap_time,
        "lap_durations": lap_durations,
        "performance_loss_sec_per_lap": performance_loss,
        "issue_worsens_over_stint": issue_worsens,
        "steering_demand": _category_from_steering(avg_abs_steering),
        "speed_category": _category_from_speed(avg_speed),
        "front_tire_temp_trend_category": front_temp_category,
        "rear_tire_temp_trend_category": rear_temp_category,
        "front_tire_wear_avg": _mean(front_wear_values),
        "rear_tire_wear_avg": _mean(rear_wear_values),
        "tire_wear_avg": _mean(tire_wear_values),
        "tire_wear_trend": tire_wear_slope,
        "brake_temp_avg": _mean(brake_temp_values),
        "brake_temp_trend": brake_temp_slope,
        "brake_temp_trend_category": brake_temp_category,
        "front_locking_index": front_locking_index or 0.0,
        "rear_locking_index": rear_locking_index or 0.0,
        "wheelspin_index": wheelspin_index or 0.0,
        "traction_loss_index": traction_loss_index,
        "brake_instability_index": brake_instability_index,
        "braking_distance_index": braking_distance_index,
        "brake_balance_sensitivity_index": brake_balance_sensitivity_index,
        "straight_line_speed_deficit_index": straight_line_speed_deficit_index,
        "acceleration_deficit_index": acceleration_deficit_index,
        "drag_index": drag_index,
        "aero_balance_shift_index": aero_balance_shift_index,
        "ers_deploy_avg": ers_deploy_avg,
        "rpm_avg": rpm_avg,
        "throttle_aggression": throttle_aggression,
        "brake_release_aggression": brake_release_aggression,
        "lap_time_variability": lap_time_variability,
        "steering_noise_index": steering_noise_index,
        "suspension_variation": suspension_variation,
        "platform_variation_index": platform_variation_index,
        **suspension_features,
        "platform_signal_available": suspension_variation is not None,
        "data_quality_flag": len(loader_warnings) > 12,
        "affected_segments": affected_segments,
        "feature_warnings": loader_warnings,
    }
