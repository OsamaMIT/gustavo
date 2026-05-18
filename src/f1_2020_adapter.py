"""Optional F1 2020 telemetry conversion into the internal CSV schema."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

from .synthetic_data import CSV_COLUMNS


def _first(record: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return default


def _tyre_value(values: Any, index: int) -> Any:
    if isinstance(values, list) and len(values) > index:
        return values[index]
    if isinstance(values, dict):
        order_keys = [
            ("rear_left", "rl", "0"),
            ("rear_right", "rr", "1"),
            ("front_left", "fl", "2"),
            ("front_right", "fr", "3"),
        ]
        for key in order_keys[index]:
            if key in values:
                return values[key]
    return None


def _normalise_steer(value: Any) -> Any:
    if not isinstance(value, (int, float)):
        return value
    value = float(value)
    if -1.1 <= value <= 1.1:
        return value * 22.0
    return value


def convert_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one flexible F1 2020 JSON record into internal telemetry columns."""

    surface_temps = _first(
        record, "tyresSurfaceTemperature", "m_tyresSurfaceTemperature", default=[]
    )
    tyre_wear = _first(record, "tyresWear", "m_tyresWear", default=[])
    brake_temps = _first(record, "brakesTemperature", "m_brakesTemperature", default=[])
    wheel_slip = _first(record, "wheelSlip", "m_wheelSlip", default=[])

    row = {column: "" for column in CSV_COLUMNS}
    row.update(
        {
            "timestamp": _first(record, "timestamp", "sessionTime", "m_sessionTime", default=0.0),
            "lap_number": _first(record, "lap_number", "currentLapNum", "m_currentLapNum", default=1),
            "distance": _first(record, "distance", "lapDistance", "m_lapDistance", default=""),
            "speed": _first(record, "speed", "m_speed", default=""),
            "steering_angle": _normalise_steer(_first(record, "steer", "m_steer", "steering_angle", default="")),
            "throttle": _first(record, "throttle", "m_throttle", default=""),
            "brake": _first(record, "brake", "m_brake", default=""),
            "lateral_accel": _first(record, "gForceLateral", "m_gForceLateral", "lateral_accel", default=""),
            "longitudinal_accel": _first(record, "gForceLongitudinal", "m_gForceLongitudinal", "longitudinal_accel", default=""),
            "yaw_rate": _first(record, "yawRate", "m_yawRate", "yaw_rate", default=""),
            "front_left_temp": _tyre_value(surface_temps, 2),
            "front_right_temp": _tyre_value(surface_temps, 3),
            "rear_left_temp": _tyre_value(surface_temps, 0),
            "rear_right_temp": _tyre_value(surface_temps, 1),
            "tire_wear_fl": _tyre_value(tyre_wear, 2),
            "tire_wear_fr": _tyre_value(tyre_wear, 3),
            "tire_wear_rl": _tyre_value(tyre_wear, 0),
            "tire_wear_rr": _tyre_value(tyre_wear, 1),
            "brake_temp_fl": _tyre_value(brake_temps, 2),
            "brake_temp_fr": _tyre_value(brake_temps, 3),
            "brake_temp_rl": _tyre_value(brake_temps, 0),
            "brake_temp_rr": _tyre_value(brake_temps, 1),
            "wheel_slip_fl": _tyre_value(wheel_slip, 2),
            "wheel_slip_fr": _tyre_value(wheel_slip, 3),
            "wheel_slip_rl": _tyre_value(wheel_slip, 0),
            "wheel_slip_rr": _tyre_value(wheel_slip, 1),
            "gear": _first(record, "gear", "m_gear", default=""),
            "rpm": _first(record, "engineRPM", "m_engineRPM", "rpm", default=""),
            "ers_deploy": _first(record, "ersDeployMode", "m_ersDeployMode", "ers_deploy", default=""),
            "ers_energy": _first(record, "ersStoreEnergy", "m_ersStoreEnergy", "ers_energy", default=""),
            "fuel_load": _first(record, "fuelInTank", "m_fuelInTank", "fuel_load", default=""),
            "drs": _first(record, "drs", "m_drs", default=""),
            "setup_id": _first(record, "setup_id", default="f1_2020_import"),
            "corner_id": _first(record, "corner_id", default="unknown"),
            "segment_id": _first(record, "segment_id", default="unknown"),
        }
    )
    return row


def convert_f1_2020_jsonl_to_csv(input_path: str | Path, output_path: str | Path) -> Path:
    """Convert JSONL telemetry logs into the internal CSV schema."""

    source = Path(input_path)
    target = Path(output_path)
    if not source.exists():
        raise FileNotFoundError(f"F1 2020 JSONL telemetry file not found: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    converted_rows: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {source}") from exc
            converted_rows.append(convert_record(record))

    if not converted_rows:
        raise ValueError(f"No telemetry records found in {source}")

    with target.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(converted_rows)

    return target

