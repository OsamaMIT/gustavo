"""CSV telemetry loading with tolerant optional-column handling."""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "timestamp",
    "lap_number",
    "speed",
    "steering_angle",
    "throttle",
    "brake",
    "lateral_accel",
}

EXPECTED_COLUMNS = REQUIRED_COLUMNS | {
    "distance",
    "longitudinal_accel",
    "front_left_temp",
    "front_right_temp",
    "rear_left_temp",
    "rear_right_temp",
    "tire_wear_fl",
    "tire_wear_fr",
    "tire_wear_rl",
    "tire_wear_rr",
    "suspension_fl",
    "suspension_fr",
    "suspension_rl",
    "suspension_rr",
    "brake_temp_fl",
    "brake_temp_fr",
    "brake_temp_rl",
    "brake_temp_rr",
    "wheel_slip_fl",
    "wheel_slip_fr",
    "wheel_slip_rl",
    "wheel_slip_rr",
    "yaw_rate",
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
}

TEXT_COLUMNS = {"setup_id", "corner_id", "segment_id"}


@dataclass(frozen=True)
class TelemetryDataset:
    """Loaded telemetry rows plus metadata about CSV coverage."""

    rows: list[dict[str, Any]]
    columns: set[str]
    warnings: list[str]
    source: Path | None = None


def _parse_value(column: str, value: str | None) -> Any:
    if value is None or value == "":
        return None
    if column in TEXT_COLUMNS:
        return value
    try:
        number = float(value)
    except ValueError:
        return value
    if column == "lap_number":
        return int(number)
    return number


def load_telemetry_csv(path: str | Path) -> TelemetryDataset:
    """Load simulator telemetry from CSV.

    Required columns produce a clear error when missing. Optional expected
    columns produce warnings and are represented as unavailable features later.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Telemetry CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Telemetry CSV has no header row: {csv_path}")

        columns = {str(column) for column in reader.fieldnames}
        missing_required = sorted(REQUIRED_COLUMNS - columns)
        if missing_required:
            raise ValueError(
                "Telemetry CSV is missing required columns: "
                + ", ".join(missing_required)
            )

        warning_messages: list[str] = []
        for column in sorted(EXPECTED_COLUMNS - REQUIRED_COLUMNS - columns):
            message = f"Optional telemetry column missing: {column}"
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            warning_messages.append(message)

        rows = [
            {column: _parse_value(column, value) for column, value in row.items()}
            for row in reader
        ]

    if not rows:
        raise ValueError(f"Telemetry CSV contains no data rows: {csv_path}")

    return TelemetryDataset(
        rows=rows,
        columns=columns,
        warnings=warning_messages,
        source=csv_path,
    )
