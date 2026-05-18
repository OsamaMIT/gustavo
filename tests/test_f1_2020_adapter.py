import csv
import json

from src.f1_2020_adapter import convert_f1_2020_jsonl_to_csv


def test_f1_2020_jsonl_conversion_maps_core_fields(tmp_path) -> None:
    source = tmp_path / "session.jsonl"
    target = tmp_path / "session.csv"
    record = {
        "m_sessionTime": 12.5,
        "m_currentLapNum": 2,
        "m_speed": 240,
        "m_steer": 0.5,
        "m_throttle": 0.9,
        "m_brake": 0.1,
        "m_gForceLateral": 1.2,
        "m_gForceLongitudinal": 0.4,
        "m_tyresSurfaceTemperature": [91, 92, 96, 97],
        "m_tyresWear": [3, 4, 5, 6],
        "m_brakesTemperature": [500, 510, 520, 530],
        "m_wheelSlip": [0.02, 0.03, 0.04, 0.05],
    }
    source.write_text(json.dumps(record) + "\n", encoding="utf-8")

    convert_f1_2020_jsonl_to_csv(source, target)

    with target.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert rows[0]["speed"] == "240"
    assert float(rows[0]["steering_angle"]) == 11.0
    assert rows[0]["front_left_temp"] == "96"
    assert rows[0]["rear_right_temp"] == "92"

