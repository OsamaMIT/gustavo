from src.feature_extractor import extract_features
from src.synthetic_data import SYMPTOM_SCENARIOS, generate_synthetic_rows
from src.symptom_identifier import identify_symptoms


def test_expanded_feature_extraction_includes_key_indices() -> None:
    rows = generate_synthetic_rows("wheelspin_on_exit", laps=3, samples_per_lap=24)
    features = extract_features(rows)

    assert features["wheelspin_index"] > 0.0
    assert features["traction_loss_index"] > 0.0
    assert features["rear_tire_temp_avg"] is not None


def test_every_synthetic_symptom_produces_a_detection_candidate() -> None:
    for scenario in SYMPTOM_SCENARIOS:
        rows = generate_synthetic_rows(scenario, laps=2, samples_per_lap=16)
        detections = identify_symptoms(extract_features(rows), min_confidence=0.10)
        assert detections, scenario
        assert detections[0].confidence >= 0.10

