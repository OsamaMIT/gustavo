import pytest

from src.diagnosis_engine import DiagnosisEngine
from src.test_library import load_default_config


def test_diagnosis_engine_stops_at_confidence_threshold() -> None:
    hypotheses, tests, likelihoods = load_default_config()
    engine = DiagnosisEngine(
        hypotheses=hypotheses,
        tests=tests,
        likelihoods=likelihoods,
        confidence_threshold=0.80,
        max_tests=5,
    )
    initial_belief = {
        "front_aero_load_limitation": 0.04,
        "front_tire_thermal_saturation": 0.84,
        "platform_aero_sensitivity": 0.04,
        "mechanical_balance_limitation": 0.04,
        "driver_input_contribution": 0.04,
    }

    result = engine.run_with_simulated_truth(
        "front_tire_thermal_saturation",
        initial_belief=initial_belief,
        seed=1,
    )

    assert result["tests_used"] == 0
    assert result["stop_reason"] == "confidence_threshold"
    assert result["final_confidence"] >= 0.80


def test_likelihood_tables_sum_to_one_for_every_test_and_hypothesis() -> None:
    hypotheses, tests, likelihoods = load_default_config()

    for test_id in tests:
        for hypothesis in hypotheses:
            assert sum(likelihoods[test_id][hypothesis].values()) == pytest.approx(1.0)

