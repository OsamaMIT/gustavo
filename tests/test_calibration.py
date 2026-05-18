import pytest

from src.calibration import rank_metrics, run_calibration_report
from src.feature_extractor import extract_features
from src.synthetic_data import SYMPTOM_SCENARIOS, generate_synthetic_rows
from src.symptom_identifier import identify_symptoms


def test_rank_metrics_reports_top_k_and_confidence() -> None:
    metrics = rank_metrics(
        "b",
        {
            "a": 0.50,
            "b": 0.30,
            "c": 0.15,
            "d": 0.05,
        },
    )

    assert metrics.rank == 2
    assert not metrics.top1
    assert metrics.top3
    assert metrics.top5
    assert metrics.top3_confidence == pytest.approx(0.95)


def test_calibration_report_contains_expected_sections() -> None:
    report = run_calibration_report(trials=5, seed=321)

    assert "summary" in report
    assert "symptom_confusion" in report
    assert "likelihood_diagnostics" in report
    assert "validation" in report
    assert len(report["symptom_rows"]) == len(SYMPTOM_SCENARIOS)
    assert report["summary"]["symptom_top3_accuracy"] >= 0.80
    assert report["summary"]["final_top3_accuracy"] >= 0.80
    assert report["likelihood_diagnostics"]["test_information"]
    assert report["likelihood_diagnostics"]["weak_hypothesis_pairs"]
    true_symptoms = {
        key.split("->", maxsplit=1)[0] for key in report["symptom_confusion"]
    }
    assert true_symptoms == set(SYMPTOM_SCENARIOS)


@pytest.mark.parametrize(
    "scenario",
    [
        "entry_understeer",
        "entry_oversteer",
        "front_locking",
        "wheelspin_on_exit",
        "high_speed_instability",
        "straight_line_speed_deficit",
        "brake_fade",
        "bottoming",
    ],
)
def test_representative_symptoms_rank_in_top_three(scenario: str) -> None:
    features = extract_features(
        generate_synthetic_rows(scenario, laps=8, samples_per_lap=80, seed=123)
    )
    ranked = [item.symptom_id for item in identify_symptoms(features, min_confidence=0.0)]

    assert scenario in ranked[:3]
