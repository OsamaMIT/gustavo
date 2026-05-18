import pytest

from src.test_library import (
    CONFIG_DIR,
    read_json,
    load_default_config,
    load_symptom_hypothesis_map,
    load_symptoms,
)
from src.synthetic_data import HYPOTHESIS_SCENARIO_MAP, SYMPTOM_SCENARIOS


def test_expanded_registry_loads_and_maps_every_symptom() -> None:
    hypotheses, tests, likelihoods = load_default_config()
    symptoms = load_symptoms()
    mapping = load_symptom_hypothesis_map()

    assert len(symptoms) >= 45
    assert len(hypotheses) >= 46
    assert len(tests) >= 18
    assert set(symptoms) == set(mapping)
    for symptom_id, candidate_hypotheses in mapping.items():
        assert candidate_hypotheses, symptom_id
        assert set(candidate_hypotheses) <= set(hypotheses)

    assert set(likelihoods) == set(tests)


def test_generated_likelihood_tables_sum_to_one_for_full_registry() -> None:
    hypotheses, tests, likelihoods = load_default_config()

    for test_id, test in tests.items():
        for hypothesis in hypotheses:
            outcomes = likelihoods[test_id][hypothesis]
            assert set(outcomes) == set(test.outcomes)
            assert sum(outcomes.values()) == pytest.approx(1.0)


def test_likelihood_overrides_reference_valid_schema() -> None:
    hypotheses, tests, likelihoods = load_default_config()
    overrides = read_json(CONFIG_DIR / "likelihood_overrides.json")

    for test_id, hypothesis_overrides in overrides.items():
        assert test_id in tests
        for hypothesis_id, outcomes in hypothesis_overrides.items():
            assert hypothesis_id in hypotheses
            assert set(outcomes) == set(tests[test_id].outcomes)
            assert sum(likelihoods[test_id][hypothesis_id].values()) == pytest.approx(1.0)


def test_every_hypothesis_has_representative_synthetic_scenario() -> None:
    hypotheses, _, _ = load_default_config()

    assert set(HYPOTHESIS_SCENARIO_MAP) == set(hypotheses)
    assert set(HYPOTHESIS_SCENARIO_MAP.values()) <= set(SYMPTOM_SCENARIOS)
