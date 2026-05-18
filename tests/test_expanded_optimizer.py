from src.hypothesis_model import uniform_belief
from src.optimizer import OBJECTIVES, explain_test, rank_tests
from src.test_library import load_default_config
import pytest


def test_optimizer_supports_all_objective_modes() -> None:
    hypotheses, tests, likelihoods = load_default_config()
    belief = uniform_belief(hypotheses)

    for objective in OBJECTIVES:
        rankings = rank_tests(
            belief,
            ["T1", "T5", "T8"],
            tests,
            likelihoods,
            objective=objective,
        )
        assert rankings
        assert rankings[0].objective == objective
        assert rankings[0].score >= 0.0


def test_optimizer_explanation_contains_outcome_posteriors() -> None:
    hypotheses, tests, likelihoods = load_default_config()
    belief = uniform_belief(hypotheses)

    explanation = explain_test(belief, "T1", tests, likelihoods)

    assert explanation.current_entropy > 0.0
    assert explanation.expected_posterior_entropy > 0.0
    assert explanation.outcome_projections
    assert sum(item.probability for item in explanation.outcome_projections) == pytest.approx(1.0)


def test_two_step_ranking_remains_valid() -> None:
    hypotheses, tests, likelihoods = load_default_config()
    belief = uniform_belief(hypotheses)

    rankings = rank_tests(
        belief,
        ["T1", "T5", "T8"],
        tests,
        likelihoods,
        use_two_step=True,
    )

    assert rankings[0].test_id in {"T1", "T5", "T8"}
