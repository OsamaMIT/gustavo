from src.hypothesis_model import uniform_belief
from src.optimizer import choose_best_test, expected_information_gain, rank_tests
from src.test_library import load_default_config


def test_expected_information_gain_is_nonnegative() -> None:
    hypotheses, tests, likelihoods = load_default_config()
    belief = uniform_belief(hypotheses)

    for test_id in tests:
        assert expected_information_gain(belief, test_id, tests, likelihoods) >= 0.0


def test_optimizer_chooses_valid_available_test() -> None:
    hypotheses, tests, likelihoods = load_default_config()
    belief = uniform_belief(hypotheses)
    available = ["T1", "T5", "T6"]

    selected = choose_best_test(belief, available, tests, likelihoods)
    rankings = rank_tests(belief, available, tests, likelihoods)

    assert selected in available
    assert rankings[0].test_id == selected

