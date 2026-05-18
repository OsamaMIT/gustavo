import math

import pytest

from src.bayes import bayesian_update, entropy, normalize_distribution
from src.test_library import load_default_config


def test_entropy_uniform_five_hypotheses() -> None:
    belief = {f"h{i}": 0.2 for i in range(5)}
    assert entropy(belief) == pytest.approx(math.log2(5))


def test_bayesian_update_normalizes_distribution() -> None:
    hypotheses, _, likelihoods = load_default_config()
    prior = normalize_distribution({hypothesis: 1.0 for hypothesis in hypotheses})

    posterior = bayesian_update(
        prior,
        "T5",
        "issue_reduced_with_smoother_inputs",
        likelihoods,
    )

    assert sum(posterior.values()) == pytest.approx(1.0)
    assert posterior["driver_input_contribution"] > prior["driver_input_contribution"]

