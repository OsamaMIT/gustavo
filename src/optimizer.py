"""Expected-information-gain optimizer for R&D test selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .bayes import Likelihoods, bayesian_update, entropy
from .hypothesis_model import top_hypothesis
from .test_library import RnDTest


OBJECTIVES = (
    "eig_per_cost",
    "eig",
    "confidence_gain_per_cost",
    "threshold_probability_per_cost",
)


@dataclass(frozen=True)
class OutcomeProjection:
    """Projected posterior state for one possible test outcome."""

    outcome: str
    probability: float
    posterior_entropy: float
    posterior_belief: dict[str, float]
    top_hypothesis: str
    top_probability: float
    reaches_threshold: bool


@dataclass(frozen=True)
class TestScore:
    """Ranking details for one candidate test."""

    test_id: str
    expected_information_gain: float
    cost: float
    score: float
    objective: str = "eig_per_cost"
    current_entropy: float = 0.0
    expected_posterior_entropy: float = 0.0
    expected_confidence_gain: float = 0.0
    threshold_reach_probability: float = 0.0
    two_step_bonus: float = 0.0
    outcome_projections: tuple[OutcomeProjection, ...] = ()


def _get_test(tests: Mapping[str, RnDTest], test_id: str) -> RnDTest:
    try:
        return tests[test_id]
    except KeyError as exc:
        raise KeyError(f"Unknown test id: {test_id}") from exc


def _outcome_probability(
    belief: Mapping[str, float],
    test_id: str,
    outcome: str,
    likelihoods: Likelihoods,
) -> float:
    return sum(
        float(probability) * float(likelihoods[test_id][hypothesis][outcome])
        for hypothesis, probability in belief.items()
    )


def explain_test(
    belief: Mapping[str, float],
    test_id: str,
    tests: Mapping[str, RnDTest],
    likelihoods: Likelihoods,
    objective: str = "eig_per_cost",
    confidence_threshold: float = 0.80,
    remaining_tests: Iterable[str] | None = None,
    use_two_step: bool = False,
) -> TestScore:
    """Return a complete explainable score for one candidate test."""

    if objective not in OBJECTIVES:
        raise ValueError(f"Unknown optimizer objective {objective!r}")

    test = _get_test(tests, test_id)
    if test.cost <= 0.0:
        raise ValueError(f"Test {test_id} has nonpositive cost: {test.cost}")

    current_entropy = entropy(belief)
    _, current_top_probability = top_hypothesis(belief)
    expected_posterior_entropy = 0.0
    expected_top_probability = 0.0
    threshold_reach_probability = 0.0
    projections: list[OutcomeProjection] = []

    for outcome in test.outcomes:
        outcome_probability = _outcome_probability(
            belief, test_id, outcome, likelihoods
        )
        if outcome_probability <= 0.0:
            continue
        posterior = bayesian_update(belief, test_id, outcome, likelihoods)
        posterior_entropy = entropy(posterior)
        top_id, top_probability = top_hypothesis(posterior)
        reaches_threshold = top_probability >= confidence_threshold
        expected_posterior_entropy += outcome_probability * posterior_entropy
        expected_top_probability += outcome_probability * top_probability
        if reaches_threshold:
            threshold_reach_probability += outcome_probability
        projections.append(
            OutcomeProjection(
                outcome=outcome,
                probability=outcome_probability,
                posterior_entropy=posterior_entropy,
                posterior_belief=posterior,
                top_hypothesis=top_id,
                top_probability=top_probability,
                reaches_threshold=reaches_threshold,
            )
        )

    eig = max(0.0, current_entropy - expected_posterior_entropy)
    expected_confidence_gain = max(0.0, expected_top_probability - current_top_probability)

    if objective == "eig":
        score = eig
    elif objective == "confidence_gain_per_cost":
        score = expected_confidence_gain / test.cost
    elif objective == "threshold_probability_per_cost":
        score = threshold_reach_probability / test.cost
    else:
        score = eig / test.cost

    two_step_bonus = 0.0
    if use_two_step and remaining_tests is not None:
        remaining = [item for item in remaining_tests if item != test_id]
        if remaining:
            expected_second_value = 0.0
            expected_second_cost = 0.0
            for projection in projections:
                ranked_second = rank_tests(
                    projection.posterior_belief,
                    remaining,
                    tests,
                    likelihoods,
                    objective=objective,
                    confidence_threshold=confidence_threshold,
                    use_two_step=False,
                )
                if ranked_second:
                    best_second = ranked_second[0]
                    expected_second_value += (
                        projection.probability * best_second.expected_information_gain
                    )
                    expected_second_cost += projection.probability * best_second.cost
            if expected_second_cost > 0.0:
                combined_score = (eig + expected_second_value) / (
                    test.cost + expected_second_cost
                )
                two_step_bonus = max(0.0, combined_score - score)
                score = combined_score

    return TestScore(
        test_id=test_id,
        expected_information_gain=eig,
        cost=test.cost,
        score=score,
        objective=objective,
        current_entropy=current_entropy,
        expected_posterior_entropy=expected_posterior_entropy,
        expected_confidence_gain=expected_confidence_gain,
        threshold_reach_probability=threshold_reach_probability,
        two_step_bonus=two_step_bonus,
        outcome_projections=tuple(projections),
    )


def expected_information_gain(
    belief: Mapping[str, float],
    test_id: str,
    tests: Mapping[str, RnDTest],
    likelihoods: Likelihoods,
) -> float:
    """Compute current entropy minus expected posterior entropy."""

    return explain_test(belief, test_id, tests, likelihoods).expected_information_gain


def score_test(
    belief: Mapping[str, float],
    test_id: str,
    tests: Mapping[str, RnDTest],
    likelihoods: Likelihoods,
    objective: str = "eig_per_cost",
    confidence_threshold: float = 0.80,
) -> float:
    """Return the selected objective score for one test."""

    return explain_test(
        belief,
        test_id,
        tests,
        likelihoods,
        objective=objective,
        confidence_threshold=confidence_threshold,
    ).score


def rank_tests(
    belief: Mapping[str, float],
    candidate_tests: Iterable[str],
    tests: Mapping[str, RnDTest],
    likelihoods: Likelihoods,
    objective: str = "eig_per_cost",
    confidence_threshold: float = 0.80,
    use_two_step: bool = False,
) -> list[TestScore]:
    """Rank candidate tests by the requested active-learning objective."""

    candidate_list = list(candidate_tests)
    rankings = [
        explain_test(
            belief,
            test_id,
            tests,
            likelihoods,
            objective=objective,
            confidence_threshold=confidence_threshold,
            remaining_tests=candidate_list,
            use_two_step=use_two_step,
        )
        for test_id in candidate_list
    ]
    return sorted(
        rankings,
        key=lambda item: (
            -item.score,
            -item.expected_information_gain,
            item.cost,
            item.test_id,
        ),
    )


def choose_best_test(
    belief: Mapping[str, float],
    candidate_tests: Iterable[str],
    tests: Mapping[str, RnDTest],
    likelihoods: Likelihoods,
    objective: str = "eig_per_cost",
    confidence_threshold: float = 0.80,
    use_two_step: bool = False,
) -> str | None:
    """Return the best available test id, or None when no tests are available."""

    ranked = rank_tests(
        belief,
        candidate_tests,
        tests,
        likelihoods,
        objective=objective,
        confidence_threshold=confidence_threshold,
        use_two_step=use_two_step,
    )
    if not ranked:
        return None
    return ranked[0].test_id

