"""Validation harness comparing active learning against baseline policies."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Mapping

from .baselines import cheapest_first, fixed_sequence, grid_all_tests, random_selection
from .bayes import Likelihoods, bayesian_update, normalize_distribution
from .feature_extractor import extract_features
from .hypothesis_model import (
    candidate_weights_from_detections,
    score_initial_belief,
    top_hypothesis,
)
from .optimizer import choose_best_test, rank_tests
from .outcome_classifier import sample_outcome
from .synthetic_data import generate_synthetic_rows
from .symptom_identifier import identify_symptoms
from .test_library import (
    Hypothesis,
    RnDTest,
    load_default_config,
    load_symptom_hypothesis_map,
    load_symptoms,
)


STRATEGIES = (
    "optimizer",
    "random",
    "cheapest_first",
    "fixed_sequence",
    "grid_all_tests",
)


@dataclass(frozen=True)
class ValidationRun:
    """Result for one strategy on one simulated true hypothesis."""

    strategy: str
    true_hypothesis: str
    tests_used: int
    total_cost: float
    final_top_hypothesis: str
    final_confidence: float
    correct_diagnosis: bool
    final_rank: int
    final_top3_correct: bool
    final_top5_correct: bool
    final_top3_confidence: float
    threshold_reached: bool
    cost_to_correct_diagnosis: float | None
    belief_history: list[dict[str, float]]
    test_sequence: list[tuple[str, str]]


def _initial_belief_for_truth(
    true_hypothesis: str,
    hypotheses: Mapping[str, Hypothesis],
    seed: int,
) -> dict[str, float]:
    rows = generate_synthetic_rows(
        scenario=true_hypothesis,
        laps=8,
        samples_per_lap=80,
        seed=seed,
    )
    features = extract_features(rows)
    detections = identify_symptoms(
        features,
        symptoms=load_symptoms(),
        symptom_hypothesis_map=load_symptom_hypothesis_map(),
    )
    return score_initial_belief(
        features,
        hypotheses,
        candidate_hypothesis_weights=candidate_weights_from_detections(detections[:3]),
    )


def _sample_trial_outcomes(
    true_hypothesis: str,
    tests: Mapping[str, RnDTest],
    likelihoods: Likelihoods,
    rng: random.Random,
) -> dict[str, str]:
    return {
        test_id: sample_outcome(test_id, true_hypothesis, likelihoods, rng)
        for test_id in tests
    }


def _ranked_hypotheses(belief: Mapping[str, float]) -> list[str]:
    return [
        hypothesis
        for hypothesis, _ in sorted(
            belief.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]


def _select_next_test(
    strategy: str,
    belief: Mapping[str, float],
    available_tests: list[str],
    tests: Mapping[str, RnDTest],
    likelihoods: Likelihoods,
    rng: random.Random,
    objective: str,
    confidence_threshold: float,
) -> str:
    if strategy == "optimizer":
        selected = choose_best_test(
            belief,
            available_tests,
            tests,
            likelihoods,
            objective=objective,
            confidence_threshold=confidence_threshold,
        )
        if selected is None:
            raise ValueError("Optimizer could not choose a test")
        return selected
    if strategy == "random":
        return random_selection(available_tests, rng)
    if strategy == "cheapest_first":
        return cheapest_first(available_tests, tests)
    if strategy == "fixed_sequence":
        return fixed_sequence(available_tests)
    if strategy == "grid_all_tests":
        return grid_all_tests(available_tests)
    raise ValueError(f"Unknown validation strategy: {strategy}")


def _run_strategy(
    strategy: str,
    true_hypothesis: str,
    initial_belief: Mapping[str, float],
    tests: Mapping[str, RnDTest],
    likelihoods: Likelihoods,
    outcomes_by_test: Mapping[str, str],
    rng: random.Random,
    confidence_threshold: float,
    max_tests: int,
    objective: str,
    min_expected_information_gain: float,
) -> ValidationRun:
    belief = normalize_distribution(initial_belief)
    available_tests = list(tests.keys())
    belief_history = [belief]
    test_sequence: list[tuple[str, str]] = []
    total_cost = 0.0
    test_limit = len(tests) if strategy == "grid_all_tests" else max_tests

    while available_tests and len(test_sequence) < test_limit:
        if strategy != "grid_all_tests":
            _, confidence = top_hypothesis(belief)
            if confidence >= confidence_threshold:
                break

        if strategy == "optimizer" and min_expected_information_gain > 0.0:
            ranked = rank_tests(
                belief,
                available_tests,
                tests,
                likelihoods,
                objective=objective,
                confidence_threshold=confidence_threshold,
            )
            if (
                not ranked
                or ranked[0].expected_information_gain < min_expected_information_gain
            ):
                break
            test_id = ranked[0].test_id
        else:
            test_id = _select_next_test(
                strategy=strategy,
                belief=belief,
                available_tests=available_tests,
                tests=tests,
                likelihoods=likelihoods,
                rng=rng,
                objective=objective,
                confidence_threshold=confidence_threshold,
            )
        outcome = outcomes_by_test[test_id]
        belief = bayesian_update(belief, test_id, outcome, likelihoods)
        total_cost += tests[test_id].cost
        test_sequence.append((test_id, outcome))
        belief_history.append(belief)
        available_tests.remove(test_id)

    final_top, final_confidence = top_hypothesis(belief)
    ranked = _ranked_hypotheses(belief)
    final_rank = ranked.index(true_hypothesis) + 1 if true_hypothesis in ranked else 999
    final_top3_confidence = sum(float(belief.get(hypothesis, 0.0)) for hypothesis in ranked[:3])
    correct = final_top == true_hypothesis
    threshold_reached = final_confidence >= confidence_threshold
    return ValidationRun(
        strategy=strategy,
        true_hypothesis=true_hypothesis,
        tests_used=len(test_sequence),
        total_cost=total_cost,
        final_top_hypothesis=final_top,
        final_confidence=final_confidence,
        correct_diagnosis=correct,
        final_rank=final_rank,
        final_top3_correct=final_rank <= 3,
        final_top5_correct=final_rank <= 5,
        final_top3_confidence=final_top3_confidence,
        threshold_reached=threshold_reached,
        cost_to_correct_diagnosis=total_cost if correct else None,
        belief_history=belief_history,
        test_sequence=test_sequence,
    )


def compare_strategies(
    trials: int = 100,
    true_hypotheses: list[str] | None = None,
    seed: int = 123,
    confidence_threshold: float = 0.80,
    max_tests: int = 5,
    objective: str = "eig_per_cost",
    min_expected_information_gain: float = 0.15,
) -> dict[str, Any]:
    """Compare optimizer against random, cheapest, fixed, and all-tests baselines."""

    if trials <= 0:
        raise ValueError("trials must be positive")

    hypotheses, tests, likelihoods = load_default_config()
    simulated_truths = true_hypotheses or list(hypotheses)
    unknown_truths = set(simulated_truths) - set(hypotheses)
    if unknown_truths:
        raise ValueError(f"Unknown simulated true hypotheses: {sorted(unknown_truths)}")

    master_rng = random.Random(seed)
    runs_by_strategy: dict[str, list[ValidationRun]] = {name: [] for name in STRATEGIES}

    for trial in range(trials):
        true_hypothesis = master_rng.choice(simulated_truths)
        initial_belief = _initial_belief_for_truth(
            true_hypothesis=true_hypothesis,
            hypotheses=hypotheses,
            seed=seed + trial,
        )
        outcome_rng = random.Random(seed * 1000 + trial)
        outcomes_by_test = _sample_trial_outcomes(
            true_hypothesis=true_hypothesis,
            tests=tests,
            likelihoods=likelihoods,
            rng=outcome_rng,
        )

        for strategy in STRATEGIES:
            strategy_rng = random.Random(seed * 100000 + trial * 10 + STRATEGIES.index(strategy))
            run = _run_strategy(
                strategy=strategy,
                true_hypothesis=true_hypothesis,
                initial_belief=initial_belief,
                tests=tests,
                likelihoods=likelihoods,
                outcomes_by_test=outcomes_by_test,
                rng=strategy_rng,
                confidence_threshold=confidence_threshold,
                max_tests=max_tests,
                objective=objective,
                min_expected_information_gain=min_expected_information_gain,
            )
            runs_by_strategy[strategy].append(run)

    summary: dict[str, Any] = {}
    for strategy, runs in runs_by_strategy.items():
        correct_costs = [
            run.cost_to_correct_diagnosis
            for run in runs
            if run.cost_to_correct_diagnosis is not None
        ]
        by_hypothesis: dict[str, dict[str, float]] = {}
        for hypothesis in simulated_truths:
            hypothesis_runs = [run for run in runs if run.true_hypothesis == hypothesis]
            if not hypothesis_runs:
                continue
            by_hypothesis[hypothesis] = {
                "trials": float(len(hypothesis_runs)),
                "accuracy": sum(run.correct_diagnosis for run in hypothesis_runs)
                / len(hypothesis_runs),
                "top3_accuracy": sum(run.final_top3_correct for run in hypothesis_runs)
                / len(hypothesis_runs),
                "avg_total_cost": sum(run.total_cost for run in hypothesis_runs)
                / len(hypothesis_runs),
                "threshold_reach_rate": sum(
                    run.threshold_reached for run in hypothesis_runs
                )
                / len(hypothesis_runs),
            }
        summary[strategy] = {
            "trials": len(runs),
            "accuracy": sum(run.correct_diagnosis for run in runs) / len(runs),
            "top3_accuracy": sum(run.final_top3_correct for run in runs) / len(runs),
            "top5_accuracy": sum(run.final_top5_correct for run in runs) / len(runs),
            "avg_tests_used": sum(run.tests_used for run in runs) / len(runs),
            "avg_total_cost": sum(run.total_cost for run in runs) / len(runs),
            "avg_final_confidence": sum(run.final_confidence for run in runs) / len(runs),
            "avg_final_top3_confidence": sum(run.final_top3_confidence for run in runs)
            / len(runs),
            "threshold_reach_rate": sum(run.threshold_reached for run in runs) / len(runs),
            "avg_cost_to_correct_diagnosis": (
                sum(correct_costs) / len(correct_costs) if correct_costs else None
            ),
            "per_hypothesis": by_hypothesis,
            "runs": [run.__dict__ for run in runs],
        }

    return {
        "trials": trials,
        "seed": seed,
        "confidence_threshold": confidence_threshold,
        "max_tests": max_tests,
        "objective": objective,
        "min_expected_information_gain": min_expected_information_gain,
        "strategies": summary,
    }
