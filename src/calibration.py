"""Calibration diagnostics for the expanded diagnostic model."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping

from .diagnosis_engine import DiagnosisEngine
from .feature_extractor import extract_features
from .hypothesis_model import top_hypothesis, uniform_belief
from .optimizer import expected_information_gain
from .synthetic_data import SYMPTOM_SCENARIOS, generate_synthetic_rows, true_hypothesis_for_scenario
from .symptom_identifier import identify_symptoms
from .test_library import Hypothesis, RnDTest, load_default_config
from .validation import compare_strategies


@dataclass(frozen=True)
class RankMetrics:
    """Rank and top-k metrics for one true id against a ranked distribution."""

    rank: int
    top1: bool
    top3: bool
    top5: bool
    top3_confidence: float


def ranked_ids_from_distribution(distribution: Mapping[str, float]) -> list[str]:
    """Return ids sorted by descending probability."""

    return [
        key
        for key, _ in sorted(
            distribution.items(),
            key=lambda item: (-float(item[1]), item[0]),
        )
    ]


def rank_metrics(
    true_id: str,
    distribution: Mapping[str, float],
    missing_rank: int = 999,
) -> RankMetrics:
    """Compute rank and top-k containment for a true id."""

    ranked = ranked_ids_from_distribution(distribution)
    rank = ranked.index(true_id) + 1 if true_id in ranked else missing_rank
    top3_confidence = sum(float(distribution.get(item, 0.0)) for item in ranked[:3])
    return RankMetrics(
        rank=rank,
        top1=rank == 1,
        top3=rank <= 3,
        top5=rank <= 5,
        top3_confidence=top3_confidence,
    )


def symptom_rank_metrics(true_symptom: str, ranked_symptoms: list[str]) -> RankMetrics:
    """Compute rank metrics for symptom detections."""

    rank = ranked_symptoms.index(true_symptom) + 1 if true_symptom in ranked_symptoms else 999
    return RankMetrics(
        rank=rank,
        top1=rank == 1,
        top3=rank <= 3,
        top5=rank <= 5,
        top3_confidence=0.0,
    )


def _rank_bucket(rank: int) -> str:
    if rank == 1:
        return "rank_1"
    if rank <= 3:
        return "rank_2_3"
    if rank <= 5:
        return "rank_4_5"
    if rank <= 10:
        return "rank_6_10"
    return "rank_gt_10"


def _diagnose_once(
    engine: DiagnosisEngine,
    true_hypothesis: str,
    features: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    result = engine.run_with_simulated_truth(
        true_hypothesis=true_hypothesis,
        initial_features=features,
        seed=seed,
    )
    final_belief = result["final_belief"]
    metrics = rank_metrics(true_hypothesis, final_belief)
    return {
        **result,
        "final_rank": metrics.rank,
        "final_top3_correct": metrics.top3,
        "final_top5_correct": metrics.top5,
        "final_top3_confidence": metrics.top3_confidence,
    }


def _total_variation_distance(
    left: Mapping[str, float],
    right: Mapping[str, float],
) -> float:
    outcomes = set(left) | set(right)
    return 0.5 * sum(
        abs(float(left.get(outcome, 0.0)) - float(right.get(outcome, 0.0)))
        for outcome in outcomes
    )


def likelihood_diagnostics(
    hypotheses: Mapping[str, Hypothesis],
    tests: Mapping[str, RnDTest],
    likelihoods: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, Any]:
    """Summarize likelihood coverage, test value, and weak hypothesis separation."""

    uniform = uniform_belief(hypotheses)
    test_information = sorted(
        [
            {
                "test_id": test_id,
                "name": test.name,
                "cost": test.cost,
                "expected_information_gain": expected_information_gain(
                    uniform, test_id, tests, likelihoods
                ),
                "eig_per_cost": expected_information_gain(
                    uniform, test_id, tests, likelihoods
                )
                / test.cost,
                "relevant_hypotheses": len(test.relevant_hypotheses),
                "outcomes": len(test.outcomes),
            }
            for test_id, test in tests.items()
        ],
        key=lambda item: (-item["eig_per_cost"], -item["expected_information_gain"], item["test_id"]),
    )

    coverage: list[dict[str, Any]] = []
    for hypothesis_id in hypotheses:
        relevant_tests = [
            test_id
            for test_id, test in tests.items()
            if hypothesis_id in test.relevant_hypotheses
        ]
        mapped_outcomes = [
            f"{test_id}:{outcome}"
            for test_id, test in tests.items()
            for outcome, mapped in test.outcome_hypothesis_map.items()
            if hypothesis_id in mapped
        ]
        coverage.append(
            {
                "hypothesis": hypothesis_id,
                "category": hypotheses[hypothesis_id].category,
                "relevant_test_count": len(relevant_tests),
                "mapped_outcome_count": len(mapped_outcomes),
                "relevant_tests": relevant_tests,
                "mapped_outcomes": mapped_outcomes,
            }
        )
    coverage = sorted(
        coverage,
        key=lambda item: (
            item["mapped_outcome_count"],
            item["relevant_test_count"],
            item["hypothesis"],
        ),
    )

    weak_pairs: list[dict[str, Any]] = []
    for left, right in combinations(hypotheses, 2):
        same_category = hypotheses[left].category == hypotheses[right].category
        shared_relevant_test = any(
            left in test.relevant_hypotheses and right in test.relevant_hypotheses
            for test in tests.values()
        )
        if not same_category and not shared_relevant_test:
            continue
        best_test = None
        best_separation = -1.0
        for test_id in tests:
            separation = _total_variation_distance(
                likelihoods[test_id][left],
                likelihoods[test_id][right],
            )
            if separation > best_separation:
                best_separation = separation
                best_test = test_id
        weak_pairs.append(
            {
                "left": left,
                "right": right,
                "category": hypotheses[left].category
                if same_category
                else "cross_category",
                "best_test": best_test,
                "best_separation": best_separation,
            }
        )
    weak_pairs = sorted(
        weak_pairs,
        key=lambda item: (item["best_separation"], item["category"], item["left"], item["right"]),
    )

    return {
        "test_information": test_information,
        "weak_hypothesis_pairs": weak_pairs[:20],
        "hypothesis_coverage": coverage,
    }


def run_calibration_report(
    trials: int = 100,
    seed: int = 123,
    confidence_threshold: float = 0.80,
    max_tests: int = 5,
    objective: str = "eig_per_cost",
    min_expected_information_gain: float = 0.15,
) -> dict[str, Any]:
    """Run calibration diagnostics across all synthetic symptom scenarios."""

    hypotheses, tests, likelihoods = load_default_config()
    engine = DiagnosisEngine(
        hypotheses=hypotheses,
        tests=tests,
        likelihoods=likelihoods,
        confidence_threshold=confidence_threshold,
        max_tests=max_tests,
        objective=objective,
        min_expected_information_gain=min_expected_information_gain,
    )

    symptom_confusion: Counter[tuple[str, str]] = Counter()
    hypothesis_confusion: Counter[tuple[str, str]] = Counter()
    initial_rank_buckets: Counter[str] = Counter()
    symptom_rows: list[dict[str, Any]] = []
    diagnosis_rows: list[dict[str, Any]] = []

    for index, scenario in enumerate(SYMPTOM_SCENARIOS):
        features = extract_features(
            generate_synthetic_rows(
                scenario,
                laps=8,
                samples_per_lap=80,
                seed=seed + index,
            )
        )
        detections = identify_symptoms(features, min_confidence=0.0)
        ranked_symptom_ids = [detection.symptom_id for detection in detections]
        detected_symptom = ranked_symptom_ids[0]
        symptom_metrics = symptom_rank_metrics(scenario, ranked_symptom_ids)
        symptom_confusion[(scenario, detected_symptom)] += 1

        true_hypothesis = true_hypothesis_for_scenario(scenario)
        initial_belief = engine.initialize_belief(features)
        initial_metrics = rank_metrics(true_hypothesis, initial_belief)
        initial_rank_buckets[_rank_bucket(initial_metrics.rank)] += 1

        diagnosis = _diagnose_once(
            engine=engine,
            true_hypothesis=true_hypothesis,
            features=features,
            seed=seed * 1000 + index,
        )
        hypothesis_confusion[
            (true_hypothesis, str(diagnosis["final_top_hypothesis"]))
        ] += 1

        symptom_rows.append(
            {
                "scenario": scenario,
                "detected_symptom": detected_symptom,
                "symptom_rank": symptom_metrics.rank,
                "symptom_top1": symptom_metrics.top1,
                "symptom_top3": symptom_metrics.top3,
                "true_hypothesis": true_hypothesis,
                "initial_true_rank": initial_metrics.rank,
                "initial_true_top1": initial_metrics.top1,
                "initial_true_top3": initial_metrics.top3,
                "initial_true_top5": initial_metrics.top5,
                "initial_top3_confidence": initial_metrics.top3_confidence,
            }
        )
        diagnosis_rows.append(diagnosis)

    validation = compare_strategies(
        trials=trials,
        seed=seed,
        confidence_threshold=confidence_threshold,
        max_tests=max_tests,
        objective=objective,
        min_expected_information_gain=min_expected_information_gain,
    )
    likelihood_report = likelihood_diagnostics(hypotheses, tests, likelihoods)
    optimizer_runs = validation["strategies"]["optimizer"]["runs"]

    def average(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    final_top3_accuracy = average(
        [1.0 if run.get("final_top3_correct") else 0.0 for run in diagnosis_rows]
    )
    if optimizer_runs and "final_top3_correct" in optimizer_runs[0]:
        final_top3_accuracy = average(
            [1.0 if run["final_top3_correct"] else 0.0 for run in optimizer_runs]
        )

    worst_symptom_confusions = [
        {"true": true, "predicted": predicted, "count": count}
        for (true, predicted), count in symptom_confusion.most_common()
        if true != predicted
    ][:10]
    worst_hypothesis_confusions = [
        {"true": true, "predicted": predicted, "count": count}
        for (true, predicted), count in hypothesis_confusion.most_common()
        if true != predicted
    ][:10]

    summary = {
        "symptom_top1_accuracy": average(
            [1.0 if row["symptom_top1"] else 0.0 for row in symptom_rows]
        ),
        "symptom_top3_accuracy": average(
            [1.0 if row["symptom_top3"] else 0.0 for row in symptom_rows]
        ),
        "initial_true_top1_accuracy": average(
            [1.0 if row["initial_true_top1"] else 0.0 for row in symptom_rows]
        ),
        "initial_true_top3_accuracy": average(
            [1.0 if row["initial_true_top3"] else 0.0 for row in symptom_rows]
        ),
        "initial_true_top5_accuracy": average(
            [1.0 if row["initial_true_top5"] else 0.0 for row in symptom_rows]
        ),
        "initial_rank_buckets": dict(initial_rank_buckets),
        "final_top1_accuracy": validation["strategies"]["optimizer"]["accuracy"],
        "final_top3_accuracy": final_top3_accuracy,
        "avg_final_top3_confidence": validation["strategies"]["optimizer"][
            "avg_final_top3_confidence"
        ],
        "threshold_reach_rate": validation["strategies"]["optimizer"][
            "threshold_reach_rate"
        ],
        "avg_tests_used": validation["strategies"]["optimizer"]["avg_tests_used"],
        "avg_total_cost": validation["strategies"]["optimizer"]["avg_total_cost"],
        "avg_cost_to_correct_diagnosis": validation["strategies"]["optimizer"][
            "avg_cost_to_correct_diagnosis"
        ],
    }

    return {
        "trials": trials,
        "seed": seed,
        "confidence_threshold": confidence_threshold,
        "max_tests": max_tests,
        "objective": objective,
        "min_expected_information_gain": min_expected_information_gain,
        "summary": summary,
        "symptom_rows": symptom_rows,
        "diagnosis_rows": diagnosis_rows,
        "likelihood_diagnostics": likelihood_report,
        "symptom_confusion": {
            f"{true}->{predicted}": count
            for (true, predicted), count in symptom_confusion.items()
        },
        "hypothesis_confusion": {
            f"{true}->{predicted}": count
            for (true, predicted), count in hypothesis_confusion.items()
        },
        "worst_symptom_confusions": worst_symptom_confusions,
        "worst_hypothesis_confusions": worst_hypothesis_confusions,
        "validation": validation,
    }
