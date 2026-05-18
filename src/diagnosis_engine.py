"""End-to-end diagnosis engine for Bayesian R&D test minimization."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from .bayes import Likelihoods, bayesian_update, normalize_distribution
from .hypothesis_model import (
    candidate_weights_from_detections,
    score_initial_belief,
    top_hypothesis,
    uniform_belief,
)
from .optimizer import TestScore, choose_best_test, rank_tests
from .outcome_classifier import sample_outcome
from .symptom_identifier import identify_symptoms
from .test_library import (
    Hypothesis,
    RnDTest,
    load_symptom_hypothesis_map,
    load_symptoms,
    validate_likelihood_tables,
)


@dataclass(frozen=True)
class ExecutedTest:
    """One executed test and its observed outcome."""

    test_id: str
    outcome: str
    cost: float


class DiagnosisEngine:
    """Bayesian active-learning engine for selecting minimum-cost R&D tests."""

    def __init__(
        self,
        hypotheses: Mapping[str, Hypothesis] | Iterable[str],
        tests: Mapping[str, RnDTest],
        likelihoods: Likelihoods,
        confidence_threshold: float = 0.80,
        max_tests: int = 5,
        objective: str = "eig_per_cost",
        min_expected_information_gain: float = 0.15,
        use_two_step: bool = False,
    ) -> None:
        if isinstance(hypotheses, Mapping):
            self.hypotheses = dict(hypotheses)
            self.hypothesis_ids = list(hypotheses.keys())
        else:
            self.hypothesis_ids = [str(item) for item in hypotheses]
            self.hypotheses = {
                hypothesis_id: Hypothesis(id=hypothesis_id, meaning="")
                for hypothesis_id in self.hypothesis_ids
            }
        self.tests = dict(tests)
        self.likelihoods = likelihoods
        self.confidence_threshold = confidence_threshold
        self.max_tests = max_tests
        self.objective = objective
        self.min_expected_information_gain = min_expected_information_gain
        self.use_two_step = use_two_step
        self.symptoms = load_symptoms()
        self.symptom_hypothesis_map = load_symptom_hypothesis_map()
        validate_likelihood_tables(self.hypotheses, self.tests, dict(likelihoods))

    def initialize_belief(self, features: Mapping[str, Any]) -> dict[str, float]:
        """Create an initial hypothesis belief from extracted telemetry features."""

        if not features:
            return uniform_belief(self.hypothesis_ids)
        detections = identify_symptoms(
            features,
            symptoms=self.symptoms,
            symptom_hypothesis_map=self.symptom_hypothesis_map,
        )
        candidate_weights = candidate_weights_from_detections(detections[:3])
        return score_initial_belief(
            features,
            self.hypothesis_ids,
            candidate_hypothesis_weights=candidate_weights,
        )

    def available_tests_for_symptom(self, symptom_id: str | None) -> list[str]:
        """Return tests relevant to a symptom's candidate hypotheses."""

        if not symptom_id:
            return list(self.tests.keys())
        candidate_hypotheses = set(self.symptom_hypothesis_map.get(symptom_id, ()))
        if not candidate_hypotheses:
            return list(self.tests.keys())
        filtered = [
            test_id
            for test_id, test in self.tests.items()
            if candidate_hypotheses.intersection(test.relevant_hypotheses)
        ]
        return filtered or list(self.tests.keys())

    def _apply_dependency_boosts(
        self,
        rankings: list[TestScore],
        executed_tests: Iterable[ExecutedTest] | None,
    ) -> list[TestScore]:
        if not executed_tests:
            return rankings
        boosts: dict[str, float] = {}
        for executed in executed_tests:
            if (
                executed.test_id == "T1"
                and executed.outcome == "improves_balance_but_worsens_front_temps"
            ):
                boosts["T6"] = boosts.get("T6", 0.0) + 0.05
            if executed.test_id == "T8" and executed.outcome in {
                "forward_bias_improves",
                "rearward_bias_improves",
                "migration_change_improves",
            }:
                boosts["T12"] = boosts.get("T12", 0.0) + 0.03
            if executed.test_id == "T13" and executed.outcome == "issue_reduced_without_kerb_use":
                boosts["T14"] = boosts.get("T14", 0.0) + 0.03
        adjusted = [
            replace(score, score=score.score + boosts.get(score.test_id, 0.0))
            for score in rankings
        ]
        return sorted(
            adjusted,
            key=lambda item: (
                -item.score,
                -item.expected_information_gain,
                item.cost,
                item.test_id,
            ),
        )

    def recommend_next_test(
        self,
        belief: Mapping[str, float],
        available_tests: Iterable[str],
        executed_tests: Iterable[ExecutedTest] | None = None,
    ) -> str | None:
        """Recommend the available test with highest expected information per cost."""

        rankings = self.rank_available_tests(belief, available_tests, executed_tests)
        if not rankings:
            return None
        if rankings[0].expected_information_gain < self.min_expected_information_gain:
            return None
        return rankings[0].test_id

    def rank_available_tests(
        self,
        belief: Mapping[str, float],
        available_tests: Iterable[str],
        executed_tests: Iterable[ExecutedTest] | None = None,
    ) -> list[TestScore]:
        """Return detailed test rankings for explainable CLI output."""

        rankings = rank_tests(
            belief,
            available_tests,
            self.tests,
            self.likelihoods,
            objective=self.objective,
            confidence_threshold=self.confidence_threshold,
            use_two_step=self.use_two_step,
        )
        return self._apply_dependency_boosts(rankings, executed_tests)

    def update_after_outcome(
        self, belief: Mapping[str, float], test_id: str, outcome: str
    ) -> dict[str, float]:
        """Update belief after a test outcome."""

        return bayesian_update(belief, test_id, outcome, self.likelihoods)

    def _confidence_reached(self, belief: Mapping[str, float]) -> bool:
        _, probability = top_hypothesis(belief)
        return probability >= self.confidence_threshold

    def run_with_simulated_truth(
        self,
        true_hypothesis: str,
        initial_features: Mapping[str, Any] | None = None,
        initial_belief: Mapping[str, float] | None = None,
        seed: int = 7,
    ) -> dict[str, Any]:
        """Run the active-learning loop with outcomes sampled from a true cause."""

        if true_hypothesis not in self.hypothesis_ids:
            raise ValueError(f"Unknown true hypothesis: {true_hypothesis}")

        rng = random.Random(seed)
        belief = (
            normalize_distribution(initial_belief)
            if initial_belief is not None
            else self.initialize_belief(initial_features or {})
        )
        available_tests = list(self.tests.keys())
        executed: list[ExecutedTest] = []
        belief_history = [belief]
        ranking_history: list[list[TestScore]] = []

        while (
            available_tests
            and len(executed) < self.max_tests
            and not self._confidence_reached(belief)
        ):
            ranked = self.rank_available_tests(belief, available_tests, executed)
            if (
                not ranked
                or ranked[0].expected_information_gain < self.min_expected_information_gain
            ):
                break
            ranking_history.append(ranked)
            next_test = ranked[0].test_id
            outcome = sample_outcome(next_test, true_hypothesis, self.likelihoods, rng)
            belief = self.update_after_outcome(belief, next_test, outcome)
            executed.append(
                ExecutedTest(
                    test_id=next_test,
                    outcome=outcome,
                    cost=self.tests[next_test].cost,
                )
            )
            belief_history.append(belief)
            available_tests.remove(next_test)

        final_top, final_confidence = top_hypothesis(belief)
        return {
            "true_hypothesis": true_hypothesis,
            "tests": executed,
            "tests_used": len(executed),
            "total_cost": sum(item.cost for item in executed),
            "final_belief": belief,
            "final_top_hypothesis": final_top,
            "final_confidence": final_confidence,
            "correct_diagnosis": final_top == true_hypothesis,
            "belief_history": belief_history,
            "ranking_history": ranking_history,
            "stop_reason": (
                "confidence_threshold"
                if final_confidence >= self.confidence_threshold
                else "no_tests_remaining"
                if not available_tests
                else "low_expected_value"
                if ranking_history
                and ranking_history[-1][0].expected_information_gain
                < self.min_expected_information_gain
                else "max_tests"
            ),
        }

    def run_interactive(self, initial_features: Mapping[str, Any]) -> dict[str, Any]:
        """Run an interactive manual-outcome loop in a terminal."""

        belief = self.initialize_belief(initial_features)
        available_tests = list(self.tests.keys())
        executed: list[ExecutedTest] = []
        belief_history = [belief]

        while (
            available_tests
            and len(executed) < self.max_tests
            and not self._confidence_reached(belief)
        ):
            ranked = self.rank_available_tests(belief, available_tests, executed)
            if (
                not ranked
                or ranked[0].expected_information_gain < self.min_expected_information_gain
            ):
                break
            next_test = ranked[0].test_id
            test = self.tests[next_test]
            print(f"Recommended test: {next_test} - {test.name}")
            print(f"Allowed outcomes: {', '.join(test.outcomes)}")
            outcome = input("Enter observed outcome: ").strip()
            if outcome not in test.outcomes:
                raise ValueError(
                    f"Outcome {outcome!r} is not valid for {next_test}. "
                    f"Expected one of: {', '.join(test.outcomes)}"
                )
            belief = self.update_after_outcome(belief, next_test, outcome)
            executed.append(ExecutedTest(next_test, outcome, test.cost))
            belief_history.append(belief)
            available_tests.remove(next_test)

        final_top, final_confidence = top_hypothesis(belief)
        return {
            "tests": executed,
            "tests_used": len(executed),
            "total_cost": sum(item.cost for item in executed),
            "final_belief": belief,
            "final_top_hypothesis": final_top,
            "final_confidence": final_confidence,
            "belief_history": belief_history,
        }
