"""Structured R&D test and configuration loading utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


@dataclass(frozen=True)
class Hypothesis:
    """A supported root-cause hypothesis."""

    id: str
    meaning: str
    category: str = "general"


@dataclass(frozen=True)
class SymptomDefinition:
    """A config-driven observable symptom definition."""

    id: str
    category: str
    description: str
    segment_type: str
    required_features: tuple[str, ...]
    optional_features: tuple[str, ...]
    signals: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RnDTest:
    """A structured R&D test with Assetto Corsa validation proxy metadata."""

    test_id: str
    name: str
    description: str
    f1_interpretation: str
    f1_2020_proxy: str
    cost: float
    outcomes: tuple[str, ...]
    relevant_hypotheses: tuple[str, ...]
    outcome_hypothesis_map: dict[str, tuple[str, ...]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RnDTest":
        outcome_hypothesis_map = {
            str(outcome): tuple(str(item) for item in hypotheses)
            for outcome, hypotheses in data.get("outcome_hypothesis_map", {}).items()
        }
        return cls(
            test_id=str(data["test_id"]),
            name=str(data["name"]),
            description=str(data["description"]),
            f1_interpretation=str(data["f1_interpretation"]),
            f1_2020_proxy=str(
                data.get("f1_2020_proxy")
                or data.get("assetto_corsa_proxy")
                or "CSV telemetry proxy"
            ),
            cost=float(data["cost"]),
            outcomes=tuple(str(item) for item in data["outcomes"]),
            relevant_hypotheses=tuple(
                str(item) for item in data["relevant_hypotheses"]
            ),
            outcome_hypothesis_map=outcome_hypothesis_map,
        )


def read_json(path: Path) -> Any:
    """Read a JSON file with a helpful path-specific error."""

    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Configuration file not found: {path}") from exc


def load_hypotheses(path: Path | None = None) -> dict[str, Hypothesis]:
    """Load supported hypotheses keyed by hypothesis id."""

    raw = read_json(path or CONFIG_DIR / "hypotheses.json")
    return {
        str(item["id"]): Hypothesis(
            id=str(item["id"]),
            meaning=str(item["meaning"]),
            category=str(item.get("category", "general")),
        )
        for item in raw
    }


def load_symptoms(path: Path | None = None) -> dict[str, SymptomDefinition]:
    """Load symptom definitions keyed by symptom id."""

    raw = read_json(path or CONFIG_DIR / "symptoms.json")
    symptoms: dict[str, SymptomDefinition] = {}
    for item in raw:
        symptom = SymptomDefinition(
            id=str(item["id"]),
            category=str(item["category"]),
            description=str(item["description"]),
            segment_type=str(item["segment_type"]),
            required_features=tuple(str(value) for value in item["required_features"]),
            optional_features=tuple(str(value) for value in item["optional_features"]),
            signals=tuple(dict(signal) for signal in item["signals"]),
        )
        symptoms[symptom.id] = symptom
    return symptoms


def load_symptom_hypothesis_map(path: Path | None = None) -> dict[str, tuple[str, ...]]:
    """Load symptom-to-candidate-hypotheses mappings."""

    raw = read_json(path or CONFIG_DIR / "symptom_hypothesis_map.json")
    return {
        str(symptom): tuple(str(hypothesis) for hypothesis in hypotheses)
        for symptom, hypotheses in raw.items()
    }


def load_tests(path: Path | None = None) -> dict[str, RnDTest]:
    """Load R&D test definitions keyed by test id."""

    raw = read_json(path or CONFIG_DIR / "tests.json")
    tests = {item["test_id"]: RnDTest.from_dict(item) for item in raw}
    return dict(sorted(tests.items()))


def load_likelihoods(path: Path | None = None) -> dict[str, dict[str, dict[str, float]]]:
    """Load manual P(outcome | hypothesis, test) likelihood tables."""

    if path is None and (CONFIG_DIR / "likelihood_templates.json").exists():
        hypotheses = load_hypotheses()
        tests = load_tests()
        return generate_likelihoods(hypotheses, tests)

    raw = read_json(path or CONFIG_DIR / "likelihoods.json")
    return {
        str(test_id): {
            str(hypothesis): {
                str(outcome): float(probability)
                for outcome, probability in outcomes.items()
            }
            for hypothesis, outcomes in hypotheses.items()
        }
        for test_id, hypotheses in raw.items()
    }


def load_default_config() -> tuple[
    dict[str, Hypothesis], dict[str, RnDTest], dict[str, dict[str, dict[str, float]]]
]:
    """Load hypotheses, tests, and likelihoods from the default config directory."""

    hypotheses = load_hypotheses()
    tests = load_tests()
    validate_test_references(hypotheses, tests)
    likelihoods = generate_likelihoods(hypotheses, tests)
    validate_likelihood_tables(hypotheses, tests, likelihoods)
    return hypotheses, tests, likelihoods


def generate_likelihoods(
    hypotheses: dict[str, Hypothesis],
    tests: dict[str, RnDTest],
    template_path: Path | None = None,
    overrides_path: Path | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Generate complete P(outcome | hypothesis, test) tables from templates."""

    template = read_json(template_path or CONFIG_DIR / "likelihood_templates.json")
    overrides_path = overrides_path or CONFIG_DIR / "likelihood_overrides.json"
    overrides = read_json(overrides_path) if overrides_path.exists() else {}

    base_weight = float(template.get("base_weight", 1.0))
    mapped_boost = float(template.get("mapped_outcome_boost", 5.0))
    relevant_boost = float(template.get("relevant_hypothesis_boost", 1.0))
    non_relevant_no_change_boost = float(
        template.get("non_relevant_no_change_boost", 3.0)
    )
    unmapped_relevant_no_change_boost = float(
        template.get("unmapped_relevant_no_change_boost", 1.6)
    )
    no_change_patterns = tuple(str(item) for item in template.get("no_change_patterns", []))

    def is_no_change(outcome: str) -> bool:
        return any(pattern in outcome for pattern in no_change_patterns)

    generated: dict[str, dict[str, dict[str, float]]] = {}
    for test_id, test in tests.items():
        generated[test_id] = {}
        mapped_hypotheses = {
            hypothesis
            for hypotheses_for_outcome in test.outcome_hypothesis_map.values()
            for hypothesis in hypotheses_for_outcome
        }
        for hypothesis_id, hypothesis in hypotheses.items():
            weights: dict[str, float] = {}
            is_relevant = hypothesis_id in test.relevant_hypotheses
            category_related = any(
                hypotheses[relevant].category == hypothesis.category
                for relevant in test.relevant_hypotheses
                if relevant in hypotheses
            )
            for outcome in test.outcomes:
                weight = base_weight
                if is_relevant:
                    weight += relevant_boost
                elif category_related:
                    weight += relevant_boost * 0.35

                mapped_for_outcome = test.outcome_hypothesis_map.get(outcome, ())
                if hypothesis_id in mapped_for_outcome:
                    weight += mapped_boost
                elif not is_relevant and is_no_change(outcome):
                    weight += non_relevant_no_change_boost
                elif is_relevant and hypothesis_id not in mapped_hypotheses and is_no_change(outcome):
                    weight += unmapped_relevant_no_change_boost

                weights[outcome] = max(weight, 0.001)

            total = sum(weights.values())
            generated[test_id][hypothesis_id] = {
                outcome: probability / total for outcome, probability in weights.items()
            }

    for test_id, hypothesis_overrides in overrides.items():
        if test_id not in generated:
            raise ValueError(f"Likelihood override references unknown test {test_id}")
        for hypothesis_id, outcomes in hypothesis_overrides.items():
            if hypothesis_id not in generated[test_id]:
                raise ValueError(
                    f"Likelihood override references unknown hypothesis {hypothesis_id}"
                )
            total = sum(float(value) for value in outcomes.values())
            if total <= 0.0:
                raise ValueError(
                    f"Likelihood override for {test_id}/{hypothesis_id} must be positive"
                )
            generated[test_id][hypothesis_id] = {
                str(outcome): float(probability) / total
                for outcome, probability in outcomes.items()
            }

    return generated


def validate_test_references(
    hypotheses: dict[str, Hypothesis], tests: dict[str, RnDTest]
) -> None:
    """Validate that tests only reference known hypotheses and outcomes."""

    hypothesis_ids = set(hypotheses)
    for test_id, test in tests.items():
        unknown = set(test.relevant_hypotheses) - hypothesis_ids
        if unknown:
            raise ValueError(
                f"Test {test_id} references unknown hypotheses: {sorted(unknown)}"
            )
        unknown_mapped = {
            hypothesis
            for outcome, mapped in test.outcome_hypothesis_map.items()
            for hypothesis in mapped
            if hypothesis not in hypothesis_ids
        }
        if unknown_mapped:
            raise ValueError(
                f"Test {test_id} outcome map references unknown hypotheses: "
                f"{sorted(unknown_mapped)}"
            )
        unknown_outcomes = set(test.outcome_hypothesis_map) - set(test.outcomes)
        if unknown_outcomes:
            raise ValueError(
                f"Test {test_id} outcome map references unknown outcomes: "
                f"{sorted(unknown_outcomes)}"
            )


def validate_likelihood_tables(
    hypotheses: dict[str, Hypothesis],
    tests: dict[str, RnDTest],
    likelihoods: dict[str, dict[str, dict[str, float]]],
    tolerance: float = 1e-9,
) -> None:
    """Validate coverage and normalization of the manual likelihood tables."""

    for test_id, test in tests.items():
        if test_id not in likelihoods:
            raise ValueError(f"Likelihood table missing test {test_id}")
        for hypothesis_id in hypotheses:
            if hypothesis_id not in likelihoods[test_id]:
                raise ValueError(
                    f"Likelihood table missing hypothesis {hypothesis_id} for {test_id}"
                )
            outcomes = likelihoods[test_id][hypothesis_id]
            missing = set(test.outcomes) - set(outcomes)
            extra = set(outcomes) - set(test.outcomes)
            if missing or extra:
                raise ValueError(
                    f"Likelihood outcomes for {test_id}/{hypothesis_id} do not "
                    f"match test outcomes. Missing={sorted(missing)}, extra={sorted(extra)}"
                )
            total = sum(outcomes.values())
            if abs(total - 1.0) > tolerance:
                raise ValueError(
                    f"Likelihoods for {test_id}/{hypothesis_id} sum to {total}, not 1"
                )
