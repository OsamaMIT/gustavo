"""Command line interface for the telemetry-driven R&D test minimization engine."""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Mapping

from .diagnosis_engine import DiagnosisEngine, ExecutedTest
from .f1_2020_adapter import convert_f1_2020_jsonl_to_csv
from .feature_extractor import extract_features
from .hypothesis_model import top_hypothesis
from .outcome_classifier import sample_outcome
from .calibration import run_calibration_report
from .symptom_identifier import identify_symptoms
from .synthetic_data import (
    SCENARIOS,
    generate_all_synthetic,
    generate_synthetic_rows,
    true_hypothesis_for_scenario,
)
from .telemetry_loader import load_telemetry_csv
from .test_library import load_default_config
from .validation import compare_strategies


def _format_belief(belief: Mapping[str, float]) -> str:
    return "\n".join(
        f"  {hypothesis}: {probability:.3f}"
        for hypothesis, probability in sorted(
            belief.items(), key=lambda item: item[1], reverse=True
        )
    )


def _print_feature_summary(features: Mapping[str, Any]) -> None:
    keys = (
        "avg_speed",
        "min_speed",
        "avg_abs_steering",
        "avg_abs_lateral_accel",
        "oversteer_index",
        "rotation_deficit_index",
        "understeer_index",
        "front_tire_temp_avg",
        "rear_tire_temp_avg",
        "front_tire_temp_trend_category",
        "rear_tire_temp_trend_category",
        "wheelspin_index",
        "brake_instability_index",
        "straight_line_speed_deficit_index",
        "acceleration_deficit_index",
        "drag_index",
        "performance_loss_sec_per_lap",
        "issue_worsens_over_stint",
        "steering_demand",
        "speed_category",
        "steering_noise_index",
        "suspension_variation",
    )
    print("Feature summary:")
    for key in keys:
        value = features.get(key)
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


def _print_rankings(engine: DiagnosisEngine, belief: Mapping[str, float], available: list[str]) -> None:
    rankings = engine.rank_available_tests(belief, available)
    print("Ranked candidate tests:")
    for rank, item in enumerate(rankings, start=1):
        test = engine.tests[item.test_id]
        print(
            f"  {rank}. {item.test_id} {test.name} "
            f"score={item.score:.4f}, eig={item.expected_information_gain:.4f}, "
            f"cost={item.cost:g}"
        )


def generate_synthetic_command(args: argparse.Namespace) -> None:
    paths = generate_all_synthetic(
        output_dir=args.output,
        laps=args.laps,
        samples_per_lap=args.samples_per_lap,
        seed=args.seed,
    )
    print("Generated synthetic telemetry CSVs:")
    for path in paths:
        print(f"  {path}")


def _build_engine(args: argparse.Namespace) -> DiagnosisEngine:
    hypotheses, tests, likelihoods = load_default_config()
    return DiagnosisEngine(
        hypotheses=hypotheses,
        tests=tests,
        likelihoods=likelihoods,
        confidence_threshold=args.confidence_threshold,
        max_tests=args.max_tests,
        objective=args.objective,
        min_expected_information_gain=getattr(args, "min_expected_information_gain", 0.0),
        use_two_step=getattr(args, "two_step", False),
    )


def diagnose_command(args: argparse.Namespace) -> None:
    engine = _build_engine(args)
    rows = generate_synthetic_rows(
        scenario=args.scenario,
        laps=args.laps,
        samples_per_lap=args.samples_per_lap,
        seed=args.seed,
    )
    features = extract_features(rows)
    symptoms = identify_symptoms(features)
    symptom = symptoms[0]
    belief = engine.initialize_belief(features)
    available = engine.available_tests_for_symptom(symptom.symptom_id)
    rng = random.Random(args.seed)
    executed: list[tuple[str, str, float]] = []
    executed_objects = []
    true_hypothesis = true_hypothesis_for_scenario(args.scenario)

    print(f"Synthetic scenario: {args.scenario}")
    print(f"Simulated true hypothesis: {true_hypothesis}")
    _print_feature_summary(features)
    print(f"Identified symptom: {symptom.symptom_id} confidence={symptom.confidence:.3f}")
    print("Top symptom candidates:")
    for candidate in symptoms[:5]:
        print(f"  {candidate.symptom_id}: {candidate.confidence:.3f}")
    print("Evidence:")
    for item in symptom.evidence:
        print(f"  - {item}")
    print("Initial belief:")
    print(_format_belief(belief))

    while available and len(executed) < engine.max_tests:
        top, confidence = top_hypothesis(belief)
        if confidence >= engine.confidence_threshold:
            break
        _print_rankings(engine, belief, available)
        test_id = engine.recommend_next_test(belief, available, executed_objects)
        if test_id is None:
            break
        outcome = sample_outcome(test_id, true_hypothesis, engine.likelihoods, rng)
        print(f"Recommended test: {test_id} {engine.tests[test_id].name}")
        print(f"Simulated outcome: {outcome}")
        belief = engine.update_after_outcome(belief, test_id, outcome)
        print("Updated belief:")
        print(_format_belief(belief))
        executed.append((test_id, outcome, engine.tests[test_id].cost))
        executed_objects.append(ExecutedTest(test_id, outcome, engine.tests[test_id].cost))
        available.remove(test_id)

    final_top, final_confidence = top_hypothesis(belief)
    print("Final diagnosis:")
    print(f"  top_hypothesis: {final_top}")
    print(f"  confidence: {final_confidence:.3f}")
    print(f"  tests_used: {len(executed)}")
    print(f"  total_cost: {sum(item[2] for item in executed):g}")


def diagnose_csv_command(args: argparse.Namespace) -> None:
    engine = _build_engine(args)
    dataset = load_telemetry_csv(args.file)
    features = extract_features(dataset)
    symptoms = identify_symptoms(features)
    symptom = symptoms[0]
    belief = engine.initialize_belief(features)
    available = engine.available_tests_for_symptom(symptom.symptom_id)

    print(f"Telemetry file: {Path(args.file)}")
    if dataset.warnings:
        print("Loader warnings:")
        for warning in dataset.warnings:
            print(f"  - {warning}")
    _print_feature_summary(features)
    print(f"Identified symptom: {symptom.symptom_id} confidence={symptom.confidence:.3f}")
    print("Top symptom candidates:")
    for candidate in symptoms[:5]:
        print(f"  {candidate.symptom_id}: {candidate.confidence:.3f}")
    print("Initial belief:")
    print(_format_belief(belief))
    _print_rankings(engine, belief, available)

    recommended = engine.recommend_next_test(belief, available)
    if recommended is None:
        print("No test recommendation available.")
        return
    print(f"Recommended test: {recommended} {engine.tests[recommended].name}")

    if args.outcome:
        test_id = args.test_id or recommended
        if test_id not in engine.tests:
            raise ValueError(f"Unknown test id: {test_id}")
        if args.outcome not in engine.tests[test_id].outcomes:
            raise ValueError(
                f"Outcome {args.outcome!r} is invalid for {test_id}. "
                f"Expected one of: {', '.join(engine.tests[test_id].outcomes)}"
            )
        updated = engine.update_after_outcome(belief, test_id, args.outcome)
        print(f"Manual outcome applied: {test_id} -> {args.outcome}")
        print("Updated belief:")
        print(_format_belief(updated))
    else:
        print("No outcome supplied; pass --outcome and optional --test-id to update belief.")


def validate_command(args: argparse.Namespace) -> None:
    result = compare_strategies(
        trials=args.trials,
        seed=args.seed,
        confidence_threshold=args.confidence_threshold,
        max_tests=args.max_tests,
        objective=args.objective,
        min_expected_information_gain=args.min_expected_information_gain,
    )
    print(
        f"Validation trials={result['trials']} seed={result['seed']} "
        f"threshold={result['confidence_threshold']:.2f}"
    )
    print("strategy              top1   top3   avg_tests  avg_cost  avg_conf  top3_conf  reach_rate  cost_correct")
    for strategy, summary in result["strategies"].items():
        cost_correct = summary["avg_cost_to_correct_diagnosis"]
        cost_correct_text = f"{cost_correct:.2f}" if cost_correct is not None else "n/a"
        print(
            f"{strategy:21s} "
            f"{summary['accuracy']:.3f}     "
            f"{summary['top3_accuracy']:.3f}  "
            f"{summary['avg_tests_used']:.2f}       "
            f"{summary['avg_total_cost']:.2f}     "
            f"{summary['avg_final_confidence']:.3f}     "
            f"{summary['avg_final_top3_confidence']:.3f}      "
            f"{summary['threshold_reach_rate']:.3f}      "
            f"{cost_correct_text}"
        )


def convert_f1_2020_command(args: argparse.Namespace) -> None:
    path = convert_f1_2020_jsonl_to_csv(args.input, args.output)
    print(f"Converted F1 2020 JSONL telemetry to CSV: {path}")


def calibrate_command(args: argparse.Namespace) -> None:
    report = run_calibration_report(
        trials=args.trials,
        seed=args.seed,
        confidence_threshold=args.confidence_threshold,
        max_tests=args.max_tests,
        objective=args.objective,
        min_expected_information_gain=args.min_expected_information_gain,
    )
    summary = report["summary"]
    print(
        f"Calibration trials={report['trials']} seed={report['seed']} "
        f"objective={report['objective']} threshold={report['confidence_threshold']:.2f}"
    )
    print("Symptom detection:")
    print(f"  top1_accuracy: {summary['symptom_top1_accuracy']:.3f}")
    print(f"  top3_accuracy: {summary['symptom_top3_accuracy']:.3f}")
    print("Initial true-hypothesis rank:")
    print(f"  top1_accuracy: {summary['initial_true_top1_accuracy']:.3f}")
    print(f"  top3_accuracy: {summary['initial_true_top3_accuracy']:.3f}")
    print(f"  top5_accuracy: {summary['initial_true_top5_accuracy']:.3f}")
    print(f"  rank_buckets: {summary['initial_rank_buckets']}")
    print("Final optimizer diagnosis:")
    print(f"  top1_accuracy: {summary['final_top1_accuracy']:.3f}")
    print(f"  top3_accuracy: {summary['final_top3_accuracy']:.3f}")
    print(f"  avg_top3_confidence: {summary['avg_final_top3_confidence']:.3f}")
    print(f"  threshold_reach_rate: {summary['threshold_reach_rate']:.3f}")
    print(f"  avg_tests_used: {summary['avg_tests_used']:.2f}")
    print(f"  avg_total_cost: {summary['avg_total_cost']:.2f}")
    cost_correct = summary["avg_cost_to_correct_diagnosis"]
    print(
        "  avg_cost_to_correct_diagnosis: "
        + (f"{cost_correct:.2f}" if cost_correct is not None else "n/a")
    )
    if report["worst_symptom_confusions"]:
        print("Worst symptom confusions:")
        for item in report["worst_symptom_confusions"][:8]:
            print(f"  {item['true']} -> {item['predicted']}: {item['count']}")
    if report["worst_hypothesis_confusions"]:
        print("Worst hypothesis confusions:")
        for item in report["worst_hypothesis_confusions"][:8]:
            print(f"  {item['true']} -> {item['predicted']}: {item['count']}")
    likelihood = report.get("likelihood_diagnostics", {})
    if likelihood:
        print("Likelihood calibration:")
        print("  highest-value tests:")
        for item in likelihood.get("test_information", [])[:5]:
            print(
                f"    {item['test_id']} eig={item['expected_information_gain']:.3f} "
                f"eig_per_cost={item['eig_per_cost']:.3f}"
            )
        weak_pairs = likelihood.get("weak_hypothesis_pairs", [])
        if weak_pairs:
            print("  weakest separated hypothesis pairs:")
            for item in weak_pairs[:5]:
                print(
                    f"    {item['left']} vs {item['right']} "
                    f"best_test={item['best_test']} separation={item['best_separation']:.3f}"
                )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Telemetry-driven Formula 1 R&D test minimization prototype."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-synthetic")
    generate.add_argument("--output", default="data/synthetic")
    generate.add_argument("--laps", type=int, default=8)
    generate.add_argument("--samples-per-lap", type=int, default=120)
    generate.add_argument("--seed", type=int, default=42)
    generate.set_defaults(func=generate_synthetic_command)

    diagnose = subparsers.add_parser("diagnose")
    diagnose.add_argument("--scenario", choices=SCENARIOS, required=True)
    diagnose.add_argument("--laps", type=int, default=8)
    diagnose.add_argument("--samples-per-lap", type=int, default=120)
    diagnose.add_argument("--seed", type=int, default=42)
    diagnose.add_argument("--confidence-threshold", type=float, default=0.80)
    diagnose.add_argument("--max-tests", type=int, default=5)
    diagnose.add_argument("--objective", default="eig_per_cost")
    diagnose.add_argument("--min-expected-information-gain", type=float, default=0.15)
    diagnose.add_argument("--two-step", action="store_true")
    diagnose.set_defaults(func=diagnose_command)

    diagnose_csv = subparsers.add_parser("diagnose-csv")
    diagnose_csv.add_argument("--file", required=True)
    diagnose_csv.add_argument("--outcome")
    diagnose_csv.add_argument("--test-id")
    diagnose_csv.add_argument("--confidence-threshold", type=float, default=0.80)
    diagnose_csv.add_argument("--max-tests", type=int, default=5)
    diagnose_csv.add_argument("--objective", default="eig_per_cost")
    diagnose_csv.add_argument("--min-expected-information-gain", type=float, default=0.15)
    diagnose_csv.add_argument("--two-step", action="store_true")
    diagnose_csv.set_defaults(func=diagnose_csv_command)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--trials", type=int, default=100)
    validate.add_argument("--seed", type=int, default=123)
    validate.add_argument("--confidence-threshold", type=float, default=0.80)
    validate.add_argument("--max-tests", type=int, default=5)
    validate.add_argument("--objective", default="eig_per_cost")
    validate.add_argument("--min-expected-information-gain", type=float, default=0.15)
    validate.set_defaults(func=validate_command)

    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--trials", type=int, default=100)
    calibrate.add_argument("--seed", type=int, default=123)
    calibrate.add_argument("--confidence-threshold", type=float, default=0.80)
    calibrate.add_argument("--max-tests", type=int, default=5)
    calibrate.add_argument("--objective", default="eig_per_cost")
    calibrate.add_argument("--min-expected-information-gain", type=float, default=0.15)
    calibrate.set_defaults(func=calibrate_command)

    convert = subparsers.add_parser("convert-f1-2020")
    convert.add_argument("--input", required=True)
    convert.add_argument("--output", required=True)
    convert.set_defaults(func=convert_f1_2020_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
