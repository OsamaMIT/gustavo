"""Streamlit dashboard for the expanded R&D test-minimization engine."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit as st
except Exception:  # pragma: no cover - optional dependency import guard
    st = None  # type: ignore[assignment]

from src import fastf1_adapter
from src.diagnosis_engine import DiagnosisEngine
from src.feature_extractor import extract_features
from src.hypothesis_model import top_hypothesis
from src.calibration import run_calibration_report
from src.symptom_identifier import identify_symptoms
from src.synthetic_data import SCENARIOS, generate_synthetic_rows, true_hypothesis_for_scenario
from src.telemetry_loader import load_telemetry_csv
from src.test_library import load_default_config
from src.validation import compare_strategies


def _safe_slug(value: Any) -> str:
    slug = "".join(
        character.lower() if character.isalnum() else "_"
        for character in str(value).strip()
    ).strip("_")
    return slug or "unknown"


def _fastf1_dashboard_output(
    year: int,
    event: str,
    session_name: str,
    driver: str,
    lap: int | None,
) -> Path:
    lap_suffix = f"_lap_{lap}" if lap is not None else ""
    filename = (
        f"fastf1_{year}_{_safe_slug(event)}_{_safe_slug(session_name)}_"
        f"{_safe_slug(driver)}{lap_suffix}.csv"
    )
    return PROJECT_ROOT / "data" / "raw" / filename


def _default_option_index(options: list[str], preferred: str) -> int:
    preferred_lower = preferred.lower()
    for index, option in enumerate(options):
        if preferred_lower in option.lower():
            return index
    return 0


def _cached_fastf1_events(year: int, cache_dir: str) -> list[str]:
    return list(fastf1_adapter.list_fastf1_events(year, cache_dir=cache_dir))


def _cached_fastf1_session_options(
    year: int,
    event: str,
    session_name: str,
    cache_dir: str,
) -> dict[str, Any]:
    options = fastf1_adapter.list_fastf1_session_lap_options(
        year,
        event,
        session_name,
        cache_dir=cache_dir,
    )
    return {
        "drivers": list(options.drivers),
        "laps_by_driver": {
            driver: list(laps)
            for driver, laps in options.laps_by_driver.items()
        },
    }


if st is not None:
    _cached_fastf1_events = st.cache_data(ttl=3600, show_spinner=False)(
        _cached_fastf1_events
    )
    _cached_fastf1_session_options = st.cache_data(ttl=900, show_spinner=False)(
        _cached_fastf1_session_options
    )


def build_diagnosis_state_from_dataset(
    dataset: Any,
    *,
    objective: str = "eig_per_cost",
    true_hypothesis: str | None = None,
) -> dict[str, Any]:
    """Build diagnosis state from telemetry rows or a loaded dataset."""

    hypotheses, tests, likelihoods = load_default_config()
    engine = DiagnosisEngine(hypotheses, tests, likelihoods, objective=objective)
    features = extract_features(dataset)
    symptoms = identify_symptoms(features)
    symptom = symptoms[0]
    belief = engine.initialize_belief(features)
    available = engine.available_tests_for_symptom(symptom.symptom_id)
    rankings = engine.rank_available_tests(belief, available)
    top_id, top_probability = top_hypothesis(belief)
    return {
        "features": features,
        "symptoms": symptoms,
        "belief": belief,
        "rankings": rankings,
        "top_hypothesis": top_id,
        "top_probability": top_probability,
        "true_hypothesis": true_hypothesis,
        "engine": engine,
    }


def build_dashboard_state(scenario: str) -> dict[str, Any]:
    """Build diagnosis state for smoke tests and dashboard rendering."""

    rows = generate_synthetic_rows(scenario)
    return build_diagnosis_state_from_dataset(
        rows,
        true_hypothesis=true_hypothesis_for_scenario(scenario),
    )


def _render_belief_chart(belief: dict[str, float]) -> None:
    ordered = dict(sorted(belief.items(), key=lambda item: item[1], reverse=True)[:12])
    st.bar_chart(ordered)


def _render_rankings(rankings: list[Any], engine: DiagnosisEngine) -> None:
    rows = []
    for item in rankings:
        rows.append(
            {
                "test_id": item.test_id,
                "name": engine.tests[item.test_id].name,
                "score": round(item.score, 4),
                "eig": round(item.expected_information_gain, 4),
                "expected_posterior_entropy": round(item.expected_posterior_entropy, 4),
                "confidence_gain": round(item.expected_confidence_gain, 4),
                "threshold_probability": round(item.threshold_reach_probability, 4),
                "cost": item.cost,
            }
        )
    st.dataframe(rows, width="stretch")


def main() -> None:
    if st is None:
        raise RuntimeError(
            "Streamlit is not installed. Install dependencies with `pip install -r requirements.txt`."
        )

    st.set_page_config(page_title="F1 R&D Test Minimization", layout="wide", page_icon="🏎️")
    st.title("F1 R&D Test Minimization")

    source = st.sidebar.radio(
        "Telemetry source",
        ["Synthetic scenario", "CSV upload", "FastF1 session"],
    )
    objective = st.sidebar.selectbox(
        "Optimizer objective",
        [
            "eig_per_cost",
            "eig",
            "confidence_gain_per_cost",
            "threshold_probability_per_cost",
        ],
    )

    hypotheses, tests, likelihoods = load_default_config()
    engine = DiagnosisEngine(hypotheses, tests, likelihoods, objective=objective)

    if source == "Synthetic scenario":
        scenario = st.sidebar.selectbox("Scenario", SCENARIOS)
        rows = generate_synthetic_rows(scenario)
        features = extract_features(rows)
        true_hypothesis = true_hypothesis_for_scenario(scenario)
        st.caption(f"Simulated true hypothesis: `{true_hypothesis}`")
    elif source == "CSV upload":
        uploaded = st.sidebar.file_uploader("Upload internal telemetry CSV", type=["csv"])
        if uploaded is None:
            st.info("Upload a telemetry CSV or switch to a synthetic scenario.")
            return
        temp_path = Path("/tmp/f1_rd_uploaded.csv")
        temp_path.write_bytes(uploaded.getvalue())
        dataset = load_telemetry_csv(temp_path)
        features = extract_features(dataset)
    else:
        year = int(st.sidebar.number_input("Year", min_value=2018, max_value=2100, value=2023))
        cache_dir = st.sidebar.text_input(
            "Cache directory",
            str(fastf1_adapter.DEFAULT_FASTF1_CACHE_DIR),
        )
        try:
            events = _cached_fastf1_events(year, cache_dir)
        except Exception as exc:
            st.error(f"FastF1 event list failed: {exc}")
            return
        event = st.sidebar.selectbox(
            "Event",
            events,
            index=_default_option_index(events, "Bahrain"),
        )
        session_name = st.sidebar.selectbox("Session", ["Q", "Race", "FP1", "FP2", "FP3", "SQ"])
        try:
            session_options = _cached_fastf1_session_options(
                year,
                event,
                session_name,
                cache_dir,
            )
        except Exception as exc:
            st.error(f"FastF1 driver/lap list failed: {exc}")
            return
        drivers = session_options["drivers"]
        if not drivers:
            st.error("FastF1 returned no drivers for this session.")
            return
        driver = st.sidebar.selectbox(
            "Driver",
            drivers,
            index=_default_option_index(drivers, "VER"),
        )
        lap_numbers = session_options["laps_by_driver"].get(driver, [])
        lap_choices = ["Fastest lap"] + [str(lap_number) for lap_number in lap_numbers]
        lap_choice = st.sidebar.selectbox("Lap", lap_choices)
        lap = None if lap_choice == "Fastest lap" else int(lap_choice)
        if not st.sidebar.button("Load FastF1 session"):
            st.info("Load a FastF1 session or switch telemetry source.")
            return
        try:
            fastf1_path = fastf1_adapter.fetch_fastf1_to_csv(
                year=year,
                event=event,
                session_name=session_name,
                driver=driver,
                lap=lap,
                output=_fastf1_dashboard_output(year, event, session_name, driver, lap),
                cache_dir=cache_dir,
            )
            dataset = load_telemetry_csv(fastf1_path)
            features = extract_features(dataset)
        except Exception as exc:
            st.error(f"FastF1 load failed: {exc}")
            return
        st.caption(f"FastF1 telemetry CSV: `{fastf1_path}`")

    symptoms = identify_symptoms(features)
    symptom = symptoms[0]
    belief = engine.initialize_belief(features)
    available = engine.available_tests_for_symptom(symptom.symptom_id)
    rankings = engine.rank_available_tests(belief, available)

    top_id, top_probability = top_hypothesis(belief)
    st.metric("Top hypothesis", top_id, f"{top_probability:.3f}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Detected Symptoms")
        st.dataframe(
            [
                {
                    "symptom": item.symptom_id,
                    "confidence": round(item.confidence, 3),
                    "category": item.category,
                    "evidence": "; ".join(item.evidence[:4]),
                }
                for item in symptoms[:8]
            ],
            width="stretch",
        )

    with right:
        st.subheader("Belief Distribution")
        _render_belief_chart(belief)

    st.subheader("Ranked R&D Tests")
    _render_rankings(rankings, engine)

    if rankings:
        selected = st.selectbox("Apply manual outcome for test", [item.test_id for item in rankings])
        outcome = st.selectbox("Outcome", engine.tests[selected].outcomes)
        if st.button("Update belief"):
            updated = engine.update_after_outcome(belief, selected, outcome)
            st.subheader("Updated Belief")
            _render_belief_chart(updated)
            updated_top, updated_probability = top_hypothesis(updated)
            st.metric("Updated top hypothesis", updated_top, f"{updated_probability:.3f}")

    st.subheader("Validation Snapshot")
    result = compare_strategies(trials=50, objective=objective)
    st.dataframe(
        [
            {
                "strategy": strategy,
                "top1_accuracy": summary["accuracy"],
                "top3_accuracy": summary["top3_accuracy"],
                "avg_tests": summary["avg_tests_used"],
                "avg_cost": summary["avg_total_cost"],
                "top3_confidence": summary["avg_final_top3_confidence"],
                "threshold_reach_rate": summary["threshold_reach_rate"],
            }
            for strategy, summary in result["strategies"].items()
        ],
        width="stretch",
    )

    if st.button("Run calibration sweep"):
        report = run_calibration_report(trials=25, objective=objective)
        summary = report["summary"]
        cols = st.columns(4)
        cols[0].metric("Symptom top-1", f"{summary['symptom_top1_accuracy']:.3f}")
        cols[1].metric("Symptom top-3", f"{summary['symptom_top3_accuracy']:.3f}")
        cols[2].metric("Initial hypothesis top-3", f"{summary['initial_true_top3_accuracy']:.3f}")
        cols[3].metric("Final diagnosis top-3", f"{summary['final_top3_accuracy']:.3f}")
        st.dataframe(report["worst_symptom_confusions"], width="stretch")


if __name__ == "__main__":
    main()
