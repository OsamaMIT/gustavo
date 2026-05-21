from dashboard.app import build_dashboard_state, build_diagnosis_state_from_dataset
from src.synthetic_data import generate_synthetic_rows


def test_dashboard_state_builds_without_streamlit_runtime() -> None:
    state = build_dashboard_state("entry_understeer")

    assert state["symptoms"]
    assert state["belief"]
    assert state["rankings"]


def test_dashboard_shared_state_builder_accepts_telemetry_rows() -> None:
    rows = generate_synthetic_rows("wheelspin_on_exit", laps=2, samples_per_lap=16)
    state = build_diagnosis_state_from_dataset(rows)

    assert state["features"]["wheelspin_index"] > 0.0
    assert state["symptoms"]
    assert state["rankings"]
