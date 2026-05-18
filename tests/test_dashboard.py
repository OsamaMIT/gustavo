from dashboard.app import build_dashboard_state


def test_dashboard_state_builds_without_streamlit_runtime() -> None:
    state = build_dashboard_state("entry_understeer")

    assert state["symptoms"]
    assert state["belief"]
    assert state["rankings"]

