"""Config-driven rule-based symptom identification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .test_library import (
    SymptomDefinition,
    load_symptom_hypothesis_map,
    load_symptoms,
)


SUPPORTED_SYMPTOM_ID = "medium_speed_entry_to_apex_understeer"


@dataclass(frozen=True)
class SymptomDetection:
    """Detected symptom and supporting telemetry evidence."""

    symptom_id: str
    confidence: float
    evidence: tuple[str, ...]
    affected_segments: tuple[str, ...]
    performance_loss_sec_per_lap: float | None
    candidate_hypotheses: tuple[str, ...] = ()
    category: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "symptom_id": self.symptom_id,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "affected_segments": list(self.affected_segments),
            "performance_loss_sec_per_lap": self.performance_loss_sec_per_lap,
            "candidate_hypotheses": list(self.candidate_hypotheses),
            "category": self.category,
        }


def _condition_matches(value: Any, op: str, threshold: Any) -> bool:
    if value is None:
        return False
    if op == "==":
        return value == threshold
    if op == "!=":
        return value != threshold
    if not isinstance(value, (int, float)) or not isinstance(threshold, (int, float)):
        return False
    if op == ">=":
        return float(value) >= float(threshold)
    if op == ">":
        return float(value) > float(threshold)
    if op == "<=":
        return float(value) <= float(threshold)
    if op == "<":
        return float(value) < float(threshold)
    raise ValueError(f"Unsupported symptom rule operator: {op}")


def _detect_from_definition(
    definition: SymptomDefinition,
    features: Mapping[str, Any],
    candidate_hypotheses: tuple[str, ...],
) -> SymptomDetection:
    confidence = 0.05
    evidence: list[str] = []

    for signal in definition.signals:
        feature = str(signal["feature"])
        op = str(signal["op"])
        threshold = signal["threshold"]
        if _condition_matches(features.get(feature), op, threshold):
            confidence += float(signal["weight"])
            evidence.append(str(signal["evidence"]))

    available_required = [
        feature
        for feature in definition.required_features
        if features.get(feature) is not None
    ]
    if available_required:
        confidence += min(0.10, 0.02 * len(available_required))

    oversteer_index = features.get("oversteer_index")
    understeer_index = features.get("understeer_index")
    if (
        "understeer" in definition.id
        and isinstance(oversteer_index, (int, float))
        and oversteer_index >= 1.30
    ):
        confidence *= 0.45
    wheelspin_index = features.get("wheelspin_index")
    traction_loss_index = features.get("traction_loss_index")
    if (
        "understeer" in definition.id
        and (
            isinstance(wheelspin_index, (int, float))
            and wheelspin_index >= 0.20
            or isinstance(traction_loss_index, (int, float))
            and traction_loss_index >= 0.24
        )
    ):
        confidence *= 0.55
    if (
        "oversteer" in definition.id
        and isinstance(oversteer_index, (int, float))
        and oversteer_index >= 1.30
    ):
        confidence += 0.12
    if (
        "oversteer" in definition.id
        and isinstance(understeer_index, (int, float))
        and understeer_index >= 14.0
        and not (
            isinstance(oversteer_index, (int, float))
            and oversteer_index >= 1.30
        )
    ):
        confidence *= 0.55

    front_locking_index = features.get("front_locking_index")
    rear_locking_index = features.get("rear_locking_index")
    if definition.id == "brake_locking" and (
        isinstance(front_locking_index, (int, float))
        and isinstance(rear_locking_index, (int, float))
        and front_locking_index >= 0.12
        and rear_locking_index >= 0.12
    ):
        confidence += 0.14
    if definition.id == "front_locking" and (
        isinstance(rear_locking_index, (int, float))
        and rear_locking_index >= 0.12
    ):
        confidence *= 0.80
    if definition.id == "rear_locking" and (
        isinstance(front_locking_index, (int, float))
        and front_locking_index >= 0.12
    ):
        confidence *= 0.80

    rotation_deficit_index = features.get("rotation_deficit_index")
    if definition.id == "poor_rotation" and (
        isinstance(rotation_deficit_index, (int, float))
        and rotation_deficit_index >= 0.40
    ):
        confidence += 0.12

    front_temp_trend = features.get("front_tire_temp_trend_category")
    rear_temp_trend = features.get("rear_tire_temp_trend_category")
    if (
        definition.id == "thermal_degradation"
        and front_temp_trend == "rising"
        and rear_temp_trend == "rising"
    ):
        confidence += 0.10

    drag_index = features.get("drag_index")
    straight_line_deficit = features.get("straight_line_speed_deficit_index")
    if (
        definition.id == "drag_sensitivity"
        and isinstance(drag_index, (int, float))
        and isinstance(straight_line_deficit, (int, float))
        and drag_index >= 0.22
        and straight_line_deficit >= 0.18
    ):
        confidence += 0.06

    avg_throttle = features.get("avg_throttle")
    if (
        definition.category == "straight_line"
        and isinstance(avg_throttle, (int, float))
        and avg_throttle < 0.55
    ):
        confidence *= 0.50

    affected_segments = tuple(str(item) for item in features.get("affected_segments", ()))
    if affected_segments:
        accepted_segments = {definition.segment_type}
        if definition.segment_type == "apex":
            accepted_segments.add("mid_corner")
        if definition.segment_type == "entry_to_apex":
            accepted_segments.add("entry_to_apex")
        if definition.segment_type == "braking_entry":
            accepted_segments.update({"entry", "braking_entry"})
        if definition.segment_type == "high_speed_corner":
            accepted_segments.add("high_speed")
        if definition.segment_type == "low_speed_exit":
            accepted_segments.update({"low_speed", "low_speed_exit"})
        if definition.segment_type == "kerb":
            accepted_segments.add("platform")
        if definition.segment_type in {"stint", "platform", "driver", "straight"}:
            accepted_segments.add(definition.segment_type)

        if set(affected_segments).intersection(accepted_segments):
            confidence += 0.18
            if definition.segment_type in affected_segments:
                confidence += 0.05
        else:
            confidence *= 0.45

        if (
            "straight" in affected_segments
            and definition.category != "straight_line"
            and definition.segment_type != "driver"
        ):
            confidence *= 0.40
        if (
            "braking" in affected_segments
            and definition.category not in {"braking", "entry"}
            and definition.segment_type != "driver"
        ):
            confidence *= 0.55

    performance_loss = features.get("performance_loss_sec_per_lap")
    return SymptomDetection(
        symptom_id=definition.id,
        confidence=min(confidence, 0.98),
        evidence=tuple(evidence),
        affected_segments=affected_segments,
        performance_loss_sec_per_lap=(
            float(performance_loss)
            if isinstance(performance_loss, (int, float))
            else None
        ),
        candidate_hypotheses=candidate_hypotheses,
        category=definition.category,
    )


def identify_symptoms(
    features: Mapping[str, Any],
    symptoms: Mapping[str, SymptomDefinition] | None = None,
    symptom_hypothesis_map: Mapping[str, tuple[str, ...]] | None = None,
    min_confidence: float = 0.20,
) -> list[SymptomDetection]:
    """Return ranked symptom candidates from extracted features."""

    definitions = dict(symptoms or load_symptoms())
    mapping = dict(symptom_hypothesis_map or load_symptom_hypothesis_map())
    detections = [
        _detect_from_definition(
            definition=definition,
            features=features,
            candidate_hypotheses=tuple(mapping.get(symptom_id, ())),
        )
        for symptom_id, definition in definitions.items()
    ]
    ranked = sorted(
        (detection for detection in detections if detection.confidence >= min_confidence),
        key=lambda item: (-item.confidence, item.symptom_id),
    )
    if ranked:
        return ranked
    return sorted(detections, key=lambda item: (-item.confidence, item.symptom_id))[:1]


def identify_symptom(features: Mapping[str, Any]) -> SymptomDetection:
    """Return the top-ranked symptom candidate for backward compatibility."""

    return identify_symptoms(features)[0]


def detect_medium_speed_entry_to_apex_understeer(
    features: Mapping[str, Any],
) -> SymptomDetection:
    """Detect the legacy v1 symptom through the config-driven detector."""

    definitions = load_symptoms()
    mapping = load_symptom_hypothesis_map()
    return _detect_from_definition(
        definitions[SUPPORTED_SYMPTOM_ID],
        features,
        tuple(mapping.get(SUPPORTED_SYMPTOM_ID, ())),
    )
