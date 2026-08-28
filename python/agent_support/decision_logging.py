"""Utility helpers for standardized decision logs."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class DecisionMeta:
    decision_mode: str
    repair_attempted: bool
    repair_success: bool
    unmapped_keys: list[str]
    missing_keys: list[str]
    minimal_set_coverage: float
    final_action_source: str
    calibration_applied: bool = False
    calibration_overrode: bool = False
    calibration_reason: str = ""
    objective_risk_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)
