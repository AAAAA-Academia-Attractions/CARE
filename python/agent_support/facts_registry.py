"""Canonical FACTS key registry.

This module keeps a strict key whitelist to avoid free-text drift between
evidence planning request generation and DB-backed FACTS retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


def sofa_cardiovascular_enabled() -> bool:
    return os.getenv("ENABLE_SOFA_CARDIOVASCULAR", "true").strip().lower() not in {"0", "false", "no", "off"}


@dataclass(frozen=True)
class FactSpec:
    key: str
    request_phrase: str
    display: str
    meaning: str
    unit: str
    col: str


FACT_SPECS: dict[str, FactSpec] = {
    "map_median_last1h": FactSpec(
        key="map_median_last1h",
        request_phrase="mean arterial pressure",
        display="MAP Median (last 1h)",
        meaning="Median mean arterial pressure over the last 1 hour; lower values suggest more sustained hypotension burden.",
        unit="mmHg",
        col="map_median_last1h",
    ),
    "has_map_coverage_last1h": FactSpec(
        key="has_map_coverage_last1h",
        request_phrase="map coverage",
        display="MAP Coverage Available (last 1h)",
        meaning="Whether any MAP data covered the last 1 hour; 1 means low-MAP burden is interpretable, 0 means hypotension burden may be unknown rather than truly zero.",
        unit="0/1",
        col="has_map_coverage_last1h",
    ),
    "map_covered_minutes_last1h": FactSpec(
        key="map_covered_minutes_last1h",
        request_phrase="map covered minutes",
        display="MAP Covered Minutes (last 1h)",
        meaning="Number of minutes in the last 1 hour with any MAP coverage; distinguishes true zero hypotension from missing blood pressure coverage.",
        unit="min",
        col="map_covered_minutes_last1h",
    ),
    "map_low_minutes_last1h_thr65": FactSpec(
        key="map_low_minutes_last1h_thr65",
        request_phrase="map < 65 sustained minutes",
        display="MAP <65 Sustained (last 1h)",
        meaning="Cumulative minutes during the last 1 hour with MAP below 65 mmHg; captures hypotension burden rather than a single reading.",
        unit="min",
        col="map_low_minutes_last1h_thr65",
    ),
    "map_low_minutes_last1h_thr60": FactSpec(
        key="map_low_minutes_last1h_thr60",
        request_phrase="map < 60 sustained minutes",
        display="MAP <60 Sustained (last 1h)",
        meaning="Cumulative minutes during the last 1 hour with MAP below 60 mmHg; flags more severe hypotension exposure.",
        unit="min",
        col="map_low_minutes_last1h_thr60",
    ),
    "hr_median_last1h": FactSpec(
        key="hr_median_last1h",
        request_phrase="heart rate",
        display="Heart Rate (median last 1h)",
        meaning="Median heart rate over the last 1 hour; persistent tachycardia can be a stress or shock signal.",
        unit="bpm",
        col="hr_median_last1h",
    ),
    "sofa_total": FactSpec(
        key="sofa_total",
        request_phrase="total sofa score",
        display="Total SOFA Score",
        meaning="Overall organ dysfunction burden at the evaluation hour using the SOFA framework.",
        unit="points",
        col="sofa_total",
    ),
    "sofa_resp": FactSpec(
        key="sofa_resp",
        request_phrase="respiratory sofa",
        display="Respiratory SOFA",
        meaning="Respiratory component of SOFA; higher values imply worse oxygenation or ventilatory failure.",
        unit="points",
        col="sofa_resp",
    ),
    "sofa_coag": FactSpec(
        key="sofa_coag",
        request_phrase="coagulation sofa",
        display="Coagulation SOFA",
        meaning="Coagulation component of SOFA; higher values imply worse platelet-related dysfunction.",
        unit="points",
        col="sofa_coag",
    ),
    "sofa_liver": FactSpec(
        key="sofa_liver",
        request_phrase="liver sofa",
        display="Liver SOFA",
        meaning="Liver component of SOFA; higher values imply worse bilirubin-defined hepatic dysfunction.",
        unit="points",
        col="sofa_liver",
    ),
    "sofa_cns": FactSpec(
        key="sofa_cns",
        request_phrase="cns sofa",
        display="CNS SOFA",
        meaning="Central nervous system component of SOFA; higher values imply worse neurologic dysfunction.",
        unit="points",
        col="sofa_cns",
    ),
    "sofa_renal": FactSpec(
        key="sofa_renal",
        request_phrase="renal sofa",
        display="Renal SOFA",
        meaning="Renal component of SOFA; higher values imply worse kidney dysfunction.",
        unit="points",
        col="sofa_renal",
    ),
    "sofa_cardiovascular": FactSpec(
        key="sofa_cardiovascular",
        request_phrase="cardiovascular sofa",
        display="Cardiovascular SOFA",
        meaning="Cardiovascular component of SOFA; higher values imply more vasopressor requirement or circulatory failure.",
        unit="points",
        col="sofa_cardiovascular",
    ),
    "lactate_latest_6h": FactSpec(
        key="lactate_latest_6h",
        request_phrase="lactate",
        display="Lactate (latest <=6h)",
        meaning="Most recent lactate within the last 6 hours; elevation suggests possible tissue hypoperfusion or metabolic stress.",
        unit="mmol/L",
        col="lactate_latest_6h",
    ),
    "urine_output_mlkghr_6h": FactSpec(
        key="urine_output_mlkghr_6h",
        request_phrase="urine output",
        display="Urine Output (6h rolling, latest <=6h)",
        meaning="Urine output normalized by body weight over a rolling 6-hour window; low values suggest oliguria or poor perfusion.",
        unit="mL/kg/hr",
        col="urine_output_mlkghr_6h",
    ),
    "norepi_eq_dose_max_1h": FactSpec(
        key="norepi_eq_dose_max_1h",
        request_phrase="vasopressor",
        display="NE Eq Dose (max overlap <=1h)",
        meaning="Maximum norepinephrine-equivalent vasopressor dose overlapping the last hour; higher values imply more hemodynamic support.",
        unit="mcg/kg/min eq",
        col="norepi_eq_dose_max_1h",
    ),
    "spo2_latest_1h": FactSpec(
        key="spo2_latest_1h",
        request_phrase="spo2",
        display="SpO2 (latest <=1h)",
        meaning="Most recent peripheral oxygen saturation within the last hour; low values suggest hypoxemia.",
        unit="%",
        col="spo2_latest_1h",
    ),
    "temperature_latest_4h": FactSpec(
        key="temperature_latest_4h",
        request_phrase="temperature",
        display="Temperature (latest <=4h)",
        meaning="Most recent body temperature within the last 4 hours; extremes can support infection or physiologic stress.",
        unit="C",
        col="temperature_latest_4h",
    ),
    "wbc_latest_24h": FactSpec(
        key="wbc_latest_24h",
        request_phrase="wbc",
        display="WBC (latest <=24h)",
        meaning="Most recent white blood cell count within the last 24 hours; marked high or low values can support inflammatory or infectious burden.",
        unit="K/uL",
        col="wbc_latest_24h",
    ),
    "rhythm_recent_6h": FactSpec(
        key="rhythm_recent_6h",
        request_phrase="telemetry",
        display="ECG / Rhythm (latest <=6h)",
        meaning="Most recent rhythm or telemetry description within the last 6 hours; can reveal arrhythmia or conduction instability.",
        unit="",
        col="rhythm_recent_6h",
    ),
}

ALLOWED_FACT_KEYS = tuple(sorted(FACT_SPECS.keys()))


def format_fact_key_line(key: str) -> str:
    spec = FACT_SPECS[key]
    unit_text = f" Unit: {spec.unit}." if spec.unit else ""
    return (
        f"`{spec.key}` -> column `{spec.col}`: {spec.display}. "
        f"{spec.meaning}{unit_text}"
    )


def allowed_fact_key_lines(keys: tuple[str, ...] | list[str] = ALLOWED_FACT_KEYS) -> tuple[str, ...]:
    return tuple(format_fact_key_line(key) for key in keys if key in FACT_SPECS)

# Core objective bundle injected for intervention branch only.
CORE_OBJECTIVE_KEYS = (
    "map_median_last1h",
    "has_map_coverage_last1h",
    "map_covered_minutes_last1h",
    "map_low_minutes_last1h_thr65",
    "map_low_minutes_last1h_thr60",
    "hr_median_last1h",
    "sofa_total",
    "lactate_latest_6h",
    "urine_output_mlkghr_6h",
    "norepi_eq_dose_max_1h",
)

# Minimal set gating for reliable OBSERVE decisions.
MINIMAL_SET_KEYS = (
    "has_map_coverage_last1h",
    "map_low_minutes_last1h_thr65",
    "map_low_minutes_last1h_thr60",
    "sofa_total",
    "lactate_latest_6h",
    "urine_output_mlkghr_6h",
)


def dedupe_keep_order(keys: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        out.append(key)
        seen.add(key)
    return out


def coerce_allowed(keys: list[str]) -> tuple[list[str], list[str]]:
    """Return (valid_keys, invalid_keys)."""
    valid: list[str] = []
    invalid: list[str] = []
    for key in keys:
        if key in FACT_SPECS:
            valid.append(key)
        else:
            invalid.append(key)
    return dedupe_keep_order(valid), dedupe_keep_order(invalid)
