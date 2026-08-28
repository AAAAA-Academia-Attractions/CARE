import os
from pathlib import Path

import duckdb
from jinja2 import Template
import pandas as pd
from loguru import logger
from package_runtime import resolve_feature_source, resolve_feature_store_for_sample, resolve_package_sample_path


def _fact(col: str, display: str, unit: str, meaning: str) -> dict[str, str]:
    return {"col": col, "display": display, "unit": unit, "meaning": meaning}


MAP_LT65 = _fact(
    "map_low_minutes_last1h_thr65",
    "MAP <65 Sustained (last 1h)",
    "min",
    "Cumulative minutes during the last 1 hour with MAP below 65 mmHg; captures hypotension burden rather than a single reading.",
)
MAP_LT60 = _fact(
    "map_low_minutes_last1h_thr60",
    "MAP <60 Sustained (last 1h)",
    "min",
    "Cumulative minutes during the last 1 hour with MAP below 60 mmHg; flags more severe hypotension exposure.",
)
MAP_MEDIAN = _fact(
    "map_median_last1h",
    "MAP Median (last 1h)",
    "mmHg",
    "Median mean arterial pressure over the last 1 hour; lower values suggest more sustained hypotension burden.",
)
MAP_COVERAGE = _fact(
    "has_map_coverage_last1h",
    "MAP Coverage Available (last 1h)",
    "0/1",
    "Whether any MAP data covered the last 1 hour; 1 means low-MAP burden is interpretable, 0 means hypotension burden may be unknown rather than truly zero.",
)
MAP_COVERED_MIN = _fact(
    "map_covered_minutes_last1h",
    "MAP Covered Minutes (last 1h)",
    "min",
    "Number of minutes in the last 1 hour with any MAP coverage; distinguishes true zero hypotension from missing blood pressure coverage.",
)
HR_MEDIAN = _fact(
    "hr_median_last1h",
    "Heart Rate (median last 1h)",
    "bpm",
    "Median heart rate over the last 1 hour; persistent tachycardia can be a stress or shock signal.",
)
SOFA_TOTAL = _fact(
    "sofa_total",
    "Total SOFA Score",
    "points",
    "Overall organ dysfunction burden at the evaluation hour using the SOFA framework.",
)
SOFA_RESP = _fact(
    "sofa_resp",
    "Respiratory SOFA",
    "points",
    "Respiratory component of SOFA; higher values imply worse oxygenation or ventilatory failure.",
)
SOFA_COAG = _fact(
    "sofa_coag",
    "Coagulation SOFA",
    "points",
    "Coagulation component of SOFA; higher values imply worse platelet-related dysfunction.",
)
SOFA_LIVER = _fact(
    "sofa_liver",
    "Liver SOFA",
    "points",
    "Liver component of SOFA; higher values imply worse bilirubin-defined hepatic dysfunction.",
)
SOFA_CNS = _fact(
    "sofa_cns",
    "CNS SOFA",
    "points",
    "Central nervous system component of SOFA; higher values imply worse neurologic dysfunction.",
)
SOFA_RENAL = _fact(
    "sofa_renal",
    "Renal SOFA",
    "points",
    "Renal component of SOFA; higher values imply worse kidney dysfunction.",
)
SOFA_CARDIO = _fact(
    "sofa_cardiovascular",
    "Cardiovascular SOFA",
    "points",
    "Cardiovascular component of SOFA; higher values imply more vasopressor requirement or circulatory failure.",
)
LACTATE = _fact(
    "lactate_latest_6h",
    "Lactate (latest <=6h)",
    "mmol/L",
    "Most recent lactate within the last 6 hours; elevation suggests possible tissue hypoperfusion or metabolic stress.",
)
URINE_OUTPUT = _fact(
    "urine_output_mlkghr_6h",
    "Urine Output (6h rolling, latest <=6h)",
    "mL/kg/hr",
    "Urine output normalized by body weight over a rolling 6-hour window; low values suggest oliguria or poor perfusion.",
)
SPO2 = _fact(
    "spo2_latest_1h",
    "SpO2 (latest <=1h)",
    "%",
    "Most recent peripheral oxygen saturation within the last hour; low values suggest hypoxemia.",
)
TEMPERATURE = _fact(
    "temperature_latest_4h",
    "Temperature (latest <=4h)",
    "C",
    "Most recent body temperature within the last 4 hours; extremes can support infection or physiologic stress.",
)
WBC = _fact(
    "wbc_latest_24h",
    "WBC (latest <=24h)",
    "K/uL",
    "Most recent white blood cell count within the last 24 hours; marked high or low values can support inflammatory or infectious burden.",
)
RHYTHM = _fact(
    "rhythm_recent_6h",
    "ECG / Rhythm (latest <=6h)",
    "",
    "Most recent rhythm or telemetry description within the last 6 hours; can reveal arrhythmia or conduction instability.",
)
NOREPI_EQ = _fact(
    "norepi_eq_dose_max_1h",
    "NE Eq Dose (max overlap <=1h)",
    "mcg/kg/min eq",
    "Maximum norepinephrine-equivalent vasopressor dose overlapping the last hour; higher values imply more hemodynamic support.",
)

# Maps free-text requests to supported DB-backed indicators.
FACTS_MAP = {
    "map < 65 sustained minutes": MAP_LT65,
    "map < 60 sustained minutes": MAP_LT60,
    "map <65 sustained minutes": MAP_LT65,
    "map <60 sustained minutes": MAP_LT60,
    "map<65": MAP_LT65,
    "map<60": MAP_LT60,
    "mean arterial pressure": MAP_MEDIAN,
    "blood pressure": MAP_MEDIAN,
    "arterial pressure": MAP_MEDIAN,
    "map": MAP_MEDIAN,
    "map coverage": MAP_COVERAGE,
    "blood pressure coverage": MAP_COVERAGE,
    "arterial pressure coverage": MAP_COVERAGE,
    "map data coverage": MAP_COVERAGE,
    "has_map_coverage_last1h": MAP_COVERAGE,
    "map covered minutes": MAP_COVERED_MIN,
    "blood pressure covered minutes": MAP_COVERED_MIN,
    "map_covered_minutes_last1h": MAP_COVERED_MIN,
    "heart rate": HR_MEDIAN,
    "hr": HR_MEDIAN,
    "total sofa score": SOFA_TOTAL,
    "total sofa": SOFA_TOTAL,
    "sofa total": SOFA_TOTAL,
    "respiratory sofa": SOFA_RESP,
    "sofa_resp": SOFA_RESP,
    "coagulation sofa": SOFA_COAG,
    "sofa_coag": SOFA_COAG,
    "liver sofa": SOFA_LIVER,
    "sofa_liver": SOFA_LIVER,
    "cns sofa": SOFA_CNS,
    "sofa_cns": SOFA_CNS,
    "renal sofa": SOFA_RENAL,
    "sofa_renal": SOFA_RENAL,
    "kidney": SOFA_RENAL,
    "cardiovascular sofa": SOFA_CARDIO,
    "sofa_cardiovascular": SOFA_CARDIO,
    "cv sofa": SOFA_CARDIO,
    "lactate": LACTATE,
    "serum lactate": LACTATE,
    "urine output": URINE_OUTPUT,
    "urine": URINE_OUTPUT,
    "spo2": SPO2,
    "oxygen saturation": SPO2,
    "temperature": TEMPERATURE,
    "wbc": WBC,
    "white blood cell": WBC,
    "ecg": RHYTHM,
    "telemetry": RHYTHM,
    "rhythm": RHYTHM,
    "vasopressor": NOREPI_EQ,
    "norepinephrine equivalent": NOREPI_EQ,
}

UNAVAILABLE_MSG = "N/A (not available in current CARE facts interface)"

# Intervention-agent only: enforce key objective indicators needed to reproduce C-like guardrails.
CORE_OBJECTIVE_FEATURES = [
    "mean arterial pressure",
    "map < 65 sustained minutes",
    "map < 60 sustained minutes",
    "map coverage",
    "map covered minutes",
    "heart rate",
    "total sofa score",
    "urine output",
    "lactate",
    "vasopressor",
]


class FactsGenerator:
    """
    Framework for Automated Clinical Template Synthesis (FACTS) Engine.
    Translates NLP feature requests into DB-backed indicators and renders
    a structured objective report for final decision reasoning.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._feature_store_df: pd.DataFrame | None = None
        self._feature_store_path = None

    def augment_with_core_objective_bundle(self, requested_features: list[str]) -> list[str]:
        """
        Adds a fixed safety-critical objective bundle for the intervention agent.
        """
        requested = [str(x).strip() for x in (requested_features or []) if str(x).strip()]
        merged = requested + CORE_OBJECTIVE_FEATURES
        deduped = []
        seen = set()
        for feat in merged:
            key = feat.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(feat)
        return deduped

    def _match_requests(self, requested_features: list[str]) -> list[dict]:
        """
        Maps free-text feature requests to known indicators.
        """
        matches = []
        seen = set()
        ordered_items = sorted(FACTS_MAP.items(), key=lambda kv: len(kv[0]), reverse=True)

        for req in requested_features:
            req_lower = str(req).lower().strip()
            if not req_lower:
                continue
            for key, val in ordered_items:
                if key in req_lower or req_lower in key:
                    marker = (val.get("col"), val["display"])
                    if marker not in seen:
                        matches.append(val)
                        seen.add(marker)
                    break
            else:
                logger.warning(f"Feature request '{req}' not mapped. Suppressed to prevent hallucinated retrieval.")
        return matches

    def _query_indicators_db(self, stay_id: int, hr: int, mapped_features: list[dict]) -> dict:
        columns_to_select = {feat["col"] for feat in mapped_features if feat.get("col")}
        if not columns_to_select:
            return {}

        column_sql = {
            "map_low_minutes_last1h_thr65": "sl.map_low_minutes_last1h_thr65",
            "map_low_minutes_last1h_thr60": "sl.map_low_minutes_last1h_thr60",
            "has_map_coverage_last1h": "sl.has_map_coverage_last1h",
            "map_covered_minutes_last1h": "sl.map_covered_minutes_last1h",
            "hr_median_last1h": """
                COALESCE(
                    sl.hr_median_last1h,
                    (
                        SELECT median(v.heart_rate)
                        FROM mimiciv_derived.vitalsign v
                        WHERE v.stay_id = a.stay_id
                          AND v.heart_rate IS NOT NULL
                          AND v.charttime BETWEEN a.t_eval - INTERVAL '1 hour' AND a.t_eval
                    )
                ) AS hr_median_last1h
            """,
            "map_median_last1h": """
                (
                    SELECT median(COALESCE(v.mbp, v.mbp_ni))
                    FROM mimiciv_derived.vitalsign v
                    WHERE v.stay_id = a.stay_id
                      AND COALESCE(v.mbp, v.mbp_ni) IS NOT NULL
                      AND v.charttime BETWEEN a.t_eval - INTERVAL '1 hour' AND a.t_eval
                ) AS map_median_last1h
            """,
            "sofa_total": "sh.sofa_total",
            "sofa_resp": "sh.sofa_resp",
            "sofa_coag": "sh.sofa_coag",
            "sofa_liver": "sh.sofa_liver",
            "sofa_cns": "sh.sofa_cns",
            "sofa_renal": "sh.sofa_renal",
            "sofa_cardiovascular": "sh.sofa_cardiovascular",
            "lactate_latest_6h": """
                (
                    SELECT b.lactate
                    FROM mimiciv_derived.bg b
                    WHERE b.hadm_id = a.hadm_id
                      AND b.lactate IS NOT NULL
                      AND b.charttime BETWEEN a.t_eval - INTERVAL '6 hour' AND a.t_eval
                    ORDER BY b.charttime DESC
                    LIMIT 1
                ) AS lactate_latest_6h
            """,
            "urine_output_mlkghr_6h": """
                (
                    SELECT u.uo_mlkghr_6hr
                    FROM mimiciv_derived.urine_output_rate u
                    WHERE u.stay_id = a.stay_id
                      AND u.uo_mlkghr_6hr IS NOT NULL
                      AND u.charttime BETWEEN a.t_eval - INTERVAL '6 hour' AND a.t_eval
                    ORDER BY u.charttime DESC
                    LIMIT 1
                ) AS urine_output_mlkghr_6h
            """,
            "spo2_latest_1h": """
                (
                    SELECT v.spo2
                    FROM mimiciv_derived.vitalsign v
                    WHERE v.stay_id = a.stay_id
                      AND v.spo2 IS NOT NULL
                      AND v.charttime BETWEEN a.t_eval - INTERVAL '1 hour' AND a.t_eval
                    ORDER BY v.charttime DESC
                    LIMIT 1
                ) AS spo2_latest_1h
            """,
            "temperature_latest_4h": """
                (
                    SELECT v.temperature
                    FROM mimiciv_derived.vitalsign v
                    WHERE v.stay_id = a.stay_id
                      AND v.temperature IS NOT NULL
                      AND v.charttime BETWEEN a.t_eval - INTERVAL '4 hour' AND a.t_eval
                    ORDER BY v.charttime DESC
                    LIMIT 1
                ) AS temperature_latest_4h
            """,
            "wbc_latest_24h": """
                (
                    SELECT cbc.wbc
                    FROM mimiciv_derived.complete_blood_count cbc
                    WHERE cbc.hadm_id = a.hadm_id
                      AND cbc.wbc IS NOT NULL
                      AND cbc.charttime BETWEEN a.t_eval - INTERVAL '24 hour' AND a.t_eval
                    ORDER BY cbc.charttime DESC
                    LIMIT 1
                ) AS wbc_latest_24h
            """,
            "rhythm_recent_6h": """
                (
                    SELECT
                        COALESCE(r.heart_rhythm, 'Unknown')
                        || CASE
                               WHEN r.ectopy_type IS NOT NULL
                                   THEN '; ectopy ' || r.ectopy_type || COALESCE(' (' || r.ectopy_frequency || ')', '')
                               ELSE ''
                           END
                    FROM mimiciv_derived.rhythm r
                    WHERE r.subject_id = a.subject_id
                      AND r.charttime BETWEEN a.t_eval - INTERVAL '6 hour' AND a.t_eval
                    ORDER BY r.charttime DESC
                    LIMIT 1
                ) AS rhythm_recent_6h
            """,
            "norepi_eq_dose_max_1h": """
                (
                    SELECT max(n.norepinephrine_equivalent_dose)
                    FROM mimiciv_derived.norepinephrine_equivalent_dose n
                    WHERE n.stay_id = a.stay_id
                      AND n.endtime >= a.t_eval - INTERVAL '1 hour'
                      AND n.starttime <= a.t_eval
                ) AS norepi_eq_dose_max_1h
            """,
        }

        sql_cols = []
        for col in sorted(columns_to_select):
            expr = column_sql.get(col)
            if expr:
                sql_cols.append(expr.strip())
            else:
                logger.warning(f"Requested FACTS column '{col}' has no SQL mapping; suppressing.")

        if not sql_cols:
            return {}

        sql_cols_str = ",\n                ".join(sql_cols)

        query = f"""
            WITH anchor AS (
                SELECT
                    i.stay_id,
                    i.hadm_id,
                    i.subject_id,
                    i.intime + INTERVAL '{hr} hour' AS t_eval
                FROM mimiciv_icu.icustays i
                WHERE i.stay_id = {stay_id}
                LIMIT 1
            )
            SELECT
                {sql_cols_str}
            FROM anchor a
            LEFT JOIN mimiciv_derived.occult_hypoperfusion_slice sl
                ON sl.stay_id = a.stay_id AND sl.hr = {hr}
            LEFT JOIN mimiciv_derived.sofa_hourly sh
                ON sh.stay_id = a.stay_id AND sh.hr = {hr}
            LIMIT 1
        """

        try:
            with duckdb.connect(self.db_path, read_only=True) as con:
                df = con.execute(query).df()
                if not df.empty:
                    return df.iloc[0].to_dict()
                return {}
        except Exception as e:
            logger.error(f"FACTS SQL execution error: {e}")
            return {"_db_error": str(e)}

    def _load_feature_store(self) -> pd.DataFrame | None:
        sample_env = os.getenv("SAMPLE_LOCK_FILE", "").strip()
        sample_path = None
        if sample_env:
            sample_path = resolve_package_sample_path(sample_env)
        feature_store_path = resolve_feature_store_for_sample(sample_path)
        if feature_store_path is None or not feature_store_path.exists():
            return None
        if self._feature_store_df is None or self._feature_store_path != feature_store_path:
            self._feature_store_df = pd.read_csv(feature_store_path)
            self._feature_store_path = feature_store_path
        return self._feature_store_df

    def _query_indicators_csv(self, stay_id: int, hr: int, mapped_features: list[dict]) -> dict:
        feature_store_df = self._load_feature_store()
        if feature_store_df is None:
            return {"_db_error": "locked_csv requested but feature store CSV could not be resolved"}

        row = feature_store_df[
            (pd.to_numeric(feature_store_df["stay_id"], errors="coerce") == int(stay_id))
            & (pd.to_numeric(feature_store_df["t_eval"], errors="coerce") == int(hr))
        ]
        if row.empty:
            return {}
        values = row.iloc[0].to_dict()
        columns_to_select = {feat["col"] for feat in mapped_features if feat.get("col")}
        return {col: values.get(col) for col in columns_to_select}

    def _query_indicators(self, stay_id: int, hr: int, mapped_features: list[dict]) -> dict:
        feature_source = resolve_feature_source()
        if feature_source in {"locked_csv", "auto"}:
            csv_result = self._query_indicators_csv(stay_id, hr, mapped_features)
            if feature_source == "locked_csv":
                return csv_result
            if "_db_error" not in csv_result:
                return csv_result
        return self._query_indicators_db(stay_id, hr, mapped_features)

    @staticmethod
    def _format_value(val):
        if val is None or pd.isna(val):
            return "N/A"
        if isinstance(val, bool):
            return "1" if val else "0"
        if isinstance(val, float):
            if val.is_integer():
                return str(int(val))
            return f"{val:.3f}".rstrip("0").rstrip(".")
        return str(val)

    def generate(self, stay_id: int, hr: int, requested_features: list[str]) -> str:
        """
        Takes raw requested terms, queries DB-backed indicators, and synthesizes markdown.
        """
        if not requested_features:
            return "*(Agent requested no additional data.)*"

        mapped_features = self._match_requests(requested_features)
        if not mapped_features:
            return "*(No valid clinical indicators could be mapped from the request.)*"

        row_dict = self._query_indicators(stay_id, hr, mapped_features)
        if "_db_error" in row_dict:
            return f"*(Database error fetching indicators: {row_dict['_db_error']})*"

        render_features = []
        for feat in mapped_features:
            col = feat.get("col")
            if col is None:
                val = UNAVAILABLE_MSG
            else:
                val = self._format_value(row_dict.get(col))
            render_features.append(
                {
                    "display": feat["display"],
                    "meaning": feat["meaning"],
                    "value": val,
                    "unit": feat["unit"],
                }
            )

        template = Template(
            """### FACTS Objective Report (CARE)
The system retrieved objective indicators requested by the agent:

| Indicator | Clinical Meaning | Value |
| :--- | :--- | :--- |
{% for feat in features %}| {{ feat.display }} | {{ feat.meaning }} | **{{ feat.value }}** {{ feat.unit }} |
{% endfor %}
> If a value is shown as N/A, treat it as unknown (not normal).
"""
        )
        return template.render(features=render_features)
