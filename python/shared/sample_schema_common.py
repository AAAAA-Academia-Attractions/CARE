import hashlib
import os
from pathlib import Path

import duckdb
import pandas as pd
from package_runtime import (
    PACKAGE_ROOT,
    SAMPLES_DIR,
    resolve_feature_source,
    resolve_package_sample_path,
    resolve_feature_store_for_sample,
)
try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)


MAP_MEDIAN_COL = "map_median_last1h"
PAIN_MAX_COL = "pain_max_last1h"
RASS_N_COL = "rass_n_last1h"
RASS_MAX_COL = "rass_max_last1h"
RASS_MIN_COL = "rass_min_last1h"

SAMPLE_OPTIONAL_COLUMNS = (
    "dataset_fp",
    "subject_id",
    "t_eval_ts",
    "occult_hypoperfusion_slice_thr60",
    "sofa_cardiovascular",
    "has_cam",
    "has_cpot",
    "has_dyspnea",
    "has_cows_anxiety_irritability",
    "has_cows_restlessness",
    "has_map_raw",
    "has_map_coverage_last1h",
    "has_hr",
    "has_hr_crit",
    "has_rr",
    "has_rr_crit",
    "has_spo2",
    "has_spo2_crit",
    "has_uo",
    "has_vent_support",
    "has_pressor",
    "delirium_last",
    "cpot_total_last",
    "dyspnea_last",
    "cows_anxiety_irritability_last",
    "cows_restlessness_last",
    "map_covered_minutes_last1h",
    "rhythm_recent_6h",
    "so_category",
    "so_alignment_group",
)

SAMPLE_INTEGER_COLUMNS = (
    "stay_id",
    "subject_id",
    "t_eval",
    "ground_truth_deterioration",
    RASS_N_COL,
    "has_cam",
    "has_cpot",
    "has_dyspnea",
    "has_cows_anxiety_irritability",
    "has_cows_restlessness",
    "has_map_raw",
    "has_map_coverage_last1h",
    "has_hr",
    "has_hr_crit",
    "has_rr",
    "has_rr_crit",
    "has_spo2",
    "has_spo2_crit",
    "has_uo",
    "has_vent_support",
    "has_pressor",
)

SAMPLE_FLOAT_COLUMNS = (
    PAIN_MAX_COL,
    RASS_MAX_COL,
    RASS_MIN_COL,
    "hr_median_last1h",
    MAP_MEDIAN_COL,
    "map_low_minutes_last1h_thr60",
    "map_low_minutes_last1h_thr65",
    "map_covered_minutes_last1h",
    "sofa_resp",
    "sofa_coag",
    "sofa_liver",
    "sofa_cns",
    "sofa_renal",
    "sofa_cardiovascular",
    "sofa_total",
    "cpot_total_last",
    "dyspnea_last",
    "cows_anxiety_irritability_last",
    "cows_restlessness_last",
)

SAMPLE_BOOLISH_COLUMNS = ("occult_hypoperfusion_slice_thr60",)

SAMPLE_COLUMN_ALIASES = {
    "class_label": "ground_truth_deterioration",
    "label": "ground_truth_deterioration",
    "y": "ground_truth_deterioration",
    "pain_last_last1h": PAIN_MAX_COL,
    "rass_last_last1h": RASS_MAX_COL,
}


def sofa_cardiovascular_enabled() -> bool:
    return os.getenv("ENABLE_SOFA_CARDIOVASCULAR", "true").strip().lower() not in {"0", "false", "no", "off"}


def _reference_full_numeric_feature_columns() -> list[str]:
    cols = [
        "map_low_minutes_last1h_thr60",
        "map_low_minutes_last1h_thr65",
        "map_median_last1h",
        "hr_median_last1h",
        "sofa_total",
        "sofa_resp",
        "sofa_coag",
        "sofa_liver",
        "sofa_cns",
        "sofa_renal",
        "lactate_latest_6h",
        "urine_output_mlkghr_6h",
        "norepi_eq_dose_max_1h",
        "spo2_latest_1h",
        "temperature_latest_4h",
        "wbc_latest_24h",
        "map_current",
        "map_min_1h",
        "map_min_3h",
        "map_mean_3h",
        "time_below_map65_3h",
        "map_worsening_3h",
        "lactate_current",
        "lactate_prev",
        "delta_lactate",
        "pct_change_lactate",
        "hours_since_last_lactate",
        "uo_total_6h",
        "uo_declining",
        "oliguria_6h_std",
        "any_vasopressor_now",
        "norepi_current",
        "norepi_equiv_current",
        "delta_norepi_3h",
        "pressor_started_within_3h",
        "pressor_escalating_3h",
        "vasopressin_added",
    ]
    if sofa_cardiovascular_enabled():
        cols.append("sofa_cardiovascular")
    return cols


def _compact_numeric_feature_columns() -> list[str]:
    cols = [
        "map_median_last1h",
        "has_map_coverage_last1h",
        "map_covered_minutes_last1h",
        "map_low_minutes_last1h_thr65",
        "map_low_minutes_last1h_thr60",
        "hr_median_last1h",
        "sofa_total",
        "sofa_resp",
        "sofa_coag",
        "sofa_liver",
        "sofa_cns",
        "sofa_renal",
        "lactate_latest_6h",
        "urine_output_mlkghr_6h",
        "norepi_eq_dose_max_1h",
        "spo2_latest_1h",
        "temperature_latest_4h",
        "wbc_latest_24h",
    ]
    if sofa_cardiovascular_enabled():
        cols.append("sofa_cardiovascular")
    return cols


FEATURE_PROFILE_FULL = "reference_full"
FEATURE_PROFILE_COMPACT = "compact"
FEATURE_PROFILES = {
    FEATURE_PROFILE_FULL,
    FEATURE_PROFILE_COMPACT,
}


def resolve_feature_profile(feature_profile: str | None = None) -> str:
    profile = (feature_profile or os.getenv("OBJECTIVE_FEATURE_PROFILE", FEATURE_PROFILE_FULL)).strip().lower()
    if profile not in FEATURE_PROFILES:
        raise ValueError(
            f"Unsupported OBJECTIVE_FEATURE_PROFILE={profile}. "
            f"Use one of: {', '.join(sorted(FEATURE_PROFILES))}."
        )
    return profile


def reference_numeric_feature_columns(feature_profile: str | None = None) -> list[str]:
    """Shared numeric feature set for structured reference models and structured ML."""
    profile = resolve_feature_profile(feature_profile)
    if profile == FEATURE_PROFILE_COMPACT:
        return _compact_numeric_feature_columns()
    return _reference_full_numeric_feature_columns()


def reference_categorical_feature_columns(feature_profile: str | None = None) -> list[str]:
    profile = resolve_feature_profile(feature_profile)
    if profile == FEATURE_PROFILE_COMPACT:
        return ["rhythm_recent_6h"]
    return []


def reference_all_feature_columns(feature_profile: str | None = None) -> list[str]:
    return reference_numeric_feature_columns(feature_profile) + reference_categorical_feature_columns(feature_profile)


def all_enrichable_feature_columns() -> list[str]:
    """All feature columns that enrich_objective_features can materialize."""
    return [
        "has_map_coverage_last1h",
        "map_covered_minutes_last1h",
        "map_low_minutes_last1h_thr65",
        "map_low_minutes_last1h_thr60",
        "hr_median_last1h",
        "map_median_last1h",
        "map_current",
        "map_min_1h",
        "map_min_3h",
        "map_mean_3h",
        "time_below_map65_3h",
        "map_worsening_3h",
        "sofa_total",
        "sofa_resp",
        "sofa_coag",
        "sofa_liver",
        "sofa_cns",
        "sofa_renal",
        "sofa_cardiovascular",
        "lactate_current",
        "lactate_prev",
        "delta_lactate",
        "pct_change_lactate",
        "hours_since_last_lactate",
        "lactate_latest_6h",
        "uo_total_6h",
        "urine_output_mlkghr_6h",
        "uo_declining",
        "oliguria_6h_std",
        "any_vasopressor_now",
        "norepi_current",
        "norepi_equiv_current",
        "norepi_eq_dose_max_1h",
        "delta_norepi_3h",
        "pressor_started_within_3h",
        "pressor_escalating_3h",
        "vasopressin_added",
        "spo2_latest_1h",
        "temperature_latest_4h",
        "wbc_latest_24h",
        "rhythm_recent_6h",
    ]


def sample_fingerprint(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    tmp = df[["stay_id", "t_eval"]].copy()
    tmp["stay_id"] = tmp["stay_id"].astype(int)
    tmp["t_eval"] = tmp["t_eval"].astype(int)
    tmp = tmp.sort_values(["stay_id", "t_eval"], kind="stable")
    raw = ";".join(f"{r.stay_id}:{r.t_eval}" for r in tmp.itertuples(index=False))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def resolve_default_paths() -> tuple[Path, Path]:
    version_root = Path(__file__).resolve().parent
    project_root = PACKAGE_ROOT
    return version_root, project_root


def resolve_sample_path(version_root: Path, sample_size: int) -> Path:
    default_sample = SAMPLES_DIR / f"care_sample_n{sample_size}_seed42.csv"
    sample_env = os.getenv("SAMPLE_LOCK_FILE")
    if sample_env:
        sample_path = Path(sample_env)
        if not sample_path.is_absolute():
            sample_path = (PACKAGE_ROOT / sample_path).resolve()
    else:
        sample_path = default_sample
    return sample_path


def resolve_feature_store_path(sample_path: Path | None = None) -> Path | None:
    return resolve_feature_store_for_sample(sample_path)


def _coerce_nullable_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _coerce_nullable_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_boolish(series: pd.Series) -> pd.Series:
    truthy = {"1", "true", "t", "yes", "y"}
    falsy = {"0", "false", "f", "no", "n"}

    if pd.api.types.is_bool_dtype(series):
        return series.astype("Int64")

    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map(
        lambda v: 1 if v in truthy else (0 if v in falsy else pd.NA)
    )
    return pd.Series(mapped, index=series.index, dtype="Int64")


def normalize_sample_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]

    rename_map: dict[str, str] = {}
    for old_name, new_name in SAMPLE_COLUMN_ALIASES.items():
        if old_name in out.columns and new_name not in out.columns:
            rename_map[old_name] = new_name
    if rename_map:
        out = out.rename(columns=rename_map)

    if RASS_MAX_COL in out.columns and RASS_MIN_COL not in out.columns:
        out[RASS_MIN_COL] = out[RASS_MAX_COL]
    if RASS_MAX_COL in out.columns and RASS_N_COL not in out.columns:
        out[RASS_N_COL] = out[RASS_MAX_COL].notna().astype("Int64")
    if PAIN_MAX_COL in out.columns and "pain_last_last1h" not in out.columns:
        out["pain_last_last1h"] = out[PAIN_MAX_COL]
    if RASS_MAX_COL in out.columns and "rass_last_last1h" not in out.columns:
        out["rass_last_last1h"] = out[RASS_MAX_COL]

    for col in SAMPLE_OPTIONAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    for col in SAMPLE_INTEGER_COLUMNS:
        if col in out.columns:
            out[col] = _coerce_nullable_int(out[col])

    for col in SAMPLE_FLOAT_COLUMNS:
        if col in out.columns:
            out[col] = _coerce_nullable_float(out[col])

    for col in SAMPLE_BOOLISH_COLUMNS:
        if col in out.columns:
            out[col] = _coerce_boolish(out[col])

    preferred_order = [
        "dataset_fp",
        "stay_id",
        "subject_id",
        "t_eval",
        "ground_truth_deterioration",
        PAIN_MAX_COL,
        RASS_N_COL,
        RASS_MAX_COL,
        RASS_MIN_COL,
        "hr_median_last1h",
        MAP_MEDIAN_COL,
        "map_low_minutes_last1h_thr60",
        "map_low_minutes_last1h_thr65",
        "has_map_coverage_last1h",
        "map_covered_minutes_last1h",
        "occult_hypoperfusion_slice_thr60",
        "sofa_resp",
        "sofa_coag",
        "sofa_liver",
        "sofa_cns",
        "sofa_renal",
        "sofa_cardiovascular",
        "sofa_total",
        "t_eval_ts",
        "has_cam",
        "has_cpot",
        "has_dyspnea",
        "has_cows_anxiety_irritability",
        "has_cows_restlessness",
        "has_map_raw",
        "has_hr",
        "has_hr_crit",
        "has_rr",
        "has_rr_crit",
        "has_spo2",
        "has_spo2_crit",
        "has_uo",
        "has_vent_support",
        "has_pressor",
        "delirium_last",
        "cpot_total_last",
        "dyspnea_last",
        "cows_anxiety_irritability_last",
        "cows_restlessness_last",
        "so_category",
        "so_alignment_group",
    ]
    ordered = [col for col in preferred_order if col in out.columns]
    extras = [col for col in out.columns if col not in ordered]
    return out[ordered + extras].copy()


def format_rass_window_last1h(row: pd.Series | dict[str, object]) -> str | None:
    rass_n = pd.to_numeric(pd.Series([row.get(RASS_N_COL)]), errors="coerce").iloc[0]
    rass_max = pd.to_numeric(pd.Series([row.get(RASS_MAX_COL)]), errors="coerce").iloc[0]
    rass_min = pd.to_numeric(pd.Series([row.get(RASS_MIN_COL)]), errors="coerce").iloc[0]

    if pd.isna(rass_n) or int(rass_n) <= 0 or (pd.isna(rass_max) and pd.isna(rass_min)):
        return None
    if pd.isna(rass_min):
        rass_min = rass_max
    if pd.isna(rass_max):
        rass_max = rass_min
    if float(rass_min) == float(rass_max):
        return f"{float(rass_max):g} (n={int(rass_n)})"
    return f"{float(rass_min):g} to {float(rass_max):g} (n={int(rass_n)})"


def build_subjective_prompt_context(row: pd.Series | dict[str, object]) -> dict[str, object]:
    pain_value = row.get(PAIN_MAX_COL)
    rass_display = format_rass_window_last1h(row)
    return {
        PAIN_MAX_COL: pain_value,
        RASS_N_COL: row.get(RASS_N_COL),
        RASS_MAX_COL: row.get(RASS_MAX_COL),
        RASS_MIN_COL: row.get(RASS_MIN_COL),
        "pain_last_last1h": pain_value,
        "rass_last_last1h": rass_display,
        "rass_window_last1h": rass_display,
    }


def load_locked_sample(version_root: Path, sample_size: int) -> pd.DataFrame:
    sample_path = resolve_sample_path(version_root, sample_size)
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    df = normalize_sample_schema(pd.read_csv(sample_path))
    required = {"stay_id", "t_eval", "ground_truth_deterioration"}
    missing = required.difference(set(df.columns))
    if missing:
        raise RuntimeError(f"Sample file missing required columns: {sorted(missing)}")

    max_rows = int(os.getenv("MAX_ROWS", "0"))
    if max_rows > 0:
        full_df = df.copy()
        y = df["ground_truth_deterioration"].astype(int)
        if y.nunique() >= 2 and max_rows >= 2:
            half = max_rows // 2
            pos = df[y == 1].head(half)
            neg = df[y == 0].head(half)
            selected = pd.concat([pos, neg], axis=0)
            if len(selected) < max_rows:
                remain = max_rows - len(selected)
                rest = full_df.drop(index=selected.index, errors="ignore")
                selected = pd.concat([selected, rest.head(remain)], axis=0)
            df = selected.head(max_rows).copy()
        else:
            df = df.head(max_rows).copy()

    fp = sample_fingerprint(df)
    pos = int((df["ground_truth_deterioration"].astype(int) == 1).sum())
    neg = int((df["ground_truth_deterioration"].astype(int) == 0).sum())
    logger.info(f"Loaded sample: {sample_path.name} n={len(df)} pos={pos} neg={neg} fp={fp}")
    return df


def _load_feature_store_df(feature_store_path: Path) -> pd.DataFrame:
    df = normalize_sample_schema(pd.read_csv(feature_store_path))
    if "stay_id" not in df.columns or "t_eval" not in df.columns:
        raise RuntimeError(f"Feature store missing key columns: {feature_store_path}")
    return df


def _merge_feature_store(
    sample_df: pd.DataFrame,
    feature_store_df: pd.DataFrame,
    requested_features: list[str] | None = None,
) -> pd.DataFrame:
    requested = set(requested_features or [])
    sample_df = normalize_sample_schema(sample_df)
    feature_store_df = normalize_sample_schema(feature_store_df)

    merge_columns = {"stay_id", "t_eval"}
    if requested:
        merge_columns.update(requested)
    else:
        merge_columns.update(c for c in feature_store_df.columns if c not in {"dataset_fp"})

    available_cols = [c for c in feature_store_df.columns if c in merge_columns]
    feat_df = feature_store_df[available_cols].copy()
    merged = sample_df.merge(feat_df, on=["stay_id", "t_eval"], how="left", suffixes=("", "_pkg"))

    for col in available_cols:
        if col in {"stay_id", "t_eval"}:
            continue
        pkg_col = f"{col}_pkg"
        if col in merged.columns and pkg_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[pkg_col])
            merged = merged.drop(columns=[pkg_col])
        elif pkg_col in merged.columns and col not in merged.columns:
            merged = merged.rename(columns={pkg_col: col})

    return normalize_sample_schema(merged)


def enrich_objective_features(
    db_path: str,
    sample_df: pd.DataFrame,
    requested_features: list[str] | None = None,
) -> pd.DataFrame:
    sample_df = normalize_sample_schema(sample_df)
    if requested_features and all(col in sample_df.columns for col in requested_features):
        return sample_df
    feature_source = resolve_feature_source()
    sample_path_env = os.getenv("SAMPLE_LOCK_FILE", "").strip()
    feature_store_path = resolve_feature_store_path(resolve_package_sample_path(sample_path_env) if sample_path_env else None)

    if feature_source in {"locked_csv", "auto"} and feature_store_path is not None and feature_store_path.exists():
        logger.info(f"Loading features from locked CSV feature store: {feature_store_path.name}")
        feature_store_df = _load_feature_store_df(feature_store_path)
        sample_df = _merge_feature_store(sample_df, feature_store_df, requested_features=requested_features)
        if feature_source == "locked_csv":
            return sample_df
    elif feature_source == "locked_csv":
        raise FileNotFoundError("FEATURE_SOURCE=locked_csv but no package feature store CSV could be resolved.")

    full_query_cols = [
        "has_map_coverage_last1h",
        "map_covered_minutes_last1h",
        "map_low_minutes_last1h_thr65",
        "map_low_minutes_last1h_thr60",
        "hr_median_last1h",
        "map_median_last1h",
        "map_current",
        "map_min_1h",
        "map_min_3h",
        "map_mean_3h",
        "time_below_map65_3h",
        "map_worsening_3h",
        "sofa_total",
        "sofa_resp",
        "sofa_coag",
        "sofa_liver",
        "sofa_cns",
        "sofa_renal",
        "sofa_cardiovascular",
        "lactate_current",
        "lactate_prev",
        "delta_lactate",
        "pct_change_lactate",
        "hours_since_last_lactate",
        "lactate_latest_6h",
        "uo_total_6h",
        "urine_output_mlkghr_6h",
        "uo_declining",
        "oliguria_6h_std",
        "any_vasopressor_now",
        "norepi_current",
        "norepi_equiv_current",
        "norepi_eq_dose_max_1h",
        "delta_norepi_3h",
        "pressor_started_within_3h",
        "pressor_escalating_3h",
        "vasopressin_added",
        "spo2_latest_1h",
        "temperature_latest_4h",
        "wbc_latest_24h",
        "rhythm_recent_6h",
    ]
    query_cols = [c for c in full_query_cols if requested_features is None or c in set(requested_features)]

    query = """
        WITH anchor AS (
            SELECT
                i.stay_id,
                i.hadm_id,
                i.subject_id,
                i.intime + (? * INTERVAL '1 hour') AS t_eval
            FROM mimiciv_icu.icustays i
            WHERE i.stay_id = ?
            LIMIT 1
        )
        SELECT
            sl.has_map_coverage_last1h,
            sl.map_covered_minutes_last1h,
            sl.map_low_minutes_last1h_thr65,
            sl.map_low_minutes_last1h_thr60,
            COALESCE(
                sl.hr_median_last1h,
                (
                    SELECT median(v.heart_rate)
                    FROM mimiciv_derived.vitalsign v
                    WHERE v.stay_id = a.stay_id
                      AND v.heart_rate IS NOT NULL
                      AND v.charttime BETWEEN a.t_eval - INTERVAL '1 hour' AND a.t_eval
                )
            ) AS hr_median_last1h,
            COALESCE(
                (
                    SELECT median(COALESCE(v.mbp, v.mbp_ni))
                    FROM mimiciv_derived.vitalsign v
                    WHERE v.stay_id = a.stay_id
                      AND COALESCE(v.mbp, v.mbp_ni) IS NOT NULL
                      AND v.charttime BETWEEN a.t_eval - INTERVAL '1 hour' AND a.t_eval
                )
            ) AS map_median_last1h,
            (
                SELECT COALESCE(v.mbp, v.mbp_ni)
                FROM mimiciv_derived.vitalsign v
                WHERE v.stay_id = a.stay_id
                  AND COALESCE(v.mbp, v.mbp_ni) IS NOT NULL
                  AND v.charttime BETWEEN a.t_eval - INTERVAL '1 hour' AND a.t_eval
                ORDER BY v.charttime DESC
                LIMIT 1
            ) AS map_current,
            (
                SELECT min(COALESCE(v.mbp, v.mbp_ni))
                FROM mimiciv_derived.vitalsign v
                WHERE v.stay_id = a.stay_id
                  AND COALESCE(v.mbp, v.mbp_ni) IS NOT NULL
                  AND v.charttime BETWEEN a.t_eval - INTERVAL '1 hour' AND a.t_eval
            ) AS map_min_1h,
            (
                SELECT min(COALESCE(v.mbp, v.mbp_ni))
                FROM mimiciv_derived.vitalsign v
                WHERE v.stay_id = a.stay_id
                  AND COALESCE(v.mbp, v.mbp_ni) IS NOT NULL
                  AND v.charttime BETWEEN a.t_eval - INTERVAL '3 hour' AND a.t_eval
            ) AS map_min_3h,
            (
                SELECT avg(COALESCE(v.mbp, v.mbp_ni))
                FROM mimiciv_derived.vitalsign v
                WHERE v.stay_id = a.stay_id
                  AND COALESCE(v.mbp, v.mbp_ni) IS NOT NULL
                  AND v.charttime BETWEEN a.t_eval - INTERVAL '3 hour' AND a.t_eval
            ) AS map_mean_3h,
            (
                WITH points AS (
                    SELECT
                        v.charttime,
                        COALESCE(v.mbp, v.mbp_ni) AS map_value,
                        LEAD(v.charttime) OVER (ORDER BY v.charttime) AS next_charttime
                    FROM mimiciv_derived.vitalsign v
                    WHERE v.stay_id = a.stay_id
                      AND COALESCE(v.mbp, v.mbp_ni) IS NOT NULL
                      AND v.charttime BETWEEN a.t_eval - INTERVAL '3 hour' AND a.t_eval
                )
                SELECT COALESCE(
                    SUM(
                        CASE
                            WHEN map_value < 65 THEN GREATEST(
                                0,
                                DATE_DIFF(
                                    'minute',
                                    charttime,
                                    LEAST(
                                        COALESCE(next_charttime, a.t_eval),
                                        charttime + INTERVAL '30 minute',
                                        a.t_eval
                                    )
                                )
                            )
                            ELSE 0
                        END
                    ),
                    0
                )
                FROM points
            ) AS time_below_map65_3h,
            (
                CASE
                    WHEN (
                        SELECT COALESCE(v.mbp, v.mbp_ni)
                        FROM mimiciv_derived.vitalsign v
                        WHERE v.stay_id = a.stay_id
                          AND COALESCE(v.mbp, v.mbp_ni) IS NOT NULL
                          AND v.charttime BETWEEN a.t_eval - INTERVAL '1 hour' AND a.t_eval
                        ORDER BY v.charttime DESC
                        LIMIT 1
                    ) <= (
                        SELECT avg(COALESCE(v.mbp, v.mbp_ni))
                        FROM mimiciv_derived.vitalsign v
                        WHERE v.stay_id = a.stay_id
                          AND COALESCE(v.mbp, v.mbp_ni) IS NOT NULL
                          AND v.charttime BETWEEN a.t_eval - INTERVAL '3 hour' AND a.t_eval
                    ) - 5
                    THEN 1 ELSE 0
                END
            ) AS map_worsening_3h,
            sh.sofa_total,
            sh.sofa_resp,
            sh.sofa_coag,
            sh.sofa_liver,
            sh.sofa_cns,
            sh.sofa_renal,
            sh.sofa_cardiovascular,
            (
                SELECT b.lactate
                FROM mimiciv_derived.bg b
                WHERE b.hadm_id = a.hadm_id
                  AND b.lactate IS NOT NULL
                  AND b.charttime BETWEEN a.t_eval - INTERVAL '24 hour' AND a.t_eval
                ORDER BY b.charttime DESC
                LIMIT 1
            ) AS lactate_current,
            (
                WITH lact AS (
                    SELECT b.charttime, b.lactate
                    FROM mimiciv_derived.bg b
                    WHERE b.hadm_id = a.hadm_id
                      AND b.lactate IS NOT NULL
                      AND b.charttime BETWEEN a.t_eval - INTERVAL '24 hour' AND a.t_eval
                    ORDER BY b.charttime DESC
                )
                SELECT lactate
                FROM lact
                OFFSET 1
                LIMIT 1
            ) AS lactate_prev,
            (
                WITH lact AS (
                    SELECT b.charttime, b.lactate
                    FROM mimiciv_derived.bg b
                    WHERE b.hadm_id = a.hadm_id
                      AND b.lactate IS NOT NULL
                      AND b.charttime BETWEEN a.t_eval - INTERVAL '24 hour' AND a.t_eval
                    ORDER BY b.charttime DESC
                    LIMIT 2
                )
                SELECT
                    CASE
                        WHEN COUNT(*) = 2 THEN MAX(lactate) FILTER (WHERE charttime = (SELECT MAX(charttime) FROM lact))
                             - MIN(lactate) FILTER (WHERE charttime <> (SELECT MAX(charttime) FROM lact))
                        ELSE NULL
                    END
                FROM lact
            ) AS delta_lactate,
            (
                WITH lact AS (
                    SELECT b.charttime, b.lactate
                    FROM mimiciv_derived.bg b
                    WHERE b.hadm_id = a.hadm_id
                      AND b.lactate IS NOT NULL
                      AND b.charttime BETWEEN a.t_eval - INTERVAL '24 hour' AND a.t_eval
                    ORDER BY b.charttime DESC
                    LIMIT 2
                ),
                latest AS (
                    SELECT lactate
                    FROM lact
                    ORDER BY charttime DESC
                    LIMIT 1
                ),
                prev AS (
                    SELECT lactate
                    FROM lact
                    ORDER BY charttime DESC
                    OFFSET 1 LIMIT 1
                )
                SELECT
                    CASE
                        WHEN (SELECT lactate FROM prev) IS NULL OR (SELECT lactate FROM prev) = 0 THEN NULL
                        ELSE ((SELECT lactate FROM latest) - (SELECT lactate FROM prev))
                             / (SELECT lactate FROM prev) * 100
                    END
            ) AS pct_change_lactate,
            (
                SELECT DATE_DIFF('minute', MAX(b.charttime), a.t_eval) / 60.0
                FROM mimiciv_derived.bg b
                WHERE b.hadm_id = a.hadm_id
                  AND b.lactate IS NOT NULL
                  AND b.charttime <= a.t_eval
            ) AS hours_since_last_lactate,
            (
                SELECT b.lactate
                FROM mimiciv_derived.bg b
                WHERE b.hadm_id = a.hadm_id
                  AND b.lactate IS NOT NULL
                  AND b.charttime BETWEEN a.t_eval - INTERVAL '6 hour' AND a.t_eval
                ORDER BY b.charttime DESC
                LIMIT 1
            ) AS lactate_latest_6h,
            (
                SELECT u.uo_mlkghr_6hr
                FROM mimiciv_derived.urine_output_rate u
                WHERE u.stay_id = a.stay_id
                  AND u.uo_mlkghr_6hr IS NOT NULL
                  AND u.charttime BETWEEN a.t_eval - INTERVAL '6 hour' AND a.t_eval
                ORDER BY u.charttime DESC
                LIMIT 1
            ) AS urine_output_mlkghr_6h,
            (
                SELECT u.urineoutput_6hr
                FROM mimiciv_derived.urine_output_rate u
                WHERE u.stay_id = a.stay_id
                  AND u.urineoutput_6hr IS NOT NULL
                  AND u.charttime BETWEEN a.t_eval - INTERVAL '6 hour' AND a.t_eval
                ORDER BY u.charttime DESC
                LIMIT 1
            ) AS uo_total_6h,
            (
                WITH latest AS (
                    SELECT u.uo_mlkghr_6hr AS rate
                    FROM mimiciv_derived.urine_output_rate u
                    WHERE u.stay_id = a.stay_id
                      AND u.uo_mlkghr_6hr IS NOT NULL
                      AND u.charttime BETWEEN a.t_eval - INTERVAL '6 hour' AND a.t_eval
                    ORDER BY u.charttime DESC
                    LIMIT 1
                ),
                prev AS (
                    SELECT u.uo_mlkghr_6hr AS rate
                    FROM mimiciv_derived.urine_output_rate u
                    WHERE u.stay_id = a.stay_id
                      AND u.uo_mlkghr_6hr IS NOT NULL
                      AND u.charttime BETWEEN a.t_eval - INTERVAL '12 hour' AND a.t_eval - INTERVAL '6 hour'
                    ORDER BY u.charttime DESC
                    LIMIT 1
                )
                SELECT
                    CASE
                        WHEN (SELECT rate FROM latest) IS NULL OR (SELECT rate FROM prev) IS NULL THEN 0
                        WHEN (SELECT rate FROM latest) <= (SELECT rate FROM prev) * 0.8 THEN 1
                        WHEN (SELECT rate FROM prev) - (SELECT rate FROM latest) >= 0.15 THEN 1
                        ELSE 0
                    END
            ) AS uo_declining,
            (
                CASE
                    WHEN (
                        SELECT u.uo_mlkghr_6hr
                        FROM mimiciv_derived.urine_output_rate u
                        WHERE u.stay_id = a.stay_id
                          AND u.uo_mlkghr_6hr IS NOT NULL
                          AND u.charttime BETWEEN a.t_eval - INTERVAL '6 hour' AND a.t_eval
                        ORDER BY u.charttime DESC
                        LIMIT 1
                    ) < 0.5 THEN 1 ELSE 0
                END
            ) AS oliguria_6h_std,
            (
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM mimiciv_derived.vasoactive_agent v
                        WHERE v.stay_id = a.stay_id
                          AND v.starttime <= a.t_eval
                          AND v.endtime >= a.t_eval
                          AND (
                              COALESCE(v.dopamine, 0) > 0
                              OR COALESCE(v.epinephrine, 0) > 0
                              OR COALESCE(v.norepinephrine, 0) > 0
                              OR COALESCE(v.phenylephrine, 0) > 0
                              OR COALESCE(v.vasopressin, 0) > 0
                          )
                    ) THEN 1 ELSE 0
                END
            ) AS any_vasopressor_now,
            (
                SELECT MAX(n.vaso_rate)
                FROM mimiciv_derived.norepinephrine n
                WHERE n.stay_id = a.stay_id
                  AND n.starttime <= a.t_eval
                  AND n.endtime >= a.t_eval
            ) AS norepi_current,
            (
                SELECT MAX(n.norepinephrine_equivalent_dose)
                FROM mimiciv_derived.norepinephrine_equivalent_dose n
                WHERE n.stay_id = a.stay_id
                  AND n.starttime <= a.t_eval
                  AND n.endtime >= a.t_eval
            ) AS norepi_equiv_current,
            (
                SELECT max(n.norepinephrine_equivalent_dose)
                FROM mimiciv_derived.norepinephrine_equivalent_dose n
                WHERE n.stay_id = a.stay_id
                  AND n.endtime >= a.t_eval - INTERVAL '1 hour'
                  AND n.starttime <= a.t_eval
            ) AS norepi_eq_dose_max_1h,
            (
                WITH current AS (
                    SELECT MAX(n.norepinephrine_equivalent_dose) AS dose
                    FROM mimiciv_derived.norepinephrine_equivalent_dose n
                    WHERE n.stay_id = a.stay_id
                      AND n.starttime <= a.t_eval
                      AND n.endtime >= a.t_eval
                ),
                prev AS (
                    SELECT MAX(n.norepinephrine_equivalent_dose) AS dose
                    FROM mimiciv_derived.norepinephrine_equivalent_dose n
                    WHERE n.stay_id = a.stay_id
                      AND n.starttime <= a.t_eval - INTERVAL '3 hour'
                      AND n.endtime >= a.t_eval - INTERVAL '3 hour'
                )
                SELECT
                    CASE
                        WHEN (SELECT dose FROM current) IS NULL THEN NULL
                        ELSE (SELECT dose FROM current) - COALESCE((SELECT dose FROM prev), 0)
                    END
            ) AS delta_norepi_3h,
            (
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM mimiciv_derived.norepinephrine_equivalent_dose n
                        WHERE n.stay_id = a.stay_id
                          AND n.starttime BETWEEN a.t_eval - INTERVAL '3 hour' AND a.t_eval
                          AND n.norepinephrine_equivalent_dose > 0
                    ) THEN 1
                    WHEN EXISTS (
                        SELECT 1
                        FROM mimiciv_derived.vasopressin v
                        WHERE v.stay_id = a.stay_id
                          AND v.starttime BETWEEN a.t_eval - INTERVAL '3 hour' AND a.t_eval
                          AND COALESCE(v.vaso_rate, 0) > 0
                    ) THEN 1
                    ELSE 0
                END
            ) AS pressor_started_within_3h,
            (
                WITH current AS (
                    SELECT MAX(n.norepinephrine_equivalent_dose) AS dose
                    FROM mimiciv_derived.norepinephrine_equivalent_dose n
                    WHERE n.stay_id = a.stay_id
                      AND n.starttime <= a.t_eval
                      AND n.endtime >= a.t_eval
                ),
                prev AS (
                    SELECT MAX(n.norepinephrine_equivalent_dose) AS dose
                    FROM mimiciv_derived.norepinephrine_equivalent_dose n
                    WHERE n.stay_id = a.stay_id
                      AND n.starttime <= a.t_eval - INTERVAL '3 hour'
                      AND n.endtime >= a.t_eval - INTERVAL '3 hour'
                )
                SELECT
                    CASE
                        WHEN (SELECT dose FROM current) IS NULL THEN 0
                        WHEN COALESCE((SELECT dose FROM prev), 0) = 0 AND (SELECT dose FROM current) >= 0.05 THEN 1
                        WHEN (SELECT dose FROM current) - COALESCE((SELECT dose FROM prev), 0) >= 0.05 THEN 1
                        WHEN COALESCE((SELECT dose FROM prev), 0) > 0
                         AND (SELECT dose FROM current) >= COALESCE((SELECT dose FROM prev), 0) * 1.5
                         THEN 1
                        ELSE 0
                    END
            ) AS pressor_escalating_3h,
            (
                CASE
                    WHEN EXISTS (
                        SELECT 1
                        FROM mimiciv_derived.vasopressin v
                        WHERE v.stay_id = a.stay_id
                          AND v.starttime BETWEEN a.t_eval - INTERVAL '3 hour' AND a.t_eval
                          AND COALESCE(v.vaso_rate, 0) > 0
                    ) THEN 1 ELSE 0
                END
            ) AS vasopressin_added,
            (
                SELECT v.spo2
                FROM mimiciv_derived.vitalsign v
                WHERE v.stay_id = a.stay_id
                  AND v.spo2 IS NOT NULL
                  AND v.charttime BETWEEN a.t_eval - INTERVAL '1 hour' AND a.t_eval
                ORDER BY v.charttime DESC
                LIMIT 1
            ) AS spo2_latest_1h,
            (
                SELECT v.temperature
                FROM mimiciv_derived.vitalsign v
                WHERE v.stay_id = a.stay_id
                  AND v.temperature IS NOT NULL
                  AND v.charttime BETWEEN a.t_eval - INTERVAL '4 hour' AND a.t_eval
                ORDER BY v.charttime DESC
                LIMIT 1
            ) AS temperature_latest_4h,
            (
                SELECT cbc.wbc
                FROM mimiciv_derived.complete_blood_count cbc
                WHERE cbc.hadm_id = a.hadm_id
                  AND cbc.wbc IS NOT NULL
                  AND cbc.charttime BETWEEN a.t_eval - INTERVAL '24 hour' AND a.t_eval
                ORDER BY cbc.charttime DESC
                LIMIT 1
            ) AS wbc_latest_24h,
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
        FROM anchor a
        LEFT JOIN mimiciv_derived.occult_hypoperfusion_slice sl
            ON sl.stay_id = a.stay_id AND sl.hr = ?
        LEFT JOIN mimiciv_derived.sofa_hourly sh
            ON sh.stay_id = a.stay_id AND sh.hr = ?
        LIMIT 1
    """

    rows = []
    with duckdb.connect(db_path, read_only=True) as con:
        for r in sample_df.itertuples(index=False):
            stay_id = int(r.stay_id)
            t_eval = int(r.t_eval)
            recs = con.execute(query, [t_eval, stay_id, t_eval, t_eval]).fetchdf()
            obj = recs.iloc[0].to_dict() if not recs.empty else {}
            obj["stay_id"] = stay_id
            obj["t_eval"] = t_eval
            rows.append(obj)

    feat_df = pd.DataFrame(rows)
    merged = sample_df.merge(feat_df, on=["stay_id", "t_eval"], how="left", suffixes=("", "_qry"))
    for col in query_cols:
        qcol = f"{col}_qry"
        if col in merged.columns and qcol in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[qcol])
            merged = merged.drop(columns=[qcol])
        elif qcol in merged.columns and col not in merged.columns:
            merged = merged.rename(columns={qcol: col})
    return merged
