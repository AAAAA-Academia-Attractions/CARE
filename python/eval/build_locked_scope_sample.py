from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import duckdb
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = THIS_DIR.parent
SHARED_DIR = PYTHON_ROOT / "shared"
for _p in (PYTHON_ROOT, SHARED_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from package_runtime import PACKAGE_ROOT, resolve_duckdb_path
from sample_schema_common import normalize_sample_schema

LABEL_WINDOW_OPTIONS = {"6_12", "0_6", "0_12"}
OVERLAP_MODE_OPTIONS = {"pair", "stay", "subject"}


def _fingerprint(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    tmp = df[["stay_id", "t_eval"]].copy()
    tmp["stay_id"] = tmp["stay_id"].astype(int)
    tmp["t_eval"] = tmp["t_eval"].astype(int)
    tmp = tmp.sort_values(["stay_id", "t_eval"], kind="stable")
    raw = ";".join(f"{r.stay_id}:{r.t_eval}" for r in tmp.itertuples(index=False))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _relative_to_package(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PACKAGE_ROOT.resolve()))
    except Exception:
        return path.name


def _label_cte_sql(label_window: str) -> str:
    if label_window == "6_12":
        return """
    labels AS (
      SELECT
        stay_id,
        t_base,
        y_deteriorate_delta2_6_12 AS ground_truth_deterioration
      FROM mimiciv_derived.sofa_labels
      WHERE y_deteriorate_delta2_6_12 IS NOT NULL
    ),
    """
    if label_window == "0_6":
        return """
    labels AS (
      WITH future_window AS (
        SELECT
          stay_id,
          hr AS t_base,
          sofa_total AS sofa_t,
          MAX(sofa_total) OVER (
            PARTITION BY stay_id
            ORDER BY hr
            ROWS BETWEEN 1 FOLLOWING AND 6 FOLLOWING
          ) AS sofa_future_max_0_6,
          COUNT(sofa_total) OVER (
            PARTITION BY stay_id
            ORDER BY hr
            ROWS BETWEEN 1 FOLLOWING AND 6 FOLLOWING
          ) AS valid_future_hours_0_6
        FROM mimiciv_derived.sofa_hourly
      )
      SELECT
        stay_id,
        t_base,
        CASE
          WHEN sofa_t IS NULL THEN NULL
          WHEN valid_future_hours_0_6 < 4 THEN NULL
          WHEN (sofa_future_max_0_6 - sofa_t) >= 2 THEN 1
          ELSE 0
        END AS ground_truth_deterioration
      FROM future_window
      WHERE t_base >= 0
        AND sofa_t IS NOT NULL
        AND valid_future_hours_0_6 >= 4
    ),
    """
    if label_window == "0_12":
        return """
    labels AS (
      WITH future_window AS (
        SELECT
          stay_id,
          hr AS t_base,
          sofa_total AS sofa_t,
          MAX(sofa_total) OVER (
            PARTITION BY stay_id
            ORDER BY hr
            ROWS BETWEEN 1 FOLLOWING AND 12 FOLLOWING
          ) AS sofa_future_max_0_12,
          COUNT(sofa_total) OVER (
            PARTITION BY stay_id
            ORDER BY hr
            ROWS BETWEEN 1 FOLLOWING AND 12 FOLLOWING
          ) AS valid_future_hours_0_12
        FROM mimiciv_derived.sofa_hourly
      )
      SELECT
        stay_id,
        t_base,
        CASE
          WHEN sofa_t IS NULL THEN NULL
          WHEN valid_future_hours_0_12 < 4 THEN NULL
          WHEN (sofa_future_max_0_12 - sofa_t) >= 2 THEN 1
          ELSE 0
        END AS ground_truth_deterioration
      FROM future_window
      WHERE t_base >= 0
        AND sofa_t IS NOT NULL
        AND valid_future_hours_0_12 >= 4
    ),
    """
    raise ValueError(f"Unsupported label_window={label_window}")


def _query_current_base_pool(db_path: str, *, label_window: str) -> pd.DataFrame:
    label_cte = _label_cte_sql(label_window)
    query = f"""
    WITH
    {label_cte}
    base AS (
      SELECT
        lbl.stay_id,
        icu.subject_id,
        lbl.t_base AS t_eval,
        lbl.ground_truth_deterioration,
        sl.pain_max_last1h,
        sl.rass_n_last1h,
        sl.rass_max_last1h,
        sl.rass_min_last1h,
        sl.hr_median_last1h,
        (
            SELECT median(COALESCE(v.mbp, v.mbp_ni))
            FROM mimiciv_derived.vitalsign v
            WHERE v.stay_id = lbl.stay_id
              AND COALESCE(v.mbp, v.mbp_ni) IS NOT NULL
              AND v.charttime BETWEEN (icu.intime + lbl.t_base * INTERVAL '1 hour') - INTERVAL '1 hour'
                                  AND (icu.intime + lbl.t_base * INTERVAL '1 hour')
        ) AS map_median_last1h,
        sl.map_low_minutes_last1h_thr60,
        sl.map_low_minutes_last1h_thr65,
        sl.has_map_coverage_last1h,
        sl.map_covered_minutes_last1h,
        sl.occult_hypoperfusion_slice_thr60,
        sh.sofa_resp,
        sh.sofa_coag,
        sh.sofa_liver,
        sh.sofa_cns,
        sh.sofa_renal,
        sh.sofa_cardiovascular,
        sh.sofa_total,
        icu.intime + lbl.t_base * INTERVAL '1 hour' AS t_eval_ts
      FROM labels lbl
      JOIN mimiciv_icu.icustays icu
        ON lbl.stay_id = icu.stay_id
      JOIN mimiciv_derived.occult_hypoperfusion_slice sl
        ON lbl.stay_id = sl.stay_id AND lbl.t_base = sl.hr
      JOIN mimiciv_derived.sofa_hourly sh
        ON lbl.stay_id = sh.stay_id AND lbl.t_base = sh.hr
      WHERE sl.pain_max_last1h = 0
        AND sl.rass_n_last1h >= 1
        AND sl.rass_max_last1h <= 0
        AND sl.rass_min_last1h > -3
        AND sl.map_low_minutes_last1h_thr65 > 5
    )
    SELECT *
    FROM base
    ORDER BY stay_id, t_eval
    """
    with duckdb.connect(db_path, read_only=True) as con:
        return con.execute(query).df()


def _load_exclusion_frames(csv_paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pair_frames: list[pd.DataFrame] = []
    stay_frames: list[pd.DataFrame] = []
    subject_frames: list[pd.DataFrame] = []
    for path in csv_paths:
        df = normalize_sample_schema(pd.read_csv(path))
        pair_frames.append(df[["stay_id", "t_eval"]].drop_duplicates().copy())
        stay_frames.append(df[["stay_id"]].drop_duplicates().copy())
        if "subject_id" in df.columns:
            subject_frames.append(df[["subject_id"]].dropna().drop_duplicates().copy())

    pair_df = pd.concat(pair_frames, ignore_index=True).drop_duplicates() if pair_frames else pd.DataFrame(columns=["stay_id", "t_eval"])
    stay_df = pd.concat(stay_frames, ignore_index=True).drop_duplicates() if stay_frames else pd.DataFrame(columns=["stay_id"])
    subject_df = pd.concat(subject_frames, ignore_index=True).drop_duplicates() if subject_frames else pd.DataFrame(columns=["subject_id"])

    if not pair_df.empty:
        pair_df["stay_id"] = pair_df["stay_id"].astype(int)
        pair_df["t_eval"] = pair_df["t_eval"].astype(int)
    if not stay_df.empty:
        stay_df["stay_id"] = stay_df["stay_id"].astype(int)
    if not subject_df.empty:
        subject_df["subject_id"] = subject_df["subject_id"].astype(int)
    return pair_df, stay_df, subject_df


def _stable_select(df: pd.DataFrame, seed: int, label: int, n: int) -> pd.DataFrame:
    tmp = df.copy()
    tmp["_ord"] = [
        hashlib.md5(f"{int(r.stay_id)}:{int(r.t_eval)}:{label}:{seed}".encode("utf-8")).hexdigest()
        for r in tmp.itertuples(index=False)
    ]
    tmp = tmp.sort_values(["_ord", "stay_id", "t_eval"], kind="stable").reset_index(drop=True)
    return tmp.iloc[:n].drop(columns=["_ord"]).reset_index(drop=True)


def _overlap_stats(sample_df: pd.DataFrame, exclusion_csvs: list[Path]) -> tuple[dict[str, dict[str, int]], dict[str, int]]:
    sample_pairs = sample_df[["stay_id", "t_eval"]].drop_duplicates().copy()
    sample_pairs["stay_id"] = sample_pairs["stay_id"].astype(int)
    sample_pairs["t_eval"] = sample_pairs["t_eval"].astype(int)
    sample_pair_set = set(map(tuple, sample_pairs.itertuples(index=False, name=None)))
    sample_stay_set = set(sample_df["stay_id"].astype(int).drop_duplicates().tolist())
    sample_subject_set = set(sample_df["subject_id"].astype(int).drop_duplicates().tolist())

    per_file: dict[str, dict[str, int]] = {}
    union_pair_set: set[tuple[int, int]] = set()
    union_stay_set: set[int] = set()
    union_subject_set: set[int] = set()

    for path in exclusion_csvs:
        df = normalize_sample_schema(pd.read_csv(path))
        pairs = df[["stay_id", "t_eval"]].drop_duplicates().copy()
        pairs["stay_id"] = pairs["stay_id"].astype(int)
        pairs["t_eval"] = pairs["t_eval"].astype(int)
        pair_set = set(map(tuple, pairs.itertuples(index=False, name=None)))
        stay_set = set(df["stay_id"].astype(int).drop_duplicates().tolist())
        subject_set = set(df["subject_id"].dropna().astype(int).drop_duplicates().tolist()) if "subject_id" in df.columns else set()

        union_pair_set |= pair_set
        union_stay_set |= stay_set
        union_subject_set |= subject_set

        per_file[path.name] = {
            "pair_overlap": len(sample_pair_set & pair_set),
            "stay_overlap": len(sample_stay_set & stay_set),
            "subject_overlap": len(sample_subject_set & subject_set),
        }

    overall = {
        "pair_overlap": len(sample_pair_set & union_pair_set),
        "stay_overlap": len(sample_stay_set & union_stay_set),
        "subject_overlap": len(sample_subject_set & union_subject_set),
    }
    return per_file, overall


def build_sample(*, db_path: str, label_window: str, seed: int, sample_size: int | None, exclusion_csvs: list[Path], overlap_mode: str) -> tuple[pd.DataFrame, dict[str, int]]:
    pool_df = _query_current_base_pool(db_path, label_window=label_window)
    pool_df = normalize_sample_schema(pool_df)
    pool_df["stay_id"] = pool_df["stay_id"].astype(int)
    pool_df["subject_id"] = pool_df["subject_id"].astype(int)
    pool_df["t_eval"] = pool_df["t_eval"].astype(int)
    pool_df["ground_truth_deterioration"] = pool_df["ground_truth_deterioration"].astype(int)
    pool_df = pool_df.sort_values(["stay_id", "t_eval"], kind="stable").reset_index(drop=True)

    excluded_pairs, excluded_stays, excluded_subjects = _load_exclusion_frames(exclusion_csvs)
    filtered = pool_df.copy()
    if not excluded_pairs.empty:
        filtered = filtered.merge(excluded_pairs.assign(_pair_excluded=1), on=["stay_id", "t_eval"], how="left")
        filtered = filtered[filtered["_pair_excluded"].isna()].drop(columns=["_pair_excluded"])
    if overlap_mode in {"stay", "subject"} and not excluded_stays.empty:
        filtered = filtered.merge(excluded_stays.assign(_stay_excluded=1), on=["stay_id"], how="left")
        filtered = filtered[filtered["_stay_excluded"].isna()].drop(columns=["_stay_excluded"])
    if overlap_mode == "subject" and not excluded_subjects.empty:
        filtered = filtered.merge(excluded_subjects.assign(_subject_excluded=1), on=["subject_id"], how="left")
        filtered = filtered[filtered["_subject_excluded"].isna()].drop(columns=["_subject_excluded"])

    pos = filtered[filtered["ground_truth_deterioration"] == 1].copy()
    neg = filtered[filtered["ground_truth_deterioration"] == 0].copy()
    available = {
        "rows": int(len(filtered)),
        "pos": int(len(pos)),
        "neg": int(len(neg)),
        "unique_stays": int(filtered["stay_id"].nunique()),
        "unique_subjects": int(filtered["subject_id"].nunique()),
    }

    half = min(len(pos), len(neg)) if sample_size is None else sample_size // 2
    if sample_size is not None and (sample_size <= 0 or sample_size % 2 != 0):
        raise ValueError("sample_size must be a positive even integer.")
    if len(pos) < half or len(neg) < half:
        raise RuntimeError(
            f"Insufficient rows after exclusions for balanced sample: need half={half}, got pos={len(pos)} neg={len(neg)}."
        )

    sample_pos = _stable_select(pos, seed + 11, 1, half)
    sample_neg = _stable_select(neg, seed + 29, 0, half)
    sample_df = pd.concat([sample_pos, sample_neg], axis=0)
    sample_df["_ord"] = [
        hashlib.md5(f"{int(r.stay_id)}:{int(r.t_eval)}:{seed + 47}".encode("utf-8")).hexdigest()
        for r in sample_df.itertuples(index=False)
    ]
    sample_df = sample_df.sort_values(["_ord", "stay_id", "t_eval"], kind="stable").drop(columns=["_ord"]).reset_index(drop=True)
    return sample_df, available


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a locked current_base sample and manifest for the retained CARE workflows.")
    parser.add_argument("--label-window", required=True, choices=sorted(LABEL_WINDOW_OPTIONS))
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--sample-size", type=int, default=0, help="Balanced total sample size. Use 0 with --max-balanced.")
    parser.add_argument("--max-balanced", action="store_true")
    parser.add_argument("--overlap-mode", choices=sorted(OVERLAP_MODE_OPTIONS), default="subject")
    parser.add_argument("--exclude-csv", action="append", default=[])
    parser.add_argument("--manifest", default="")
    parser.add_argument("--db", default="", help="DuckDB path. Defaults to MIMIC_DB_PATH / package default.")
    args = parser.parse_args()

    db_path = str(Path(args.db).resolve()) if args.db else str(resolve_duckdb_path())
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (PACKAGE_ROOT / output_path).resolve()
    manifest_path = Path(args.manifest) if args.manifest else output_path.with_name(f"{output_path.stem}_manifest.json")
    if not manifest_path.is_absolute():
        manifest_path = (PACKAGE_ROOT / manifest_path).resolve()

    exclusion_csvs = []
    for raw in args.exclude_csv:
        path = Path(raw)
        if not path.is_absolute():
            path = (PACKAGE_ROOT / path).resolve()
        exclusion_csvs.append(path)

    requested_size = None if args.max_balanced else args.sample_size
    df, available = build_sample(
        db_path=db_path,
        label_window=args.label_window,
        seed=args.seed,
        sample_size=requested_size,
        exclusion_csvs=exclusion_csvs,
        overlap_mode=args.overlap_mode,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    overlap_by_file, overall_overlap = _overlap_stats(df, exclusion_csvs)
    manifest = {
        "dataset_name": output_path.name,
        "sample_csv": _relative_to_package(output_path),
        "manifest_json": _relative_to_package(manifest_path),
        "builder": "build_locked_scope_sample.py",
        "cohort": "current_base",
        "label_window": args.label_window,
        "overlap_mode": args.overlap_mode,
        "seed": args.seed,
        "sample_size": int(len(df)),
        "class_balance": {
            "pos": int((df["ground_truth_deterioration"] == 1).sum()),
            "neg": int((df["ground_truth_deterioration"] == 0).sum()),
        },
        "unique_entities": {
            "pairs": int(df[["stay_id", "t_eval"]].drop_duplicates().shape[0]),
            "stays": int(df["stay_id"].nunique()),
            "subjects": int(df["subject_id"].nunique()),
        },
        "fingerprint": _fingerprint(df),
        "available_after_exclusion": available,
        "excludes": [_relative_to_package(p) for p in exclusion_csvs],
        "overlap_by_file": overlap_by_file,
        "overall_overlap": overall_overlap,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"[OK] output={output_path}")
    print(f"[OK] manifest={manifest_path}")
    print(f"[OK] rows={len(df)} pos={(df['ground_truth_deterioration'] == 1).sum()} neg={(df['ground_truth_deterioration'] == 0).sum()}")
    print(f"[OK] unique_stays={df['stay_id'].nunique()} unique_subjects={df['subject_id'].nunique()}")
    print(f"[OK] available_after_exclusion={available}")
    print(f"[OK] fingerprint={_fingerprint(df)}")
    print(f"[OK] overall_overlap={overall_overlap}")


if __name__ == "__main__":
    main()
