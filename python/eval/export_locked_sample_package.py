from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = THIS_DIR.parent
SHARED_DIR = PYTHON_ROOT / "shared"
for _p in (PYTHON_ROOT, SHARED_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

import pandas as pd

from package_runtime import PACKAGE_ROOT, bootstrap_import_paths, load_package_env, resolve_duckdb_path
from sample_schema_common import (
    all_enrichable_feature_columns,
    enrich_objective_features,
    normalize_sample_schema,
    sample_fingerprint,
)


def _default_feature_output_path(sample_path: Path) -> Path:
    return sample_path.with_name(f"{sample_path.stem}__feature_package.csv")


def _default_manifest_path(sample_path: Path) -> Path:
    return sample_path.with_name(f"{sample_path.stem}__manifest.json")


def _ordered_columns(existing: list[str], enrichable: list[str]) -> list[str]:
    ordered = list(existing)
    for col in enrichable:
        if col not in ordered:
            ordered.append(col)
    return ordered


def _resolve_under_package(path_str: str | None, default_path: Path) -> Path:
    if not path_str:
        return default_path
    path = Path(path_str)
    if not path.is_absolute():
        path = (PACKAGE_ROOT / path).resolve()
    return path


def _class_balance(df: pd.DataFrame) -> dict[str, int]:
    labels = df["ground_truth_deterioration"].astype(int)
    return {
        "pos": int((labels == 1).sum()),
        "neg": int((labels == 0).sum()),
    }


def _relative_to_package(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PACKAGE_ROOT.resolve()))
    except Exception:
        return path.name


def build_manifest(*, sample_path: Path, feature_path: Path, db_path: Path, sample_df: pd.DataFrame, feature_df: pd.DataFrame) -> dict:
    sample_balance = _class_balance(sample_df)
    feature_balance = _class_balance(feature_df)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sample_csv": _relative_to_package(sample_path),
        "feature_package_csv": _relative_to_package(feature_path),
        "duckdb_path": _relative_to_package(db_path),
        "sample_size": int(len(sample_df)),
        "sample_fingerprint": sample_fingerprint(sample_df),
        "class_balance": sample_balance,
        "feature_package": {
            "rows": int(len(feature_df)),
            "columns": int(len(feature_df.columns)),
            "fingerprint": sample_fingerprint(feature_df),
            "class_balance": feature_balance,
        },
    }


def main() -> None:
    bootstrap_import_paths()
    load_package_env()

    parser = argparse.ArgumentParser(
        description="Export a locked sample package: feature-enriched CSV plus a compact manifest JSON."
    )
    parser.add_argument("--sample", required=True, help="Input locked sample CSV path.")
    parser.add_argument("--feature-output", help="Output path for enriched feature CSV.")
    parser.add_argument("--manifest-output", help="Output path for manifest JSON.")
    parser.add_argument("--db", help="DuckDB path. Defaults to MIMIC_DB_PATH / package default.")
    args = parser.parse_args()

    sample_path = Path(args.sample)
    if not sample_path.is_absolute():
        sample_path = (PACKAGE_ROOT / sample_path).resolve()
    feature_output = _resolve_under_package(args.feature_output, _default_feature_output_path(sample_path))
    manifest_output = _resolve_under_package(args.manifest_output, _default_manifest_path(sample_path))
    db_path = Path(args.db).resolve() if args.db else resolve_duckdb_path()

    sample_df = normalize_sample_schema(pd.read_csv(sample_path))
    original_columns = list(sample_df.columns)
    enrichable = all_enrichable_feature_columns()
    feature_df = enrich_objective_features(str(db_path), sample_df, requested_features=enrichable)
    feature_df = feature_df[_ordered_columns(original_columns, enrichable)].copy()

    feature_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(feature_output, index=False)

    manifest = build_manifest(
        sample_path=sample_path,
        feature_path=feature_output,
        db_path=db_path,
        sample_df=sample_df,
        feature_df=feature_df,
    )
    manifest_output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"WROTE feature_package={feature_output}")
    print(f"WROTE manifest={manifest_output}")
    print(
        f"rows={len(feature_df)} pos={manifest['class_balance']['pos']} "
        f"neg={manifest['class_balance']['neg']} fp={manifest['sample_fingerprint']}"
    )
    print(f"feature_cols={len(feature_df.columns)}")


if __name__ == "__main__":
    main()
