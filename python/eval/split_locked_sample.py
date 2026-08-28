from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _sample_fingerprint(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    tmp = df[["stay_id", "t_eval"]].copy()
    tmp["stay_id"] = tmp["stay_id"].astype(int)
    tmp["t_eval"] = tmp["t_eval"].astype(int)
    tmp = tmp.sort_values(["stay_id", "t_eval"], kind="stable")
    raw = ";".join(f"{r.stay_id}:{r.t_eval}" for r in tmp.itertuples(index=False))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _stable_shuffle(df: pd.DataFrame, seed: int, label: int) -> pd.DataFrame:
    tmp = df.copy()
    tmp["_ord"] = [
        hashlib.md5(f"{int(r.stay_id)}:{int(r.t_eval)}:{label}:{seed}".encode("utf-8")).hexdigest()
        for r in tmp.itertuples(index=False)
    ]
    tmp = tmp.sort_values(["_ord", "stay_id", "t_eval"], kind="stable")
    return tmp.drop(columns=["_ord"]).reset_index(drop=True)


def _split_class(df: pd.DataFrame, num_splits: int) -> list[pd.DataFrame]:
    if len(df) % num_splits != 0:
        raise ValueError(
            f"Class size {len(df)} is not divisible by num_splits={num_splits}; cannot create exactly balanced chunks."
        )
    chunk_size = len(df) // num_splits
    return [
        df.iloc[i * chunk_size : (i + 1) * chunk_size].copy().reset_index(drop=True)
        for i in range(num_splits)
    ]


def _detect_overlaps(split_frames: list[pd.DataFrame]) -> dict[str, int]:
    pair_sets = []
    stay_sets = []
    subject_sets = []
    for df in split_frames:
        pair_sets.append(set(zip(df["stay_id"].astype(int), df["t_eval"].astype(int))))
        stay_sets.append(set(df["stay_id"].astype(int)))
        subject_sets.append(set(df["subject_id"].astype(int))) if "subject_id" in df.columns else subject_sets.append(set())

    pair_overlap = 0
    stay_overlap = 0
    subject_overlap = 0
    for i in range(len(split_frames)):
        for j in range(i + 1, len(split_frames)):
            pair_overlap += len(pair_sets[i].intersection(pair_sets[j]))
            stay_overlap += len(stay_sets[i].intersection(stay_sets[j]))
            if subject_sets[i] and subject_sets[j]:
                subject_overlap += len(subject_sets[i].intersection(subject_sets[j]))
    return {
        "pair_overlap": int(pair_overlap),
        "stay_overlap": int(stay_overlap),
        "subject_overlap": int(subject_overlap),
    }


def build_splits(input_csv: Path, output_dir: Path, num_splits: int, seed: int) -> dict:
    df = pd.read_csv(input_csv)
    required = {"stay_id", "t_eval", "ground_truth_deterioration"}
    missing = required.difference(set(df.columns))
    if missing:
        raise RuntimeError(f"Input sample missing required columns: {sorted(missing)}")

    df["ground_truth_deterioration"] = df["ground_truth_deterioration"].astype(int)
    pos = _stable_shuffle(df[df["ground_truth_deterioration"] == 1].copy(), seed, 1)
    neg = _stable_shuffle(df[df["ground_truth_deterioration"] == 0].copy(), seed, 0)

    pos_splits = _split_class(pos, num_splits)
    neg_splits = _split_class(neg, num_splits)

    output_dir.mkdir(parents=True, exist_ok=True)
    split_frames = []
    split_meta = []
    stem = input_csv.stem

    for idx in range(num_splits):
        split_df = pd.concat([pos_splits[idx], neg_splits[idx]], axis=0)
        split_df = split_df.sort_values(["stay_id", "t_eval"], kind="stable").reset_index(drop=True)
        split_frames.append(split_df)

        split_name = f"{stem}_split{idx + 1:02d}_n{len(split_df)}.csv"
        split_path = output_dir / split_name
        split_df.to_csv(split_path, index=False)

        split_meta.append(
            {
                "split_index": idx + 1,
                "file": split_name,
                "rows": int(len(split_df)),
                "pos": int((split_df["ground_truth_deterioration"] == 1).sum()),
                "neg": int((split_df["ground_truth_deterioration"] == 0).sum()),
                "unique_stays": int(split_df["stay_id"].nunique()),
                "unique_subjects": int(split_df["subject_id"].nunique()) if "subject_id" in split_df.columns else None,
                "fingerprint": _sample_fingerprint(split_df),
            }
        )

    overlap_stats = _detect_overlaps(split_frames)
    manifest = {
        "source_file": input_csv.name,
        "source_rows": int(len(df)),
        "source_pos": int((df["ground_truth_deterioration"] == 1).sum()),
        "source_neg": int((df["ground_truth_deterioration"] == 0).sum()),
        "source_fingerprint": _sample_fingerprint(df),
        "num_splits": int(num_splits),
        "seed": int(seed),
        "rows_per_split": int(len(df) // num_splits),
        "exact_class_balance_per_split": True,
        "split_overlap_summary": overlap_stats,
        "splits": split_meta,
    }

    manifest_path = output_dir / f"{stem}_splits_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a locked sample into deterministic balanced chunks.")
    parser.add_argument("input_csv", type=Path, help="Source locked sample CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for split CSVs and manifest.")
    parser.add_argument("--num-splits", type=int, default=5, help="Number of output splits.")
    parser.add_argument("--seed", type=int, default=73005, help="Deterministic split seed.")
    args = parser.parse_args()

    input_csv = args.input_csv.resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = pd.read_csv(input_csv)
    default_output_dir = input_csv.parent / f"{input_csv.stem}_splits_{args.num_splits}x{len(df) // args.num_splits}"
    output_dir = (args.output_dir or default_output_dir).resolve()

    manifest = build_splits(input_csv, output_dir, args.num_splits, args.seed)
    print(json.dumps({"output_dir": str(output_dir), **manifest}, indent=2))


if __name__ == "__main__":
    main()
