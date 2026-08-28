from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = ROOT / "data" / "mimiciv.duckdb"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "dataset_flow"


def _compute_counts(db_path: Path, manifest_path: Path) -> dict[str, int]:
    con = duckdb.connect(str(db_path), read_only=True)

    icu_stays = con.execute("SELECT COUNT(*) FROM mimiciv_icu.icustays").fetchone()[0]
    hourly_states = con.execute("SELECT COUNT(*) FROM mimiciv_derived.sofa_hourly").fetchone()[0]

    label_q = """
    WITH labels AS (
        WITH base AS (
          SELECT stay_id, hr AS t_eval, sofa_total AS sofa_t
          FROM mimiciv_derived.sofa_hourly
          WHERE sofa_total IS NOT NULL
        ),
        future AS (
          SELECT b.stay_id, b.t_eval,
                 COUNT(f.sofa_total) AS n_future_nonnull,
                 MAX(f.sofa_total) AS sofa_future_max_0_12
          FROM base b
          LEFT JOIN mimiciv_derived.sofa_hourly f
            ON f.stay_id = b.stay_id
           AND f.hr BETWEEN b.t_eval + 1 AND b.t_eval + 12
           AND f.sofa_total IS NOT NULL
          GROUP BY 1,2
        )
        SELECT b.stay_id, b.t_eval,
               CASE
                 WHEN fu.n_future_nonnull < 4 THEN NULL
                 WHEN (fu.sofa_future_max_0_12 - b.sofa_t) >= 2 THEN 1
                 ELSE 0
               END AS y
        FROM base b
        JOIN future fu ON fu.stay_id = b.stay_id AND fu.t_eval = b.t_eval
    ),
    joined AS (
        SELECT lbl.stay_id, lbl.t_eval, lbl.y,
               sl.pain_max_last1h, sl.rass_n_last1h, sl.rass_max_last1h,
               sl.rass_min_last1h, sl.map_low_minutes_last1h_thr65
        FROM labels lbl
        JOIN mimiciv_derived.occult_hypoperfusion_slice sl
          ON lbl.stay_id = sl.stay_id AND lbl.t_eval = sl.hr
        WHERE lbl.y IS NOT NULL
    )
    SELECT
      COUNT(*) AS labeled_hours,
      SUM(CASE WHEN pain_max_last1h IS NOT NULL THEN 1 ELSE 0 END) AS pain_nonnull,
      SUM(CASE WHEN pain_max_last1h = 0 THEN 1 ELSE 0 END) AS pain_eq_0,
      SUM(CASE WHEN pain_max_last1h = 0 AND rass_n_last1h >= 1 THEN 1 ELSE 0 END) AS rass_n_ge_1,
      SUM(CASE WHEN pain_max_last1h = 0 AND rass_n_last1h >= 1 AND rass_max_last1h <= 0 THEN 1 ELSE 0 END) AS rass_max_le_0,
      SUM(CASE WHEN pain_max_last1h = 0 AND rass_n_last1h >= 1 AND rass_max_last1h <= 0 AND rass_min_last1h > -3 THEN 1 ELSE 0 END) AS rass_min_gt_neg3,
      SUM(CASE WHEN pain_max_last1h = 0 AND rass_n_last1h >= 1 AND rass_max_last1h <= 0 AND rass_min_last1h > -3 AND map_low_minutes_last1h_thr65 > 5 THEN 1 ELSE 0 END) AS current_base_pool
    FROM joined
    """
    labeled = con.execute(label_q).fetchone()
    con.close()

    manifest = json.loads(manifest_path.read_text())
    eval_pool = manifest.get("available_after_exclusion", {}).get("rows", 0)
    eval_pool_pos = manifest.get("available_after_exclusion", {}).get("pos", 0)
    eval_pool_neg = manifest.get("available_after_exclusion", {}).get("neg", 0)
    eval_final = manifest.get("sample_size", 0)
    eval_final_pos = manifest.get("class_balance", {}).get("pos", 0)
    eval_final_neg = manifest.get("class_balance", {}).get("neg", 0)

    return {
        "icu_stays": icu_stays,
        "hourly_states": hourly_states,
        "labeled_hours": labeled[0],
        "pain_nonnull": labeled[1],
        "pain_eq_0": labeled[2],
        "rass_n_ge_1": labeled[3],
        "rass_max_le_0": labeled[4],
        "rass_min_gt_neg3": labeled[5],
        "current_base_pool": labeled[6],
        "eval_pool": eval_pool,
        "eval_pool_pos": eval_pool_pos,
        "eval_pool_neg": eval_pool_neg,
        "eval_final": eval_final,
        "eval_final_pos": eval_final_pos,
        "eval_final_neg": eval_final_neg,
    }


def _fmt(n: int) -> str:
    return f"{n:,}"


def _draw_box(ax, x: float, y: float, w: float, h: float, text: str, fs: int = 22) -> None:
    rect = Rectangle((x - w / 2, y - h / 2), w, h, facecolor="white", edgecolor="black", linewidth=1.6)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, family="serif")


def _draw_arrow(ax, x: float, y_top: float, y_bottom: float) -> None:
    arrow = FancyArrowPatch((x, y_top), (x, y_bottom), arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color="black")
    ax.add_patch(arrow)


def _draw_note(ax, x: float, y: float, text: str, fs: int = 18) -> None:
    ax.text(x, y, text, ha="left", va="center", fontsize=fs, family="serif")


def generate(*, db_path: Path, manifest_path: Path, out_dir: Path) -> tuple[Path, Path, Path]:
    counts = _compute_counts(db_path, manifest_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "dataset_construction_flow_eval1000.png"
    svg_path = out_dir / "dataset_construction_flow_eval1000.svg"
    md_path = out_dir / "dataset_construction_flow_eval1000_summary.md"

    boxes = [
        f"MIMIC-IV source\n{_fmt(counts['icu_stays'])} ICU stays",
        f"{_fmt(counts['hourly_states'])}\nhourly SOFA-aligned\npatient-states",
        f"{_fmt(counts['labeled_hours'])}\nlabeled patient-hours",
        f"{_fmt(counts['pain_nonnull'])}\nwith non-null pain",
        f"{_fmt(counts['pain_eq_0'])}\nwith pain = 0",
        f"{_fmt(counts['rass_n_ge_1'])}\nwith at least 1 RASS",
        f"{_fmt(counts['rass_max_le_0'])}\nwith RASS max <= 0",
        f"{_fmt(counts['rass_min_gt_neg3'])}\nwith RASS min > -3",
        f"{_fmt(counts['current_base_pool'])}\ncurrent_base cohort\nbefore exclusions",
        f"{_fmt(counts['eval_pool'])}\neval-eligible pairs\nafter exclusions",
        f"{_fmt(counts['eval_final'])}\nlocked eval pairs\n({_fmt(counts['eval_final_pos'])} positive / {_fmt(counts['eval_final_neg'])} negative)",
    ]
    notes = [
        "",
        "Align ICU stays to hourly SOFA states",
        "Keep hours with current SOFA and at least 4 valid\nfuture SOFA observations in the next 12 hours;\nlabel positive if future max SOFA - current SOFA >= 2",
        "Require bedside pain assessment in the preceding hour",
        "Retain only bedside states with no recorded pain",
        "Require at least one RASS assessment in the preceding hour",
        "Exclude agitated states",
        "Exclude deeply sedated states",
        "Require more than 5 minutes of MAP <65 mmHg\nin the preceding hour",
        f"Exclude prior locked overlaps;\nremaining pool = {_fmt(counts['eval_pool_pos'])} positive + {_fmt(counts['eval_pool_neg'])} negative",
        "Deterministic pair-level balanced locking\nfor the canonical 1000-sample evaluation split",
    ]

    fig, ax = plt.subplots(figsize=(14, 24))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 24)
    ax.axis("off")

    x_box = 4.2
    x_note = 7.2
    box_w = 4.7
    box_h = 1.35
    y_positions = list(reversed([1.6 + i * 2.0 for i in range(len(boxes))]))

    for i, (text, y) in enumerate(zip(boxes, y_positions)):
        _draw_box(ax, x_box, y, box_w, box_h, text, fs=24 if i < 3 else 22)
        if i < len(boxes) - 1:
            next_y = y_positions[i + 1]
            _draw_arrow(ax, x_box, y - box_h / 2, next_y + box_h / 2)
            if notes[i + 1]:
                _draw_note(ax, x_note, (y + next_y) / 2, notes[i + 1], fs=18)

    fig.tight_layout()
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    plt.close(fig)

    summary = f"""# Dataset Construction Flow (Eval 1000)\n\n- ICU stays in local MIMIC-IV build: {_fmt(counts['icu_stays'])}\n- Hourly SOFA-aligned patient-states: {_fmt(counts['hourly_states'])}\n- Labeled patient-hours for `y0_12`: {_fmt(counts['labeled_hours'])}\n- Non-null pain: {_fmt(counts['pain_nonnull'])}\n- Pain = 0: {_fmt(counts['pain_eq_0'])}\n- At least one RASS: {_fmt(counts['rass_n_ge_1'])}\n- RASS max <= 0: {_fmt(counts['rass_max_le_0'])}\n- RASS min > -3: {_fmt(counts['rass_min_gt_neg3'])}\n- `current_base` cohort before exclusions: {_fmt(counts['current_base_pool'])}\n- Eval-eligible pool after exclusions: {_fmt(counts['eval_pool'])} ({_fmt(counts['eval_pool_pos'])} positive / {_fmt(counts['eval_pool_neg'])} negative)\n- Locked eval split: {_fmt(counts['eval_final'])} ({_fmt(counts['eval_final_pos'])} positive / {_fmt(counts['eval_final_neg'])} negative)\n"""
    md_path.write_text(summary, encoding="utf-8")
    return png_path, svg_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the dataset construction flow figure from a DuckDB build and a sample manifest.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="DuckDB path.")
    parser.add_argument("--manifest", required=True, help="Manifest JSON path for the locked sample.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for PNG/SVG/summary.")
    args = parser.parse_args()

    db_path = Path(args.db).resolve()
    manifest_path = Path(args.manifest).resolve()
    out_dir = Path(args.out_dir).resolve()
    png, svg, md = generate(db_path=db_path, manifest_path=manifest_path, out_dir=out_dir)
    print(png)
    print(svg)
    print(md)


if __name__ == "__main__":
    main()
