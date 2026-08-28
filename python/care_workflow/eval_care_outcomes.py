import glob
import json
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = THIS_DIR.parent
SHARED_DIR = PYTHON_ROOT / "shared"
for _p in (PYTHON_ROOT, SHARED_DIR):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

import pandas as pd
from sklearn.metrics import confusion_matrix
from package_runtime import LOGS_DIR

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger(__name__)

VERSION_ID = "care_main"


def _parse_final_action(planner: dict | None, decision: dict | None):
    for obj in (decision, planner):
        if not isinstance(obj, dict):
            continue
        v = str(obj.get("final_action", "")).upper().strip()
        if v == "INVESTIGATE_O":
            return 1
        if v in {"OBSERVE", "TREAT_S"}:
            return 0
    return None


def _is_error(planner: dict | None, decision: dict | None):
    for obj in (planner, decision):
        if not isinstance(obj, dict):
            continue
        if obj.get("error"):
            return True
        if str(obj.get("final_action", "")).upper() == "ERROR":
            return True
    return False


def _log_success(msg: str) -> None:
    getattr(logger, "success", logger.info)(msg)


def evaluate_care_outputs() -> None:
    logs_dir = LOGS_DIR

    pattern = os.getenv("CARE_LOG_PATTERN", "care_main_*.jsonl")
    files = glob.glob(str(logs_dir / pattern))
    if not files:
        logger.error(f"No matching logs found for pattern '{pattern}'.")
        return

    latest = max(files, key=os.path.getctime)
    metrics_path = latest.replace(".jsonl", "_metrics.md")
    logger.info(f"Evaluating CARE log: {latest}")

    recs = []
    with open(latest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))

    df = pd.DataFrame(recs)
    if df.empty:
        logger.warning("Log file is empty.")
        return

    df["ground_truth"] = df["ground_truth"].astype(int)
    # care focuses on the agent branch. Reference checks are handled separately.
    df["pred_agent"] = df.apply(lambda r: _parse_final_action(r.get("agent_planner"), r.get("agent_decision")), axis=1)
    df["agent_error"] = df.apply(lambda r: _is_error(r.get("agent_planner"), r.get("agent_decision")), axis=1)
    
    # Telemetry for transition advisory outputs
    df["initial_severity"] = df["agent_meta"].apply(lambda x: x.get("care_initial_severity", 0) if isinstance(x, dict) else 0)
    df["final_severity"] = df["agent_meta"].apply(lambda x: x.get("care_final_severity", 0) if isinstance(x, dict) else 0)
    df["category_jumped"] = df["final_severity"] > df["initial_severity"]

    df["agent_mode"] = df["agent_meta"].apply(lambda x: x.get("decision_mode", "unknown") if isinstance(x, dict) else "unknown")
    df["agent_repair"] = df["agent_meta"].apply(lambda x: bool(x.get("repair_attempted", False)) if isinstance(x, dict) else False)
    df["agent_unmapped_n"] = df["agent_meta"].apply(lambda x: len(x.get("unmapped_keys", [])) if isinstance(x, dict) else 0)
    df["sufficiency_insufficient"] = df["agent_meta"].apply(
        lambda x: bool(x.get("sufficiency_is_sufficient") is False) if isinstance(x, dict) else False
    )
    df["advisor_num_categories"] = df["agent_meta"].apply(
        lambda x: int(x.get("advisor_num_categories", 0) or 0) if isinstance(x, dict) else 0
    )
    df["advisor_raw_categories"] = df["agent_meta"].apply(
        lambda x: int(x.get("advisor_raw_categories", 0) or 0) if isinstance(x, dict) else 0
    )
    df["advisor_filtered_out"] = df["agent_meta"].apply(
        lambda x: int(x.get("advisor_filtered_out", 0) or 0) if isinstance(x, dict) else 0
    )
    df["advisor_guidance_present"] = df["agent_meta"].apply(
        lambda x: bool(x.get("advisor_guidance_present", False)) if isinstance(x, dict) else False
    )
    df["advisor_empty"] = (~df["advisor_guidance_present"]) & (df["advisor_num_categories"] <= 0)
    df["transition_remote_candidate_used"] = df["agent_meta"].apply(
        lambda x: bool(x.get("transition_remote_candidate_used", False)) if isinstance(x, dict) else False
    )
    df["transition_remote_uplifted"] = df["agent_meta"].apply(
        lambda x: bool(x.get("transition_remote_uplifted", False)) if isinstance(x, dict) else False
    )
    df["planner_tokens"] = df["agent_meta"].apply(lambda x: int(x.get("planner_tokens", 0) or 0) if isinstance(x, dict) else 0)
    df["planner_input_tokens"] = df["agent_meta"].apply(lambda x: int(x.get("planner_input_tokens", 0) or 0) if isinstance(x, dict) else 0)
    df["planner_output_tokens"] = df["agent_meta"].apply(lambda x: int(x.get("planner_output_tokens", 0) or 0) if isinstance(x, dict) else 0)
    df["planner_token_split_available"] = df["agent_meta"].apply(lambda x: bool(x.get("planner_token_split_available", False)) if isinstance(x, dict) else False)
    df["advisor_tokens"] = df["agent_meta"].apply(lambda x: int(x.get("advisor_tokens", 0) or 0) if isinstance(x, dict) else 0)
    df["advisor_input_tokens"] = df["agent_meta"].apply(lambda x: int(x.get("advisor_input_tokens", 0) or 0) if isinstance(x, dict) else 0)
    df["advisor_output_tokens"] = df["agent_meta"].apply(lambda x: int(x.get("advisor_output_tokens", 0) or 0) if isinstance(x, dict) else 0)
    df["advisor_token_split_available"] = df["agent_meta"].apply(lambda x: bool(x.get("advisor_token_split_available", False)) if isinstance(x, dict) else False)
    df["decision_tokens"] = df["agent_meta"].apply(lambda x: int(x.get("decision_tokens", 0) or 0) if isinstance(x, dict) else 0)
    df["decision_input_tokens"] = df["agent_meta"].apply(lambda x: int(x.get("decision_input_tokens", 0) or 0) if isinstance(x, dict) else 0)
    df["decision_output_tokens"] = df["agent_meta"].apply(lambda x: int(x.get("decision_output_tokens", 0) or 0) if isinstance(x, dict) else 0)
    df["decision_token_split_available"] = df["agent_meta"].apply(lambda x: bool(x.get("decision_token_split_available", False)) if isinstance(x, dict) else False)
    df["branch_input_tokens"] = df["agent_meta"].apply(lambda x: int(x.get("branch_input_tokens", 0) or 0) if isinstance(x, dict) else 0)
    df["branch_output_tokens"] = df["agent_meta"].apply(lambda x: int(x.get("branch_output_tokens", 0) or 0) if isinstance(x, dict) else 0)
    df["branch_total_tokens"] = df["agent_meta"].apply(lambda x: int(x.get("branch_total_tokens", 0) or 0) if isinstance(x, dict) else 0)
    df["planner_invalid_key_count"] = df["agent_meta"].apply(
        lambda x: int(x.get("planner_invalid_key_count", 0) or 0) if isinstance(x, dict) else 0
    )
    df["planner_has_invalid_keys"] = df["planner_invalid_key_count"] > 0
    df["planner_need_data_true_but_no_effective_keys"] = df["agent_meta"].apply(
        lambda x: bool(x.get("planner_need_data_true_but_no_effective_keys", False)) if isinstance(x, dict) else False
    )
    df["sufficiency_recovery_attempted"] = df["agent_meta"].apply(
        lambda x: bool(x.get("sufficiency_recovery_attempted", False)) if isinstance(x, dict) else False
    )
    df["sufficiency_recovery_success"] = df["agent_meta"].apply(
        lambda x: bool(x.get("sufficiency_recovery_success", False)) if isinstance(x, dict) else False
    )
    df["sufficiency_recovery_failed"] = df["agent_meta"].apply(
        lambda x: bool(x.get("sufficiency_recovery_failed", False)) if isinstance(x, dict) else False
    )
    df["sufficiency_recovery_added_key_count"] = df["agent_meta"].apply(
        lambda x: int(x.get("sufficiency_recovery_added_key_count", 0) or 0) if isinstance(x, dict) else 0
    )

    total = len(df)
    out = []
    out.append(f"# CARE Workflow Metrics ({VERSION_ID})")
    out.append(f"**Log File:** `{Path(latest).name}` | **Total Cases:** {total}\n")

    out.append("## 0. Run Quality")
    out.append(f"- Agent error cases: {int(df['agent_error'].sum())}/{total} ({df['agent_error'].mean()*100:.1f}%)\n")

    out.append("## 1. Coverage / Repair / Degraded")
    out.append(f"- Agent repair_rate: {df['agent_repair'].mean()*100:.1f}%")
    out.append(f"- Agent degraded_rate: {(df['agent_mode'].str.startswith('degraded')).mean()*100:.1f}%")
    out.append(f"- Agent unmapped_rate (rows with unmapped_keys): {(df['agent_unmapped_n']>0).mean()*100:.1f}%\n")

    out.append("## Retrieval / Advisory Quality")
    out.append(
        f"- sufficiency recheck insufficient_rate: {df['sufficiency_insufficient'].mean()*100:.1f}% "
        f"({int(df['sufficiency_insufficient'].sum())}/{total})"
    )
    out.append(
        f"- transition advisory empty_guidance_rate: {df['advisor_empty'].mean()*100:.1f}% "
        f"({int(df['advisor_empty'].sum())}/{total})"
    )
    out.append(
        f"- transition advisory filtered_category_rate (filtered_out>0): {(df['advisor_filtered_out']>0).mean()*100:.1f}% "
        f"({int((df['advisor_filtered_out']>0).sum())}/{total})"
    )
    out.append(f"- transition advisory guidance_present_rate: {df['advisor_guidance_present'].mean()*100:.1f}%")
    out.append(f"- transition advisory avg raw_categories_per_row: {df['advisor_raw_categories'].mean():.2f}")
    out.append(f"- transition advisory avg valid_categories_per_row: {df['advisor_num_categories'].mean():.2f}")
    out.append(f"- local transition update remote_candidate_used_rate: {df['transition_remote_candidate_used'].mean()*100:.1f}%")
    out.append(f"- local transition update remote_uplift_rate: {df['transition_remote_uplifted'].mean()*100:.1f}%\n")

    out.append("## Token Usage")
    out.append(f"- Experiment total tokens: {int(df['branch_total_tokens'].sum())}")
    out.append(f"- Experiment input tokens: {int(df['branch_input_tokens'].sum())}")
    out.append(f"- Experiment output tokens: {int(df['branch_output_tokens'].sum())}")
    out.append(f"- evidence planning total/input/output: {int(df['planner_tokens'].sum())} / {int(df['planner_input_tokens'].sum())} / {int(df['planner_output_tokens'].sum())}")
    out.append(f"- transition advisory total/input/output: {int(df['advisor_tokens'].sum())} / {int(df['advisor_input_tokens'].sum())} / {int(df['advisor_output_tokens'].sum())}")
    out.append(f"- final decision total/input/output: {int(df['decision_tokens'].sum())} / {int(df['decision_input_tokens'].sum())} / {int(df['decision_output_tokens'].sum())}")
    out.append(f"- evidence planning token_split_available_rate: {df['planner_token_split_available'].mean()*100:.1f}%")
    out.append(f"- transition advisory token_split_available_rate: {df['advisor_token_split_available'].mean()*100:.1f}%")
    out.append(f"- final decision token_split_available_rate: {df['decision_token_split_available'].mean()*100:.1f}%\n")

    out.append("## facts_keys Interface Health")
    out.append(
        f"- Invalid key row rate: {df['planner_has_invalid_keys'].mean()*100:.1f}% "
        f"({int(df['planner_has_invalid_keys'].sum())}/{total})"
    )
    out.append(f"- Invalid key total count: {int(df['planner_invalid_key_count'].sum())}")
    out.append(
        f"- need_data=true but no effective keys rate: {df['planner_need_data_true_but_no_effective_keys'].mean()*100:.1f}% "
        f"({int(df['planner_need_data_true_but_no_effective_keys'].sum())}/{total})\n"
    )

    out.append("## sufficiency recheck Recovery")
    out.append(
        f"- Recovery attempted_rate: {df['sufficiency_recovery_attempted'].mean()*100:.1f}% "
        f"({int(df['sufficiency_recovery_attempted'].sum())}/{total})"
    )
    out.append(
        f"- Recovery success_rate: {df['sufficiency_recovery_success'].mean()*100:.1f}% "
        f"({int(df['sufficiency_recovery_success'].sum())}/{total})"
    )
    out.append(
        f"- Recovery failed_rate: {df['sufficiency_recovery_failed'].mean()*100:.1f}% "
        f"({int(df['sufficiency_recovery_failed'].sum())}/{total})"
    )
    out.append(f"- Avg recovery added keys per row: {df['sufficiency_recovery_added_key_count'].mean():.2f}\n")

    out.append("## 2. Remote Rubric (CARE Neuro-System) Statistics")
    out.append(f"- Mean Initial Severity: {df['initial_severity'].mean():.2f}/5")
    out.append(f"- Mean Final Severity: {df['final_severity'].mean():.2f}/5")
    out.append(f"- Cases where Objective rules worsened category: {df['category_jumped'].sum()} ({df['category_jumped'].mean()*100:.1f}%)")

    out.append("\n## 3. Final Triage Performance (Agent Branch)")

    agent_eval = df[df["pred_agent"].notna()].copy()
    if agent_eval.empty:
        out.append("- Branch skipped or no valid agent predictions in this log.")
    else:
        try:
            y_true = agent_eval["ground_truth"]
            y_pred = agent_eval["pred_agent"].astype(int)
            tn_agent, fp_agent, fn_agent, tp_agent = confusion_matrix(y_true, y_pred).ravel()
            out.append(f"- Evaluated rows: {len(agent_eval)}")
            out.append(f"- Confusion Matrix: TN={tn_agent}, FP={fp_agent}, FN={fn_agent}, TP={tp_agent}")
            out.append(f"- TPR: {tp_agent/(tp_agent+fn_agent) if (tp_agent+fn_agent)>0 else 0:.4f}")
            out.append(f"- FNR: {fn_agent/(tp_agent+fn_agent) if (tp_agent+fn_agent)>0 else 0:.4f}")

            fn_rows = agent_eval[(agent_eval["ground_truth"] == 1) & (agent_eval["pred_agent"] == 0)]
            fp_rows = agent_eval[(agent_eval["ground_truth"] == 0) & (agent_eval["pred_agent"] == 1)]
            fn_n = len(fn_rows)
            fp_n = len(fp_rows)
            fn_sufficiency = int(fn_rows["sufficiency_insufficient"].sum())
            fp_sufficiency = int(fp_rows["sufficiency_insufficient"].sum())
            fn_advisor_empty = int(fn_rows["advisor_empty"].sum())
            fp_advisor_empty = int(fp_rows["advisor_empty"].sum())

            out.append("\n### Error Attribution by Component")
            out.append(
                f"- FN from sufficiency recheck insufficient: {fn_sufficiency}/{fn_n} "
                f"({(fn_sufficiency/fn_n*100) if fn_n else 0:.1f}%)"
            )
            out.append(
                f"- FP from sufficiency recheck insufficient: {fp_sufficiency}/{fp_n} "
                f"({(fp_sufficiency/fp_n*100) if fp_n else 0:.1f}%)"
            )
            out.append(
                f"- FN with transition advisory empty guidance: {fn_advisor_empty}/{fn_n} "
                f"({(fn_advisor_empty/fn_n*100) if fn_n else 0:.1f}%)"
            )
            out.append(
                f"- FP with transition advisory empty guidance: {fp_advisor_empty}/{fp_n} "
                f"({(fp_advisor_empty/fp_n*100) if fp_n else 0:.1f}%)"
            )
        except Exception as e:
            out.append(f"- Warning: agent confusion matrix unavailable ({e})")

    out_text = "\n".join(out)
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(out_text)

    print("\n" + "=" * 60)
    print(out_text)
    print("=" * 60 + "\n")
    _log_success(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    evaluate_care_outputs()
