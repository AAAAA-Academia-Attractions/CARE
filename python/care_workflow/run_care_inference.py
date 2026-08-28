import asyncio
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

THIS_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = THIS_DIR.parent
for _p in (
    PYTHON_ROOT,
    PYTHON_ROOT / "shared",
    PYTHON_ROOT / "eval",
    PYTHON_ROOT / "agent_support",
    PYTHON_ROOT / "care_common",
):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from openai import AsyncOpenAI
from package_runtime import (
    LOGS_DIR,
    PACKAGE_ROOT,
    SAMPLES_DIR,
    bootstrap_import_paths,
    ensure_runtime_dirs,
    load_package_env,
    resolve_duckdb_path,
    resolve_feature_source,
    resolve_prompt_dir,
    tee_console_to_file,
)
from care_instrumentation import llm_usage_from_raw, planner_facts_keys_health, sufficiency_recovery_stats

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging
    import types
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    _base_logger = logging.getLogger(__name__)
    class _LoguruShim:
        def __getattr__(self, name):
            if name == "success": return _base_logger.info
            return getattr(_base_logger, name, _base_logger.info)
    logger = _LoguruShim()
    sys.modules.setdefault("loguru", types.SimpleNamespace(logger=logger))

PROJECT_ROOT = PACKAGE_ROOT


# Import reusable infrastructure from shared modules
from facts_generator import FactsGenerator
from sample_schema_common import normalize_sample_schema
from decision_logging import DecisionMeta
from facts_adapter import FactsAdapter
from facts_registry import allowed_fact_key_lines, dedupe_keep_order, FACT_SPECS

# Import local shared workflow helpers
from local_workflow_common import _build_base_df, _deterministic_sample, _extract_final_action, _extract_json, evaluate_vignette, _sample_fingerprint, _load_project_env

# Import CARE-specific modules
from response_schema import normalize_planner_response, normalize_advisor_response
from risk_rubric_engine import RiskRubricEngine


def _get_feature_descriptions(keys: list[str]) -> list[str]:
    """Return human-readable feature descriptions (meaning + unit) for the given keys.
    Contains NO patient values — safe for REMOTE prompts."""
    descs = []
    for k in keys:
        spec = FACT_SPECS.get(k)
        if spec:
            unit_text = f" (Unit: {spec.unit})" if spec.unit else ""
            descs.append(f"`{k}`: {spec.display} — {spec.meaning}{unit_text}")
    return descs


def _get_rubric_category_summaries(rubric_engine: RiskRubricEngine) -> list[dict]:
    """Return category name/severity/description from the rubric. No patient values."""
    cats = rubric_engine.rubric.get("categories", [])
    return [
        {"name": c["name"], "severity": c.get("severity_level", 0), "description": c.get("description", "")}
        for c in sorted(cats, key=lambda x: x.get("severity_level", 0), reverse=True)
    ]

bootstrap_import_paths()
load_package_env()
_load_project_env()

_RUNTIME_USAGE = {
    "calls": 0,
    "reported_tokens": 0,
}


def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _provider_from_base_url(base_url: str) -> str:
    raw = str(base_url or "").strip().lower()
    if "127.0.0.1" in raw or "localhost" in raw:
        return "local"
    return "standard"


def _require_loopback_url(base_url: str) -> str:
    value = str(base_url or "").strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError(
            "LOCAL_BASE_URL must point to a locally hosted endpoint using localhost, 127.0.0.1, or ::1."
        )
    return value


def _sufficiency_fallback_keys() -> list[str]:
    preferred = [
        "map_median_last1h",
        "map_low_minutes_last1h_thr65",
        "map_low_minutes_last1h_thr60",
        "lactate_latest_6h",
        "sofa_total",
        "sofa_cardiovascular",
        "urine_output_mlkghr_6h",
        "norepi_eq_dose_max_1h",
    ]
    return [k for k in preferred if k in FACT_SPECS]


def _initial_patient_state(row: dict) -> dict:
    keys = [
        "hr_median_last1h",
        "pain_max_last1h",
        "pain_last_last1h",
        "rass_max_last1h",
        "rass_min_last1h",
        "rass_n_last1h",
    ]
    return {k: row.get(k) for k in keys}


def _initial_state_has_structured_risk(row: dict) -> bool:
    def _f(x):
        try:
            return float(x)
        except Exception:
            return None

    map65 = _f(row.get("map_low_minutes_last1h_thr65"))
    map60 = _f(row.get("map_low_minutes_last1h_thr60"))
    map_med = _f(row.get("map_median_last1h"))
    sofa_total = _f(row.get("sofa_total"))
    sofa_cv = _f(row.get("sofa_cardiovascular"))
    return bool(
        (map65 is not None and map65 >= 10)
        or (map60 is not None and map60 >= 5)
        or (map_med is not None and map_med < 65)
        or (sofa_total is not None and sofa_total >= 8)
        or (sofa_cv is not None and sofa_cv >= 3)
    )


def _category_requirement_groups(category: str, row: dict | None = None) -> list[list[str]]:
    hemo = ["map_low_minutes_last1h_thr65", "map_low_minutes_last1h_thr60", "map_median_last1h"]
    perf = ["lactate_latest_6h", "urine_output_mlkghr_6h"]
    support = ["norepi_eq_dose_max_1h", "sofa_cardiovascular"]
    cat = str(category or "").strip().upper()
    if cat in {"VERY_LIKELY_WORSENING", "LIKELY_WORSENING"}:
        groups = [hemo, perf, support]
    elif cat == "POTENTIAL_OCCULT_SHOCK":
        groups = [hemo, perf, support]
    else:
        groups = [hemo]
        if row and _initial_state_has_structured_risk(row):
            groups.extend([perf, support])
    return [[k for k in group if k in FACT_SPECS] for group in groups if any(k in FACT_SPECS for k in group)]


def _missing_requirement_keys(requested_keys: list[str], category: str, row: dict | None = None) -> tuple[list[str], list[list[str]]]:
    groups = _category_requirement_groups(category, row=row)
    requested = set(requested_keys)
    missing: list[str] = []
    for group in groups:
        if not any(k in requested for k in group):
            missing.append(group[0])
    return missing, groups


def _candidate_categories_fallback(initial_category: str) -> list[str]:
    ordered = [
        "VERY_LIKELY_WORSENING",
        "LIKELY_WORSENING",
        "POTENTIAL_OCCULT_SHOCK",
        "LIKELY_STABLE",
        "VERY_LIKELY_STABLE",
    ]
    if initial_category in ordered:
        start = ordered.index(initial_category)
        return ordered[: start + 1]
    return ["POTENTIAL_OCCULT_SHOCK", "LIKELY_STABLE"]


def _build_remote_advisor_context(
    *,
    initial_rubric_eval: dict,
    eff_keys: list[str],
    rubric_engine: RiskRubricEngine,
) -> dict:
    ctx = {
        "initial_category": initial_rubric_eval["category"],
        "initial_severity": initial_rubric_eval["severity"],
        "initial_category_reason": initial_rubric_eval["reason"],
        "feature_descriptions": _get_feature_descriptions(eff_keys),
        "rubric_categories": _get_rubric_category_summaries(rubric_engine),
    }
    return ctx


def _objective_domain_flags(values: dict) -> dict[str, bool]:
    def _f(x):
        try:
            return float(x)
        except Exception:
            return None

    map65 = _f(values.get("map_low_minutes_last1h_thr65"))
    map60 = _f(values.get("map_low_minutes_last1h_thr60"))
    map_med = _f(values.get("map_median_last1h"))
    lact = _f(values.get("lactate_latest_6h"))
    uo = _f(values.get("urine_output_mlkghr_6h"))
    norepi = _f(values.get("norepi_eq_dose_max_1h"))
    sofa_total = _f(values.get("sofa_total"))
    sofa_cv = _f(values.get("sofa_cardiovascular"))
    return {
        "hemodynamic": (map65 is not None and map65 >= 10) or (map60 is not None and map60 >= 5) or (map_med is not None and map_med < 65),
        "perfusion": lact is not None and lact >= 2.0,
        "renal": uo is not None and uo < 0.5,
        "pressor": norepi is not None and norepi >= 0.1,
        "organ": (sofa_total is not None and sofa_total >= 8) or (sofa_cv is not None and sofa_cv >= 3),
    }


def _merge_remote_candidates_with_local_rubric(local_eval: dict, candidate_categories: list[str], values: dict) -> tuple[dict, dict]:
    ordered = [
        "VERY_LIKELY_STABLE",
        "LIKELY_STABLE",
        "POTENTIAL_OCCULT_SHOCK",
        "LIKELY_WORSENING",
        "VERY_LIKELY_WORSENING",
    ]
    merged = dict(local_eval)
    meta = {
        "transition_merge_source": "local_only",
        "transition_advisory_candidate_used": False,
        "transition_advisory_uplifted": False,
        "transition_advisory_target": None,
    }
    if not candidate_categories:
        return merged, meta

    valid = [c for c in candidate_categories if c in ordered]
    if not valid:
        return merged, meta

    remote_target = max(valid, key=lambda c: ordered.index(c))
    meta["transition_advisory_target"] = remote_target

    flags = _objective_domain_flags(values)
    if not (flags.get("hemodynamic") or flags.get("perfusion")):
        meta["transition_merge_source"] = "remote_ignored_no_local_support"
        return merged, meta

    local_category = str(local_eval.get("category") or "").strip().upper()
    if local_category not in ordered:
        meta["transition_merge_source"] = "remote_ignored_invalid_local_category"
        return merged, meta

    local_rank = ordered.index(local_category)
    remote_rank = ordered.index(remote_target)
    if remote_rank <= local_rank:
        meta["transition_merge_source"] = "remote_no_uplift"
        return merged, meta

    uplift_rank = min(local_rank + 1, remote_rank)
    uplift_category = ordered[uplift_rank]
    merged["category"] = uplift_category
    merged["severity"] = uplift_rank + 1
    merged["reason"] = (
        f"{local_eval.get('reason', '')} "
        f"[REMOTE_CANDIDATE_MERGE] Local rubric was uplifted one level toward remote candidate "
        f"{remote_target} because hemodynamic/perfusion support was present."
    ).strip()
    meta["transition_merge_source"] = "remote_candidate_uplift"
    meta["transition_advisory_candidate_used"] = True
    meta["transition_advisory_uplifted"] = True
    return merged, meta

def _apply_decision_balance_gate(raw_decision: dict | None, values: dict, final_rubric_eval: dict) -> tuple[dict | None, dict]:
    if not isinstance(raw_decision, dict):
        return raw_decision, {"decision_balance_gate": "not_applicable"}
    action = _extract_final_action(raw_decision)
    flags = _objective_domain_flags(values)
    support_count = sum(1 for v in flags.values() if v)
    gate = {
        "decision_balance_gate": "none",
        "decision_support_count": support_count,
        "decision_support_flags": flags,
    }
    if action == "INVESTIGATE_O":
        if support_count == 0:
            raw_decision = dict(raw_decision)
            raw_decision["final_action"] = "OBSERVE"
            raw_decision["differential_diagnosis"] = (
                f"{raw_decision.get('differential_diagnosis', '')} "
                "[LOCAL_BALANCE_GATE] Escalation downgraded because objective multi-domain support was absent."
            ).strip()
            gate["decision_balance_gate"] = "downgrade_to_observe"
        elif support_count == 1 and final_rubric_eval.get("category") != "VERY_LIKELY_WORSENING":
            raw_decision = dict(raw_decision)
            raw_decision["final_action"] = "TREAT_S"
            raw_decision["differential_diagnosis"] = (
                f"{raw_decision.get('differential_diagnosis', '')} "
                "[LOCAL_BALANCE_GATE] Escalation softened because only one objective risk domain was clearly abnormal."
            ).strip()
            gate["decision_balance_gate"] = "downgrade_to_treat_s"
    return raw_decision, gate

def _trace_enabled() -> bool:
    return _env_bool("CARE_TRACE_PROGRESS", True)


def _record_usage(tokens: int) -> tuple[int, int]:
    tk = max(0, int(tokens or 0))
    _RUNTIME_USAGE["calls"] += 1
    _RUNTIME_USAGE["reported_tokens"] += tk
    return _RUNTIME_USAGE["calls"], _RUNTIME_USAGE["reported_tokens"]


def _trace_step(
    *,
    stay_id: int,
    step: str,
    event: str,
    call_tokens: int = 0,
    branch_tokens: int = 0,
    attempt: int | None = None,
    extra: str = "",
) -> None:
    if not _trace_enabled():
        return
    msg = f"care trace | stay_id={stay_id} step={step} event={event}"
    if attempt is not None:
        msg += f" attempt={attempt}"
    if call_tokens:
        msg += f" call_tokens={int(call_tokens)}"
    if branch_tokens:
        msg += f" branch_tokens={int(branch_tokens)}"
    if event == "done":
        calls, total = _record_usage(call_tokens)
        msg += f" run_calls={calls} run_reported_tokens={total}"
    if extra:
        msg += f" | {extra}"
    logger.info(msg)

async def _run_care_case(
    *,
    local_client: AsyncOpenAI,
    local_provider: str,
    local_model_name: str,
    remote_client: AsyncOpenAI,
    remote_provider: str,
    remote_model_name: str,
    row: dict,
    adapter: FactsAdapter,
    rubric_engine: RiskRubricEngine,
    t_planner,
    t_advisor,
    t_decision,
) -> tuple[dict, dict | None, dict, int]:
    """
    Executes the CARE Neuro-symbolic loop for a single patient.
    Privacy boundary: evidence planning and final decision = LOCAL LLM (can see values).
                      transition advisory  = REMOTE LLM (NO patient values).
    """
    tokens = 0
    input_tokens = 0
    output_tokens = 0
    stay_id = int(row["stay_id"])
    t0 = time.monotonic()
    
    # ---------------------------------------------------------
    # initial risk screen: Local Initial Category Computation (PROGRAMMATIC)
    # ---------------------------------------------------------
    patient_reference_state = _initial_patient_state(row)
    initial_rubric_eval = rubric_engine.evaluate_patient(patient_reference_state)
    
    # ---------------------------------------------------------
    # evidence planning: Category-Aware Data Acquisition Planning
    # PRIVACY: LOCAL — receives actual patient values
    # ---------------------------------------------------------
    ctx_base_planner = {
        "hr_median_last1h": row.get("hr_median_last1h"),
        "pain_max_last1h": row.get("pain_max_last1h"),
        "rass_max_last1h": row.get("rass_max_last1h"),
        "map_median_last1h": row.get("map_median_last1h"),
        "map_low_minutes_last1h_thr65": row.get("map_low_minutes_last1h_thr65"),
        "map_low_minutes_last1h_thr60": row.get("map_low_minutes_last1h_thr60"),
        "has_map_coverage_last1h": row.get("has_map_coverage_last1h"),
        "sofa_total": row.get("sofa_total"),
        "sofa_cardiovascular": row.get("sofa_cardiovascular"),
        "current_category": initial_rubric_eval["category"],
        "current_severity": initial_rubric_eval["severity"],
        "current_category_reason": initial_rubric_eval["reason"]
    }

    prompt_planner = t_planner.render(**ctx_base_planner, allowed_keys=allowed_fact_key_lines())
    _trace_step(stay_id=stay_id, step="planner", event="start")
    raw_planner = await evaluate_vignette(
        local_client,
        local_provider,
        local_model_name,
        prompt_planner,
        component="planner",
    )  # PRIVACY: LOCAL
    planner_usage = llm_usage_from_raw(raw_planner)
    planner_tokens = planner_usage["total_tokens"]
    tokens += planner_tokens
    input_tokens += planner_usage["input_tokens"]
    output_tokens += planner_usage["output_tokens"]
    advisor_input_tokens = 0
    advisor_output_tokens = 0
    advisor_token_split_available = True
    advisor_total_tokens = 0
    _trace_step(
        stay_id=stay_id,
        step="planner",
        event="done",
        call_tokens=planner_tokens,
        branch_tokens=tokens,
        extra=f"elapsed={time.monotonic()-t0:.1f}s",
    )

    norm = normalize_planner_response(raw_planner)
    facts_keys = list(norm.facts_keys)

    # ---------------------------------------------------------
    # sufficiency recheck: Data Sufficiency Recheck (LOCAL SYSTEM)
    # ---------------------------------------------------------
    requested_keys = dedupe_keep_order(facts_keys)
    sufficiency_remaining_requested_keys, category_requirement_groups = _missing_requirement_keys(
        requested_keys, initial_rubric_eval["category"], row=row
    )
    category_requirements = [k for group in category_requirement_groups for k in group]
    sufficiency_is_sufficient = bool(norm.need_data and bool(requested_keys) and not sufficiency_remaining_requested_keys)

    eff_keys = dedupe_keep_order(requested_keys + sufficiency_remaining_requested_keys)
    if not eff_keys and norm.need_data:
        eff_keys = dedupe_keep_order(category_requirements + _sufficiency_fallback_keys())
    elif sufficiency_remaining_requested_keys:
        eff_keys = dedupe_keep_order(eff_keys + _sufficiency_fallback_keys())

    key_health = planner_facts_keys_health(
        facts_keys=facts_keys,
        valid_key_lookup=FACT_SPECS,
        need_data=norm.need_data,
        eff_keys=eff_keys,
    )
    recovery_stats = sufficiency_recovery_stats(
        valid_requested_keys=key_health["planner_valid_requested_keys"],
        effective_keys=key_health["planner_effective_keys"],
        sufficiency_remaining_requested_keys=sufficiency_remaining_requested_keys,
        need_data=norm.need_data,
    )

    if not eff_keys:
        planner_out = {
            "need_data": True,
            "facts_keys": list(facts_keys),
            "category_requirements": category_requirements,
            "sufficiency_is_sufficient": False,
            "sufficiency_remaining_requested_keys": sufficiency_remaining_requested_keys,
            "sufficiency_updated_available_keys": [],
            "reasoning": raw_planner.get("reasoning", ""),
        }
        meta = {
            "care_initial_category": initial_rubric_eval["category"],
            "care_initial_severity": initial_rubric_eval["severity"],
            "initial_state_keys": sorted(patient_reference_state.keys()),
            "initial_state_nonnull": sum(v is not None for v in patient_reference_state.values()),
            "category_requirements": category_requirements,
            "sufficiency_is_sufficient": False,
            "sufficiency_remaining_requested_keys": sufficiency_remaining_requested_keys,
            "sufficiency_updated_available_keys": [],
            "advisor_attempts": 0,
            "advisor_raw_categories": 0,
            "advisor_filtered_out": 0,
            "advisor_num_categories": 0,
            "advisor_guidance_present": False,
            "planner_input_tokens": planner_usage["input_tokens"],
            "planner_output_tokens": planner_usage["output_tokens"],
            "planner_token_split_available": planner_usage["token_split_available"],
            "planner_tokens": planner_tokens,
            "advisor_input_tokens": 0,
            "advisor_output_tokens": 0,
            "advisor_token_split_available": False,
            "advisor_tokens": 0,
            "decision_input_tokens": 0,
            "decision_output_tokens": 0,
            "decision_token_split_available": False,
            "decision_tokens": 0,
            "branch_input_tokens": input_tokens,
            "branch_output_tokens": output_tokens,
            "branch_total_tokens": tokens,
            "planner_model": local_model_name,
            "advisor_model": remote_model_name,
            "decision_model": local_model_name,
            **key_health,
            **recovery_stats,
            "decision_mode": "sufficiency_insufficient",
        }
        return planner_out, {
            "final_action": "INVESTIGATE_O",
            "differential_diagnosis": "[INSUFFICIENT_EVIDENCE] Insufficient requested evidence after fallback key synthesis.",
        }, meta, tokens

    values = adapter.fetch_values(int(row["stay_id"]), int(row["t_eval"]), eff_keys)

    # ---------------------------------------------------------
    # transition advisory: Remote Transition Policy Suggestion
    # PRIVACY: REMOTE — receives ZERO patient values.
    #   Only: category label, key NAMES, feature DESCRIPTIONS, rubric definitions.
    # ---------------------------------------------------------
    ctx_advisor = _build_remote_advisor_context(
        initial_rubric_eval=initial_rubric_eval,
        eff_keys=eff_keys,
        rubric_engine=rubric_engine,
    )
    advisor_retry = max(0, int(os.getenv("CARE_ADVISOR_RETRY", "1")))
    require_nonempty_advisor = _env_bool("CARE_REQUIRE_NONEMPTY_ADVISOR", False)
    prompt_advisor = t_advisor.render(**ctx_advisor)
    advisor_attempts = 1 + advisor_retry
    raw_advisor = {}
    norm_advisor = {
        "candidate_target_categories": [],
        "transition_guidance": "",
        "transition_reasoning": "",
        "raw_category_count": 0,
        "filtered_out_count": 0,
    }
    candidate_categories: list[str] = []

    for attempt in range(1, advisor_attempts + 1):
        _trace_step(stay_id=stay_id, step="advisor", event="start", attempt=attempt)
        raw_advisor = await evaluate_vignette(
            remote_client,
            remote_provider,
            remote_model_name,
            prompt_advisor,
            component="advisor",
        )  # PRIVACY: REMOTE
        advisor_usage = llm_usage_from_raw(raw_advisor)
        advisor_tokens = advisor_usage["total_tokens"]
        tokens += advisor_tokens
        advisor_total_tokens += advisor_tokens
        input_tokens += advisor_usage["input_tokens"]
        output_tokens += advisor_usage["output_tokens"]
        advisor_input_tokens += advisor_usage["input_tokens"]
        advisor_output_tokens += advisor_usage["output_tokens"]
        advisor_token_split_available = advisor_token_split_available and advisor_usage["token_split_available"]
        norm_advisor = normalize_advisor_response(raw_advisor)
        candidate_categories = list(norm_advisor.get("candidate_target_categories", []))
        _trace_step(
            stay_id=stay_id,
            step="advisor",
            event="done",
            attempt=attempt,
            call_tokens=advisor_tokens,
            branch_tokens=tokens,
            extra=(
                f"raw_categories={int(norm_advisor.get('raw_category_count', 0) or 0)} "
                f"valid_categories={len(candidate_categories)} filtered_out={int(norm_advisor.get('filtered_out_count', 0) or 0)} "
                f"guidance_present={bool(norm_advisor.get('transition_guidance'))}"
            ),
        )
        if candidate_categories or norm_advisor.get("transition_guidance"):
            break
        if attempt < advisor_attempts:
            logger.warning(
                "CARE transition advisory returned no usable guidance: "
                f"attempt={attempt}/{advisor_attempts}; retrying."
            )
    
    # ---------------------------------------------------------
    # local transition update: Local Transition Computation (PROGRAMMATIC)
    # Combines remote advisory guidance with locally available objective values.
    # The final updated category is computed locally from the shared rubric.
    # ---------------------------------------------------------
    patient_full_state = {**patient_reference_state, **values}
    local_rubric_eval = rubric_engine.evaluate_patient(patient_full_state)
    guidance_text = norm_advisor.get("transition_guidance", "").strip()
    guidance_reasoning = norm_advisor.get("transition_reasoning", "").strip()
    if require_nonempty_advisor and not guidance_text and not candidate_categories:
        guidance_text = (
            "Remote transition guidance unavailable; local system should rely on updated objective rubric "
            "and prioritize hemodynamic, perfusion, and renal convergence."
        )
    final_rubric_eval, transition_merge_meta = _merge_remote_candidates_with_local_rubric(
        local_rubric_eval, candidate_categories, values
    )
    
    # ---------------------------------------------------------
    # final decision: Local Final Decision
    # PRIVACY: LOCAL — receives actual patient values + all reasoning
    # ---------------------------------------------------------
    report = adapter.render_report(eff_keys, values)
    
    dynamic_summary = final_rubric_eval.get("reason", "")

    ctx_base_decision = {
        **ctx_base_planner,
        "dynamic_facts_report": report,
        "initial_category": initial_rubric_eval["category"],
        "updated_category": final_rubric_eval["category"],
        "updated_severity": final_rubric_eval["severity"],
        "transition_reason": dynamic_summary,
        "transition_guidance": (
            f"Updated category computed locally as {final_rubric_eval['category']} after objective review. "
            "This CARE variant does not pass remote guidance text into final decision."
        ),
        "transition_reasoning": (
            "final decision receives only the locally merged category state; remote candidate categories are consumed "
            "inside local transition update only."
        ),
    }
    
    prompt_decision = t_decision.render(**ctx_base_decision)
    _trace_step(stay_id=stay_id, step="decision", event="start")
    raw_decision = await evaluate_vignette(
        local_client,
        local_provider,
        local_model_name,
        prompt_decision,
        component="decision",
    )  # PRIVACY: LOCAL
    decision_usage = llm_usage_from_raw(raw_decision)
    decision_tokens = decision_usage["total_tokens"]
    tokens += decision_tokens
    input_tokens += decision_usage["input_tokens"]
    output_tokens += decision_usage["output_tokens"]
    _trace_step(
        stay_id=stay_id,
        step="decision",
        event="done",
        call_tokens=decision_tokens,
        branch_tokens=tokens,
        extra=f"elapsed={time.monotonic()-t0:.1f}s",
    )
    raw_decision, balance_gate_meta = _apply_decision_balance_gate(raw_decision, values, final_rubric_eval)

    planner_out = {
        "need_data": True,
        "facts_keys": facts_keys,
        "category_requirements": category_requirements,
        "category_requirement_groups": category_requirement_groups,
        "sufficiency_is_sufficient": sufficiency_is_sufficient,
        "sufficiency_remaining_requested_keys": sufficiency_remaining_requested_keys,
        "sufficiency_updated_available_keys": eff_keys,
        "reasoning": raw_planner.get("reasoning", ""),
        "care_engine_initial": initial_rubric_eval,
        "care_engine_local_final": local_rubric_eval,
        "care_engine_final": final_rubric_eval,
        "care_transition_candidates": candidate_categories,
        "care_transition_guidance": guidance_text,
        "care_transition_reasoning": guidance_reasoning,
    }
    
    meta = {
        "care_initial_category": initial_rubric_eval["category"], 
        "care_initial_severity": initial_rubric_eval["severity"],
        "care_local_final_category": local_rubric_eval["category"],
        "care_local_final_severity": local_rubric_eval["severity"],
        "care_final_category": final_rubric_eval["category"], 
        "care_final_severity": final_rubric_eval["severity"],
        "initial_state_keys": sorted(patient_reference_state.keys()),
        "initial_state_nonnull": sum(v is not None for v in patient_reference_state.values()),
        "category_requirements": category_requirements,
        "category_requirement_groups": category_requirement_groups,
        "sufficiency_is_sufficient": sufficiency_is_sufficient,
        "sufficiency_remaining_requested_keys": sufficiency_remaining_requested_keys,
        "sufficiency_updated_available_keys": eff_keys,
        "advisor_attempts": advisor_attempts,
        "advisor_raw_categories": int(norm_advisor.get("raw_category_count", 0) or 0),
        "advisor_filtered_out": int(norm_advisor.get("filtered_out_count", 0) or 0),
        "advisor_num_categories": len(candidate_categories),
        "advisor_guidance_present": bool(guidance_text),
        "advisor_candidate_categories": candidate_categories,
        "care_workflow_variant": "transition_advisory_to_local_update",
        "planner_input_tokens": planner_usage["input_tokens"],
        "planner_output_tokens": planner_usage["output_tokens"],
        "planner_token_split_available": planner_usage["token_split_available"],
        "planner_tokens": planner_tokens,
        "advisor_input_tokens": advisor_input_tokens,
        "advisor_output_tokens": advisor_output_tokens,
        "advisor_token_split_available": advisor_total_tokens > 0 and advisor_token_split_available,
        "advisor_tokens": advisor_total_tokens,
        "decision_input_tokens": decision_usage["input_tokens"],
        "decision_output_tokens": decision_usage["output_tokens"],
        "decision_token_split_available": decision_usage["token_split_available"],
        "decision_tokens": decision_tokens,
        "branch_input_tokens": input_tokens,
        "branch_output_tokens": output_tokens,
        "branch_total_tokens": tokens,
        "planner_model": local_model_name,
        "advisor_model": remote_model_name,
        "decision_model": local_model_name,
        **key_health,
        **recovery_stats,
        **transition_merge_meta,
        **balance_gate_meta,
        "decision_mode": "normal" if sufficiency_is_sufficient else "sufficiency_insufficient_recovered"
    }

    return planner_out, raw_decision, meta, tokens

async def run_care_inference(sample_size: int = 200, sample_lock_file: str | None = None) -> None:
    _RUNTIME_USAGE["calls"] = 0
    _RUNTIME_USAGE["reported_tokens"] = 0
    project_root = PROJECT_ROOT
    feature_source = resolve_feature_source()
    db_path = str(resolve_duckdb_path()) if feature_source != "locked_csv" else ""
    if db_path and not Path(db_path).is_absolute():
         db_path = str((project_root / db_path).resolve())

    runs_dir = LOGS_DIR
    ensure_runtime_dirs()

    local_base_url = _require_loopback_url(os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:8000/v1"))
    remote_base_url = os.getenv("REMOTE_BASE_URL", "").strip().rstrip("/")
    remote_api_key = os.getenv("REMOTE_API_KEY", "").strip()
    if not remote_base_url:
        raise RuntimeError("REMOTE_BASE_URL is required.")
    if not remote_api_key:
        raise RuntimeError("REMOTE_API_KEY is required.")
    if local_base_url == remote_base_url:
        raise RuntimeError("LOCAL_BASE_URL and REMOTE_BASE_URL must use separate endpoints.")

    local_model_name = os.getenv("LOCAL_MODEL", "your_local_model")
    remote_model_name = os.getenv("REMOTE_MODEL", "your_remote_model")
    local_provider_name = "local"
    remote_provider_name = os.getenv("REMOTE_PROVIDER", _provider_from_base_url(remote_base_url))

    local_client = AsyncOpenAI(api_key="local-runtime", base_url=local_base_url)
    remote_client = AsyncOpenAI(api_key=remote_api_key, base_url=remote_base_url)
    logger.info(
        "Hybrid routing: local_provider=%s local_model=%s remote_provider=%s remote_model=%s",
        local_provider_name,
        local_model_name,
        remote_provider_name,
        remote_model_name,
    )

    batch_size = int(os.getenv("BATCH_SIZE", "5"))
    max_token_limit = int(os.getenv("MAX_TOKEN_LIMIT", "300000"))
    use_concordant = os.getenv("USE_CONCORDANT_COHORT", "false").lower() == "true"
    force_resample = os.getenv("FORCE_RESAMPLE", "false").lower() == "true"
    sample_seed = int(os.getenv("SAMPLE_HASH_SEED", "42003"))
    sample_lock_override = sample_lock_file or os.getenv("SAMPLE_LOCK_FILE")

    sample_lock_path: Path | None = None
    if sample_lock_override:
        sample_lock_path = Path(sample_lock_override)
        if not sample_lock_path.is_absolute():
            sample_lock_path = (project_root / sample_lock_path).resolve()
        if sample_lock_path.exists() and not force_resample:
            cohort_df = normalize_sample_schema(pd.read_csv(sample_lock_path))
            logger.info(f"Using locked sample file for care: {sample_lock_path.name} (n={len(cohort_df)}).")
        elif sample_lock_path.exists() and force_resample:
            if feature_source == "locked_csv":
                raise RuntimeError(
                    "FEATURE_SOURCE=locked_csv does not allow FORCE_RESAMPLE=true. "
                    "CSV mode requires a fixed locked sample and will not rebuild it from DuckDB."
                )
            logger.info(f"Rebuilding locked sample for care: {sample_lock_path.name} (n={sample_size}).")
            base_df = _build_base_df(db_path, use_concordant=use_concordant)
            cohort_df = _deterministic_sample(
                base_df,
                sample_size=sample_size,
                seed=sample_seed,
                occult_min_per_class=0,
            )
            sample_lock_path.parent.mkdir(parents=True, exist_ok=True)
            cohort_df.to_csv(sample_lock_path, index=False)
        else:
            raise FileNotFoundError(
                f"SAMPLE_LOCK_FILE does not exist: {sample_lock_path}. "
                "Pass an existing locked CSV or set FORCE_RESAMPLE=true to create one."
            )
    else:
        if feature_source == "locked_csv":
            raise FileNotFoundError(
                "FEATURE_SOURCE=locked_csv requires SAMPLE_LOCK_FILE to be set to an existing locked CSV. "
                "CSV mode will not build a cohort from DuckDB."
            )
        logger.info(f"Building cohort sample for care test... (n={sample_size})")
        base_df = _build_base_df(db_path, use_concordant=use_concordant)
        cohort_df = _deterministic_sample(
            base_df,
            sample_size=sample_size,
            seed=sample_seed,
            occult_min_per_class=0,
        )
        sample_lock_path = (SAMPLES_DIR / f"care_sample_n{sample_size}_seed42.csv").resolve()

    required = {
        "stay_id",
        "t_eval",
        "ground_truth_deterioration",
        "pain_max_last1h",
        "rass_max_last1h",
        "hr_median_last1h",
        "occult_hypoperfusion_slice_thr60",
    }
    missing = required.difference(set(cohort_df.columns))
    if missing:
        raise RuntimeError(f"care sample missing required columns: {sorted(missing)}")

    fp = _sample_fingerprint(cohort_df)
    pos = int((cohort_df["ground_truth_deterioration"].astype(int) == 1).sum())
    neg = int((cohort_df["ground_truth_deterioration"].astype(int) == 0).sum())
    occ = int(cohort_df["occult_hypoperfusion_slice_thr60"].fillna(False).astype(bool).sum())
    logger.info(f"Locked sample summary: n={len(cohort_df)} pos={pos} neg={neg} occ={occ} fp={fp}")

    env = Environment(loader=FileSystemLoader(str(resolve_prompt_dir())))
    t_agent_planner = env.get_template("evidence_planning.j2")
    t_agent_advisor = env.get_template("transition_advisory.j2")
    t_agent_decision = env.get_template("final_decision.j2")

    facts_engine = FactsGenerator(db_path)
    adapter = FactsAdapter(facts_engine)
    rubric_engine = RiskRubricEngine((THIS_DIR / "config_rubric.json").resolve())

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    jsonl_path = runs_dir / f"care_main_{ts}.jsonl"
    md_path = runs_dir / f"care_main_{ts}_review.md"
    console_log_path = runs_dir / f"care_main_{ts}.console.log"

    rows = cohort_df.to_dict("records")
    total_tokens = 0

    with tee_console_to_file(console_log_path):
        logger.info(f"Console log mirrored to {console_log_path}")
        with open(jsonl_path, "w", encoding="utf-8") as f_jsonl, open(md_path, "w", encoding="utf-8") as f_md:
            f_md.write(
                f"# CARE Workflow Review Log: {ts} "
                f"(Local: {local_model_name}, Remote: {remote_model_name})\n\n"
            )
            f_md.write(f"Sample fingerprint: `{fp}`\n\n")
            if sample_lock_path is not None:
                f_md.write(f"Locked sample file: `{sample_lock_path}`\n\n")

            for i in range(0, len(rows), batch_size):
                batch = rows[i : i + batch_size]
                logger.info(f"Processing Batch {i // batch_size + 1} / {(len(rows) + batch_size - 1) // batch_size}")

                tasks_agent = [
                    _run_care_case(
                        local_client=local_client,
                        local_provider=local_provider_name,
                        local_model_name=local_model_name,
                        remote_client=remote_client,
                        remote_provider=remote_provider_name,
                        remote_model_name=remote_model_name,
                        row=row,
                        adapter=adapter,
                        rubric_engine=rubric_engine,
                        t_planner=t_agent_planner,
                        t_advisor=t_agent_advisor,
                        t_decision=t_agent_decision
                    )
                    for row in batch
                ]
                res_agent = await asyncio.gather(*tasks_agent)

                for j, row in enumerate(batch):
                    a_planner, a_decision, a_meta, tk_a = res_agent[j]
                    total_tokens += tk_a

                    entry = {
                        "stay_id": int(row["stay_id"]),
                        "hr": int(row["t_eval"]),
                        "ground_truth": int(row["ground_truth_deterioration"]),
                        "is_hypoperfusion": bool(row.get("occult_hypoperfusion_slice_thr60", False)),
                        "agent_planner": a_planner,
                        "agent_decision": a_decision,
                        "agent_meta": a_meta,
                    }
                    f_jsonl.write(json.dumps(entry) + "\n")

                    f_md.write(f"### Patient: stay_id={entry['stay_id']} hr={entry['hr']} (GT: {entry['ground_truth']})\n")
                    f_md.write(
                        f"- initial risk screen Computed: **{a_meta.get('care_initial_category')}** "
                        f"-> local transition update Computed: **{a_meta.get('care_final_category', 'N/A')}**\n"
                    )
                    f_md.write(f"- Final Action: `{_extract_final_action(a_decision) or _extract_final_action(a_planner)}`\n\n")

                if total_tokens >= max_token_limit:
                    logger.warning(f"Token limit reached: {total_tokens} >= {max_token_limit}; stopping early.")
                    break

        # Call the evaluation script natively
        previous_pattern = os.environ.get("CARE_LOG_PATTERN")
        os.environ["CARE_LOG_PATTERN"] = jsonl_path.name
        try:
            from eval_care_outcomes import evaluate_care_outputs
            evaluate_care_outputs()
        finally:
            if previous_pattern is None:
                os.environ.pop("CARE_LOG_PATTERN", None)
            else:
                os.environ["CARE_LOG_PATTERN"] = previous_pattern

        logger.success(f"CARE run complete. JSONL={jsonl_path} REVIEW={md_path} CONSOLE={console_log_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--sample-lock-file", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(
        run_care_inference(
            sample_size=args.sample_size,
            sample_lock_file=args.sample_lock_file,
        )
    )
