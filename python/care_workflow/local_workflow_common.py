import hashlib
import json
import os
import re
import sys
from typing import Any

import duckdb
import pandas as pd
from openai import AsyncOpenAI

from package_runtime import bootstrap_import_paths, load_package_env

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging
    import types

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    _base_logger = logging.getLogger(__name__)

    class _LoguruShim:
        def __getattr__(self, name):
            if name == "success":
                return _base_logger.info
            return getattr(_base_logger, name, _base_logger.info)

    logger = _LoguruShim()
    sys.modules.setdefault("loguru", types.SimpleNamespace(logger=logger))


def _load_project_env() -> None:
    bootstrap_import_paths()
    load_package_env()


def _extract_json(text: str) -> dict:
    text = text or ""
    if "</think>" in text:
        text = text.split("</think>", 1)[-1]
    m = re.search(r"\{.*\}", text, re.DOTALL)
    payload = m.group(0) if m else text.strip()
    return json.loads(payload)


def _usage_tokens(response: Any) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {
            "total_tokens": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "token_split_available": False,
        }

    total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    input_tokens = getattr(usage, "prompt_tokens", None)
    if input_tokens is None:
        input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    if output_tokens is None:
        output_tokens = getattr(usage, "output_tokens", None)
    token_split_available = input_tokens is not None and output_tokens is not None

    return {
        "total_tokens": total_tokens,
        "input_tokens": int(input_tokens or 0),
        "output_tokens": int(output_tokens or 0),
        "token_split_available": bool(token_split_available),
    }


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _component_completion_cap(component: str) -> int:
    defaults = {
        "planner": 15000,
        "advisor": 12000,
        "decision": 20000,
    }
    env_map = {
        "planner": "CARE_PLANNER_MAX_COMPLETION_TOKENS",
        "advisor": "CARE_ADVISOR_MAX_COMPLETION_TOKENS",
        "decision": "CARE_DECISION_MAX_COMPLETION_TOKENS",
    }
    env_name = env_map.get(component, "CARE_MAX_COMPLETION_TOKENS")
    default_value = defaults.get(component, 20000)
    return int(os.getenv(env_name, os.getenv("CARE_MAX_COMPLETION_TOKENS", str(default_value))))


def _build_completion_kwargs(provider: str, model: str, completion_token_cap: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "max_tokens": completion_token_cap,
        "max_completion_tokens": completion_token_cap,
    }
    return kwargs


async def evaluate_vignette(
    client: AsyncOpenAI,
    provider: str,
    model: str,
    prompt_text: str,
    *,
    component: str,
) -> dict:
    raw_text = ""
    completion_token_cap = _component_completion_cap(component)
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Clinical Reasoning Agent. Output your response entirely in valid JSON format.",
                },
                {"role": "user", "content": prompt_text},
            ],
            **_build_completion_kwargs(provider, model, completion_token_cap),
        )
    except Exception as e:  # pragma: no cover
        logger.error(f"API Error ({model}): {e} | Raw={raw_text[:120] if raw_text else 'None'}")
        return {
            "error": str(e),
            "action": "ERROR",
            "final_action": "ERROR",
            "_total_tokens": 0,
            "_input_tokens": 0,
            "_output_tokens": 0,
            "_token_split_available": False,
        }

    raw_text = response.choices[0].message.content
    usage = _usage_tokens(response)
    try:
        parsed = _extract_json(raw_text)
    except Exception as e:  # pragma: no cover
        logger.error(f"Parse Error ({model}): {e} | Raw={raw_text[:120] if raw_text else 'None'}")
        return {
            "error": str(e),
            "action": "ERROR",
            "final_action": "ERROR",
            "_total_tokens": usage["total_tokens"],
            "_input_tokens": usage["input_tokens"],
            "_output_tokens": usage["output_tokens"],
            "_token_split_available": usage["token_split_available"],
            "_raw_text_present": bool(raw_text),
        }
    parsed["_total_tokens"] = usage["total_tokens"]
    parsed["_input_tokens"] = usage["input_tokens"]
    parsed["_output_tokens"] = usage["output_tokens"]
    parsed["_token_split_available"] = usage["token_split_available"]
    return parsed


def _sample_fingerprint(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    tmp = df[["stay_id", "t_eval"]].copy()
    tmp["stay_id"] = tmp["stay_id"].astype(int)
    tmp["t_eval"] = tmp["t_eval"].astype(int)
    tmp = tmp.sort_values(["stay_id", "t_eval"], kind="stable")
    raw = ";".join(f"{r.stay_id}:{r.t_eval}" for r in tmp.itertuples(index=False))
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _stable_key(stay_id: int, t_eval: int, seed: int) -> str:
    raw = f"{stay_id}:{t_eval}:{seed}".encode("utf-8")
    return hashlib.md5(raw).hexdigest()


def _deterministic_sample(
    base_df: pd.DataFrame,
    sample_size: int,
    seed: int,
    occult_min_per_class: int,
) -> pd.DataFrame:
    half = sample_size // 2
    out_parts = []

    for y in [1, 0]:
        cls = base_df[base_df["ground_truth_deterioration"].astype(int) == y].copy()
        if cls.empty:
            continue
        cls["_ord"] = [
            _stable_key(int(r.stay_id), int(r.t_eval), seed + y * 1000)
            for r in cls.itertuples(index=False)
        ]
        cls = cls.sort_values("_ord", kind="stable")

        if occult_min_per_class > 0:
            occ = cls[cls["occult_hypoperfusion_slice_thr60"].fillna(False).astype(bool)].copy()
            non = cls[~cls["occult_hypoperfusion_slice_thr60"].fillna(False).astype(bool)].copy()
            take_occ = min(occult_min_per_class, half, len(occ))
            chosen = pd.concat([occ.head(take_occ), non.head(max(half - take_occ, 0))], axis=0)
            if len(chosen) < half:
                remain = half - len(chosen)
                rest = cls.drop(index=chosen.index, errors="ignore")
                chosen = pd.concat([chosen, rest.head(remain)], axis=0)
        else:
            chosen = cls.head(half)
        out_parts.append(chosen)

    out = pd.concat(out_parts, axis=0) if out_parts else pd.DataFrame()
    out = out.drop(columns=["_ord"], errors="ignore")

    if len(out) < sample_size:
        remain = sample_size - len(out)
        rest = base_df.drop(index=out.index, errors="ignore").copy()
        rest["_ord"] = [
            _stable_key(int(r.stay_id), int(r.t_eval), seed + 777)
            for r in rest.itertuples(index=False)
        ]
        rest = rest.sort_values("_ord", kind="stable")
        out = pd.concat([out, rest.head(remain)], axis=0)

    out = out.head(sample_size).copy()
    out["stay_id"] = out["stay_id"].astype(int)
    out["t_eval"] = out["t_eval"].astype(int)
    out = out.sort_values(["stay_id", "t_eval"], kind="stable")
    return out


def _build_base_df(db_path: str, use_concordant: bool) -> pd.DataFrame:
    if use_concordant:
        where_clause = """
        WHERE lbl.y_deteriorate_delta2_6_12 IS NOT NULL
          AND sl.pain_max_last1h IS NOT NULL
          AND sl.rass_n_last1h >= 1
          AND (CAST(sl.pain_max_last1h AS FLOAT) >= 6
               OR CAST(sl.rass_max_last1h AS FLOAT) >= 2
               OR CAST(sl.rass_min_last1h AS FLOAT) <= -3)
        """
    else:
        where_clause = """
        WHERE lbl.y_deteriorate_delta2_6_12 IS NOT NULL
          AND sl.pain_max_last1h IS NOT NULL
          AND sl.rass_n_last1h >= 1
          AND sl.pain_max_last1h = 0
          AND sl.rass_max_last1h <= 0
          AND sl.rass_min_last1h > -3
          AND sl.map_low_minutes_last1h_thr65 > 5
        """

    query = f"""
    SELECT
        lbl.stay_id,
        lbl.t_base AS t_eval,
        lbl.y_deteriorate_delta2_6_12 AS ground_truth_deterioration,
        sl.pain_max_last1h,
        sl.rass_max_last1h,
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
        sl.occult_hypoperfusion_slice_thr60,
        sh.sofa_resp,
        sh.sofa_coag,
        sh.sofa_liver,
        sh.sofa_cns,
        sh.sofa_renal,
        sh.sofa_cardiovascular,
        sh.sofa_total
    FROM mimiciv_derived.sofa_labels lbl
    JOIN mimiciv_icu.icustays icu
      ON lbl.stay_id = icu.stay_id
    JOIN mimiciv_derived.occult_hypoperfusion_slice sl
      ON lbl.stay_id = sl.stay_id AND lbl.t_base = sl.hr
    JOIN mimiciv_derived.sofa_hourly sh
      ON lbl.stay_id = sh.stay_id AND lbl.t_base = sh.hr
    {where_clause}
    """

    with duckdb.connect(db_path, read_only=True) as con:
        return con.execute(query).df()


def _extract_final_action(obj: dict | None) -> str | None:
    if not isinstance(obj, dict):
        return None
    for k in ("final_action", "action"):
        v = obj.get(k)
        if not v:
            continue
        s = str(v).strip().upper()
        if s in {"OBSERVE", "TREAT_S", "INVESTIGATE_O"}:
            return s
    return None
