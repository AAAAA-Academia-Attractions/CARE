from __future__ import annotations

from typing import Any


def _dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def llm_usage_from_raw(raw: dict | None) -> dict[str, Any]:
    raw = raw or {}
    total_tokens = int(raw.get("_total_tokens", 0) or 0)
    input_tokens = int(raw.get("_input_tokens", 0) or 0)
    output_tokens = int(raw.get("_output_tokens", 0) or 0)
    token_split_available = bool(raw.get("_token_split_available", False))
    return {
        "total_tokens": total_tokens,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "token_split_available": token_split_available,
    }


def planner_facts_keys_health(
    *,
    facts_keys: list[str],
    valid_key_lookup: dict[str, Any],
    need_data: bool,
    eff_keys: list[str],
) -> dict[str, Any]:
    facts_keys = list(facts_keys or [])
    invalid_keys = [k for k in facts_keys if k not in valid_key_lookup]
    valid_requested_keys = _dedupe_keep_order([k for k in facts_keys if k in valid_key_lookup])
    effective_keys = _dedupe_keep_order([k for k in (eff_keys or []) if k in valid_key_lookup])
    return {
        "planner_facts_keys_count": len(facts_keys),
        "planner_invalid_keys": invalid_keys,
        "planner_invalid_key_count": len(invalid_keys),
        "planner_has_invalid_keys": bool(invalid_keys),
        "planner_valid_requested_keys": valid_requested_keys,
        "planner_valid_requested_key_count": len(valid_requested_keys),
        "planner_need_data_true": bool(need_data),
        "planner_effective_keys": effective_keys,
        "planner_effective_key_count": len(effective_keys),
        "planner_need_data_true_but_no_effective_keys": bool(need_data) and not bool(effective_keys),
    }


def sufficiency_recovery_stats(
    *,
    valid_requested_keys: list[str],
    effective_keys: list[str],
    sufficiency_remaining_requested_keys: list[str],
    need_data: bool,
) -> dict[str, Any]:
    valid_requested_keys = _dedupe_keep_order(list(valid_requested_keys or []))
    effective_keys = _dedupe_keep_order(list(effective_keys or []))
    sufficiency_remaining_requested_keys = list(sufficiency_remaining_requested_keys or [])

    added_keys = [k for k in effective_keys if k not in set(valid_requested_keys)]
    attempted = bool(sufficiency_remaining_requested_keys) or (bool(need_data) and not bool(valid_requested_keys))
    success = attempted and bool(effective_keys)
    failed = attempted and not bool(effective_keys)
    return {
        "sufficiency_recovery_attempted": attempted,
        "sufficiency_recovery_added_keys": added_keys,
        "sufficiency_recovery_added_key_count": len(added_keys),
        "sufficiency_recovery_success": success,
        "sufficiency_recovery_failed": failed,
    }
