from pydantic import BaseModel, Field, ValidationError
from typing import List
import json


class PlannerResponse(BaseModel):
    """Structured response for evidence acquisition planning."""

    reasoning: str = Field(description="Clinical rationale for requesting objective keys")
    need_data: bool = Field(description="Whether additional objective data is needed")
    facts_keys: List[str] = Field(default_factory=list, description="Exact keys requested")


class PlannerNormalized(BaseModel):
    need_data: bool
    facts_keys: List[str] = Field(default_factory=list)
    invalid_keys: List[str] = Field(default_factory=list)


def normalize_planner_response(raw_json: dict) -> PlannerNormalized:
    try:
        obj = PlannerResponse(**raw_json)
        return PlannerNormalized(
            need_data=obj.need_data,
            facts_keys=list(obj.facts_keys),
            invalid_keys=list(),
        )
    except ValidationError:
        need_data = raw_json.get("need_data", True) if isinstance(raw_json, dict) else True
        keys = raw_json.get("facts_keys", []) if isinstance(raw_json, dict) else []
        if not isinstance(keys, list):
            keys = []
        return PlannerNormalized(
            need_data=bool(need_data),
            facts_keys=[str(k) for k in keys],
            invalid_keys=list(),
        )


class AdvisorResponse(BaseModel):
    """Structured response for privacy-preserving transition advisory guidance."""

    candidate_target_categories: List[str] = Field(
        default_factory=list,
        description="Ordered plausible target categories to evaluate locally.",
    )
    transition_guidance: str = Field(
        default="",
        description="Short guidance about which transitions matter and why.",
    )
    transition_reasoning: str = Field(
        default="",
        description="Short explanation for the suggested transition priorities.",
    )


def normalize_advisor_response(raw_json: dict) -> dict:
    try:
        payload = raw_json
        if isinstance(payload, str):
            payload = json.loads(payload.strip())
        if not isinstance(payload, dict):
            payload = {}

        cats = payload.get("candidate_target_categories", [])
        if not isinstance(cats, list):
            cats = []
        valid_cats: list[str] = []
        for cat in cats:
            text = str(cat).strip()
            if text:
                valid_cats.append(text)

        return {
            "candidate_target_categories": valid_cats,
            "transition_guidance": str(payload.get("transition_guidance", "") or "").strip(),
            "transition_reasoning": str(payload.get("transition_reasoning", "") or "").strip(),
            "raw_category_count": len(cats),
            "filtered_out_count": max(0, len(cats) - len(valid_cats)),
        }
    except Exception:
        return {
            "candidate_target_categories": [],
            "transition_guidance": "",
            "transition_reasoning": "",
            "raw_category_count": 0,
            "filtered_out_count": 0,
        }
