import json
import logging
import operator
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# String to operator mapping
OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    "<": operator.lt,
    ">=": operator.ge,
    "<=": operator.le,
}

class RiskRubricEngine:
    """
    Programmatic evaluator for initial risk screen (Initial Category) and local transition update (Transition).
    It compares patient features against conditions defined in config_rubric.json.
    """

    def __init__(self, rubric_path: str | Path):
        self.rubric_path = Path(rubric_path)
        self.rubric = self._load_rubric()
        
    def _load_rubric(self) -> Dict[str, Any]:
        if not self.rubric_path.exists():
            raise FileNotFoundError(f"Rubric config not found: {self.rubric_path}")
        with open(self.rubric_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _evaluate_leaf_rule(self, rule: Dict[str, Any], patient_data: Dict[str, Any]) -> bool:
        feature = rule["feature"]
        op_str = rule["op"]
        threshold = rule["value"]

        val = patient_data.get(feature)
        # If the feature is completely missing (not even fetched), we cannot evaluate this rule.
        # Strict logic: Missing data fails the specific rule checking for it.
        if val is None or str(val).upper() == "N/A" or val == "":
            return False

        try:
            val_float = float(val)
            threshold_float = float(threshold)
            op_func = OPERATORS.get(op_str)
            if not op_func:
                logger.warning(f"Engine: Unknown operator '{op_str}'")
                return False
            return op_func(val_float, threshold_float)
        except ValueError:
            # Handle non-numerical edge cases gracefully if necessary
            return False

    def _evaluate_condition_node(self, node: Dict[str, Any], patient_data: Dict[str, Any]) -> bool:
        # It's a leaf node if it has 'feature'
        if "feature" in node:
            return self._evaluate_leaf_rule(node, patient_data)
        
        # Otherwise, it's a logical node (AND/OR) containing 'rules'
        logic = node.get("logic", "AND").upper()
        rules = node.get("rules", [])
        
        if not rules:
             # Empty rules array means condition is vacuously met
             return True
             
        results = [self._evaluate_condition_node(r, patient_data) for r in rules]
        
        if logic == "AND":
            return all(results)
        elif logic == "OR":
            return any(results)
        else:
             logger.warning(f"Engine: Unknown logic gate '{logic}'")
             return False

    def evaluate_patient(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the patient data against all categories in descending order of severity.
        Returns the highest severity category matched.
        Used for initial risk screen (initial category with static rubric).
        """
        categories = sorted(self.rubric.get("categories", []), key=lambda x: x.get("severity_level", 0), reverse=True)
        
        for category in categories:
            conditions = category.get("conditions", [])
            
            # If a category has no conditions, it is the default fallback.
            if not conditions:
                return {
                    "matched": True,
                    "category": category["name"],
                    "severity": category["severity_level"],
                    "reason": f"Fallback to {category['name']} (No specific threshold met)."
                }
                
            # For multiple top-level conditions in a category, we treat them as OR by default, 
            # though usually there's only 1 top-level condition node inside the array.
            category_met = any(self._evaluate_condition_node(cond, patient_data) for cond in conditions)
            
            if category_met:
                 return {
                    "matched": True,
                    "category": category["name"],
                    "severity": category["severity_level"],
                    "reason": category["description"]
                 }
                
        # Failsafe
        return {
            "matched": False,
            "category": "UNKNOWN",
            "severity": 0,
            "reason": "Failed to match any category thresholds."
        }

    def evaluate_patient_dynamic(
        self, patient_data: Dict[str, Any], transition_rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate the patient against LLM-generated transition rules from transition advisory.
        Used for local transition update (transition computation with dynamic rules).

        Each rule in transition_rules has:
          - target_category: str
          - severity_level: int
          - description: str
          - rules: list of {feature, op, value, reason}
          - logic: "AND" or "OR"
          - min_matches: int (for OR logic, how many must be true)

        Returns the highest severity category matched, or falls back to static rubric.
        """
        if not transition_rules:
            logger.warning("No dynamic transition rules provided; falling back to static rubric.")
            return self.evaluate_patient(patient_data)

        # Sort by severity descending — check highest first
        sorted_rules = sorted(transition_rules, key=lambda x: x.get("severity_level", 0), reverse=True)

        for rule_def in sorted_rules:
            target = rule_def.get("target_category", "UNKNOWN")
            severity = rule_def.get("severity_level", 0)
            description = rule_def.get("description", "")
            rules = rule_def.get("rules", [])
            logic = rule_def.get("logic", "AND").upper()
            min_matches = rule_def.get("min_matches", 1)

            if not rules:
                continue

            # Evaluate each leaf rule
            results = [self._evaluate_leaf_rule(r, patient_data) for r in rules]
            matched_count = sum(results)
            matched_reasons = [
                r.get("reason", "")
                for r, res in zip(rules, results) if res
            ]

            category_met = False
            if logic == "AND":
                category_met = all(results)
            elif logic == "OR":
                category_met = matched_count >= min_matches
            else:
                logger.warning(f"Dynamic rules: unknown logic '{logic}', treating as AND")
                category_met = all(results)

            if category_met:
                reason_str = f"{description} [Matched {matched_count}/{len(rules)} rules"
                if matched_reasons:
                    reason_str += f": {'; '.join(matched_reasons[:3])}"
                reason_str += "]"
                return {
                    "matched": True,
                    "category": target,
                    "severity": severity,
                    "reason": reason_str,
                    "dynamic": True,
                    "matched_count": matched_count,
                    "total_rules": len(rules),
                }

        # No dynamic rule matched — fall back to lowest severity
        return {
            "matched": False,
            "category": "VERY_LIKELY_STABLE",
            "severity": 1,
            "reason": "No dynamic transition rule matched; patient remains in initial category.",
            "dynamic": True,
            "matched_count": 0,
            "total_rules": 0,
        }

if __name__ == "__main__":
    import os
    # Quick test
    test_rubric_path = Path(__file__).parent / "config_rubric.json"
    engine = RiskRubricEngine(test_rubric_path)
    
    # Test case 1: initial risk screen (Only subjective and reference HR available)
    # The patient looks perfectly fine subjectively, but HR is slightly elevated
    patient_initial_screen = {
        "hr_median_last1h": 105,
        "pain_last_last1h": 0,
        "rass_max_last1h": 0,
        # Objective features haven't been fetched yet!
    }
    res_initial_screen = engine.evaluate_patient(patient_initial_screen)
    print(f"initial risk screen Result: {res_initial_screen['category']}")  # Should hit POTENTIAL_OCCULT_SHOCK
    
    # Test case 2: local transition update (Objective facts fetched)
    # The agent requested MAP low minutes and Lactate, and they came back terrible
    patient_transition_update = {
        "hr_median_last1h": 105,
        "pain_last_last1h": 0,
        "rass_max_last1h": 0,
        "map_low_minutes_last1h_thr65": 25,  # Terribly low MAP
        "lactate_latest_6h": 4.1
    }
    res_transition_update = engine.evaluate_patient(patient_transition_update)
    print(f"local transition update Result: {res_transition_update['category']}") # Should jump to VERY_LIKELY_WORSENING
