"""Adapter that bridges canonical fact keys to existing FACTS SQL retrieval."""

from __future__ import annotations

import math
from typing import Any

from jinja2 import Template

from facts_registry import FACT_SPECS


class FactsAdapter:
    def __init__(self, base_facts_engine: Any):
        # base_facts_engine is the shared FactsGenerator implementation.
        self.base = base_facts_engine

    def fetch_values(self, stay_id: int, hr: int, fact_keys: list[str]) -> dict[str, Any]:
        mapped = []
        seen_cols = set()
        for k in fact_keys:
            spec = FACT_SPECS.get(k)
            if not spec:
                continue
            if spec.col in seen_cols:
                continue
            seen_cols.add(spec.col)
            mapped.append(
                {
                    "col": spec.col,
                    "display": spec.display,
                    "meaning": spec.meaning,
                    "unit": spec.unit,
                }
            )

        if not mapped:
            return {}

        row = self.base._query_indicators(stay_id, hr, mapped)  # noqa: SLF001 (intentional adapter call)
        if not isinstance(row, dict):
            return {}

        out: dict[str, Any] = {}
        for k in fact_keys:
            spec = FACT_SPECS.get(k)
            if not spec:
                continue
            out[k] = row.get(spec.col)
        if "_db_error" in row:
            out["_db_error"] = row["_db_error"]
        return out

    @staticmethod
    def _fmt(v: Any) -> str:
        if v is None:
            return "N/A"
        if isinstance(v, bool):
            return "1" if v else "0"
        if isinstance(v, float):
            if math.isnan(v):
                return "N/A"
            if v.is_integer():
                return str(int(v))
            return f"{v:.3f}".rstrip("0").rstrip(".")
        return str(v)

    def render_report(self, fact_keys: list[str], values: dict[str, Any]) -> str:
        rows = []
        for k in fact_keys:
            spec = FACT_SPECS.get(k)
            if not spec:
                continue
            rows.append(
                {
                    "key": k,
                    "display": spec.display,
                    "meaning": spec.meaning,
                    "unit": spec.unit,
                    "value": self._fmt(values.get(k)),
                }
            )

        tmpl = Template(
            """### FACTS Objective Report (CARE)
| Key | Indicator | Clinical Meaning | Value |
| :--- | :--- | :--- | :--- |
{% for r in rows %}| `{{ r.key }}` | {{ r.display }} | {{ r.meaning }} | **{{ r.value }}** {{ r.unit }} |
{% endfor %}
> `N/A` means unknown; treat as missing, not normal.
"""
        )
        return tmpl.render(rows=rows)
