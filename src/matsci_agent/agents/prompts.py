"""Central place for system prompts and node instructions."""

from __future__ import annotations

SEARCH_SYSTEM = (
    "You are a materials-science literature search expert. Be precise and concrete; "
    "prefer specific material names, formulas, properties, and measurement techniques."
)

GENERATE_QUERIES_PROMPT = """Research goal: {goal}

Generate 3 to 5 search queries that would find the most relevant materials science papers.
Consider: specific material names or formulas, the key property being sought, the
application domain, and the experimental or computational technique used to measure it.
Each query should be a short phrase, not a sentence."""

RELEVANCE_PROMPT = """Research goal: {goal}

Below are numbered passages retrieved from papers. Score each one from 0.0 to 1.0 for how
relevant it is to the research goal:
- 1.0: directly reports data or findings that bear on the goal
- 0.5: related topic or method but does not address the goal itself
- 0.0: unrelated

Return one score for every passage, using its number.

{passages}"""

REFORMULATE_PROMPT = """Research goal: {goal}

Queries already tried (do not repeat them):
{tried}

The results so far were weak. The few relevant passages found were about:
{found}

Write ONE improved search query that is likely to surface different, more relevant papers."""

EXTRACTION_SYSTEM = (
    "You extract structured materials-science data from paper excerpts. You only report values "
    "explicitly stated in the excerpt and never guess or infer numbers."
)

EXTRACTION_PROMPT = """Paper: {title}
DOI: {doi}

Excerpt:
{text}

Extract every material property value reported in this excerpt. For each one return:
- material_name: the material as written in the text
- material_formula: its chemical formula if given, else null
- property_name: snake_case name, e.g. ionic_conductivity, bandgap, formation_energy,
  electrochemical_stability_window, density, activation_energy
- value_raw: the value exactly as written (e.g. "3.4 x 10^-3")
- value_numeric: that value as a single float in the units as written, or null if it is a range
- unit: the unit as written
- conditions: measurement conditions (temperature, method, synthesis route), or ""
- confidence: "high" if the number is stated explicitly for this material, "medium" if it
  needed interpretation, "low" if uncertain
- excerpt: the exact sentence the value came from

If the excerpt reports no property values, return an empty list."""
