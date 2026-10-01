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

EXTRACTION_PROMPT = """Numbered passages from papers:

{passages}

Extract every material property value reported in these passages. For each one return:
- passage: the number of the passage the value comes from
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

If the passages report no property values, return an empty list."""

CRITERIA_SYSTEM = (
    "You convert a materials-research goal into precise, checkable screening criteria. "
    "Only include requirements the researcher actually stated."
)

CRITERIA_PROMPT = """Research goal: {goal}

List every requirement in the goal as a criterion:
- kind "numeric": a threshold or range on a measurable property. Give property_name (e.g.
  ionic_conductivity, bandgap, electrochemical_stability_window, density, formation_energy),
  operator (">", ">=", "<", "<=", "between"), value (and value_high for "between") exactly as
  written, and unit as written (e.g. "S/cm", "eV", "V").
- kind "exclude_elements": the material must not contain certain elements. List their chemical
  symbols in `elements` ("no rare earth elements" -> the rare earths: Sc, Y, La, Ce, Pr, Nd, Pm,
  Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu).
- kind "measured_evidence": the material must be demonstrated experimentally, not only predicted.
- kind "qualitative": anything that cannot be checked from numbers (e.g. air stability).

`label` is the requirement in plain words. Use null for fields that do not apply."""

HYPOTHESIS_SYSTEM = (
    "You are a materials scientist who proposes testable, evidence-grounded hypotheses. Every "
    "hypothesis must cite the evidence items that support it and must be falsifiable by a "
    "concrete experiment. Never invent evidence or numbers."
)

HYPOTHESIS_PROMPT = """Research goal: {goal}

Screening result ({n_candidates} candidates):
{candidates}

Evidence items (cite by id, e.g. E1 or R2):
{evidence}

Write 3 to 5 hypotheses about WHY the top candidates satisfy (or fail) the goal, based on
patterns you can see across them (shared structure, chemistry, dopants, trade-offs).
{scarcity_instruction}
For each hypothesis give:
- statement: one or two sentences
- evidence_ids: the ids of the evidence items that support it (at least one)
- suggested_experiment: a concrete experiment or calculation that would test it
- confidence: "high", "medium" or "low", based on how directly the evidence supports it

Also list knowledge_gaps: important things the goal needs that the evidence does not show."""

SCARCITY_INSTRUCTION = (
    "Fewer than 3 candidates were found, so make the hypotheses about WHY candidates are scarce "
    "(which criteria are hardest to satisfy together) and what would widen the search."
)
