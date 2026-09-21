"""Property-name canonicalisation, value parsing, and unit normalisation for materials data."""

from __future__ import annotations

import re
from dataclasses import dataclass

# canonical property -> (canonical unit, name aliases). Longest alias wins on a match.
PROPERTIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "ionic_conductivity": ("S/cm", ("ionic conductivity", "li ion conductivity", "lithium ion conductivity", "conductivity")),
    "bandgap": ("eV", ("band gap", "bandgap", "optical band gap", "energy gap")),
    "formation_energy": ("eV/atom", ("formation energy", "formation energy per atom")),
    "energy_above_hull": ("eV/atom", ("energy above hull", "e hull", "hull energy")),
    "density": ("g/cm3", ("density",)),
    "electrochemical_stability_window": ("V", ("electrochemical stability window", "electrochemical window", "stability window", "electrochemical stability")),
    "activation_energy": ("eV", ("activation energy",)),
    "bulk_modulus": ("GPa", ("bulk modulus",)),
    "shear_modulus": ("GPa", ("shear modulus",)),
    "youngs_modulus": ("GPa", ("young s modulus", "youngs modulus", "elastic modulus")),
    "critical_temperature": ("K", ("critical temperature", "superconducting transition temperature", "transition temperature")),
    "power_conversion_efficiency": ("%", ("power conversion efficiency", "pce")),
    "capacity": ("mAh/g", ("specific capacity", "discharge capacity", "capacity")),
}

# Aliases too generic to match inside a longer name ("thermal conductivity", "current density").
_EXACT_ONLY = frozenset({"conductivity", "capacity", "density", "pce"})

# canonical property -> {normalised unit string: factor to the canonical unit}
_ENERGY = {"ev": 1.0, "mev": 1e-3, "kev": 1e3, "kj/mol": 0.010364, "kcal/mol": 0.043364}
_PRESSURE = {"gpa": 1.0, "mpa": 1e-3, "kpa": 1e-6, "pa": 1e-9, "tpa": 1e3}
UNIT_FACTORS: dict[str, dict[str, float]] = {
    "ionic_conductivity": {"s/cm": 1.0, "ms/cm": 1e-3, "us/cm": 1e-6, "s/m": 1e-2, "ms/m": 1e-5},
    "bandgap": _ENERGY,
    "activation_energy": _ENERGY,
    "formation_energy": {"ev/atom": 1.0, "mev/atom": 1e-3, "kj/mol": 0.010364, "kcal/mol": 0.043364},
    "energy_above_hull": {"ev/atom": 1.0, "mev/atom": 1e-3},
    "density": {"g/cm3": 1.0, "kg/m3": 1e-3, "mg/cm3": 1e-3},
    "electrochemical_stability_window": {"v": 1.0, "mv": 1e-3},
    "bulk_modulus": _PRESSURE,
    "shear_modulus": _PRESSURE,
    "youngs_modulus": _PRESSURE,
    "critical_temperature": {"k": 1.0},  # Celsius handled separately (offset, not a factor)
    "power_conversion_efficiency": {"%": 1.0, "percent": 1.0},
    "capacity": {"mah/g": 1.0, "ah/kg": 1.0},
}

# Regexes for finding 'number unit' pairs in running text (shared by extraction and screening).
_POWER_DIGITS = r"[-−+]?[\d⁻⁺⁰¹²³⁴⁵⁶⁷⁸⁹]+"
VALUE_PATTERN = (
    r"(?P<val>[<>~≈≥≤]?\s*(?:"
    r"10\s*(?:\^\s*\{?" + _POWER_DIGITS + r"\}?|[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+)"  # bare power: 10^-3, 10⁻³
    r"|\d+(?:\.\d+)?(?:\s*[–-]\s*\d+(?:\.\d+)?)?"  # number or range
    r"(?:\s*[×xX·*]\s*10\s*\^?\s*\{?" + _POWER_DIGITS + r"\}?|[eE][-+]?\d+)?"  # x 10^n or e-notation
    r"))"
)
UNIT_PATTERNS = {
    "ionic_conductivity": r"(?:[mµμu]?S\s?/\s?cm|[mµμu]?S\s?cm(?:\^?-1|⁻¹)|S\s?/\s?m)",
    "bandgap": r"m?eV",
    "formation_energy": r"(?:eV/atom|meV/atom|kJ/mol)",
    "density": r"(?:g\s?/\s?cm(?:3|³)|g\s?cm(?:\^?-3|⁻³)|kg\s?/\s?m(?:3|³))",
    "electrochemical_stability_window": r"V",
    "activation_energy": r"m?eV",
    "critical_temperature": r"K",
    "power_conversion_efficiency": r"%",
    "capacity": r"(?:mAh\s?/\s?g|mAh\s?g(?:\^?-1|⁻¹))",
}
_SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")
_SUPERSCRIPT_RUN = re.compile(r"[⁻⁺]?[⁰¹²³⁴⁵⁶⁷⁸⁹]+")
_NUM = r"\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_SCI = re.compile(rf"({_NUM})?\s*[×xX*·]\s*10\s*\^?\s*\{{?\s*([-+]?\d+)\s*\}}?")
_BARE_POWER = re.compile(r"(?<![\d.])10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?")
_RANGE = re.compile(rf"({_NUM})\s*(?:–|—|-|to)\s*({_NUM})")
_PLAIN = re.compile(_NUM)
_QUALIFIERS = (
    (re.compile(r"^\s*(?:>=|≥|>|above|over|greater than|at least)"), ">"),
    (re.compile(r"^\s*(?:<=|≤|<|below|under|less than|at most)"), "<"),
    (re.compile(r"^\s*(?:~|≈|about|approximately|around)"), "~"),
)


@dataclass(frozen=True)
class ParsedValue:
    value: float | None  # point estimate; midpoint for ranges
    low: float | None = None
    high: float | None = None
    qualifier: str = ""  # "", ">", "<", "~"


def parse_value(raw: str) -> ParsedValue:
    """Parse '3.4 × 10⁻³', '2–4', '> 1e-3', '3.4 ± 0.1' and friends into numbers."""
    # Real values are short; the cap bounds the regexes' worst case on hostile input.
    text = _SUPERSCRIPT_RUN.sub(lambda m: "^" + m.group().translate(_SUPERSCRIPTS), raw.strip()[:200])
    text = text.replace("−", "-").replace(" ", " ")
    qualifier = ""
    for pattern, symbol in _QUALIFIERS:
        match = pattern.match(text)
        if match:
            qualifier, text = symbol, text[match.end():]
            break
    text = text.split("±")[0]

    exponent = None
    sci = _SCI.search(text)
    if sci:
        exponent = int(sci.group(2))
        head = text[: sci.start()] + (sci.group(1) or "1") + text[sci.end():]
    elif _BARE_POWER.search(text):
        bare = _BARE_POWER.search(text)
        exponent = int(bare.group(1))
        head = text[: bare.start()] + "1" + text[bare.end():]
    else:
        head = text
    scale = 10.0 ** exponent if exponent is not None else 1.0

    negative = re.match(r"\s*-\s*(?=\d)", head)  # '-0.88' is a value, not a range
    if negative:
        head = head[negative.end():]
    ranged = None if negative else _RANGE.search(head)
    if ranged:
        low, high = float(ranged.group(1)) * scale, float(ranged.group(2)) * scale
        return ParsedValue((low + high) / 2, low, high, qualifier or "~")
    number = _PLAIN.search(head)
    if not number:
        return ParsedValue(None, qualifier=qualifier)
    value = float(number.group()) * scale
    return ParsedValue(-value if negative else value, qualifier=qualifier)


def canonical_property(name: str) -> str:
    """Map free-text property names ('ionic conductivity', 'Band-gap') to canonical keys."""
    cleaned = re.sub(r"[^a-z0-9]+", " ", name.lower().replace("'", " ")).strip()
    if cleaned.replace(" ", "_") in PROPERTIES:
        return cleaned.replace(" ", "_")
    matches = [
        (len(alias), key)
        for key, (_, aliases) in PROPERTIES.items()
        for alias in aliases
        if alias == cleaned
        or (alias not in _EXACT_ONLY and re.search(rf"\b{re.escape(alias)}\b", cleaned))
    ]
    if matches:
        return max(matches)[1]
    return cleaned.replace(" ", "_") or "unknown"


def _normalise_unit(unit: str) -> str:
    u = unit.strip().lower().replace("μ", "u").replace("µ", "u")
    u = u.translate(_SUPERSCRIPTS).replace("·", " ").replace("*", " ").replace("^", "")
    u = re.sub(r"\s+", " ", u).replace("°", "")
    return re.sub(
        r"^([a-z]+) ?([a-z]+)-(\d)$", lambda m: f"{m[1]}/{m[2]}{'' if m[3] == '1' else m[3]}", u
    )


def _tidy(value: float | None) -> float | None:
    """Drop float noise from unit conversion (1.64 * 1.0 -> 1.6400000000000001)."""
    return None if value is None else float(f"{value:.10g}")


def normalise(property_name: str, value: float | None, unit: str) -> tuple[str, float | None, str, bool]:
    """Return (canonical property, value, unit, converted?) with the value in the canonical unit.

    Unknown properties or units are returned unchanged with converted=False, never guessed.
    """
    prop = canonical_property(property_name)
    if prop not in PROPERTIES:
        return prop, value, unit, False
    canonical_unit = PROPERTIES[prop][0]
    norm = _normalise_unit(unit)
    if prop == "critical_temperature" and norm in {"c", "celsius"}:
        return prop, _tidy(None if value is None else value + 273.15), canonical_unit, True
    factor = UNIT_FACTORS.get(prop, {}).get(norm)
    if factor is None:
        return prop, value, unit, False
    return prop, _tidy(None if value is None else value * factor), canonical_unit, True
