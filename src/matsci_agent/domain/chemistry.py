"""Chemical formula helpers: validation, element sets, and detection in free text."""

from __future__ import annotations

import re

ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As "
    "Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu "
    "Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np "
    "Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)
# Sc, Y and the lanthanides: the usual definition of "rare earth elements".
RARE_EARTH = frozenset("Sc Y La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split())

_TOKEN = r"[A-Z][a-z]?\d*(?:\.\d+)?"
_FORMULA_CANDIDATE = re.compile(rf"(?<![A-Za-z0-9])(?:{_TOKEN}|\((?:{_TOKEN})+\)\d*)(?:{_TOKEN}|\((?:{_TOKEN})+\)\d*)+(?![A-Za-z0-9])")
_ELEMENT_TOKEN = re.compile(r"[A-Z][a-z]?")
# Two-element acronyms that are valid element pairs but almost never mean a compound.
_NOT_FORMULAS = frozenset({"CV", "PV", "SOC", "IV", "OCV", "NP", "PS", "HF", "IR", "SI", "NS"})


def elements_in(formula: str) -> set[str] | None:
    """Element symbols in a formula, or None if it is not a valid formula."""
    stripped = re.sub(r"[\d.()]", "", formula)
    symbols = _ELEMENT_TOKEN.findall(stripped)
    if not symbols or "".join(symbols) != stripped or not set(symbols) <= ELEMENTS:
        return None
    return set(symbols)


def has_rare_earth(formula: str) -> bool:
    return bool(elements_in(formula) and elements_in(formula) & RARE_EARTH)


def find_formulas(text: str) -> list[str]:
    """Distinct chemical formulas mentioned in `text`, in order of first appearance."""
    found: dict[str, None] = {}
    for match in _FORMULA_CANDIDATE.finditer(text):
        candidate = match.group()
        elems = elements_in(candidate)
        if elems and len(elems) >= 2 and candidate not in _NOT_FORMULAS:
            found.setdefault(candidate)
    return list(found)
