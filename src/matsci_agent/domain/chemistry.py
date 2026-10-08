"""Chemical formula helpers: validation, element sets, and detection in free text."""

from __future__ import annotations

import re

ELEMENTS = frozenset(
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As "
    "Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu "
    "Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np "
    "Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og".split()
)
# No stable isotopes: never suggested as practical candidates.
RADIOACTIVE = frozenset("Tc Pm Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())
# Sc, Y and the lanthanides: the usual definition of "rare earth elements".
RARE_EARTH = frozenset("Sc Y La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split())

_TOKEN = r"[A-Z][a-z]?\d*(?:\.\d+)?"
_FORMULA_CANDIDATE = re.compile(rf"(?<![A-Za-z0-9])(?:{_TOKEN}|\((?:{_TOKEN})+\)\d*)(?:{_TOKEN}|\((?:{_TOKEN})+\)\d*)+(?![A-Za-z0-9])")
_ELEMENT_TOKEN = re.compile(r"[A-Z][a-z]?")
# Two-element acronyms that are valid element pairs but almost never mean a compound.
_NOT_FORMULAS = frozenset({"CV", "PV", "SOC", "IV", "OCV", "NP", "PS", "HF", "IR", "SI", "NS"})


# Organic cations written as abbreviations in halide-perovskite papers, with their composition.
CATION_ABBREVIATIONS = {
    "MA": "CH6N",    # methylammonium, CH3NH3+
    "FA": "CH5N2",   # formamidinium, HC(NH2)2+
    "GA": "CH6N3",   # guanidinium, C(NH2)3+
    "EA": "C2H8N",   # ethylammonium
    "DMA": "C2H8N",  # dimethylammonium
    "PEA": "C8H12N", # phenethylammonium
    "BA": "C4H12N",  # butylammonium
}
# An abbreviation is a capital-letter run not glued to a lowercase letter, so "BAl" (B, Al) and "FAs" are untouched.
_ABBREVIATION = re.compile(r"(?<![A-Za-z])(PEA|DMA|MA|FA|GA|EA|BA)(?![a-z])")


def expand_abbreviations(formula: str) -> str:
    """'MAPbI3' -> 'CH6NPbI3', so hybrid perovskites parse as compositions."""
    return _ABBREVIATION.sub(lambda m: CATION_ABBREVIATIONS[m.group(1)], formula)


def elements_in(formula: str) -> set[str] | None:
    """Element symbols in a formula, or None if it is not a valid formula."""
    stripped = re.sub(r"[\d.()]", "", expand_abbreviations(formula))
    symbols = _ELEMENT_TOKEN.findall(stripped)
    if not symbols or "".join(symbols) != stripped or not set(symbols) <= ELEMENTS:
        return None
    return set(symbols)


_SUBSCRIPTS = str.maketrans("\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089", "0123456789")
_LATEX_SCRIPT = re.compile(r"[_^]\{([^{}]*)\}")


def clean_formula(text: str) -> str:
    """'Li$_{7}$La$_{3}$O$_{12}$' and 'Li\u2087La\u2083' -> 'Li7La3...': abstracts write formulas as LaTeX or unicode."""
    text = text.replace("$", "").translate(_SUBSCRIPTS)
    text = _LATEX_SCRIPT.sub(lambda m: m.group(1), text)
    text = re.sub(r"[_^](\w)", r"\1", text)
    return re.sub(r"\\(?:mathrm|text|rm)\{([^{}]*)\}", r"\1", text).replace("{", "").replace("}", "").strip()


# Words that name a class of materials, not one material. Plurals ("oxides") and intrinsically vague
# words ("candidate", "composition") qualify; a singular "zinc oxide" or "silicon carbide" is a compound.
_GENERIC_WORDS = frozenset(
    "material materials compound compounds electrolyte electrolytes oxides sulfides halides phosphates thiophosphates "
    "nitrides carbides glasses ceramics family families class classes candidate candidates composition compositions "
    "system systems alloys polymers phases structures semiconductors perovskite perovskites films layers cathodes "
    "anodes conductor conductors".split()
)


def looks_generic(name: str) -> bool:
    """True when a material 'name' describes a class or a whole sentence rather than one specific material."""
    words = re.findall(r"[A-Za-z]+", name.lower())
    return len(name.split()) > 4 or any(w in _GENERIC_WORDS for w in words) or "$" in name


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
