"""Client for the Materials Project API.

Without an API key the client serves a small offline reference table of experimental band
gaps (see `domain/reference_data.py`), clearly labelled as such. It never invents values:
unknown formulas return nothing, and a failed live call returns nothing rather than mock data.

Materials Project stores many polymorphs per formula (44 for GaN), most of them hypothetical
and unstable. Lookups return only the ground state (lowest energy above hull); otherwise the
screener, which takes the most favourable reported value, could pick a structure that does
not exist in practice.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from matsci_agent.core.config import settings
from matsci_agent.domain.chemistry import RADIOACTIVE, elements_in
from matsci_agent.domain.reference_data import OFFLINE_BANDGAPS, OFFLINE_SOURCE
from matsci_agent.domain.schemas import MaterialProperty

logger = logging.getLogger(__name__)

# `crystal_system` is not a top-level summary field; it lives under `symmetry`.
SUMMARY_FIELDS = [
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "density",
    "symmetry",
    "is_gap_direct",
]
MP_DOI = "https://doi.org/10.1063/1.4812323"  # Jain et al., APL Materials 2013
# The API caps exclude_elements at 60 characters, so only the radioactive elements that are common in
# Materials Project are excluded server-side; the full set is filtered client-side.
# Elements that are gases or molecular solids under ambient conditions (solid Ar, Br2 ...).
NON_SOLID_ELEMENTS = frozenset("H He N O F Ne Cl Ar Br Kr I Xe".split())
SERVER_EXCLUDE = ["Ac", "Th", "U", "Pa", "Np", "Pu", "Tc", "Pm", "Am", "Cm"]


def _summary(mpr: Any) -> Any:
    """The summary endpoint; newer clients moved it under `materials` (the old path is deprecated)."""
    materials = getattr(mpr, "materials", None)
    return getattr(materials, "summary", None) or mpr.summary


def _crystal_system(doc: Any) -> Optional[str]:
    system = getattr(getattr(doc, "symmetry", None), "crystal_system", None)
    return None if system is None else str(getattr(system, "value", system))


def _hull(doc: Any) -> float:
    value = getattr(doc, "energy_above_hull", None)
    return float("inf") if value is None else float(value)


def ground_states(docs: list[Any]) -> list[Any]:
    """The most stable entry per formula (lowest energy above hull), in input order of formulas."""
    best: dict[str, Any] = {}
    for doc in docs:
        key = doc.formula_pretty
        if key not in best or _hull(doc) < _hull(best[key]):
            best[key] = doc
    return list(best.values())


def _practical(formula: str) -> bool:
    """No radioactive elements, and not a gas or molecular solid in elemental form."""
    elements = elements_in(formula) or set()
    if elements & RADIOACTIVE:
        return False
    return not (len(elements) == 1 and elements <= NON_SOLID_ELEMENTS)


class MaterialsProjectClient:
    """Wraps the Materials Project API. Provides clean, typed access to material properties."""

    def __init__(self, api_key: Optional[str] = settings.materials_project_api_key):
        self.api_key = api_key

    @property
    def online(self) -> bool:
        """True when backed by the live Materials Project API rather than the offline table."""
        return bool(self.api_key)

    def _search(self, **query: Any) -> list[Any]:
        from mp_api.client import MPRester

        with MPRester(self.api_key) as mpr:
            return list(_summary(mpr).search(**query))

    def get_properties(
        self, formula: str, properties: Optional[list[str]] = None
    ) -> list[MaterialProperty]:
        """Properties of the ground-state entry for a formula (e.g. 'GaN', 'Li7La3Zr2O12')."""
        if not self.online:
            return self._offline_properties(formula)

        requested = properties or SUMMARY_FIELDS
        try:
            docs = self._search(
                formula=formula,
                fields=list(dict.fromkeys(["material_id", "formula_pretty", "energy_above_hull", *requested])),
            )
        except Exception as exc:
            logger.error(f"Materials Project API error for {formula}: {exc}")
            return []

        results: list[MaterialProperty] = []
        for doc in ground_states(docs):
            base: dict[str, Any] = {
                "material_id": str(doc.material_id), "formula": doc.formula_pretty, "doi": MP_DOI,
            }
            if "band_gap" in requested and getattr(doc, "band_gap", None) is not None:
                results.append(MaterialProperty(
                    **base, property_name="bandgap", value=round(doc.band_gap, 4), unit="eV",
                    is_direct_gap=getattr(doc, "is_gap_direct", None),
                ))
            if "formation_energy_per_atom" in requested and getattr(doc, "formation_energy_per_atom", None) is not None:
                results.append(MaterialProperty(
                    **base, property_name="formation_energy",
                    value=round(doc.formation_energy_per_atom, 4), unit="eV/atom",
                ))
            if "energy_above_hull" in requested and getattr(doc, "energy_above_hull", None) is not None:
                results.append(MaterialProperty(
                    **base, property_name="energy_above_hull",
                    value=round(doc.energy_above_hull, 4), unit="eV/atom",
                ))
            if "density" in requested and getattr(doc, "density", None) is not None:
                results.append(MaterialProperty(
                    **base, property_name="density", value=round(doc.density, 4), unit="g/cm3",
                ))
            crystal = _crystal_system(doc)
            if "symmetry" in requested and crystal:
                results.append(MaterialProperty(**base, property_name="crystal_system", value=crystal, unit=""))
        return results

    def screen_by_criteria(
        self,
        bandgap_min: Optional[float] = None,
        bandgap_max: Optional[float] = None,
        is_stable: bool = True,
        crystal_system: Optional[str] = None,
        max_results: int = 20,
    ) -> list[dict]:
        """Screen materials by band gap / stability / crystal system (one row per formula).

        Only experimentally known structures (`theoretical=False`) without radioactive elements are
        considered. The API cannot sort, so a large page is fetched and ranked here: most stable
        first, then simplest (fewest elements). Without that, results come back alphabetically
        (actinium compounds first).
        """
        if not self.online:
            return self._offline_screen(bandgap_min, bandgap_max, crystal_system, max_results)

        query: dict[str, Any] = {
            "fields": [
                "material_id", "formula_pretty", "band_gap", "formation_energy_per_atom",
                "energy_above_hull", "symmetry", "nelements",
            ],
            "theoretical": False,
            "exclude_elements": SERVER_EXCLUDE,
            "num_chunks": 1,
            "chunk_size": 1000,
        }
        if bandgap_min is not None or bandgap_max is not None:
            query["band_gap"] = (bandgap_min or 0.0, bandgap_max or 20.0)
        if is_stable:
            query["energy_above_hull"] = (0.0, 0.1)
        if crystal_system:
            query["crystal_system"] = crystal_system.capitalize()  # a valid filter, unlike the field
        try:
            docs = self._search(**query)
        except Exception as exc:
            logger.error(f"Materials Project screening error: {exc}")
            return []
        # ponytail: the API cannot sort, so the 1000-row page is biased toward early-alphabet formulas;
        # fetch several pages (or randomise the offset) if seed diversity ever matters more than latency.
        safe = [d for d in ground_states(docs) if _practical(d.formula_pretty)]
        rows = sorted(safe, key=lambda d: (_hull(d), getattr(d, "nelements", 99) or 99))[:max_results]
        return [
            {
                "material_id": str(d.material_id),
                "formula": d.formula_pretty,
                "bandgap_eV": d.band_gap,
                "formation_energy_eV_per_atom": d.formation_energy_per_atom,
                "stability_eV": d.energy_above_hull,
                "crystal_system": _crystal_system(d),
                "source": "Materials Project",
            }
            for d in rows
        ]

    # ---- offline reference table -----------------------------------------------------

    def _offline_properties(self, formula: str) -> list[MaterialProperty]:
        entry = OFFLINE_BANDGAPS.get(formula)
        if entry is None:
            return []
        gap, direct, _ = entry
        return [MaterialProperty(
            material_id="offline-reference", formula=formula, property_name="bandgap",
            value=gap, unit="eV", source=OFFLINE_SOURCE, doi="", is_direct_gap=direct,
            confidence="medium",
        )]

    def _offline_screen(
        self, gap_min: Optional[float], gap_max: Optional[float],
        crystal_system: Optional[str], max_results: int,
    ) -> list[dict]:
        rows = [
            {
                "material_id": "offline-reference",
                "formula": formula,
                "bandgap_eV": gap,
                "formation_energy_eV_per_atom": None,
                "stability_eV": None,
                "crystal_system": system,
                "source": OFFLINE_SOURCE,
            }
            for formula, (gap, _, system) in OFFLINE_BANDGAPS.items()
            if (gap_min is None or gap >= gap_min)
            and (gap_max is None or gap <= gap_max)
            and (crystal_system is None or system == crystal_system.lower())
        ]
        return sorted(rows, key=lambda r: r["bandgap_eV"])[:max_results]
