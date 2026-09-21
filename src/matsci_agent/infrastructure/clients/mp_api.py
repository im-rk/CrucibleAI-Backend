"""Client for the Materials Project API.

Without an API key the client serves a small offline reference table of experimental band
gaps (see `domain/reference_data.py`), clearly labelled as such. It never invents values:
unknown formulas return nothing, and a failed live call returns nothing rather than mock data.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from matsci_agent.core.config import settings
from matsci_agent.domain.reference_data import OFFLINE_BANDGAPS, OFFLINE_SOURCE
from matsci_agent.domain.schemas import MaterialProperty

logger = logging.getLogger(__name__)

SUMMARY_FIELDS = [
    "band_gap",
    "formation_energy_per_atom",
    "energy_above_hull",
    "density",
    "crystal_system",
    "is_gap_direct",
]


class MaterialsProjectClient:
    """Wraps the Materials Project API. Provides clean, typed access to material properties."""

    def __init__(self, api_key: Optional[str] = settings.materials_project_api_key):
        self.api_key = api_key

    @property
    def online(self) -> bool:
        """True when backed by the live Materials Project API rather than the offline table."""
        return bool(self.api_key)

    def get_properties(
        self, formula: str, properties: Optional[list[str]] = None
    ) -> list[MaterialProperty]:
        """Get properties for a material by formula (e.g. 'GaN', 'Li7La3Zr2O12')."""
        if not self.online:
            return self._offline_properties(formula)

        requested = properties or SUMMARY_FIELDS
        try:
            from mp_api.client import MPRester

            with MPRester(self.api_key) as mpr:
                docs = mpr.summary.search(
                    formula=formula, fields=["material_id", "formula_pretty"] + requested
                )
        except Exception as exc:
            logger.error(f"Materials Project API error for {formula}: {exc}")
            return []

        results: list[MaterialProperty] = []
        for doc in docs or []:
            base: dict[str, Any] = {"material_id": str(doc.material_id), "formula": doc.formula_pretty}
            if "band_gap" in requested and getattr(doc, "band_gap", None) is not None:
                results.append(MaterialProperty(
                    **base, property_name="bandgap", value=round(doc.band_gap, 4), unit="eV",
                    is_direct_gap=getattr(doc, "is_gap_direct", None),
                ))
            if getattr(doc, "formation_energy_per_atom", None) is not None and (
                "formation_energy_per_atom" in requested
            ):
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
            if "crystal_system" in requested and getattr(doc, "crystal_system", None) is not None:
                crystal = doc.crystal_system
                results.append(MaterialProperty(
                    **base, property_name="crystal_system",
                    value=str(getattr(crystal, "value", crystal)), unit="",
                ))
        return results

    def screen_by_criteria(
        self,
        bandgap_min: Optional[float] = None,
        bandgap_max: Optional[float] = None,
        is_stable: bool = True,
        crystal_system: Optional[str] = None,
        max_results: int = 20,
    ) -> list[dict]:
        """Screen materials by band gap / stability / crystal system."""
        if not self.online:
            return self._offline_screen(bandgap_min, bandgap_max, crystal_system, max_results)

        query: dict[str, Any] = {
            "fields": [
                "material_id", "formula_pretty", "band_gap", "formation_energy_per_atom",
                "energy_above_hull", "crystal_system",
            ]
        }
        if bandgap_min is not None or bandgap_max is not None:
            query["band_gap"] = (bandgap_min or 0.0, bandgap_max or 20.0)
        if is_stable:
            query["energy_above_hull"] = (0.0, 0.1)
        if crystal_system:
            query["crystal_system"] = crystal_system
        try:
            from mp_api.client import MPRester

            with MPRester(self.api_key) as mpr:
                docs = mpr.summary.search(**query)
        except Exception as exc:
            logger.error(f"Materials Project screening error: {exc}")
            return []
        return [
            {
                "material_id": str(d.material_id),
                "formula": d.formula_pretty,
                "bandgap_eV": d.band_gap,
                "formation_energy_eV_per_atom": d.formation_energy_per_atom,
                "stability_eV": d.energy_above_hull,
                "crystal_system": str(getattr(d.crystal_system, "value", d.crystal_system)),
                "source": "Materials Project",
            }
            for d in (docs or [])[:max_results]
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
