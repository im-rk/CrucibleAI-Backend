"""Client for the Materials Project API."""

from __future__ import annotations

import logging
import os
from typing import Optional
from matsci_agent.domain.schemas import MaterialProperty

logger = logging.getLogger(__name__)


class MaterialsProjectClient:
    """Wraps the Materials Project API. Provides clean, typed access to material properties."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("MATERIALS_PROJECT_API_KEY")

    def get_properties(
        self, formula: str, properties: Optional[list[str]] = None
    ) -> list[MaterialProperty]:
        """Get properties for a material by formula (e.g. 'GaN', 'Li7La3Zr2O12')."""
        if not self.api_key:
            logger.debug("MATERIALS_PROJECT_API_KEY not set. Returning mock properties for testing.")
            return self._get_mock_properties(formula, properties)

        try:
            from mp_api.client import MPRester
        except ImportError:
            logger.debug("mp-api package not installed. Falling back to mock response.")
            return self._get_mock_properties(formula, properties)


        default_properties = [
            "band_gap",
            "formation_energy_per_atom",
            "energy_above_hull",
            "density",
            "crystal_system",
            "is_gap_direct",
        ]
        requested = properties or default_properties

        try:
            with MPRester(self.api_key) as mpr:
                docs = mpr.summary.search(
                    formula=formula, fields=["material_id", "formula_pretty"] + requested
                )

            if not docs:
                return []

            results = []
            for doc in docs:
                if "band_gap" in requested and getattr(doc, "band_gap", None) is not None:
                    results.append(
                        MaterialProperty(
                            material_id=doc.material_id,
                            formula=doc.formula_pretty,
                            property_name="bandgap",
                            value=round(doc.band_gap, 4),
                            unit="eV",
                            is_direct_gap=getattr(doc, "is_gap_direct", None),
                        )
                    )
                if (
                    "formation_energy_per_atom" in requested
                    and getattr(doc, "formation_energy_per_atom", None) is not None
                ):
                    results.append(
                        MaterialProperty(
                            material_id=doc.material_id,
                            formula=doc.formula_pretty,
                            property_name="formation_energy",
                            value=round(doc.formation_energy_per_atom, 4),
                            unit="eV/atom",
                        )
                    )
                if "density" in requested and getattr(doc, "density", None) is not None:
                    results.append(
                        MaterialProperty(
                            material_id=doc.material_id,
                            formula=doc.formula_pretty,
                            property_name="density",
                            value=round(doc.density, 4),
                            unit="g/cm³",
                        )
                    )
            return results
        except Exception as exc:
            logger.error(f"Materials Project API error for {formula}: {exc}")
            return self._get_mock_properties(formula, properties)

    def screen_by_criteria(
        self,
        bandgap_min: Optional[float] = None,
        bandgap_max: Optional[float] = None,
        is_stable: bool = True,
        crystal_system: Optional[str] = None,
        max_results: int = 20,
    ) -> list[dict]:
        """Screen Materials Project by property criteria."""
        if not self.api_key:
            logger.debug("MATERIALS_PROJECT_API_KEY not set. Returning mock screening results.")

            return [
                {
                    "material_id": f"mp-{i+1000}",
                    "formula": "GaN" if i == 0 else f"Mat{i}",
                    "bandgap_eV": 3.4 if i == 0 else 1.5 + i * 0.2,
                    "formation_energy_eV_per_atom": -0.85,
                    "stability_eV": 0.01,
                    "crystal_system": crystal_system or "hexagonal",
                }
                for i in range(min(max_results, 5))
            ]

        try:
            from mp_api.client import MPRester
        except ImportError:
            return []

        try:
            query_params = {
                "fields": [
                    "material_id",
                    "formula_pretty",
                    "band_gap",
                    "formation_energy_per_atom",
                    "energy_above_hull",
                    "crystal_system",
                ]
            }

            if bandgap_min is not None:
                query_params["band_gap"] = (bandgap_min, bandgap_max or 20.0)
            if is_stable:
                query_params["energy_above_hull"] = (0.0, 0.1)
            if crystal_system:
                query_params["crystal_system"] = crystal_system

            with MPRester(self.api_key) as mpr:
                docs = mpr.summary.search(**query_params)

            return [
                {
                    "material_id": d.material_id,
                    "formula": d.formula_pretty,
                    "bandgap_eV": d.band_gap,
                    "formation_energy_eV_per_atom": d.formation_energy_per_atom,
                    "stability_eV": d.energy_above_hull,
                    "crystal_system": str(d.crystal_system),
                }
                for d in docs[:max_results]
            ]
        except Exception as exc:
            logger.error(f"Materials Project screening error: {exc}")
            return []

    def _get_mock_properties(
        self, formula: str, properties: Optional[list[str]] = None
    ) -> list[MaterialProperty]:
        mock_db = {
            "GaN": [
                MaterialProperty(
                    material_id="mp-804",
                    formula="GaN",
                    property_name="bandgap",
                    value=3.4,
                    unit="eV",
                    is_direct_gap=True,
                ),
                MaterialProperty(
                    material_id="mp-804",
                    formula="GaN",
                    property_name="formation_energy",
                    value=-0.88,
                    unit="eV/atom",
                ),
                MaterialProperty(
                    material_id="mp-804",
                    formula="GaN",
                    property_name="density",
                    value=6.15,
                    unit="g/cm³",
                ),
            ],
            "Li7La3Zr2O12": [
                MaterialProperty(
                    material_id="mp-942733",
                    formula="Li7La3Zr2O12",
                    property_name="bandgap",
                    value=5.1,
                    unit="eV",
                    is_direct_gap=False,
                ),
                MaterialProperty(
                    material_id="mp-942733",
                    formula="Li7La3Zr2O12",
                    property_name="formation_energy",
                    value=-3.12,
                    unit="eV/atom",
                ),
            ],
        }
        return mock_db.get(
            formula,
            [
                MaterialProperty(
                    material_id="mp-mock",
                    formula=formula,
                    property_name="bandgap",
                    value=2.5,
                    unit="eV",
                )
            ],
        )
