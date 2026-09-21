"""Direct access to the Materials Project client and the KRR predictor."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from matsci_agent.api.dependencies import get_krr_predictor, get_mp_client
from matsci_agent.domain.chemistry import elements_in

router = APIRouter(tags=["materials"])


class PredictRequest(BaseModel):
    formula: str = Field(min_length=1, max_length=100)
    property_name: str = "band_gap"


def _require_formula(formula: str) -> str:
    formula = formula.strip()
    if not elements_in(formula):
        raise HTTPException(status_code=422, detail=f"'{formula}' is not a valid chemical formula.")
    return formula


@router.post("/predict")
async def predict_property(body: PredictRequest) -> dict:
    """Kernel Ridge Regression estimate for one property of one formula."""
    formula = _require_formula(body.formula)
    result = await asyncio.to_thread(get_krr_predictor().predict, formula, body.property_name)
    if "error" in result:
        raise HTTPException(status_code=422, detail=result["error"])
    return result


@router.get("/materials/lookup")
async def lookup_material(formula: str = Query(..., min_length=1, max_length=100)) -> dict:
    """Look a formula up in Materials Project (or the labelled offline reference table)."""
    formula = _require_formula(formula)
    client = get_mp_client()
    props = await asyncio.to_thread(client.get_properties, formula)
    return {
        "formula": formula,
        "source": "Materials Project" if client.online else "offline reference table",
        "properties": [p.model_dump(exclude={"doi"} if not p.doi else set()) for p in props],
    }


@router.get("/materials/screen")
async def screen_materials(
    bandgap_min: float | None = Query(None, ge=0, le=20),
    bandgap_max: float | None = Query(None, ge=0, le=20),
    crystal_system: str | None = Query(None, max_length=30),
    limit: int = Query(20, ge=1, le=100),
) -> dict:
    """Screen materials by band gap and crystal system."""
    client = get_mp_client()
    rows = await asyncio.to_thread(
        client.screen_by_criteria, bandgap_min, bandgap_max, True, crystal_system, limit
    )
    return {"source": "Materials Project" if client.online else "offline reference table", "results": rows}
