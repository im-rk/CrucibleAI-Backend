"""Unit tests for MaterialsProjectClient."""

from __future__ import annotations

from matsci_agent.infrastructure.clients.mp_api import MaterialsProjectClient


def test_mp_client_get_properties_fallback() -> None:
    client = MaterialsProjectClient(api_key=None)
    props = client.get_properties("GaN")
    assert len(props) > 0
    assert props[0].formula == "GaN"
    assert any(p.property_name == "bandgap" for p in props)


def test_mp_client_screen_by_criteria_fallback() -> None:
    client = MaterialsProjectClient(api_key=None)
    results = client.screen_by_criteria(bandgap_min=1.0, bandgap_max=4.0, max_results=3)
    assert len(results) > 0
    assert "bandgap_eV" in results[0]
