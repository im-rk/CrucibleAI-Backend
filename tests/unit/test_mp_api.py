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


def test_offline_records_are_labelled_and_never_claim_to_be_materials_project() -> None:
    client = MaterialsProjectClient(api_key=None)
    assert client.online is False
    prop = client.get_properties("GaN")[0]
    assert prop.value == 3.4
    assert "not Materials Project" in prop.source
    assert prop.material_id == "offline-reference"


def test_offline_unknown_formula_returns_nothing_instead_of_inventing_data() -> None:
    assert MaterialsProjectClient(api_key=None).get_properties("Li7La3Zr2O12") == []


def test_offline_screening_filters_by_band_gap_and_crystal_system() -> None:
    client = MaterialsProjectClient(api_key=None)
    wide = client.screen_by_criteria(bandgap_min=3.3)
    assert {r["formula"] for r in wide} >= {"GaN", "ZnO", "AlN", "MgO"}
    assert all(r["bandgap_eV"] >= 3.3 for r in wide)
    hexagonal = client.screen_by_criteria(bandgap_min=3.3, crystal_system="hexagonal")
    assert {r["formula"] for r in hexagonal} == {"GaN", "AlN", "ZnO"}


def test_live_lookup_failure_returns_nothing_rather_than_mock_data(monkeypatch) -> None:
    import sys
    import types

    class Boom:
        def __init__(self, *args, **kwargs):
            raise ConnectionError("network down")

    fake_module = types.SimpleNamespace(MPRester=Boom)
    monkeypatch.setitem(sys.modules, "mp_api.client", fake_module)
    client = MaterialsProjectClient(api_key="fake-key")
    assert client.online is True
    assert client.get_properties("GaN") == []
    assert client.screen_by_criteria(bandgap_min=1.0) == []
