"""Unit tests for MaterialsProjectClient."""

from __future__ import annotations

import sys

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


# ---- live-client behaviour against a fake MPRester shaped like real responses -----------------

import types  # noqa: E402

from matsci_agent.infrastructure.clients.mp_api import (  # noqa: E402
    SERVER_EXCLUDE,
    _practical,
    ground_states,
)


def doc(formula, mp_id, hull, gap=2.0, system="Hexagonal", nelements=2):
    return types.SimpleNamespace(
        material_id=mp_id, formula_pretty=formula, energy_above_hull=hull, band_gap=gap,
        is_gap_direct=True, formation_energy_per_atom=-0.5, density=5.0, nelements=nelements,
        symmetry=types.SimpleNamespace(crystal_system=types.SimpleNamespace(value=system)),
    )


def fake_rester(monkeypatch, docs, captured):
    class Summary:
        def search(self, **query):
            captured.append(query)
            return docs

    class FakeMPRester:
        def __init__(self, key):
            self.materials = types.SimpleNamespace(summary=Summary())

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(sys.modules, "mp_api.client", types.SimpleNamespace(MPRester=FakeMPRester))


def test_ground_states_keep_the_most_stable_polymorph_per_formula() -> None:
    docs = [doc("GaN", "mp-830", 0.007), doc("GaN", "mp-804", 0.0), doc("ZnO", "mp-2133", 0.0), doc("GaN", "mp-1", 0.29)]
    assert [d.material_id for d in ground_states(docs)] == ["mp-804", "mp-2133"]


def test_lookup_returns_only_the_ground_state_and_never_requests_invalid_fields(monkeypatch) -> None:
    captured: list = []
    fake_rester(monkeypatch, [doc("GaN", "mp-1", 0.29, gap=0.8), doc("GaN", "mp-804", 0.0, gap=1.74)], captured)
    props = {p.property_name: p for p in MaterialsProjectClient(api_key="k").get_properties("GaN")}

    assert {p.material_id for p in props.values()} == {"mp-804"}  # not the unstable polymorph
    assert props["bandgap"].value == 1.74 and props["bandgap"].is_direct_gap is True
    assert props["crystal_system"].value == "Hexagonal"
    assert props["bandgap"].doi == "https://doi.org/10.1063/1.4812323"
    # Regression: 'crystal_system' is not a summary field; asking for it made every live call fail.
    assert "crystal_system" not in captured[0]["fields"] and "symmetry" in captured[0]["fields"]


def test_screening_asks_for_experimental_structures_within_the_api_limits(monkeypatch) -> None:
    captured: list = []
    docs = [
        doc("AlN", "mp-661", 0.0, gap=4.05), doc("AcBr3", "mp-9", 0.0, gap=4.1),
        doc("Ar", "mp-23", 0.0, gap=8.5, nelements=1), doc("Al2O3", "mp-1143", 0.0, gap=5.9, nelements=2),
        doc("ThO2", "mp-643", 0.02, gap=4.4), doc("AlN", "mp-2", 0.05, gap=5.0),
    ]
    fake_rester(monkeypatch, docs, captured)
    rows = MaterialsProjectClient(api_key="k").screen_by_criteria(bandgap_min=3.4, crystal_system="hexagonal", max_results=5)

    query = captured[0]
    assert query["theoretical"] is False and query["band_gap"] == (3.4, 20.0)
    assert query["crystal_system"] == "Hexagonal"
    assert len(",".join(query["exclude_elements"])) <= 60  # the API rejects longer lists
    assert "crystal_system" not in query["fields"]
    # radioactive (Ac, Th) and elemental gases (Ar) are dropped; AlN is its ground state only
    assert [(r["formula"], r["material_id"]) for r in rows] == [("AlN", "mp-661"), ("Al2O3", "mp-1143")]
    assert rows[0]["crystal_system"] == "Hexagonal"


def test_practical_filter() -> None:
    assert _practical("GaN") and _practical("Si") and _practical("AlSb")
    assert not _practical("AcBr3") and not _practical("UO2") and not _practical("Ar") and not _practical("Br")
    assert set(SERVER_EXCLUDE) <= {"Ac", "Th", "U", "Pa", "Np", "Pu", "Tc", "Pm", "Am", "Cm"}
