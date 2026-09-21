"""Unit tests for value parsing, unit normalisation, and chemical formula helpers."""

from __future__ import annotations

import pytest

from matsci_agent.domain.chemistry import elements_in, find_formulas, has_rare_earth
from matsci_agent.domain.units import canonical_property, normalise, parse_value


@pytest.mark.parametrize(
    "raw, value",
    [
        ("3.4 × 10⁻³", 3.4e-3),
        ("3.4×10^-3", 3.4e-3),
        ("3.4 x 10-3", 3.4e-3),
        ("2.1 · 10^{-4}", 2.1e-4),
        ("1e-3", 1e-3),
        ("10^-3", 1e-3),
        ("10⁻³", 1e-3),
        ("3.4 ± 0.1", 3.4),
        ("-0.88", -0.88),
        ("−0.88", -0.88),
        ("42", 42.0),
    ],
)
def test_parse_value_handles_scientific_notation_and_signs(raw: str, value: float) -> None:
    assert parse_value(raw).value == pytest.approx(value)


def test_parse_value_ranges_use_midpoint_and_keep_bounds() -> None:
    parsed = parse_value("2–4")
    assert (parsed.value, parsed.low, parsed.high, parsed.qualifier) == (3.0, 2.0, 4.0, "~")
    scaled = parse_value("1–5 × 10^-3")
    assert scaled.value == pytest.approx(3e-3) and scaled.high == pytest.approx(5e-3)


@pytest.mark.parametrize("raw, qualifier", [("> 1e-3", ">"), ("≥ 5", ">"), ("< 0.1", "<"), ("≈ 5", "~"), ("about 5", "~")])
def test_parse_value_keeps_qualifiers(raw: str, qualifier: str) -> None:
    assert parse_value(raw).qualifier == qualifier


def test_parse_value_returns_none_when_there_is_no_number() -> None:
    assert parse_value("n/a").value is None


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Ionic conductivity", "ionic_conductivity"),
        ("Li-ion conductivity", "ionic_conductivity"),
        ("room temperature ionic conductivity", "ionic_conductivity"),
        ("conductivity", "ionic_conductivity"),
        ("band_gap", "bandgap"),
        ("Band-gap", "bandgap"),
        ("Young's modulus", "youngs_modulus"),
        ("electrochemical stability window", "electrochemical_stability_window"),
        # generic aliases must not swallow different properties
        ("thermal conductivity", "thermal_conductivity"),
        ("current density", "current_density"),
    ],
)
def test_canonical_property(name: str, expected: str) -> None:
    assert canonical_property(name) == expected


@pytest.mark.parametrize(
    "prop, value, unit, expected",
    [
        ("ionic_conductivity", 2.0, "mS/cm", 2e-3),
        ("ionic_conductivity", 100.0, "S/m", 1.0),
        ("ionic_conductivity", 5.0, "µS/cm", 5e-6),
        ("ionic_conductivity", 3.0, "mS cm⁻¹", 3e-3),
        ("ionic_conductivity", 1.0, "S cm-1", 1.0),
        ("bandgap", 3400.0, "meV", 3.4),
        ("formation_energy", -96.485, "kJ/mol", -1.0),
        ("bulk_modulus", 150000.0, "MPa", 150.0),
        ("density", 6150.0, "kg/m3", 6.15),
        ("critical_temperature", 92.0, "K", 92.0),
        ("critical_temperature", 20.0, "°C", 293.15),
    ],
)
def test_normalise_converts_to_canonical_units(prop, value, unit, expected) -> None:
    _, converted, _, ok = normalise(prop, value, unit)
    assert ok and converted == pytest.approx(expected, rel=1e-3)


def test_normalise_reports_canonical_unit() -> None:
    assert normalise("Ionic conductivity", 1.0, "mS/cm")[0::2] == ("ionic_conductivity", "S/cm")


def test_normalise_never_guesses_unknown_units_or_properties() -> None:
    assert normalise("ionic_conductivity", 5.0, "furlongs") == ("ionic_conductivity", 5.0, "furlongs", False)
    assert normalise("thermal conductivity", 5.0, "W/mK") == ("thermal_conductivity", 5.0, "W/mK", False)


def test_elements_in_validates_formulas() -> None:
    assert elements_in("Li7La3Zr2O12") == {"Li", "La", "Zr", "O"}
    assert elements_in("Li6.4La3Zr1.4Ta0.6O12") is not None
    assert elements_in("Ca(OH)2") == {"Ca", "O", "H"}
    assert elements_in("LLZO") is None
    assert elements_in("Xx2") is None


def test_has_rare_earth() -> None:
    assert has_rare_earth("Li7La3Zr2O12")  # lanthanum
    assert has_rare_earth("Y3Al5O12")  # yttrium
    assert not has_rare_earth("LiFePO4")
    assert not has_rare_earth("not-a-formula")


def test_find_formulas_ignores_acronyms_and_words() -> None:
    text = "LLZO (Li7La3Zr2O12) and GaN show sharp CV peaks. The SOC of TiO2 electrodes rose."
    assert find_formulas(text) == ["Li7La3Zr2O12", "GaN", "TiO2"]
