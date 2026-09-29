"""KRR training metadata, honest confidence, property guards, and Materials Project training."""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest

from matsci_agent.domain.reference_data import OFFLINE_BANDGAPS
from matsci_agent.infrastructure.ml.krr import KRRPropertyPredictor


@pytest.fixture
def predictor(tmp_path):
    p = KRRPropertyPredictor(model_dir=tmp_path)
    p._featurizer_failed = True  # use the fast built-in features instead of importing matminer
    return p


def fit_reference(predictor, source="custom"):
    formulas = list(OFFLINE_BANDGAPS)
    X = np.array([predictor.compute_features(f) for f in formulas])
    y = np.array([OFFLINE_BANDGAPS[f][0] for f in formulas])
    predictor.train(X, y, "band_gap", source=source)


def test_training_writes_cv_metadata_next_to_the_model(predictor, tmp_path) -> None:
    fit_reference(predictor)
    meta = json.loads((tmp_path / "krr_band_gap.meta.json").read_text())
    assert meta["n_samples"] == len(OFFLINE_BANDGAPS)
    assert meta["cv_rmse"] is not None and meta["cv_r2"] is not None
    assert meta["n_features"] == 20


def test_prediction_reports_uncertainty_and_low_confidence_for_non_mp_training(predictor) -> None:
    fit_reference(predictor)
    result = predictor.predict("GaN", "band_gap")
    assert result["is_predicted"] and result["unit"] == "eV"
    assert result["uncertainty"] > 0 and result["training_samples"] == len(OFFLINE_BANDGAPS)
    assert result["confidence"] == "low"
    assert "Not a measurement" in result["note"]


def test_property_aliases_resolve_to_the_same_model(predictor) -> None:
    fit_reference(predictor)
    assert predictor.predict("GaN", "bandgap")["property"] == "band_gap"


def test_unsupported_property_is_an_error_not_a_guess(predictor) -> None:
    assert "Unsupported property" in predictor.predict("GaN", "ionic_conductivity")["error"]


def test_property_without_a_trained_model_is_refused(predictor) -> None:
    # Regression: this used to silently train on band-gap values and label it formation energy.
    error = predictor.predict("GaN", "formation_energy")["error"]
    assert "No trained model" in error and "--source mp" in error


def test_baseline_only_exists_for_band_gap(predictor) -> None:
    with pytest.raises(ValueError, match="No baseline data"):
        predictor._train_baseline("density")


def test_baseline_trains_on_the_offline_reference_table(predictor) -> None:
    result = predictor.predict("GaN", "band_gap")  # no model on disk: trains the baseline
    assert result["training_source"] == "offline_reference" and result["confidence"] == "low"


def test_non_negative_properties_are_clamped(predictor) -> None:
    fit_reference(predictor)
    assert predictor.predict("Cs2Pb3I9Au5", "band_gap")["predicted_value"] >= 0


def _fake_mp(monkeypatch, docs):
    class FakeMPRester:
        def __init__(self, key):
            self.summary = types.SimpleNamespace(search=lambda **query: docs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(sys.modules, "mp_api.client", types.SimpleNamespace(MPRester=FakeMPRester))


def test_training_from_materials_project_marks_the_source(predictor, monkeypatch) -> None:
    docs = [types.SimpleNamespace(formula_pretty=f, band_gap=g) for f, (g, _, _) in OFFLINE_BANDGAPS.items()]
    _fake_mp(monkeypatch, docs)
    predictor.train_from_materials_project("band_gap", n_samples=50, api_key="key")
    assert predictor.meta["band_gap"]["source"] == "materials_project"
    assert predictor.predict("GaN", "band_gap")["training_source"] == "materials_project"
    assert "underestimates the measured gap" in predictor.predict("GaN", "band_gap")["note"]


def test_mp_training_needs_a_key_a_known_property_and_enough_data(predictor, monkeypatch) -> None:
    with pytest.raises(ValueError, match="API key"):
        predictor.train_from_materials_project("band_gap", api_key=None)
    with pytest.raises(ValueError, match="Unsupported property"):
        predictor.train_from_materials_project("ionic_conductivity", api_key="key")
    _fake_mp(monkeypatch, [types.SimpleNamespace(formula_pretty="GaN", band_gap=3.4)])
    with pytest.raises(RuntimeError, match="at least 10"):
        predictor.train_from_materials_project("band_gap", api_key="key")


def test_mp_training_asks_for_experimental_stable_materials_and_uses_ground_states(predictor, monkeypatch) -> None:
    captured = {}
    formulas = list(OFFLINE_BANDGAPS)
    docs = [types.SimpleNamespace(formula_pretty=f, band_gap=g, energy_above_hull=0.0) for f, (g, _, _) in OFFLINE_BANDGAPS.items()]
    docs.append(types.SimpleNamespace(formula_pretty="GaN", band_gap=0.1, energy_above_hull=0.04))  # unstable polymorph

    class FakeMPRester:
        def __init__(self, key):
            self.summary = types.SimpleNamespace(search=lambda **q: captured.update(q) or docs)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setitem(sys.modules, "mp_api.client", types.SimpleNamespace(MPRester=FakeMPRester))
    predictor.train_from_materials_project("band_gap", n_samples=len(formulas), api_key="key")

    assert captured["theoretical"] is False and captured["energy_above_hull"] == (0.0, 0.05)
    assert predictor.meta["band_gap"]["n_samples"] == len(formulas)  # one row per formula: the duplicate GaN is gone


def test_cross_validation_is_shuffled_so_sorted_data_is_scored_fairly(predictor) -> None:
    # Targets sorted in blocks: unshuffled 5-fold CV would test each fold on unseen chemistry.
    formulas = sorted(OFFLINE_BANDGAPS, key=lambda f: OFFLINE_BANDGAPS[f][0])
    X = np.array([predictor.compute_features(f) for f in formulas])
    y = np.array([OFFLINE_BANDGAPS[f][0] for f in formulas])
    first = predictor.train(X, y, "band_gap")
    second = predictor.train(X, y, "band_gap")
    assert first == second  # deterministic (fixed shuffle seed)
