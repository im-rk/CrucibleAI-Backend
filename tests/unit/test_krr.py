"""Unit tests for KRRPropertyPredictor."""

from __future__ import annotations

import tempfile
import numpy as np
from matsci_agent.infrastructure.ml.krr import KRRPropertyPredictor


def test_krr_predictor_feature_computation() -> None:
    predictor = KRRPropertyPredictor()
    feats = predictor.compute_features("GaN")
    assert feats is not None
    assert len(feats) >= 10


def test_krr_predictor_train_and_predict() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp_dir:
        predictor = KRRPropertyPredictor(model_dir=tmp_dir)

        sample_formulas = ["GaN", "Si", "Ge", "CdTe", "TiO2"]
        sample_targets = np.array([3.4, 1.12, 0.66, 1.5, 3.2])
        X = np.array([predictor.compute_features(f) for f in sample_formulas])

        r2 = predictor.train(X, sample_targets, property_name="band_gap")
        assert isinstance(r2, float)

        res = predictor.predict("GaN", property_name="band_gap")
        assert res["formula"] == "GaN"
        assert res["property"] == "band_gap"
        assert "predicted_value" in res
        assert res["unit"] == "eV"
        assert res["is_predicted"] is True

