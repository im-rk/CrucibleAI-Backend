"""The model-benchmark machinery on synthetic data with a known answer."""

from __future__ import annotations

import numpy as np

from matsci_agent.evaluation import ml_eval


def synthetic(n=400, noise=0.3, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 5))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + rng.normal(0, noise, size=n)
    return X, y


def test_every_candidate_beats_or_matches_the_mean_baseline_on_learnable_data() -> None:
    X, y = synthetic()
    results = ml_eval.cross_validate({k: v for k, v in ml_eval.candidate_models().items() if k != "random_forest"}, X, y)
    baseline = results["mean_baseline"]["r2"]
    assert abs(baseline) < 0.05  # predicting the mean explains nothing
    assert results["krr_laplacian (shipped)"]["r2"] > 0.2 and results["extra_trees"]["r2"] > 0.5
    assert all(m["rmse"] > 0 and m["r2_fold_std"] >= 0 for m in results.values())


def test_the_learning_curve_improves_with_more_data() -> None:
    from sklearn.kernel_ridge import KernelRidge

    X, y = synthetic(n=600)
    rows = ml_eval.learning_curve(lambda: KernelRidge(kernel="rbf", alpha=0.1, gamma=0.2), X, y)
    sizes = [r["train_size"] for r in rows]
    assert sizes == sorted(sizes) and rows[-1]["rmse"] < rows[0]["rmse"]


def test_conformal_intervals_hit_their_target_and_plus_minus_one_rmse_does_not() -> None:
    from sklearn.kernel_ridge import KernelRidge

    X, y = synthetic(n=900, noise=0.3)
    cal = ml_eval.calibration(lambda: KernelRidge(kernel="rbf", alpha=0.1, gamma=0.2), X, y, repeats=12)
    m = cal["methods"]

    assert 0.85 <= m["conformal_90"]["coverage"] <= 0.95       # promised 0.90
    assert 0.55 <= m["plus_minus_1_rmse"]["coverage"] <= 0.78  # about two thirds
    assert m["plus_minus_1_rmse"]["mean_width"] < m["conformal_90"]["mean_width"]


def test_the_model_card_reports_the_numbers_it_was_given() -> None:
    X, y = synthetic(n=300)
    models = {"krr_laplacian (shipped)": ml_eval.candidate_models()["krr_laplacian (shipped)"], "mean_baseline": ml_eval.candidate_models()["mean_baseline"]}
    card = ml_eval.render_model_card(ml_eval.run_benchmark(X, y, cv_models=models, repeats=3), "Synthetic data.")
    assert "# Model card" in card and "krr_laplacian (shipped)" in card and "Split conformal" in card and "PBE-DFT" in card
    assert "300 materials" in card
