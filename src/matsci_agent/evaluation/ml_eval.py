"""Model benchmark: is the KRR band-gap model actually the right model, and can its intervals be trusted?

Three questions, answered on the same data with the same folds:

1. **Accuracy vs baselines**: 5-fold shuffled cross-validation of KRR against a mean predictor, ridge,
   k-nearest-neighbours, random forest, extra trees and gradient boosting (R², MAE, RMSE).
2. **Data efficiency**: a learning curve, i.e. error as the training set grows.
3. **Calibration**: over many random train / calibration / test splits, how often does an interval
   actually contain the truth? Compared are the split-conformal interval (the one the API reports)
   and the older "plus or minus one RMSE" reading (which promises little: about two thirds).
"""

from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from matsci_agent.infrastructure.ml.krr import KRR_PARAMS, conformal_quantile


def candidate_models(seed: int = 0) -> dict:
    scaled = lambda est: make_pipeline(StandardScaler(), est)  # noqa: E731
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge": scaled(Ridge(alpha=10.0)),
        "knn_5": scaled(KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "krr_rbf (previous)": scaled(KernelRidge(kernel="rbf", alpha=0.1, gamma=0.01)),
        "krr_laplacian (shipped)": scaled(KernelRidge(**KRR_PARAMS)),
        "random_forest": RandomForestRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(n_estimators=300, random_state=seed, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=seed),
    }


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def cross_validate(models: dict, X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> dict[str, dict]:
    """Pooled out-of-fold metrics plus the spread of R² across folds."""
    out: dict[str, dict] = {}
    splitter = KFold(folds, shuffle=True, random_state=seed)
    for name, model in models.items():
        predicted = cross_val_predict(model, X, y, cv=splitter)
        fold_r2 = [r2_score(y[test], predicted[test]) for _, test in splitter.split(X)]
        out[name] = {**metrics(y, predicted), "r2_fold_std": float(np.std(fold_r2))}
    return out


def learning_curve(model_factory, X: np.ndarray, y: np.ndarray, fractions=(0.2, 0.4, 0.6, 0.8, 1.0), seed: int = 0) -> list[dict]:
    """Held-out error as the training set grows (a fixed 20% test split, training drawn from the rest)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    n_test = len(y) // 5
    test, pool = order[:n_test], order[n_test:]
    rows = []
    for f in fractions:
        train = pool[: max(int(len(pool) * f), 10)]
        model = model_factory().fit(X[train], y[train])
        rows.append({"train_size": int(len(train)), **metrics(y[test], model.predict(X[test]))})
    return rows


def calibration(model_factory, X: np.ndarray, y: np.ndarray, repeats: int = 20, alpha: float = 0.1, seed: int = 0) -> dict:
    """Empirical coverage and width of intervals, over random 60/20/20 train / calibration / test splits."""
    rng = np.random.default_rng(seed)
    rows: dict[str, list[tuple[float, float]]] = {"conformal_90": [], "gaussian_rmse_x1.645": [], "plus_minus_1_rmse": []}
    for _ in range(repeats):
        order = rng.permutation(len(y))
        n = len(y)
        train, cal, test = order[: int(0.6 * n)], order[int(0.6 * n): int(0.8 * n)], order[int(0.8 * n):]
        model = model_factory().fit(X[train], y[train])
        cal_err = np.abs(model.predict(X[cal]) - y[cal])
        test_err = np.abs(model.predict(X[test]) - y[test])
        widths = {
            "conformal_90": conformal_quantile(cal_err, alpha),
            "gaussian_rmse_x1.645": 1.645 * float(np.sqrt(np.mean(cal_err ** 2))),
            "plus_minus_1_rmse": float(np.sqrt(np.mean(cal_err ** 2))),
        }
        for name, half in widths.items():
            rows[name].append((float(np.mean(test_err <= half)), 2 * half))
    return {
        "target_coverage": 1 - alpha, "repeats": repeats,
        "methods": {
            name: {"coverage": round(float(np.mean([c for c, _ in v])), 4), "coverage_std": round(float(np.std([c for c, _ in v])), 4),
                   "mean_width": round(float(np.mean([w for _, w in v])), 4)}
            for name, v in rows.items()
        },
    }


def run_benchmark(X: np.ndarray, y: np.ndarray, *, cv_models: dict | None = None, repeats: int = 20) -> dict:
    krr = lambda: make_pipeline(StandardScaler(), KernelRidge(**KRR_PARAMS))  # noqa: E731
    return {
        "samples": int(len(y)), "features": int(X.shape[1]),
        "target": {"mean": round(float(np.mean(y)), 3), "std": round(float(np.std(y)), 3), "min": round(float(np.min(y)), 3), "max": round(float(np.max(y)), 3)},
        "cross_validation": cross_validate(cv_models or candidate_models(), X, y),
        "learning_curve": learning_curve(krr, X, y),
        "calibration": calibration(krr, X, y, repeats=repeats),
    }


def render_model_card(r: dict, source_note: str) -> str:
    cv, cal = r["cross_validation"], r["calibration"]
    rows = "\n".join(
        f"| {name} | {m['r2']:.3f} | {m['mae']:.3f} | {m['rmse']:.3f} | {m['r2_fold_std']:.3f} |"
        for name, m in sorted(cv.items(), key=lambda kv: -kv[1]["r2"])
    )
    curve = "\n".join(f"| {row['train_size']} | {row['r2']:.3f} | {row['mae']:.3f} | {row['rmse']:.3f} |" for row in r["learning_curve"])
    labels = {"conformal_90": "Split conformal, 90% (what the API reports)", "gaussian_rmse_x1.645": "Gaussian: 1.645 x RMSE",
              "plus_minus_1_rmse": "Plus or minus one RMSE (the earlier reading)"}
    cov = "\n".join(
        f"| {labels[k]} | {v['coverage']:.3f} ± {v['coverage_std']:.3f} | {v['mean_width']:.2f} eV |" for k, v in cal["methods"].items()
    )
    t = r["target"]
    m = cal["methods"]
    commentary = (
        f"A plain \"plus or minus one RMSE\" contained the truth only {m['plus_minus_1_rmse']['coverage']:.0%} of the time against the "
        f"{cal['target_coverage']:.0%} promised; the conformal interval contained it {m['conformal_90']['coverage']:.0%} of the time "
        f"and, unlike the Gaussian reading, does not assume the errors are normal (here they are close to it, so the two agree). "
        f"The price of honesty is width: a 90% interval is about {m['conformal_90']['mean_width']:.1f} eV wide, so an estimate is a "
        "triage signal, not a measurement."
    )
    return f"""# Model card: band-gap estimator

**What it is.** A kernel ridge regression (Laplacian kernel) on composition-only descriptors that estimates a material's
**PBE-DFT band gap** from its chemical formula. It is used to fill a missing band gap when screening candidates,
and its estimate is always labelled as one: **an estimate can never make a candidate qualify.**

## Data
{source_note} {r['samples']} materials, {r['features']} features (Magpie composition statistics via matminer: mean, spread and
range of elemental properties such as electronegativity, atomic radius and valence electrons).
Target: band gap in eV, mean {t['mean']}, std {t['std']}, range {t['min']} to {t['max']}.

## Accuracy against baselines (5-fold shuffled cross-validation, identical folds)
| Model | R² | MAE (eV) | RMSE (eV) | R² spread across folds |
| --- | --- | --- | --- | --- |
{rows}

## Data efficiency (fixed 20% test split)
| Training size | R² | MAE (eV) | RMSE (eV) |
| --- | --- | --- | --- |
{curve}

## Are the intervals trustworthy?
Coverage of intervals over {cal['repeats']} random train / calibration / test splits. The promise is {cal['target_coverage']:.0%}.

| Interval | Coverage on unseen materials | Mean width |
| --- | --- | --- |
{cov}

{commentary}

## Intended use and limits
- **Intended:** triage when no measured or database value exists, with the interval shown to the user.
- **It predicts DFT, not experiment.** PBE band gaps are typically well below measured gaps (GaN: about 1.7 eV in
  Materials Project against 3.4 eV measured), and the model inherits that bias.
- **Composition only.** It cannot tell polymorphs apart; materials with the same formula get the same estimate.
- **Training set excludes metals** (band gap below 0.1 eV) and hypothetical structures, so estimates for metals or
  exotic chemistries are extrapolation. The interval is calibrated on materials like the training set, not on those.
- Errors are not uniform: wide-gap insulators and unusual element combinations are predicted worse than common
  semiconductors.
"""
