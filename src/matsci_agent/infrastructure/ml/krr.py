"""Kernel Ridge Regression for materials property prediction (Pilania et al., Sci. Rep. 2013).

Compositional features (Magpie via matminer, or a small built-in fallback) feed a kernel ridge
regression (Laplacian kernel; see docs/MODEL_CARD.md for why it replaced the RBF kernel). Every trained model writes a metadata sidecar with its cross-validated error, so each
prediction reports an uncertainty and a confidence tied to how it was actually trained.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

from matsci_agent.domain.reference_data import OFFLINE_BANDGAPS

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[4] / "models"

# Chosen by the benchmark in docs/MODEL_CARD.md: with nested cross-validation the Laplacian kernel scored
# R2 0.729 against 0.683 for the RBF kernel (tuning the RBF's alpha and gamma did not help).
KRR_PARAMS = {"kernel": "laplacian", "alpha": 0.1, "gamma": 0.01}

# Names callers use -> the key models are stored under (Materials Project summary field names).
PROPERTY_KEYS = {
    "band_gap": "band_gap",
    "bandgap": "band_gap",
    "formation_energy_per_atom": "formation_energy_per_atom",
    "formation_energy": "formation_energy_per_atom",
    "density": "density",
}
UNITS = {"band_gap": "eV", "formation_energy_per_atom": "eV/atom", "density": "g/cm3"}
NON_NEGATIVE = {"band_gap", "density"}


def conformal_quantile(abs_residuals: np.ndarray, alpha: float = 0.1) -> float:
    """Half-width of a (1 - alpha) prediction interval from held-out absolute errors (split conformal).

    Uses the finite-sample correction ceil((n + 1)(1 - alpha)) / n, so the interval covers a new point with
    probability at least 1 - alpha under exchangeability, whatever the error distribution.
    """
    n = len(abs_residuals)
    if n == 0:
        raise ValueError("Need at least one residual.")
    rank = min(n, math.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(abs_residuals)[rank - 1])


class KRRPropertyPredictor:
    """Predicts material properties using Kernel Ridge Regression."""

    # Basic elemental atomic numbers & electronegativities for the fallback feature vector
    ELEMENT_DB = {
        "H": (1.0, 2.20), "Li": (3.0, 0.98), "Be": (4.0, 1.57), "B": (5.0, 2.04),
        "C": (6.0, 2.55), "N": (7.0, 3.04), "O": (8.0, 3.44), "F": (9.0, 3.98),
        "Na": (11.0, 0.93), "Mg": (12.0, 1.31), "Al": (13.0, 1.61), "Si": (14.0, 1.90),
        "P": (15.0, 2.19), "S": (16.0, 2.58), "Cl": (17.0, 3.16), "K": (19.0, 0.82),
        "Ca": (20.0, 1.00), "Ti": (22.0, 1.54), "V": (23.0, 1.63), "Cr": (24.0, 1.66),
        "Mn": (25.0, 1.55), "Fe": (26.0, 1.83), "Co": (27.0, 1.88), "Ni": (28.0, 1.91),
        "Cu": (29.0, 1.90), "Zn": (30.0, 1.65), "Ga": (31.0, 1.81), "Ge": (32.0, 2.01),
        "As": (33.0, 2.18), "Se": (34.0, 2.55), "Zr": (40.0, 1.33), "Nb": (41.0, 1.60),
        "Mo": (42.0, 2.16), "Ag": (47.0, 1.93), "Cd": (48.0, 1.69), "In": (49.0, 1.78),
        "Sn": (50.0, 1.96), "Sb": (51.0, 2.05), "La": (57.0, 1.10), "Ta": (73.0, 1.50),
        "W": (74.0, 2.36), "Au": (79.0, 2.54), "Pb": (82.0, 2.33), "Bi": (83.0, 2.02),
    }

    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else DEFAULT_MODEL_DIR
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.models: dict[str, KernelRidge] = {}
        self.scalers: dict[str, StandardScaler] = {}
        self.meta: dict[str, dict] = {}
        self._featurizer = None
        self._featurizer_failed = False

    # ---- features ----------------------------------------------------------------------

    def compute_features(self, formula: str) -> Optional[np.ndarray]:
        """Magpie compositional descriptors, or a 20-d fallback if matminer is unavailable."""
        if not self._featurizer_failed:
            try:
                from matminer.featurizers.composition import ElementProperty
                from pymatgen.core import Composition

                from matsci_agent.domain.chemistry import expand_abbreviations

                if self._featurizer is None:
                    self._featurizer = ElementProperty.from_preset("magpie")
                    self._featurizer.set_n_jobs(1)
                return np.array(self._featurizer.featurize(Composition(expand_abbreviations(formula))), dtype=np.float64)
            except ImportError:
                self._featurizer_failed = True  # don't retry the import per formula
            except Exception:
                pass  # this formula failed to featurise; fall through to the fallback
        return self._compute_fallback_features(formula)

    def _compute_fallback_features(self, formula: str) -> Optional[np.ndarray]:
        matches = re.findall(r"([A-Z][a-z]?)(?:(\d+(?:\.\d+)?))?", formula)
        if not matches:
            return None
        counts = [float(c) if c else 1.0 for _, c in matches]
        total_atoms = sum(counts)
        if total_atoms == 0:
            return None
        fractions = [c / total_atoms for c in counts]
        info = [self.ELEMENT_DB.get(elem, (20.0, 1.50)) for elem, _ in matches]
        numbers = [i[0] for i in info]
        negativity = [i[1] for i in info]
        features = [
            np.average(numbers, weights=fractions), np.min(numbers), np.max(numbers), np.std(numbers),
            np.average(negativity, weights=fractions), np.min(negativity), np.max(negativity),
            np.std(negativity), len(matches), total_atoms,
        ]
        features.extend([0.0] * (20 - len(features)))
        return np.array(features, dtype=np.float64)

    # ---- training ----------------------------------------------------------------------

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        property_name: str = "band_gap",
        source: str = "custom",
    ) -> float:
        """Fit KRR on (X, y), save it with cross-validated error metadata, return CV R²."""
        key = PROPERTY_KEYS.get(property_name, property_name)
        logger.info(f"Training KRR for {key} on {len(y)} samples ({source})...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = KernelRidge(**KRR_PARAMS)
        model.fit(X_scaled, y)

        r2 = rmse = q90 = None
        folds = min(5, len(y))
        if folds >= 3:
            # Shuffled folds: training data can arrive sorted (MP returns formulas alphabetically),
            # and contiguous folds would then each be a different slice of chemistry.
            predicted = cross_val_predict(model, X_scaled, y, cv=KFold(folds, shuffle=True, random_state=0))
            rmse = float(np.sqrt(np.mean((predicted - y) ** 2)))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = float(1 - np.sum((predicted - y) ** 2) / ss_tot) if ss_tot > 0 else None
            # Out-of-fold errors are honest held-out errors, so their quantile calibrates the interval.
            q90 = conformal_quantile(np.abs(predicted - y), 0.1)

        meta = {
            "property": key, "n_samples": int(len(y)), "cv_r2": r2, "cv_rmse": rmse,
            "source": source, "n_features": int(X.shape[1]), "conformal_q90": q90, "kernel": KRR_PARAMS["kernel"],
        }
        joblib.dump(model, self.model_dir / f"krr_{key}.joblib")
        joblib.dump(scaler, self.model_dir / f"scaler_{key}.joblib")
        (self.model_dir / f"krr_{key}.meta.json").write_text(json.dumps(meta))
        self.models[key], self.scalers[key], self.meta[key] = model, scaler, meta
        logger.info(f"KRR {key}: CV R²={r2} RMSE={rmse}")
        return r2 if r2 is not None else float("nan")

    def train_from_materials_project(
        self, property_name: str = "band_gap", n_samples: int = 500, api_key: str | None = None
    ) -> float:
        """Train on real Materials Project data (needs MATERIALS_PROJECT_API_KEY)."""
        key = PROPERTY_KEYS.get(property_name)
        if key is None:
            raise ValueError(f"Unsupported property '{property_name}'. Choose from {sorted(set(PROPERTY_KEYS.values()))}.")
        if not api_key:
            raise ValueError("A Materials Project API key is required to train from Materials Project data.")
        X, y, _ = self.fetch_materials_project_dataset(key, n_samples, api_key)
        return self.train(X, y, key, source="materials_project")

    def fetch_materials_project_dataset(
        self, key: str, n_samples: int, api_key: str
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Features, targets and formulas of a random sample of experimentally known, near-stable materials."""
        from mp_api.client import MPRester

        # Experimentally known, (near-)stable materials only; hypothetical polymorphs add noise. MP
        # returns results alphabetically and cannot sort, so fetch a larger pool, keep the ground
        # state per formula (duplicate formulas have identical features), and sample at random.
        pool = max(3 * n_samples, 1000)
        query: dict = {
            "fields": ["formula_pretty", "energy_above_hull", key],
            "theoretical": False,
            "energy_above_hull": (0.0, 0.05),
            "num_chunks": math.ceil(pool / 1000),
            "chunk_size": 1000,
        }
        if key == "band_gap":
            query["band_gap"] = (0.1, 10.0)  # materials with a non-zero gap
        with MPRester(api_key) as mpr:
            summary = getattr(getattr(mpr, "materials", None), "summary", None) or mpr.summary
            docs = list(summary.search(**query))

        best: dict[str, object] = {}
        for doc in docs:
            if getattr(doc, key, None) is None:
                continue
            current = best.get(doc.formula_pretty)
            if current is None or (getattr(doc, "energy_above_hull", 0) or 0) < (getattr(current, "energy_above_hull", 0) or 0):
                best[doc.formula_pretty] = doc
        candidates = list(best.values())
        rng = np.random.default_rng(0)
        chosen = [candidates[i] for i in rng.permutation(len(candidates))[:n_samples]]

        features, targets, formulas = [], [], []
        for doc in chosen:
            target = getattr(doc, key, None)
            vec = self.compute_features(doc.formula_pretty) if target is not None else None
            if vec is not None and not np.any(np.isnan(vec)):
                features.append(vec)
                targets.append(float(target))
                formulas.append(doc.formula_pretty)
        if len(targets) < 10:
            raise RuntimeError(f"Only {len(targets)} usable Materials Project samples; need at least 10.")
        return np.array(features), np.array(targets), formulas

    def _train_baseline(self, property_name: str = "band_gap") -> None:
        """Train on the offline reference table. Only band gaps have reference data."""
        if PROPERTY_KEYS.get(property_name) != "band_gap":
            raise ValueError(
                f"No baseline data for '{property_name}'. Train from Materials Project: "
                "python scripts/train_krr.py --source mp --property " + property_name
            )
        rows = [(self.compute_features(f), gap) for f, (gap, _, _) in OFFLINE_BANDGAPS.items()]
        rows = [(x, y) for x, y in rows if x is not None]
        self.train(
            np.array([x for x, _ in rows]), np.array([y for _, y in rows]),
            "band_gap", source="offline_reference",
        )

    # ---- prediction --------------------------------------------------------------------

    def _load(self, key: str) -> bool:
        """Load a trained model with metadata; legacy models without metadata are retrained."""
        if key in self.models:
            return True
        model_path = self.model_dir / f"krr_{key}.joblib"
        meta_path = self.model_dir / f"krr_{key}.meta.json"
        if model_path.exists() and meta_path.exists():
            self.models[key] = joblib.load(model_path)
            self.scalers[key] = joblib.load(self.model_dir / f"scaler_{key}.joblib")
            self.meta[key] = json.loads(meta_path.read_text())
            return True
        if key == "band_gap":
            self._train_baseline(key)
            return True
        return False

    def predict(self, formula: str, property_name: str = "band_gap") -> dict:
        """Predict a property for a formula; returns an `error` entry instead of raising."""
        key = PROPERTY_KEYS.get(property_name)
        if key is None:
            return {"error": f"Unsupported property '{property_name}'. Supported: {sorted(set(PROPERTY_KEYS.values()))}"}
        if not self._load(key):
            return {"error": f"No trained model for '{key}'. Run: python scripts/train_krr.py --source mp --property {key}"}

        features = self.compute_features(formula)
        if features is None:
            return {"error": f"Cannot compute features for formula: {formula}"}
        meta = self.meta[key]
        if len(features) != meta.get("n_features", len(features)):
            return {"error": f"Feature mismatch for {formula}; retrain the {key} model in this environment."}

        value = float(self.models[key].predict(self.scalers[key].transform([features]))[0])
        if key in NON_NEGATIVE:
            value = max(0.0, value)
        r2, rmse, n = meta.get("cv_r2"), meta.get("cv_rmse"), meta.get("n_samples")
        q90 = meta.get("conformal_q90")
        interval = None
        if q90 is not None:
            low = value - q90
            interval = [round(max(0.0, low) if key in NON_NEGATIVE else low, 4), round(value + q90, 4)]
        from_mp = meta.get("source") == "materials_project"
        return {
            "formula": formula,
            "property": key,
            "predicted_value": round(value, 4),
            "unit": UNITS[key],
            "method": f"Kernel Ridge Regression ({str(meta.get('kernel', 'rbf')).title()} kernel)",
            "is_predicted": True,
            "confidence": "medium" if from_mp and r2 is not None and r2 >= 0.6 else "low",
            "uncertainty": None if rmse is None else round(rmse, 4),
            "interval_90": interval,
            "cv_r2": None if r2 is None else round(r2, 3),
            "training_samples": n,
            "training_source": meta.get("source"),
            "note": (
                f"Machine-learning estimate trained on {n} "
                f"{'Materials Project materials' if from_mp else 'reference/custom data points'}"
                f"{'' if rmse is None else f' (cross-validated RMSE ±{rmse:.2f} {UNITS[key]})'}. "
                + (f"The 90% interval {interval[0]} to {interval[1]} {UNITS[key]} is calibrated on held-out errors. " if interval else "")
                + (
                    # MP band gaps are PBE-DFT values, which underestimate measured gaps (~40%).
                    "It predicts the PBE-DFT band gap, which typically underestimates the measured gap. "
                    if from_mp and key == "band_gap" else ""
                )
                + "Not a measurement; verify with DFT or experiment."
            ),
        }

    def _get_unit(self, property_name: str) -> str:
        return UNITS.get(PROPERTY_KEYS.get(property_name, property_name), "unitless")
