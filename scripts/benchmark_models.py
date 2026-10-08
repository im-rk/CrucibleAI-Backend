"""Benchmark the band-gap model against baselines and check its prediction intervals.

    python scripts/benchmark_models.py                 # uses the cached dataset, fetching it once from Materials Project
    python scripts/benchmark_models.py --update-model  # also retrain the shipped model (adds the calibrated interval)

Writes evaluation/results/model_benchmark.json and docs/MODEL_CARD.md.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.append("src")

from matsci_agent.core.config import settings  # noqa: E402
from matsci_agent.evaluation import ml_eval  # noqa: E402
from matsci_agent.infrastructure.ml.krr import KRRPropertyPredictor  # noqa: E402

CACHE = Path("data/cache/mp_band_gap_{n}.npz")
RESULTS = Path("evaluation/results/model_benchmark.json")
CARD = Path("docs/MODEL_CARD.md")


def load_dataset(predictor: KRRPropertyPredictor, n: int) -> tuple[np.ndarray, np.ndarray]:
    path = Path(str(CACHE).format(n=n))
    if path.exists():
        data = np.load(path, allow_pickle=False)
        return data["X"], data["y"]
    if not settings.materials_project_api_key:
        sys.exit("No cached dataset and MATERIALS_PROJECT_API_KEY is not set (put it in .env).")
    print(f"Fetching {n} Materials Project band gaps and computing Magpie features (once; cached afterwards)...")
    X, y, _ = predictor.fetch_materials_project_dataset("band_gap", n, settings.materials_project_api_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--repeats", type=int, default=20, help="random splits for the calibration experiment")
    parser.add_argument("--update-model", action="store_true", help="retrain the shipped model on this dataset")
    args = parser.parse_args()

    predictor = KRRPropertyPredictor()
    X, y = load_dataset(predictor, args.samples)
    print(f"Dataset: {len(y)} materials x {X.shape[1]} features")
    result = ml_eval.run_benchmark(X, y, repeats=args.repeats)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=1), encoding="utf-8")
    CARD.parent.mkdir(exist_ok=True)
    CARD.write_text(
        ml_eval.render_model_card(
            result, "Materials Project (experimentally known, energy above hull <= 0.05 eV/atom, band gap 0.1-10 eV; ground state per formula, random sample).",
        ),
        encoding="utf-8",
    )
    if args.update_model:
        r2 = predictor.train(X, y, "band_gap", source="materials_project")
        print(f"Retrained the shipped model: CV R2 {r2:.3f}; conformal half-width {predictor.meta['band_gap']['conformal_q90']:.3f} eV")
    best = max(result["cross_validation"].items(), key=lambda kv: kv[1]["r2"])
    print(f"Best model by CV R2: {best[0]} ({best[1]['r2']:.3f}); wrote {RESULTS} and {CARD}")


if __name__ == "__main__":
    main()
