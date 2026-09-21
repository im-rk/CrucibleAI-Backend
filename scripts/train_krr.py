"""CLI entry point for training the Kernel Ridge Regression model.

  python scripts/train_krr.py                                   # offline baseline (band gap only)
  python scripts/train_krr.py --source mp --samples 500         # real Materials Project data
  python scripts/train_krr.py --source mp --property formation_energy_per_atom
"""

from __future__ import annotations

import argparse
import sys

sys.path.append("src")

from matsci_agent.core.config import settings  # noqa: E402
from matsci_agent.infrastructure.ml.krr import PROPERTY_KEYS, KRRPropertyPredictor  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Kernel Ridge Regression for materials properties.")
    parser.add_argument("--property", default="band_gap", help="band_gap | formation_energy_per_atom | density")
    parser.add_argument("--source", choices=["baseline", "mp"], default="baseline")
    parser.add_argument("--samples", type=int, default=500, help="Materials Project samples (--source mp)")
    args = parser.parse_args()

    predictor = KRRPropertyPredictor()
    if args.source == "mp":
        if not settings.materials_project_api_key:
            sys.exit("MATERIALS_PROJECT_API_KEY is not set (put it in .env).")
        r2 = predictor.train_from_materials_project(
            args.property, args.samples, settings.materials_project_api_key
        )
    else:
        predictor._train_baseline(args.property)
        r2 = predictor.meta[PROPERTY_KEYS[args.property]]["cv_r2"]

    print(f"[OK] Trained '{args.property}' from {args.source}. Cross-validated R²: {r2}")
    result = predictor.predict("GaN", args.property)
    print(f"Test prediction for GaN: {result}")


if __name__ == "__main__":
    main()
