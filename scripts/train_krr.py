"""CLI entry point for training the Kernel Ridge Regression model."""

from __future__ import annotations

import argparse
import sys
sys.path.append("src")

from matsci_agent.infrastructure.ml.krr import KRRPropertyPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Kernel Ridge Regression for materials properties.")
    parser.add_argument("--property", type=str, default="band_gap", help="Property to train (default: band_gap)")
    args = parser.parse_args()

    print(f"=== Starting KRR Training for property: {args.property} ===")
    predictor = KRRPropertyPredictor()
    predictor._train_baseline(args.property)

    print(f"[OK] Training complete for {args.property}!")

    print("\n--- Test Prediction ---")
    gan_result = predictor.predict("GaN", property_name=args.property)
    print(f"  Formula: {gan_result['formula']}")
    print(f"  Predicted Bandgap: {gan_result['predicted_value']} {gan_result['unit']}")
    print(f"  Method: {gan_result['method']}")


if __name__ == "__main__":
    main()
