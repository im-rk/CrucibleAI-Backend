# Model card: band-gap estimator

**What it is.** A kernel ridge regression (Laplacian kernel) on composition-only descriptors that estimates a material's
**PBE-DFT band gap** from its chemical formula. It is used to fill a missing band gap when screening candidates,
and its estimate is always labelled as one: **an estimate can never make a candidate qualify.**

## Data
Materials Project (experimentally known, energy above hull <= 0.05 eV/atom, band gap 0.1-10 eV; ground state per formula, random sample). 1500 materials, 132 features (Magpie composition statistics via matminer: mean, spread and
range of elemental properties such as electronegativity, atomic radius and valence electrons).
Target: band gap in eV, mean 2.601, std 1.654, range 0.104 to 9.065.

## Accuracy against baselines (5-fold shuffled cross-validation, identical folds)
| Model | R² | MAE (eV) | RMSE (eV) | R² spread across folds |
| --- | --- | --- | --- | --- |
| hist_gradient_boosting | 0.733 | 0.610 | 0.854 | 0.038 |
| krr_laplacian (shipped) | 0.728 | 0.617 | 0.862 | 0.041 |
| extra_trees | 0.724 | 0.605 | 0.869 | 0.051 |
| random_forest | 0.720 | 0.631 | 0.875 | 0.047 |
| krr_rbf (previous) | 0.683 | 0.652 | 0.931 | 0.051 |
| knn_5 | 0.613 | 0.756 | 1.029 | 0.063 |
| ridge | 0.584 | 0.824 | 1.066 | 0.059 |
| mean_baseline | -0.004 | 1.356 | 1.657 | 0.012 |

## Data efficiency (fixed 20% test split)
| Training size | R² | MAE (eV) | RMSE (eV) |
| --- | --- | --- | --- |
| 240 | 0.629 | 0.772 | 1.037 |
| 480 | 0.709 | 0.685 | 0.918 |
| 720 | 0.742 | 0.657 | 0.865 |
| 960 | 0.752 | 0.634 | 0.848 |
| 1200 | 0.765 | 0.614 | 0.825 |

## Are the intervals trustworthy?
Coverage of intervals over 20 random train / calibration / test splits. The promise is 90%.

| Interval | Coverage on unseen materials | Mean width |
| --- | --- | --- |
| Split conformal, 90% (what the API reports) | 0.900 ± 0.018 | 2.80 eV |
| Gaussian: 1.645 x RMSE | 0.906 ± 0.022 | 2.87 eV |
| Plus or minus one RMSE (the earlier reading) | 0.747 ± 0.033 | 1.74 eV |

A plain "plus or minus one RMSE" contained the truth only 75% of the time against the 90% promised; the conformal interval contained it 90% of the time and, unlike the Gaussian reading, does not assume the errors are normal (here they are close to it, so the two agree). The price of honesty is width: a 90% interval is about 2.8 eV wide, so an estimate is a triage signal, not a measurement.

## Intended use and limits
- **Intended:** triage when no measured or database value exists, with the interval shown to the user.
- **It predicts DFT, not experiment.** PBE band gaps are typically well below measured gaps (GaN: about 1.7 eV in
  Materials Project against 3.4 eV measured), and the model inherits that bias.
- **Composition only.** It cannot tell polymorphs apart; materials with the same formula get the same estimate.
- **Training set excludes metals** (band gap below 0.1 eV) and hypothetical structures, so estimates for metals or
  exotic chemistries are extrapolation. The interval is calibrated on materials like the training set, not on those.
- Errors are not uniform: wide-gap insulators and unusual element combinations are predicted worse than common
  semiconductors.
