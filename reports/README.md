# Evaluation protocol

`model_benchmark.json` is the machine-readable record of the latest reproducible candidate comparison. It records the dataset mapping, seed, group-aware partitions, model settings, validation-only thresholds, same-test baseline comparison, calibration diagnostics, per-source results, versions, and limitations.

Run the benchmark from the project root with:

```powershell
.\ai_text_detector_env\Scripts\python.exe .\train_model.py
```

The comparison trains on `mage` and `m4gt`, selects hyperparameters and score cutoffs on a disjoint source/sub-source validation fold, fits Platt scaling on a separate group fold, and reserves the `hc3` collection source for a final cross-collection test. Exact SHA-256 duplicate texts are removed across partitions. Near-duplicate detection is not performed.

After selection, `analyze_strata.py` reuses that exact test partition to evaluate raw and Platt probabilities by word-count band, sub-source, and dataset model metadata, including Brier score, log loss, ECE, and reliability-bin data. It also reports risk-controlled false-positive rate, AI recall, and coverage by length. It performs no fitting or threshold selection. Results are saved to `stratified_model_analysis.json` and are only loaded by Flask when the candidate and seed match the main report.

The AI decision threshold is selected to meet a one-sided 95% Wilson upper-bound target for human false positives. The human decision threshold has a separate AI false-negative target. Samples between those cutoffs are inconclusive. The application uses the oriented linear decision score, not a probability; the probability-calibration diagnostics are retained for evaluation and are not a reliability claim.

`model/baseline_20261025/` and `train_model_baseline.py` preserve the previous model, calibrator, metrics, and trainer. The baseline row in the benchmark is retrained on the benchmark's exact training partition and evaluated on the same untouched `hc3` test rows as the selected candidate.

This dataset and its `hc3` collection are not independent proof of real-world performance. Collection, domain, generator, language, editing, and prompt differences may alter results. Model explanations report fitted feature contributions, not authorship evidence.

Brand identity is configured in `brand_config.py` and can be overridden with `PRODUCT_NAME`, `PRODUCT_TAGLINE`, `PRODUCT_FOOTER`, `PRODUCT_PRIMARY_COLOR`, and `PRODUCT_BACKGROUND_COLOR` environment variables.

The main result presents the sign and magnitude of the raw AI-oriented Logistic Regression decision score, not a probability. Feature rows are TF-IDF values multiplied by oriented model coefficients; all active feature products plus the intercept are checked against the actual score. The app describes this as learned model association, not authorship evidence. Evidence strength is weak between the validation cutoffs, and moderate only when a conservative cutoff and the external risk gate both pass; a short passage is always limited to weak evidence. No strong tier has been validated.

To refresh only the held-out calibration/length/source diagnostics without retraining or retuning:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\ai_text_detector_env\Scripts\python.exe .\analyze_strata.py
```

This uses the cached dataset and writes `stratified_model_analysis.json`. Groups below the recorded support minimum are excluded from calibration subgroup summaries; the report lists those exclusions. The analysis uses the existing test partition and must not be used to change model selection or thresholds.
