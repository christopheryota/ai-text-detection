"""Reproduce holdout probability and decision metrics by length/source groups."""

from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import joblib
import numpy as np
from datasets import concatenate_datasets, load_dataset
import datasets
import sklearn

from benchmark_models import DATASET_NAME, MODEL_DIR, REPORT_PATH, _make_splits
from src.model_benchmark import grouped_metrics, grouped_probability_diagnostics
from src.verity_analysis import ai_decision_scores, probability_diagnostics


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "reports" / "stratified_model_analysis.json"
CALIBRATION_GROUP_MIN_SUPPORT = 30


def _length_band(word_count):
    if word_count < 80:
        return "under_80_words"
    if word_count < 200:
        return "80_to_199_words"
    if word_count < 500:
        return "200_to_499_words"
    return "500_or_more_words"


def main():
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    human_label = report["label_mapping"]["human"]
    ai_label = report["label_mapping"]["ai_generated"]
    selected = report["selection"]["selected_candidate"]
    candidate = next(item for item in report["candidates"] if item["name"] == selected)
    thresholds = candidate["validation"]["risk_controlled"]["thresholds"]

    source_dataset = load_dataset(DATASET_NAME)
    dataset = concatenate_datasets([source_dataset["train"], source_dataset["dev"]])
    labels, sources, sub_sources, partitions, duplicate_counts = _make_splits(
        dataset, human_label, ai_label
    )
    test_indices = partitions["test"]
    if len(test_indices) != report["split_sizes"]["test"]["total"]:
        raise RuntimeError("Recreated test set size does not match the saved benchmark report.")

    test_rows = dataset.select(test_indices.tolist())
    texts = test_rows["text"]
    test_labels = labels[test_indices]
    model_names = np.asarray(test_rows["model"], dtype=object)
    test_sources = sources[test_indices]
    test_sub_sources = sub_sources[test_indices]
    word_counts = np.asarray([len(text.split()) for text in texts], dtype=int)
    length_bands = np.asarray([_length_band(count) for count in word_counts], dtype=object)

    def excluded_small_groups(groups):
        return {
            str(group): int(np.count_nonzero(groups == group))
            for group in sorted(set(groups.tolist()))
            if np.count_nonzero(groups == group) < CALIBRATION_GROUP_MIN_SUPPORT
        }

    model_path = MODEL_DIR / "ai_text_detector.joblib"
    calibrator_path = MODEL_DIR / "ai_text_detector_calibrator.joblib"
    model = joblib.load(model_path)
    calibrator = joblib.load(calibrator_path)
    ai_scores = ai_decision_scores(model, texts, ai_label)
    ai_index = list(model.classes_).index(ai_label)
    raw_probabilities = model.predict_proba(texts)[:, ai_index]
    calibrated_probabilities = calibrator.predict_proba(ai_scores.reshape(-1, 1))[
        :, list(calibrator.classes_).index(ai_label)
    ]

    score_metrics_by_length = {}
    for band in sorted(set(length_bands.tolist())):
        mask = length_bands == band
        score_metrics_by_length[band] = grouped_metrics(
            test_labels[mask],
            ai_scores[mask],
            np.full(int(mask.sum()), band, dtype=object),
            human_label,
            ai_label,
            thresholds,
        )[band]

    output = {
        "dataset": DATASET_NAME,
        "selected_candidate": selected,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": report["seed"],
        "test_partition": "same untouched HC3 test partition from model_benchmark.json; no model or threshold fitting in this script",
        "test_size": len(test_indices),
        "duplicate_counts_recreated": duplicate_counts,
        "calibration_group_minimum_support": CALIBRATION_GROUP_MIN_SUPPORT,
        "calibration_groups_excluded_for_low_support": {
            "word_count": excluded_small_groups(length_bands),
            "sub_source": excluded_small_groups(test_sub_sources),
            "model_metadata": excluded_small_groups(model_names),
        },
        "probability_interpretation": "Diagnostics only. Probability estimates are not enabled because overall held-out calibration worsened after Platt scaling.",
        "raw_probability_diagnostics_overall": probability_diagnostics(
            test_labels, raw_probabilities, ai_label
        ),
        "platt_probability_diagnostics_overall": probability_diagnostics(
            test_labels, calibrated_probabilities, ai_label
        ),
        "raw_probability_diagnostics_by_word_count": grouped_probability_diagnostics(
            test_labels, raw_probabilities, length_bands, ai_label,
            min_support=CALIBRATION_GROUP_MIN_SUPPORT,
        ),
        "platt_probability_diagnostics_by_word_count": grouped_probability_diagnostics(
            test_labels, calibrated_probabilities, length_bands, ai_label,
            min_support=CALIBRATION_GROUP_MIN_SUPPORT,
        ),
        "raw_probability_diagnostics_by_sub_source": grouped_probability_diagnostics(
            test_labels, raw_probabilities, test_sub_sources, ai_label,
            min_support=CALIBRATION_GROUP_MIN_SUPPORT,
        ),
        "platt_probability_diagnostics_by_sub_source": grouped_probability_diagnostics(
            test_labels, calibrated_probabilities, test_sub_sources, ai_label,
            min_support=CALIBRATION_GROUP_MIN_SUPPORT,
        ),
        "raw_probability_diagnostics_by_model_metadata": grouped_probability_diagnostics(
            test_labels, raw_probabilities, model_names, ai_label,
            min_support=CALIBRATION_GROUP_MIN_SUPPORT,
        ),
        "platt_probability_diagnostics_by_model_metadata": grouped_probability_diagnostics(
            test_labels, calibrated_probabilities, model_names, ai_label,
            min_support=CALIBRATION_GROUP_MIN_SUPPORT,
        ),
        "risk_controlled_metrics_by_word_count": score_metrics_by_length,
        "risk_controlled_metrics_by_sub_source": grouped_metrics(
            test_labels, ai_scores, test_sub_sources, human_label, ai_label, thresholds
        ),
        "versions": {
            "python": platform.python_version(),
            "datasets": datasets.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Stratified analysis saved to {OUTPUT_PATH}")
    print(json.dumps({
        "risk_controlled_metrics_by_word_count": score_metrics_by_length,
        "raw_probability_diagnostics_by_word_count": output["raw_probability_diagnostics_by_word_count"],
        "platt_probability_diagnostics_by_word_count": output["platt_probability_diagnostics_by_word_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
