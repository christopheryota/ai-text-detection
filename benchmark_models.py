"""Compare linear text-classification candidates on a source-held-out test set."""

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import gc

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

from src.model_benchmark import evaluate_scores, grouped_metrics, select_risk_thresholds
from src.verity_analysis import (
    ai_decision_scores,
    fit_probability_calibrator,
    verify_label_mapping,
)


DATASET_NAME = "Jinyan1/COLING_2025_MGT_en"
SEED = 137
TRAIN_LIMIT = 20_000
CALIBRATION_LIMIT = 5_000
VALIDATION_LIMIT = 8_000
TEST_LIMIT = 15_000
MAX_HUMAN_FALSE_POSITIVE_RATE = 0.05
MAX_AI_FALSE_NEGATIVE_RATE = 0.10
MINIMUM_VERDICTS = 100
MINIMUM_VALIDATION_COVERAGE = 0.10
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
REPORT_DIR = BASE_DIR / "reports"
REPORT_PATH = REPORT_DIR / "model_benchmark.json"
CANDIDATE_PATH = MODEL_DIR / "ai_text_detector_candidate.joblib"
CALIBRATOR_CANDIDATE_PATH = MODEL_DIR / "ai_text_detector_calibrator_candidate.joblib"
DEPLOYED_MODEL_PATH = MODEL_DIR / "ai_text_detector.joblib"
DEPLOYED_CALIBRATOR_PATH = MODEL_DIR / "ai_text_detector_calibrator.joblib"


def _group_key(source, sub_source):
    source = str(source).strip()
    sub_source = str(sub_source).strip()
    if not sub_source or sub_source.casefold() in {"none", "nan"}:
        return f"{source}\x1fmissing-{source}"
    return f"{source}\x1f{sub_source}"


def _stratified_sample(indices, labels, limit, seed):
    indices = np.asarray(indices, dtype=int)
    if len(indices) <= limit:
        return np.sort(indices)
    selected, _ = train_test_split(
        indices,
        train_size=limit,
        random_state=seed,
        stratify=np.asarray(labels)[indices],
    )
    return np.sort(selected)


def _deduplicate(dataset, partitions):
    seen = set()
    clean = {}
    removed = {}
    for split_name in ("train", "calibration", "validation", "test"):
        selected = partitions[split_name]
        texts = dataset.select(selected.tolist())["text"]
        kept = []
        count = 0
        for index, text in zip(selected, texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Invalid or empty text in {split_name} data at row {index}.")
            fingerprint = hashlib.sha256(text.encode("utf-8")).digest()
            if fingerprint in seen:
                count += 1
            else:
                seen.add(fingerprint)
                kept.append(int(index))
        clean[split_name] = np.asarray(kept, dtype=int)
        removed[split_name] = count
    return clean, removed


def _build_model(feature_set, classifier, c_value, class_weight):
    word = TfidfVectorizer(
        max_features=30_000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    character = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        max_features=120_000,
        min_df=3,
        sublinear_tf=True,
    )

    if feature_set == "word":
        features = word
    elif feature_set == "character":
        features = character
    elif feature_set == "word_and_character":
        features = FeatureUnion([("word", word), ("character", character)])
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")

    if classifier == "logistic_regression":
        estimator = LogisticRegression(
            C=c_value,
            class_weight=class_weight,
            max_iter=1_000,
            random_state=SEED,
        )
    elif classifier == "linear_svm":
        estimator = LinearSVC(C=c_value, class_weight=class_weight, random_state=SEED)
    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    return Pipeline([("features", features), ("classifier", estimator)])


def _candidate_specs():
    return [
        {"name": "word_lr_c1_unweighted", "feature_set": "word", "classifier": "logistic_regression", "c": 1.0, "class_weight": None},
        {"name": "word_lr_c0.1_balanced", "feature_set": "word", "classifier": "logistic_regression", "c": 0.1, "class_weight": "balanced"},
        {"name": "word_lr_c1_balanced", "feature_set": "word", "classifier": "logistic_regression", "c": 1.0, "class_weight": "balanced"},
        {"name": "word_lr_c10_balanced", "feature_set": "word", "classifier": "logistic_regression", "c": 10.0, "class_weight": "balanced"},
        {"name": "character_lr_c1_balanced", "feature_set": "character", "classifier": "logistic_regression", "c": 1.0, "class_weight": "balanced"},
        {"name": "hybrid_lr_c1_balanced", "feature_set": "word_and_character", "classifier": "logistic_regression", "c": 1.0, "class_weight": "balanced"},
        {"name": "word_svm_c1_balanced", "feature_set": "word", "classifier": "linear_svm", "c": 1.0, "class_weight": "balanced"},
    ]


def _scores(model, texts, ai_label):
    return ai_decision_scores(model, texts, ai_label)


def _make_splits(dataset, human_label, ai_label):
    sources = np.asarray(dataset["source"], dtype=object)
    sub_sources = np.asarray(dataset["sub_source"], dtype=object)
    labels = np.asarray(dataset["label"], dtype=int)
    test_mask = sources == "hc3"
    development_mask = np.isin(sources, ["mage", "m4gt"])
    if not np.any(test_mask) or not np.any(development_mask):
        raise ValueError("Expected mage/m4gt development rows and a distinct hc3 source holdout.")
    development_indices = np.flatnonzero(development_mask)
    test_indices = np.flatnonzero(test_mask)
    groups = np.asarray([
        _group_key(sources[index], sub_sources[index]) for index in development_indices
    ], dtype=object)
    if len(set(groups)) < 8:
        raise ValueError("Source-aware development splitting requires at least eight groups.")

    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_cal_relative, validation_relative = next(
        outer.split(development_indices, labels[development_indices], groups)
    )
    train_cal_indices = development_indices[train_cal_relative]
    validation_indices = development_indices[validation_relative]
    train_cal_groups = groups[train_cal_relative]
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED + 1)
    train_relative, calibration_relative = next(
        inner.split(
            train_cal_indices,
            labels[train_cal_indices],
            train_cal_groups,
        )
    )
    train_indices = train_cal_indices[train_relative]
    calibration_indices = train_cal_indices[calibration_relative]
    train_indices = _stratified_sample(train_indices, labels, TRAIN_LIMIT, SEED + 2)
    calibration_indices = _stratified_sample(
        calibration_indices, labels, CALIBRATION_LIMIT, SEED + 3
    )
    validation_indices = _stratified_sample(
        validation_indices, labels, VALIDATION_LIMIT, SEED + 4
    )
    test_indices = _stratified_sample(test_indices, labels, TEST_LIMIT, SEED + 5)
    partitions = {
        "train": train_indices,
        "calibration": calibration_indices,
        "validation": validation_indices,
        "test": test_indices,
    }
    partitions, duplicates_removed = _deduplicate(dataset, partitions)

    def partition_group_set(name):
        return {
            _group_key(sources[index], sub_sources[index])
            for index in partitions[name]
        }

    if any(
        partition_group_set(left) & partition_group_set(right)
        for left, right in (("train", "calibration"), ("train", "validation"), ("calibration", "validation"))
    ):
        raise RuntimeError("Development source-group leakage detected.")
    if any(sources[index] != "hc3" for index in partitions["test"]):
        raise RuntimeError("External test set contains non-HC3 rows.")
    if any(sources[index] == "hc3" for name in ("train", "calibration", "validation")
           for index in partitions[name]):
        raise RuntimeError("HC3 leakage into development partitions detected.")
    return labels, sources, sub_sources, partitions, duplicates_removed


def _partition_data(dataset, labels, partitions):
    content = {}
    for name, indices in partitions.items():
        rows = dataset.select(indices.tolist())
        content[name] = (rows["text"], labels[indices])
    return content


def _selection_eligible(validation_report):
    controlled = validation_report["risk_controlled"]
    return (
        controlled["thresholds"]["targets_met"]
        and controlled["coverage"] >= MINIMUM_VALIDATION_COVERAGE
        and controlled["ai_recall"] > 0
        and controlled["ai_precision"] is not None
        and controlled["human_verdict_precision"] is not None
    )


def main():
    from datasets import concatenate_datasets, load_dataset
    import datasets
    import sklearn

    print(f"Loading {DATASET_NAME}...")
    source_dataset = load_dataset(DATASET_NAME)
    for split_name in ("train", "dev"):
        missing = {"text", "label", "model", "source", "sub_source"} - set(source_dataset[split_name].column_names)
        if missing:
            raise ValueError(f"Dataset split {split_name!r} is missing required columns {sorted(missing)}.")
    mapping_labels = list(source_dataset["train"]["label"]) + list(source_dataset["dev"]["label"])
    mapping_models = list(source_dataset["train"]["model"]) + list(source_dataset["dev"]["model"])
    mapping = verify_label_mapping(mapping_labels, mapping_models)
    human_label = mapping["human"]
    ai_label = mapping["ai_generated"]
    dataset = concatenate_datasets([source_dataset["train"], source_dataset["dev"]])
    if not len(dataset):
        raise ValueError("Dataset has no rows.")
    labels, sources, sub_sources, partitions, duplicates_removed = _make_splits(
        dataset, human_label, ai_label
    )
    data = _partition_data(dataset, labels, partitions)
    for name in ("train", "calibration", "validation", "test"):
        split_labels = data[name][1]
        if set(np.unique(split_labels).tolist()) != {human_label, ai_label}:
            raise ValueError(f"The {name} partition does not contain both verified classes.")

    train_texts, train_labels = data["train"]
    validation_texts, validation_labels = data["validation"]
    results = []
    candidate_specs = _candidate_specs()
    for candidate_number, spec in enumerate(candidate_specs, start=1):
        print(f"[{candidate_number}/{len(candidate_specs)}] Fitting {spec['name']}...")
        candidate = _build_model(
            spec["feature_set"], spec["classifier"], spec["c"], spec["class_weight"]
        )
        candidate.fit(train_texts, train_labels)
        validation_scores = _scores(candidate, validation_texts, ai_label)
        thresholds = select_risk_thresholds(
            validation_labels,
            validation_scores,
            human_label,
            ai_label,
            max_human_false_positive_rate=MAX_HUMAN_FALSE_POSITIVE_RATE,
            max_ai_false_negative_rate=MAX_AI_FALSE_NEGATIVE_RATE,
            minimum_verdicts=MINIMUM_VERDICTS,
        )
        validation_report = evaluate_scores(
            validation_labels, validation_scores, human_label, ai_label, thresholds
        )
        entry = {
            **spec,
            "validation": validation_report,
            "eligible_for_selection": _selection_eligible(validation_report),
        }
        results.append(entry)
        del candidate
        gc.collect()

    eligible = [item for item in results if item["eligible_for_selection"]]
    if eligible:
        selected = max(
            eligible,
            key=lambda item: (
                item["validation"]["risk_controlled"]["ai_recall"],
                item["validation"]["risk_controlled"]["coverage"],
                item["validation"]["default_zero_threshold"]["roc_auc"] or 0.0,
                item["validation"]["default_zero_threshold"]["macro_f1"],
            ),
        )
        selection_status = "candidate met both validation risk targets and minimum validation coverage"
    else:
        selected = max(
            results,
            key=lambda item: (
                item["validation"]["default_zero_threshold"]["roc_auc"] or 0.0,
                item["validation"]["default_zero_threshold"]["macro_f1"],
            ),
        )
        selection_status = "no candidate met the validation risk gate; selected highest validation ROC AUC for diagnostic testing only"

    baseline = next(item for item in results if item["name"] == "word_lr_c1_unweighted")
    selected_model = _build_model(
        selected["feature_set"], selected["classifier"], selected["c"], selected["class_weight"]
    ).fit(train_texts, train_labels)
    baseline_model = _build_model(
        baseline["feature_set"], baseline["classifier"], baseline["c"], baseline["class_weight"]
    ).fit(train_texts, train_labels)
    calibration_texts, calibration_labels = data["calibration"]
    calibrator = fit_probability_calibrator(
        selected_model, calibration_texts, calibration_labels, ai_label
    )

    test_texts, test_labels = data["test"]
    test_scores = _scores(selected_model, test_texts, ai_label)
    baseline_test_scores = _scores(baseline_model, test_texts, ai_label)
    test_results = {}
    for item, score in ((baseline, baseline_test_scores), (selected, test_scores)):
        test_results[item["name"]] = evaluate_scores(
            test_labels,
            score,
            human_label,
            ai_label,
            item["validation"]["risk_controlled"]["thresholds"],
        )
    del baseline_model
    gc.collect()

    ai_index = list(selected_model.classes_).index(ai_label)
    if hasattr(selected_model, "predict_proba"):
        raw_ai_probabilities = selected_model.predict_proba(test_texts)[:, ai_index]
    else:
        raw_ai_probabilities = None
    calibrated_probabilities = calibrator.predict_proba(
        test_scores.reshape(-1, 1)
    )[:, list(calibrator.classes_).index(ai_label)]
    from src.verity_analysis import probability_diagnostics

    calibration_report = {
        "calibration_fit_partition_size": len(calibration_labels),
        "raw_probability_diagnostics": (
            probability_diagnostics(test_labels, raw_ai_probabilities, ai_label)
            if raw_ai_probabilities is not None else None
        ),
        "platt_calibrated_probability_diagnostics": probability_diagnostics(
            test_labels, calibrated_probabilities, ai_label
        ),
        "interpretation": "Use the decision score and validation risk thresholds in the app. Calibration diagnostics are not sufficient to claim probability reliability.",
    }

    validation_source = sources[partitions["validation"]]
    validation_sub_source = sub_sources[partitions["validation"]]
    test_source = sources[partitions["test"]]
    test_sub_source = sub_sources[partitions["test"]]
    test_model_names = np.asarray(dataset.select(partitions["test"].tolist())["model"], dtype=object)
    test_results[selected["name"]]["per_collection_source"] = grouped_metrics(
        test_labels, test_scores, test_source, human_label, ai_label,
        selected["validation"]["risk_controlled"]["thresholds"],
    )
    test_results[selected["name"]]["per_sub_source"] = grouped_metrics(
        test_labels, test_scores, test_sub_source, human_label, ai_label,
        selected["validation"]["risk_controlled"]["thresholds"],
    )
    test_results[selected["name"]]["per_generator_or_human_metadata"] = grouped_metrics(
        test_labels, test_scores, test_model_names, human_label, ai_label,
        selected["validation"]["risk_controlled"]["thresholds"],
    )

    report = {
        "dataset": DATASET_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "dataset_rows": len(dataset),
        "label_mapping": {**mapping, "class_order": ["human", "ai_generated"]},
        "protocol": {
            "development_sources": ["mage", "m4gt"],
            "external_test_source": "hc3",
            "validation": "one deterministic StratifiedGroupKFold sub_source/source group held out from mage+m4gt",
            "calibration": "separate deterministic group fold from development data, used only to fit Platt scaling",
            "test": "HC3 source held entirely out from training, hyperparameter selection, calibration fitting, and threshold selection",
            "test_limit": TEST_LIMIT,
            "exact_duplicate_removal": "SHA-256 across sampled partitions; earlier partition retained",
            "near_duplicates": "not detected; source groups and exact hashes are used",
            "threshold_selection": {
                "human_false_positive_target": MAX_HUMAN_FALSE_POSITIVE_RATE,
                "ai_false_negative_rate_target": MAX_AI_FALSE_NEGATIVE_RATE,
                "confidence_bound": "one-sided 95% Wilson upper bound on validation data",
                "minimum_verdicts_per_class": MINIMUM_VERDICTS,
                "minimum_validation_coverage": MINIMUM_VALIDATION_COVERAGE,
                "selection_metric_order": ["validation AI recall", "validation coverage", "ROC AUC", "macro F1"],
            },
        },
        "split_sizes": {
            name: {
                "total": len(indices),
                "human": int(np.count_nonzero(labels[indices] == human_label)),
                "ai_generated": int(np.count_nonzero(labels[indices] == ai_label)),
                "source_counts": dict(Counter(sources[indices])),
            }
            for name, indices in partitions.items()
        },
        "duplicate_counts_removed": duplicates_removed,
        "validation_source_counts": dict(Counter(validation_source)),
        "validation_sub_source_counts": dict(Counter(validation_sub_source)),
        "candidates": results,
        "selection": {
            "selected_candidate": selected["name"],
            "status": selection_status,
            "eligible_candidate_count": len(eligible),
            "was_selected_using_test": False,
        },
        "same_test_set_comparison": {
            "baseline_candidate": baseline["name"],
            "selected_candidate": selected["name"],
            "baseline": test_results[baseline["name"]],
            "selected": test_results[selected["name"]],
        },
        "calibration": calibration_report,
        "versions": {
            "python": platform.python_version(),
            "datasets": datasets.__version__,
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
            "joblib": joblib.__version__,
        },
        "limitations": [
            "The external result is a held-out collection source within this dataset, not an independently curated benchmark corpus.",
            "The HC3 rows may share upstream text, prompts, or generation processes with other collections despite source separation.",
            "Near-duplicate detection is not performed; exact duplicate hashing and source group separation are used.",
            "Thresholds are selected on one group-held-out validation fold and may not transport to new domains or generators.",
            "A score threshold is not an authorship probability. Calibrated probability diagnostics are reported separately and are not assumed reliable.",
            "A detector can learn topic, collection, and generation artifacts; model-derived feature contributions describe score evidence, not authorship proof.",
        ],
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, CANDIDATE_PATH)
    joblib.dump(selected_model, DEPLOYED_MODEL_PATH)
    joblib.dump(calibrator, CALIBRATOR_CANDIDATE_PATH)
    joblib.dump(calibrator, DEPLOYED_CALIBRATOR_PATH)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Selected candidate: {selected['name']}")
    print(f"Selection status: {selection_status}")
    print(json.dumps({
        "baseline_same_test": test_results[baseline["name"]],
        "selected_same_test": test_results[selected["name"]],
        "calibration": calibration_report,
        "report": str(REPORT_PATH),
    }, indent=2))
    print(f"Candidate model saved to {CANDIDATE_PATH}")
    print(f"Compatible model artifact saved to {DEPLOYED_MODEL_PATH}")
    print(f"Candidate calibrator saved to {CALIBRATOR_CANDIDATE_PATH}")
    print(f"Compatible calibrator artifact saved to {DEPLOYED_CALIBRATOR_PATH}")


if __name__ == "__main__":
    main()