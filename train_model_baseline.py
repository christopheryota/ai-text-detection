"""Train and evaluate Verity AI's TF-IDF/logistic-regression baseline."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline

from src.verity_analysis import (
    ai_decision_scores,
    compute_evaluation_metrics,
    fit_probability_calibrator,
    require_minimum_class_counts,
    verify_label_mapping,
)


DATASET_NAME = "Jinyan1/COLING_2025_MGT_en"
SEED = 42
TRAIN_LIMIT = 20_000
CALIBRATION_LIMIT = 5_000
TEST_LIMIT = 5_000
MIN_CALIBRATION_PER_CLASS = 20
MIN_TEST_PER_CLASS = 20
UNCERTAINTY_BAND = (0.40, 0.60)
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "ai_text_detector.joblib"
CALIBRATOR_PATH = MODEL_DIR / "ai_text_detector_calibrator.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"


def _verified_mapping_from_dataset(dataset):
    required_columns = {"text", "label", "model"}
    all_labels = []
    all_model_names = []
    for split_name in ("train", "dev"):
        if split_name not in dataset:
            raise ValueError(f"Dataset is missing the required {split_name!r} split.")
        split = dataset[split_name]
        missing = required_columns - set(split.column_names)
        if missing:
            raise ValueError(f"Dataset split {split_name!r} is missing columns: {sorted(missing)}")
        if len(split) == 0:
            raise ValueError(f"Dataset split {split_name!r} is empty.")
        all_labels.extend(split["label"])
        all_model_names.extend(split["model"])
    return verify_label_mapping(all_labels, all_model_names)


def _group_ids(dataset):
    columns = set(dataset.column_names)
    group_column = "sub_source" if "sub_source" in columns else "source" if "source" in columns else None
    if group_column is None:
        return None, "none"

    groups = []
    identifiers = dataset["id"] if "id" in columns else None
    for index, value in enumerate(dataset[group_column]):
        normalized = str(value).strip()
        if not normalized or normalized.casefold() in {"none", "nan"}:
            identifier = identifiers[index] if identifiers is not None else index
            normalized = f"missing-source-row-{identifier}"
        groups.append(normalized)
    return np.asarray(groups, dtype=object), group_column


def _split_indices(labels, groups, seed):
    indices = np.arange(len(labels))
    if groups is None:
        remaining, test = train_test_split(
            indices, test_size=0.20, random_state=seed, stratify=labels
        )
        train, calibration = train_test_split(
            remaining,
            test_size=0.1875,
            random_state=seed + 1,
            stratify=np.asarray(labels)[remaining],
        )
    else:
        if len(set(groups)) < 5:
            raise ValueError(
                "Grouped evaluation requires at least five distinct source groups; "
                "the available data cannot support separate train/calibration/test groups."
            )
        outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        remaining, test = next(outer.split(indices, labels, groups))
        inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed + 1)
        train_relative, calibration_relative = next(
            inner.split(
                remaining,
                np.asarray(labels)[remaining],
                np.asarray(groups)[remaining],
            )
        )
        train = remaining[train_relative]
        calibration = remaining[calibration_relative]

    partitions = {"train": train, "calibration": calibration, "test": test}
    if groups is not None:
        group_sets = {
            name: set(groups[indices]) for name, indices in partitions.items()
        }
        if (group_sets["train"] & group_sets["calibration"]
                or group_sets["train"] & group_sets["test"]
                or group_sets["calibration"] & group_sets["test"]):
            raise RuntimeError("Source-group leakage detected while creating splits.")
    return partitions


def _sample_indices(indices, labels, limit, seed):
    if len(indices) <= limit:
        return np.asarray(indices, dtype=int)
    selected, _ = train_test_split(
        np.asarray(indices),
        train_size=limit,
        random_state=seed,
        stratify=np.asarray(labels)[indices],
    )
    return np.sort(selected)


def _deduplicate_partitions(dataset, partitions):
    seen = set()
    clean = {}
    removed = {}
    for split_name in ("train", "calibration", "test"):
        indices = partitions[split_name]
        texts = dataset.select(indices.tolist())["text"]
        kept = []
        duplicate_count = 0
        for index, text in zip(indices, texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"Split {split_name!r} contains an empty or invalid text.")
            fingerprint = hashlib.sha256(text.encode("utf-8")).digest()
            if fingerprint in seen:
                duplicate_count += 1
            else:
                seen.add(fingerprint)
                kept.append(int(index))
        clean[split_name] = np.asarray(kept, dtype=int)
        removed[split_name] = duplicate_count
    return clean, removed


def _counts(labels, human_label, ai_label):
    return {
        "total": len(labels),
        "human": int(np.count_nonzero(np.asarray(labels) == human_label)),
        "ai_generated": int(np.count_nonzero(np.asarray(labels) == ai_label)),
    }


def main():
    from datasets import concatenate_datasets, load_dataset
    import datasets
    import sklearn

    print(f"Loading dataset {DATASET_NAME}...")
    source_dataset = load_dataset(DATASET_NAME)
    label_mapping = _verified_mapping_from_dataset(source_dataset)
    human_label = label_mapping["human"]
    ai_label = label_mapping["ai_generated"]

    combined = concatenate_datasets([source_dataset["train"], source_dataset["dev"]])
    if len(combined) == 0:
        raise ValueError("The combined dataset is empty.")
    labels = np.asarray(combined["label"], dtype=int)
    unexpected = set(labels.tolist()) - {human_label, ai_label}
    if unexpected:
        raise ValueError(f"Unexpected labels after mapping verification: {sorted(unexpected)}")

    groups, grouping_method = _group_ids(combined)
    partitions = _split_indices(labels, groups, SEED)
    partitions = {
        "train": _sample_indices(partitions["train"], labels, TRAIN_LIMIT, SEED),
        "calibration": _sample_indices(
            partitions["calibration"], labels, CALIBRATION_LIMIT, SEED + 1
        ),
        "test": _sample_indices(partitions["test"], labels, TEST_LIMIT, SEED + 2),
    }
    partitions, duplicates_removed = _deduplicate_partitions(combined, partitions)

    train_labels = labels[partitions["train"]]
    calibration_labels = labels[partitions["calibration"]]
    test_labels = labels[partitions["test"]]
    require_minimum_class_counts(train_labels, human_label, ai_label, 2, "training")
    require_minimum_class_counts(
        calibration_labels, human_label, ai_label, MIN_CALIBRATION_PER_CLASS, "calibration"
    )
    require_minimum_class_counts(
        test_labels, human_label, ai_label, MIN_TEST_PER_CLASS, "test"
    )

    train_rows = combined.select(partitions["train"].tolist())
    calibration_rows = combined.select(partitions["calibration"].tolist())
    test_rows = combined.select(partitions["test"].tolist())
    X_train, y_train = train_rows["text"], train_labels.tolist()
    X_calibration, y_calibration = calibration_rows["text"], calibration_labels.tolist()
    X_test, y_test = test_rows["text"], test_labels.tolist()

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=30_000,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("classifier", LogisticRegression(max_iter=1_000, random_state=SEED)),
    ])
    print(f"Training on {len(X_train):,} examples...")
    model.fit(X_train, y_train)

    print(f"Fitting calibrator on {len(X_calibration):,} calibration examples...")
    calibrator = fit_probability_calibrator(model, X_calibration, y_calibration, ai_label)

    print(f"Evaluating on {len(X_test):,} untouched test examples...")
    ai_index = list(model.classes_).index(ai_label)
    raw_probabilities = model.predict_proba(X_test)[:, ai_index]
    ai_scores = ai_decision_scores(model, X_test, ai_label).reshape(-1, 1)
    calibrated_probabilities = calibrator.predict_proba(ai_scores)[
        :, list(calibrator.classes_).index(ai_label)
    ]
    metrics = compute_evaluation_metrics(
        y_test,
        raw_probabilities,
        calibrated_probabilities,
        human_label,
        ai_label,
        UNCERTAINTY_BAND,
    )

    split_sizes = {
        "train": _counts(y_train, human_label, ai_label),
        "calibration": _counts(y_calibration, human_label, ai_label),
        "test": _counts(y_test, human_label, ai_label),
    }
    group_counts = None if groups is None else {
        name: len(set(groups[indices])) for name, indices in partitions.items()
    }
    report = {
        "dataset": DATASET_NAME,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "random_seed": SEED,
        "label_mapping": {
            "human": human_label,
            "ai_generated": ai_label,
            "class_order": ["human", "ai_generated"],
            "verified_from": "dataset model metadata: human identifier versus generator/model identifiers",
        },
        "split_sizes": split_sizes,
        "source_grouping": {"columns": grouping_method, "group_counts": group_counts},
        "exact_duplicates_removed": duplicates_removed,
        "uncertainty_band": {
            "metric": "calibrated AI-generated score",
            "human_below": UNCERTAINTY_BAND[0],
            "ai_above": UNCERTAINTY_BAND[1],
            "inclusive_middle_is_inconclusive": True,
        },
        "metrics": metrics,
        "versions": {
            "python": platform.python_version(),
            "datasets": datasets.__version__,
            "scikit_learn": sklearn.__version__,
            "joblib": joblib.__version__,
            "numpy": np.__version__,
        },
        "limitations": [
            "Exact duplicate texts are removed across sampled partitions; near-duplicate detection is not performed.",
            "Grouping uses sub_source (or source when unavailable); missing group values are isolated by row ID.",
            "The held-out test measures this dataset and its source-group distribution, not all real-world writing or future generators.",
            "Feature contributions explain the fitted linear score; they are not proof of authorship or causal explanations.",
            "Calibration diagnostics are reported; probability reliability is not asserted from these values alone.",
        ],
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(calibrator, CALIBRATOR_PATH)
    METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Calibrator saved to {CALIBRATOR_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    print(json.dumps({"split_sizes": split_sizes, "test_metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()