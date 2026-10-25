"""Evaluation and explanation helpers for Verity AI's linear baseline."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


def verify_label_mapping(labels, model_names):
    """Verify human/AI labels from the dataset's documented model metadata."""
    if len(labels) != len(model_names) or len(labels) == 0:
        raise ValueError("Cannot verify labels without equally sized, non-empty label and model metadata.")

    human_labels = set()
    generated_labels = set()
    for label, model_name in zip(labels, model_names):
        if isinstance(label, (bool, np.bool_)) or not isinstance(label, (int, np.integer)):
            raise ValueError(f"Expected integer dataset labels, found {label!r}.")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("Every dataset row needs a non-empty model metadata value to verify labels.")
        if model_name.strip().casefold() in {"human", "human-written", "human_written"}:
            human_labels.add(int(label))
        else:
            generated_labels.add(int(label))

    if len(human_labels) != 1 or len(generated_labels) != 1:
        raise ValueError(
            "Could not verify a one-to-one human/generated label mapping from the dataset's model metadata. "
            f"Human metadata labels={sorted(human_labels)}, generated metadata labels={sorted(generated_labels)}."
        )
    human_label = next(iter(human_labels))
    ai_label = next(iter(generated_labels))
    if human_label == ai_label:
        raise ValueError(
            "Dataset model metadata does not establish a one-to-one mapping; "
            "human and generated text use the same label."
        )
    if set(int(label) for label in labels) != {human_label, ai_label}:
        raise ValueError("Dataset contains labels not explained by its human/generated model metadata.")
    return {"human": human_label, "ai_generated": ai_label}


def require_minimum_class_counts(labels, human_label, ai_label, minimum, split_name):
    """Fail when either class is too small for the requested split's role."""
    counts = {
        human_label: int(np.count_nonzero(np.asarray(labels) == human_label)),
        ai_label: int(np.count_nonzero(np.asarray(labels) == ai_label)),
    }
    if min(counts.values()) < minimum:
        raise ValueError(
            f"The {split_name} split needs at least {minimum} examples per class; "
            f"found human={counts[human_label]}, AI-generated={counts[ai_label]}."
        )
    imbalance_ratio = max(counts.values()) / min(counts.values())
    if imbalance_ratio > 20:
        raise ValueError(
            f"The {split_name} split is too imbalanced for this evaluation "
            f"(majority/minority ratio {imbalance_ratio:.1f}; maximum allowed is 20)."
        )
    return counts


def ai_decision_scores(model, texts, ai_label):
    """Return binary decision scores oriented toward the AI-generated class."""
    classes = list(model.classes_)
    if len(classes) != 2 or ai_label not in classes:
        raise ValueError("The explanation/calibration helpers require a fitted binary classifier.")
    scores = np.asarray(model.decision_function(texts), dtype=float).reshape(-1)
    return scores if classes[1] == ai_label else -scores


def fit_probability_calibrator(model, texts, labels, ai_label):
    """Fit Platt scaling on calibration scores only; do not pass test rows."""
    scores = ai_decision_scores(model, texts, ai_label).reshape(-1, 1)
    labels = np.asarray(labels)
    if len(scores) != len(labels) or len(set(labels.tolist())) != 2:
        raise ValueError("Calibration requires aligned scores and examples from both classes.")
    calibrator = LogisticRegression(max_iter=1_000, random_state=42)
    calibrator.fit(scores, labels)
    return calibrator


def _probability_diagnostics(labels, probabilities, positive_label, bins=10):
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    observed = (labels == positive_label).astype(int)
    if len(observed) != len(probabilities) or not len(observed):
        raise ValueError("Probability diagnostics require aligned, non-empty inputs.")
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite values between 0 and 1.")

    edges = np.linspace(0.0, 1.0, bins + 1)
    bin_data = []
    for index in range(bins):
        mask = (probabilities >= edges[index]) & (
            probabilities <= edges[index + 1] if index == bins - 1
            else probabilities < edges[index + 1]
        )
        count = int(mask.sum())
        bin_data.append({
            "lower": float(edges[index]),
            "upper": float(edges[index + 1]),
            "count": count,
            "mean_predicted_probability": float(probabilities[mask].mean()) if count else None,
            "observed_ai_fraction": float(observed[mask].mean()) if count else None,
        })

    ece = sum(
        item["count"] / len(observed)
        * abs(item["mean_predicted_probability"] - item["observed_ai_fraction"])
        for item in bin_data if item["count"]
    )
    return {
        "brier_score": float(np.mean((probabilities - observed) ** 2)),
        "log_loss": float(log_loss(observed, probabilities, labels=[0, 1])),
        "expected_calibration_error": float(ece),
        "calibration_bins": bin_data,
    }


def probability_diagnostics(labels, probabilities, positive_label, bins=10):
    """Public Brier, ECE and calibration-bin summary for held-out data."""
    return _probability_diagnostics(labels, probabilities, positive_label, bins)


def wilson_interval(successes, total, z=1.959963984540054):
    """Return a two-sided 95% Wilson score interval for a binomial rate."""
    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson interval requires 0 <= successes <= total and total > 0.")
    rate = successes / total
    denominator = 1 + z**2 / total
    center = (rate + z**2 / (2 * total)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / total + z**2 / (4 * total**2)) / denominator
    return [float(max(0.0, center - margin)), float(min(1.0, center + margin))]


def _class_metrics(labels, predictions, human_label, ai_label):
    raw_report = classification_report(
        labels,
        predictions,
        labels=[human_label, ai_label],
        target_names=["human", "ai_generated"],
        output_dict=True,
        zero_division=0,
    )
    return {
        name: {
            "precision": float(raw_report[name]["precision"]),
            "recall": float(raw_report[name]["recall"]),
            "f1_score": float(raw_report[name]["f1-score"]),
            "support": int(raw_report[name]["support"]),
        }
        for name in ("human", "ai_generated")
    }


def compute_evaluation_metrics(
    labels, raw_probabilities, calibrated_probabilities,
    human_label, ai_label, uncertainty_band,
):
    """Summarize untouched-test predictions, calibration, and abstention."""
    labels = np.asarray(labels)
    raw_probabilities = np.asarray(raw_probabilities, dtype=float)
    calibrated_probabilities = np.asarray(calibrated_probabilities, dtype=float)
    if not (len(labels) == len(raw_probabilities) == len(calibrated_probabilities)) or not len(labels):
        raise ValueError("Evaluation labels and probabilities must be aligned and non-empty.")
    if set(np.unique(labels).tolist()) != {human_label, ai_label}:
        raise ValueError("Evaluation labels must contain exactly the verified human and AI classes.")
    low, high = uncertainty_band
    if not 0 <= low < high <= 1:
        raise ValueError("The uncertainty band must satisfy 0 <= lower < upper <= 1.")

    raw_predictions = np.where(raw_probabilities >= 0.5, ai_label, human_label)
    predictions = np.where(calibrated_probabilities >= 0.5, ai_label, human_label)
    human_mask = labels == human_label
    ai_mask = labels == ai_label
    false_positive_count = int(np.count_nonzero(human_mask & (predictions == ai_label)))
    false_negative_count = int(np.count_nonzero(ai_mask & (predictions == human_label)))
    raw_false_positive_count = int(np.count_nonzero(human_mask & (raw_predictions == ai_label)))
    raw_false_negative_count = int(np.count_nonzero(ai_mask & (raw_predictions == human_label)))

    def summary(y, probabilities, predicted):
        result = {
            "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=[human_label, ai_label], average="macro", zero_division=0)),
            "per_class": _class_metrics(y, predicted, human_label, ai_label),
            "confusion_matrix": confusion_matrix(y, predicted, labels=[human_label, ai_label]).tolist(),
            "confusion_matrix_class_order": ["human", "ai_generated"],
            "probability_diagnostics": _probability_diagnostics(y, probabilities, ai_label),
        }
        try:
            result["roc_auc"] = float(roc_auc_score((y == ai_label).astype(int), probabilities))
        except ValueError:
            result["roc_auc"] = None
        return result

    decisive = (calibrated_probabilities < low) | (calibrated_probabilities > high)
    inconclusive = ~decisive
    decisive_count = int(decisive.sum())
    decisive_errors = int(np.count_nonzero(predictions[decisive] != labels[decisive]))
    return {
        "raw": summary(labels, raw_probabilities, raw_predictions),
        "calibrated": summary(labels, calibrated_probabilities, predictions),
        "human_incorrectly_flagged_as_ai": {
            "count": false_positive_count,
            "denominator": int(human_mask.sum()),
            "rate": float(false_positive_count / human_mask.sum()) if human_mask.any() else None,
            "wilson_95_percent_interval": wilson_interval(false_positive_count, int(human_mask.sum())) if human_mask.any() else None,
        },
        "ai_generated_incorrectly_labelled_human": {
            "count": false_negative_count,
            "denominator": int(ai_mask.sum()),
            "rate": float(false_negative_count / ai_mask.sum()) if ai_mask.any() else None,
        },
        "raw_human_incorrectly_flagged_as_ai": {
            "count": raw_false_positive_count,
            "denominator": int(human_mask.sum()),
            "rate": float(raw_false_positive_count / human_mask.sum()) if human_mask.any() else None,
            "wilson_95_percent_interval": wilson_interval(raw_false_positive_count, int(human_mask.sum())) if human_mask.any() else None,
        },
        "raw_ai_generated_incorrectly_labelled_human": {
            "count": raw_false_negative_count,
            "denominator": int(ai_mask.sum()),
            "rate": float(raw_false_negative_count / ai_mask.sum()) if ai_mask.any() else None,
        },
        "uncertainty_band": {
            "lower_inclusive": float(low),
            "upper_inclusive": float(high),
            "inconclusive_count": int(inconclusive.sum()),
            "inconclusive_coverage": float(inconclusive.mean()),
            "decisive_count": decisive_count,
            "decisive_coverage": float(decisive.mean()),
            "decisive_error_count": decisive_errors,
            "decisive_error_rate": float(decisive_errors / decisive_count) if decisive_count else None,
            "decisive_human_error_rate": (
                float(np.count_nonzero(human_mask & decisive & (predictions == ai_label))
                      / np.count_nonzero(human_mask & decisive))
                if np.any(human_mask & decisive) else None
            ),
            "decisive_ai_error_rate": (
                float(np.count_nonzero(ai_mask & decisive & (predictions == human_label))
                      / np.count_nonzero(ai_mask & decisive))
                if np.any(ai_mask & decisive) else None
            ),
        },
    }


def explain_prediction(model, text, ai_label, top_n=None):
    """Compute faithful signed feature contributions toward AI-generated."""
    if top_n is not None and top_n < 0:
        raise ValueError("top_n must be non-negative or None.")
    vectorizer = model.named_steps.get("features", model.named_steps.get("tfidf"))
    classifier = model.named_steps["classifier"]
    if (len(classifier.classes_) != 2 or ai_label not in classifier.classes_
            or not hasattr(classifier, "coef_") or not hasattr(classifier, "intercept_")):
        raise ValueError("Explanations require a fitted binary linear classifier with coefficients.")
    vector = vectorizer.transform([text])
    coefficients = classifier.coef_[0].copy()
    if classifier.classes_[1] != ai_label:
        coefficients *= -1
    contributions = vector.multiply(coefficients).tocoo()
    features = vectorizer.get_feature_names_out()

    ranked = []
    for feature_index, value in zip(contributions.col, contributions.data):
        feature_name = str(features[feature_index])
        if feature_name.startswith("word__"):
            feature_type = "word_ngram"
            term = feature_name.removeprefix("word__")
            description = "Word or adjacent-word sequence measured by the word TF-IDF vectorizer."
        elif feature_name.startswith("character__"):
            feature_type = "character_ngram"
            term = feature_name.removeprefix("character__")
            description = f"Character sequence ({len(term)} characters) measured by the character TF-IDF vectorizer."
        else:
            feature_type = "tfidf_feature"
            term = feature_name
            description = "Feature emitted by the fitted TF-IDF vectorizer."
        feature_value = float(vector[0, feature_index])
        coefficient = float(coefficients[feature_index])
        contribution = float(value)
        direction = "AI-associated" if contribution > 0 else "human-associated"
        ranked.append({
            "rank": 0,
            "feature_name": feature_name,
            "term": term,
            "feature_type": feature_type,
            "description": description,
            "tfidf_value": feature_value,
            "ai_oriented_coefficient": coefficient,
            "contribution": contribution,
            "absolute_contribution": abs(contribution),
            "direction": direction,
        })
    ranked.sort(key=lambda item: (-item["absolute_contribution"], item["feature_name"]))
    for rank, item in enumerate(ranked, start=1):
        item["rank"] = rank

    all_contribution_sum = float(sum(item["contribution"] for item in ranked))
    intercept = float(
        classifier.intercept_[0]
        if classifier.classes_[1] == ai_label
        else -classifier.intercept_[0]
    )
    decision_score = float(ai_decision_scores(model, [text], ai_label)[0])
    reconstruction_error = float(intercept + all_contribution_sum - decision_score)
    toward_ai = [item for item in ranked if item["contribution"] > 0]
    toward_human = [item for item in ranked if item["contribution"] < 0]
    displayed_ranked = ranked if top_n is None else ranked[:top_n]
    displayed_ai = [item for item in displayed_ranked if item["contribution"] > 0]
    displayed_human = [item for item in displayed_ranked if item["contribution"] < 0]
    ai_mass = float(sum(item["contribution"] for item in toward_ai))
    human_mass = float(-sum(item["contribution"] for item in toward_human))
    total_mass = ai_mass + human_mass
    ai_bar_width = 50.0 if total_mass == 0 else 100 * ai_mass / total_mass
    human_bar_width = 50.0 if total_mass == 0 else 100 * human_mass / total_mass
    return {
        "toward_ai": displayed_ai,
        "toward_human": displayed_human,
        "ranked_contributions": displayed_ranked,
        "all_feature_contributions": ranked,
        "active_feature_count": int(vector.nnz),
        "ai_contribution_mass": ai_mass,
        "human_contribution_mass": human_mass,
        "ai_bar_width": float(ai_bar_width),
        "human_bar_width": float(human_bar_width),
        "feature_contribution_sum": all_contribution_sum,
        "intercept": intercept,
        "decision_score": decision_score,
        "reconstructed_score": float(intercept + all_contribution_sum),
        "reconstruction_error": reconstruction_error,
        "contribution_definition": "TF-IDF feature value multiplied by its fitted logistic-regression coefficient, oriented toward AI; the sum plus intercept equals the raw decision score, which is not a probability.",
        "warning": "These are measurements of this model's learned associations, not proof of authorship or causal explanations.",
    }