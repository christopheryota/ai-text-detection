"""Risk-controlled threshold selection and evaluation for raw model scores."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.verity_analysis import probability_diagnostics, wilson_interval


ONE_SIDED_95_Z = 1.6448536269514722


def _upper_rate(errors, total):
    return wilson_interval(errors, total, z=ONE_SIDED_95_Z)[1]


def select_risk_thresholds(
    labels,
    scores,
    human_label,
    ai_label,
    max_human_false_positive_rate=0.05,
    max_ai_false_negative_rate=0.10,
    minimum_verdicts=20,
):
    """Select two decision boundaries using validation data only.

    A confident AI decision must satisfy an upper one-sided 95% Wilson bound on
    human false-positive rate. A confident human decision must satisfy a separate
    upper bound on AI false-negative rate over all AI examples. Values between
    the boundaries are abstentions. The returned score is not a probability.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("Threshold selection requires aligned, non-empty labels and scores.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Threshold selection scores must be finite.")
    if set(np.unique(labels).tolist()) != {human_label, ai_label}:
        raise ValueError("Threshold selection requires examples from both verified classes.")
    if not 0 < max_human_false_positive_rate < 1:
        raise ValueError("The human false-positive target must be between zero and one.")
    if not 0 < max_ai_false_negative_rate < 1:
        raise ValueError("The AI false-negative target must be between zero and one.")

    human_scores = np.sort(scores[labels == human_label])
    ai_scores = np.sort(scores[labels == ai_label])
    thresholds = np.unique(scores)

    ai_candidates = []
    for threshold in thresholds:
        false_positives = len(human_scores) - np.searchsorted(human_scores, threshold, side="left")
        true_positives = len(ai_scores) - np.searchsorted(ai_scores, threshold, side="left")
        predicted_ai = false_positives + true_positives
        if (predicted_ai >= minimum_verdicts
                and _upper_rate(false_positives, len(human_scores)) <= max_human_false_positive_rate):
            ai_candidates.append((true_positives, threshold, false_positives, predicted_ai))
    ai_choice = max(ai_candidates, key=lambda item: (item[0], item[1])) if ai_candidates else None

    human_candidates = []
    for threshold in thresholds:
        human_verdicts = np.searchsorted(human_scores, threshold, side="right")
        ai_false_negatives = np.searchsorted(ai_scores, threshold, side="right")
        if (human_verdicts >= minimum_verdicts
            and _upper_rate(ai_false_negatives, len(ai_scores)) <= max_ai_false_negative_rate):
            human_candidates.append((human_verdicts, threshold, ai_false_negatives))
    human_choice = max(human_candidates, key=lambda item: (item[0], -item[1])) if human_candidates else None

    ai_threshold = None if ai_choice is None else float(ai_choice[1])
    human_threshold = None if human_choice is None else float(human_choice[1])
    overlap = (
        ai_threshold is not None
        and human_threshold is not None
        and human_threshold >= ai_threshold
    )
    if overlap:
        ai_threshold = None
        human_threshold = None

    decided_human = np.zeros(len(labels), dtype=bool)
    decided_ai = np.zeros(len(labels), dtype=bool)
    if human_threshold is not None:
        decided_human = scores <= human_threshold
    if ai_threshold is not None:
        decided_ai = scores >= ai_threshold
    if np.any(decided_human & decided_ai):
        raise RuntimeError("Selected human and AI decision thresholds overlap.")

    validation_human_fp = int(np.count_nonzero((labels == human_label) & decided_ai))
    validation_ai_as_human = int(np.count_nonzero((labels == ai_label) & decided_human))
    validation_ai_count = int(np.count_nonzero(labels == ai_label))
    return {
        "human_score_at_or_below": human_threshold,
        "ai_score_at_or_above": ai_threshold,
        "target_human_false_positive_rate": max_human_false_positive_rate,
        "target_ai_false_negative_rate": max_ai_false_negative_rate,
        "confidence_bound": "one-sided 95% Wilson upper bound on validation data",
        "minimum_verdict_count": minimum_verdicts,
        "targets_met": human_threshold is not None and ai_threshold is not None,
        "human_gate_met": human_threshold is not None,
        "ai_gate_met": ai_threshold is not None,
        "thresholds_overlap_on_validation": overlap,
        "validation_decisive_coverage": float(np.mean(decided_human | decided_ai)),
        "validation_human_false_positive_rate": float(validation_human_fp / np.count_nonzero(labels == human_label)),
        "validation_human_false_positive_upper_bound": _upper_rate(
            validation_human_fp, int(np.count_nonzero(labels == human_label))
        ),
        "validation_ai_false_negative_count": validation_ai_as_human,
        "validation_ai_denominator": validation_ai_count,
        "validation_ai_false_negative_rate": float(validation_ai_as_human / validation_ai_count),
        "validation_ai_false_negative_upper_bound": _upper_rate(
            validation_ai_as_human, validation_ai_count
        ),
        "validation_human_verdict_count": int(decided_human.sum()),
        "validation_human_verdict_precision": (
            float(np.count_nonzero((labels == human_label) & decided_human) / decided_human.sum())
            if np.any(decided_human) else None
        ),
        "validation_ai_recall": float(np.count_nonzero((labels == ai_label) & decided_ai)
                                       / np.count_nonzero(labels == ai_label)),
    }


def evaluate_scores(labels, scores, human_label, ai_label, thresholds):
    """Evaluate default-score and risk-controlled decisions on one held-out set."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if len(labels) != len(scores) or not len(labels):
        raise ValueError("Evaluation requires aligned, non-empty labels and scores.")
    if set(np.unique(labels).tolist()) != {human_label, ai_label}:
        raise ValueError("Evaluation requires examples from both verified classes.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Evaluation scores must be finite.")

    human_mask = labels == human_label
    ai_mask = labels == ai_label
    default_predictions = np.where(scores >= 0, ai_label, human_label)
    try:
        auc = float(roc_auc_score(ai_mask.astype(int), scores))
    except ValueError:
        auc = None

    def binary_report(predictions):
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_f1": float(f1_score(labels, predictions, labels=[human_label, ai_label], average="macro", zero_division=0)),
            "ai_precision": float(precision_score(labels, predictions, pos_label=ai_label, zero_division=0)),
            "ai_recall": float(recall_score(labels, predictions, pos_label=ai_label, zero_division=0)),
            "roc_auc": auc,
            "confusion_matrix": confusion_matrix(labels, predictions, labels=[human_label, ai_label]).tolist(),
            "confusion_matrix_class_order": ["human", "ai_generated"],
        }

    human_threshold = thresholds["human_score_at_or_below"]
    ai_threshold = thresholds["ai_score_at_or_above"]
    decided_human = np.zeros(len(labels), dtype=bool) if human_threshold is None else scores <= human_threshold
    decided_ai = np.zeros(len(labels), dtype=bool) if ai_threshold is None else scores >= ai_threshold
    if np.any(decided_human & decided_ai):
        raise ValueError("Human and AI thresholds overlap on evaluation scores.")
    inconclusive = ~(decided_human | decided_ai)
    false_positives = int(np.count_nonzero(human_mask & decided_ai))
    ai_detected = int(np.count_nonzero(ai_mask & decided_ai))
    ai_as_human = int(np.count_nonzero(ai_mask & decided_human))
    human_correct = int(np.count_nonzero(human_mask & decided_human))
    decisive_count = int(np.count_nonzero(~inconclusive))
    decisive_errors = false_positives + ai_as_human
    ai_verdict_count = int(decided_ai.sum())
    human_verdict_count = int(decided_human.sum())
    human_count = int(human_mask.sum())
    ai_count = int(ai_mask.sum())

    return {
        "score_type": "uncalibrated oriented logistic-regression decision score; not a probability",
        "default_zero_threshold": binary_report(default_predictions),
        "risk_controlled": {
            "thresholds": thresholds,
            "verdict_counts": {
                "human": human_verdict_count,
                "ai_generated": ai_verdict_count,
                "inconclusive": int(inconclusive.sum()),
            },
            "coverage": float(1 - inconclusive.mean()),
            "inconclusive_fraction": float(inconclusive.mean()),
            "human_false_positive_count": false_positives,
            "human_denominator": human_count,
            "human_false_positive_rate": float(false_positives / human_count),
            "human_false_positive_wilson_95_percent_interval": wilson_interval(false_positives, human_count),
            "specificity_for_ai_verdict": float(1 - false_positives / human_count),
            "ai_detected_count": ai_detected,
            "ai_denominator": ai_count,
            "ai_recall": float(ai_detected / ai_count),
            "ai_precision": float(ai_detected / ai_verdict_count) if ai_verdict_count else None,
            "ai_generated_labelled_human_count": ai_as_human,
            "ai_generated_human_verdict_rate": float(ai_as_human / ai_count),
            "human_correct_count": human_correct,
            "human_denominator": human_count,
            "human_recall": float(human_correct / human_count),
            "human_verdict_precision": float(human_correct / human_verdict_count) if human_verdict_count else None,
            "decisive_error_count": decisive_errors,
            "decisive_error_rate": float(decisive_errors / decisive_count) if decisive_count else None,
            "balanced_accuracy_on_decisive": (
                float(0.5 * (human_correct / np.count_nonzero(human_mask & ~inconclusive)
                             + ai_detected / np.count_nonzero(ai_mask & ~inconclusive)))
                if np.any(human_mask & ~inconclusive) and np.any(ai_mask & ~inconclusive) else None
            ),
        },
    }


def grouped_metrics(labels, scores, groups, human_label, ai_label, thresholds):
    """Return compact per-group risk, recall, coverage, and support statistics."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    groups = np.asarray(groups, dtype=object)
    result = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        group_labels = labels[mask]
        group_scores = scores[mask]
        human_count = int(np.count_nonzero(group_labels == human_label))
        ai_count = int(np.count_nonzero(group_labels == ai_label))
        human_threshold = thresholds["human_score_at_or_below"]
        ai_threshold = thresholds["ai_score_at_or_above"]
        predicted_ai = np.zeros(len(group_labels), dtype=bool) if ai_threshold is None else group_scores >= ai_threshold
        predicted_human = np.zeros(len(group_labels), dtype=bool) if human_threshold is None else group_scores <= human_threshold
        fp = int(np.count_nonzero((group_labels == human_label) & predicted_ai))
        ai_recalled = int(np.count_nonzero((group_labels == ai_label) & predicted_ai))
        decided = predicted_ai | predicted_human
        result[str(group)] = {
            "support": len(group_labels),
            "human_support": human_count,
            "ai_support": ai_count,
            "human_false_positive_rate": float(fp / human_count) if human_count else None,
            "human_false_positive_wilson_95_percent_interval": wilson_interval(fp, human_count) if human_count else None,
            "ai_recall": float(ai_recalled / ai_count) if ai_count else None,
            "coverage": float(decided.mean()),
        }
    return result


def grouped_probability_diagnostics(labels, probabilities, groups, positive_label, min_support=1):
    """Report calibration diagnostics by a predefined source or length group."""
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities, dtype=float)
    groups = np.asarray(groups, dtype=object)
    if not (len(labels) == len(probabilities) == len(groups)) or not len(labels):
        raise ValueError("Grouped calibration requires aligned, non-empty labels, scores, and groups.")
    if min_support < 1:
        raise ValueError("min_support must be at least one.")

    result = {}
    for group in sorted(set(groups.tolist())):
        mask = groups == group
        support = int(mask.sum())
        if support < min_support:
            continue
        result[str(group)] = {
            "support": support,
            "positive_support": int(np.count_nonzero(labels[mask] == positive_label)),
            "diagnostics": probability_diagnostics(
                labels[mask], probabilities[mask], positive_label
            ),
        }
    return result