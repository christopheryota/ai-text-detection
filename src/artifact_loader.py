"""Validated loading of the model and report required by the Flask application."""

import json
from pathlib import Path

import joblib


def load_application_artifacts(model_path, report_path):
    model_path = Path(model_path)
    report_path = Path(report_path)
    missing = [str(path) for path in (model_path, report_path) if not path.is_file()]
    if missing:
        raise RuntimeError(f"Required model artifact(s) missing: {', '.join(missing)}")

    try:
        model = joblib.load(model_path)
    except Exception as error:
        raise RuntimeError(f"Could not load model artifact {model_path}: {error}") from error

    try:
        with report_path.open(encoding="utf-8") as report_file:
            report = json.load(report_file)
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError(f"Could not read evaluation report {report_path}: {error}") from error

    steps = getattr(model, "named_steps", None)
    if steps is None or "classifier" not in steps:
        raise RuntimeError("Model artifact is malformed: expected a fitted scikit-learn pipeline.")
    classifier = steps["classifier"]
    if (len(getattr(classifier, "classes_", ())) != 2
            or not hasattr(classifier, "coef_")
            or not hasattr(classifier, "intercept_")
            or not hasattr(model, "decision_function")):
        raise RuntimeError("Model artifact is malformed: expected a fitted binary linear classifier.")
    vectorizer = steps.get("features", steps.get("tfidf"))
    if (vectorizer is None or not hasattr(vectorizer, "transform")
            or not hasattr(vectorizer, "get_feature_names_out")):
        raise RuntimeError("Model artifact is malformed: fitted TF-IDF feature extraction is unavailable.")

    try:
        labels = report["label_mapping"]
        human_label = labels["human"]
        ai_label = labels["ai_generated"]
        selected_name = report["selection"]["selected_candidate"]
        selected_result = next(
            item for item in report["candidates"] if item["name"] == selected_name
        )
        thresholds = selected_result["validation"]["risk_controlled"]["thresholds"]
        external_test = report["same_test_set_comparison"]["selected"]["risk_controlled"]
        fpr_upper = external_test["human_false_positive_wilson_95_percent_interval"][1]
        ai_recall = external_test["ai_recall"]
        fpr_target = thresholds["target_human_false_positive_rate"]
    except (KeyError, IndexError, StopIteration, TypeError) as error:
        raise RuntimeError(f"Evaluation report is malformed or incomplete: {error}") from error

    if human_label == ai_label or set(model.classes_) != {human_label, ai_label}:
        raise RuntimeError("Saved model classes do not match the verified evaluation label mapping.")
    if (thresholds["human_score_at_or_below"] is not None
            and thresholds["ai_score_at_or_above"] is not None
            and thresholds["human_score_at_or_below"] >= thresholds["ai_score_at_or_above"]):
        raise RuntimeError("Evaluation report contains overlapping human and AI score thresholds.")

    risk_gate_met = bool(
        thresholds["targets_met"]
        and fpr_upper <= fpr_target
        and ai_recall >= 0.10
    )
    if risk_gate_met:
        risk_gate_reason = None
    elif not thresholds["targets_met"]:
        risk_gate_reason = "No candidate met both validation risk targets. The analysis direction is research-only."
    elif fpr_upper > fpr_target:
        risk_gate_reason = "The external test did not confirm the human false-positive target. Evidence strength is limited."
    else:
        risk_gate_reason = "The external test did not meet the minimum AI recall requirement. Evidence strength is limited."

    return {
        "model": model,
        "report": report,
        "human_label": human_label,
        "ai_label": ai_label,
        "selected_name": selected_name,
        "thresholds": thresholds,
        "risk_gate_met": risk_gate_met,
        "risk_gate_reason": risk_gate_reason,
    }
