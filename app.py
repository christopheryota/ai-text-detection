
from pathlib import Path
import math
import json

from flask import Flask, jsonify, render_template, request

from brand_config import BRAND
from src.artifact_loader import load_application_artifacts
from src.text_analysis import build_text_analysis
from src.verity_analysis import ai_decision_scores, explain_prediction


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 125_000

# Find the saved model relative to this file.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "ai_text_detector.joblib"
METRICS_PATH = BASE_DIR / "reports" / "model_benchmark.json"
STRATIFIED_METRICS_PATH = BASE_DIR / "reports" / "stratified_model_analysis.json"

ARTIFACTS = load_application_artifacts(MODEL_PATH, METRICS_PATH)
model = ARTIFACTS["model"]
evaluation_report = ARTIFACTS["report"]
HUMAN_LABEL = ARTIFACTS["human_label"]
AI_LABEL = ARTIFACTS["ai_label"]
DECISION_THRESHOLDS = ARTIFACTS["thresholds"]
RISK_GATE_MET = ARTIFACTS["risk_gate_met"]
RISK_GATE_REASON = ARTIFACTS["risk_gate_reason"]
SELECTED_MODEL_NAME = ARTIFACTS["selected_name"]
SELECTED_TEST_METRICS = evaluation_report["same_test_set_comparison"]["selected"]["risk_controlled"]
CALIBRATION_DIAGNOSTICS = evaluation_report["calibration"]
STRATIFIED_DIAGNOSTICS = None
if STRATIFIED_METRICS_PATH.is_file():
    try:
        candidate_diagnostics = json.loads(
            STRATIFIED_METRICS_PATH.read_text(encoding="utf-8")
        )
        if (candidate_diagnostics.get("selected_candidate") == SELECTED_MODEL_NAME
                and candidate_diagnostics.get("seed") == evaluation_report["seed"]):
            STRATIFIED_DIAGNOSTICS = candidate_diagnostics
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        STRATIFIED_DIAGNOSTICS = None


@app.context_processor
def inject_brand():
    return {"brand": BRAND}


@app.errorhandler(413)
def input_too_large(_error):
    return render_template(
        "index.html",
        result=None,
        analysis=None,
        decision_score=None,
        explanation=None,
        submitted_text="",
        error="Text exceeds the 30,000-character analysis limit.",
    ), 413


@app.route("/evaluation")
def evaluation():
    response = dict(evaluation_report)
    response["stratified_analysis"] = STRATIFIED_DIAGNOSTICS
    return jsonify(response)


@app.route("/", methods=["GET", "POST"])
def index():
    submitted_text = ""
    error = None
    decision_score = None
    explanation = None
    analysis = None

    if request.method == "POST":
        submitted_text = request.form.get("text", "").strip()

        if not submitted_text:
            error = "Please enter some text to analyze."
        elif len(submitted_text) > 30_000:
            error = "Text exceeds the 30,000-character analysis limit."
        else:
            word_count = len(submitted_text.split())
            try:
                decision_score = float(ai_decision_scores(model, [submitted_text], AI_LABEL)[0])
                explanation = explain_prediction(model, submitted_text, AI_LABEL)
                if (not math.isfinite(decision_score)
                        or not math.isfinite(explanation["decision_score"])
                        or abs(decision_score - explanation["decision_score"]) > 1e-8):
                    raise ValueError("Inference and explanation scores are invalid or inconsistent.")
                analysis = build_text_analysis(
                    explanation,
                    word_count,
                    DECISION_THRESHOLDS,
                    RISK_GATE_MET,
                )
                analysis["risk_gate_reason"] = RISK_GATE_REASON
                analysis["evaluation_context"] = {
                    "selected_model": SELECTED_MODEL_NAME,
                    "dataset": evaluation_report["dataset"],
                    "dataset_rows": evaluation_report["dataset_rows"],
                    "seed": evaluation_report["seed"],
                    "external_test_source": evaluation_report["protocol"]["external_test_source"],
                    "external_test_support": evaluation_report["split_sizes"]["test"]["total"],
                    "external_human_false_positive_rate": SELECTED_TEST_METRICS["human_false_positive_rate"],
                    "external_human_false_positive_interval": SELECTED_TEST_METRICS[
                        "human_false_positive_wilson_95_percent_interval"
                    ],
                    "external_ai_recall": SELECTED_TEST_METRICS["ai_recall"],
                    "external_decisive_coverage": SELECTED_TEST_METRICS["coverage"],
                    "calibration": CALIBRATION_DIAGNOSTICS,
                    "stratified_calibration": STRATIFIED_DIAGNOSTICS,
                    "thresholds": DECISION_THRESHOLDS,
                    "versions": evaluation_report["versions"],
                }
            except Exception:
                app.logger.exception("Text analysis could not produce valid model evidence")
                analysis = None
                decision_score = None
                explanation = None
                error = (
                    "Analysis is unavailable because the model returned invalid or inconsistent evidence. "
                    "No direction is reported and no authorship assessment is made."
                )

    return render_template(
        "index.html",
        analysis=analysis,
        decision_score=decision_score,
        explanation=explanation,
        submitted_text=submitted_text,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
