
from pathlib import Path

import joblib
from flask import Flask, render_template, request


app = Flask(__name__)

# Find the saved model relative to this file.
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "ai_text_detector.joblib"

# Load the trained machine-learning pipeline.
model = joblib.load(MODEL_PATH)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    submitted_text = ""
    error = None

    if request.method == "POST":
        submitted_text = request.form.get("text", "").strip()

        if not submitted_text:
            error = "Please enter some text to analyze."
        else:
            prediction = model.predict([submitted_text])[0]

            # Your training report maps class 0 to Human and class 1 to AI.
            if prediction == 0:
                result = "Human-written"
            elif prediction == 1:
                result = "AI-generated"
            else:
                error = "The model returned an unexpected prediction."

    return render_template(
        "index.html",
        result=result,
        submitted_text=submitted_text,
        error=error,
    )


if __name__ == "__main__":
    app.run(debug=True)
