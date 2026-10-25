import json
from pathlib import Path
import tempfile
import unittest

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.artifact_loader import load_application_artifacts


class ArtifactLoaderTests(unittest.TestCase):
    def test_missing_artifacts_fail_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "missing"):
                load_application_artifacts(root / "missing.joblib", root / "missing.json")

    def test_non_pipeline_model_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.joblib"
            report_path = root / "report.json"
            joblib.dump({"not": "a pipeline"}, model_path)
            report_path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "malformed.*pipeline"):
                load_application_artifacts(model_path, report_path)

    def test_invalid_report_json_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.joblib"
            report_path = root / "report.json"
            model = Pipeline([
                ("features", TfidfVectorizer(min_df=1)),
                ("classifier", LogisticRegression()),
            ]).fit(
                ["human words and personal tone", "generated words and repeated pattern"],
                [0, 1],
            )
            joblib.dump(model, model_path)
            report_path.write_text("{invalid", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "evaluation report"):
                load_application_artifacts(model_path, report_path)


if __name__ == "__main__":
    unittest.main()
