import unittest

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.verity_analysis import (
    compute_evaluation_metrics,
    explain_prediction,
    fit_probability_calibrator,
    require_minimum_class_counts,
    verify_label_mapping,
    wilson_interval,
)


class VerityAnalysisTests(unittest.TestCase):
    def test_label_mapping_comes_from_model_metadata_not_numeric_order(self):
        mapping = verify_label_mapping([1, 1, 0, 0], ["human", "human", "gpt-4", "llama"])
        self.assertEqual(mapping, {"human": 1, "ai_generated": 0})

    def test_label_mapping_rejects_ambiguous_metadata(self):
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            verify_label_mapping([0, 0], ["human", "gpt-4"])

    def test_minimum_class_counts_reject_imbalance(self):
        with self.assertRaisesRegex(ValueError, "imbalanced"):
            require_minimum_class_counts([0] * 50 + [1] * 2, 0, 1, 2, "test")

    def test_wilson_interval_is_bounded_and_contains_observed_rate(self):
        interval = wilson_interval(5, 10)
        self.assertLess(interval[0], 0.5)
        self.assertGreater(interval[1], 0.5)
        self.assertGreaterEqual(interval[0], 0.0)
        self.assertLessEqual(interval[1], 1.0)

    def test_metrics_report_false_positive_denominator_and_uncertainty_coverage(self):
        result = compute_evaluation_metrics(
            labels=[0, 0, 1, 1],
            raw_probabilities=[0.8, 0.7, 0.4, 0.1],
            calibrated_probabilities=[0.8, 0.7, 0.4, 0.1],
            human_label=0,
            ai_label=1,
            uncertainty_band=(0.4, 0.6),
        )
        self.assertEqual(result["calibrated"]["confusion_matrix"], [[0, 2], [2, 0]])
        self.assertEqual(result["human_incorrectly_flagged_as_ai"]["count"], 2)
        self.assertEqual(result["human_incorrectly_flagged_as_ai"]["denominator"], 2)
        self.assertEqual(result["human_incorrectly_flagged_as_ai"]["rate"], 1.0)
        self.assertEqual(result["uncertainty_band"]["inconclusive_count"], 1)
        self.assertEqual(result["uncertainty_band"]["inconclusive_coverage"], 0.25)
        self.assertGreaterEqual(result["calibrated"]["probability_diagnostics"]["log_loss"], 0.0)
        self.assertEqual(len(result["calibrated"]["probability_diagnostics"]["calibration_bins"]), 10)

    def test_calibrator_fits_on_given_calibration_examples(self):
        texts = [
            "human writing has personal details",
            "human writing has varied language",
            "natural human expression uses details",
            "an ai generated response repeats structure",
            "ai generated text has repeated phrasing",
            "machine generated language repeats patterns",
        ]
        labels = np.array([0, 0, 0, 1, 1, 1])
        model = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("classifier", LogisticRegression(random_state=42)),
        ]).fit(texts, labels)
        calibrator = fit_probability_calibrator(model, texts, labels, ai_label=1)
        probabilities = calibrator.predict_proba(
            model.decision_function(texts).reshape(-1, 1)
        )
        self.assertEqual(probabilities.shape, (len(texts), 2))
        self.assertTrue(np.all((probabilities >= 0) & (probabilities <= 1)))

    def test_explanations_are_signed_feature_contributions_from_model(self):
        texts = [
            "human personal voice and varied detail",
            "human natural voice with personal detail",
            "natural human writing has varied voice",
            "ai generated text repeats common structure",
            "machine generated response repeats structure",
            "ai generated language uses common patterns",
        ]
        labels = np.array([0, 0, 0, 1, 1, 1])
        model = Pipeline([
            ("tfidf", TfidfVectorizer(min_df=1)),
            ("classifier", LogisticRegression(random_state=42)),
        ]).fit(texts, labels)
        explanation = explain_prediction(model, texts[0], ai_label=1, top_n=100)
        signed_contributions = [
            item["contribution"]
            for item in explanation["toward_ai"] + explanation["toward_human"]
        ]
        reconstructed_score = explanation["intercept"] + sum(signed_contributions)
        self.assertAlmostEqual(reconstructed_score, explanation["decision_score"], places=10)
        self.assertTrue(explanation["toward_human"])
        self.assertTrue(all(item["contribution"] < 0 for item in explanation["toward_human"]))
        self.assertTrue(all(item["contribution"] > 0 for item in explanation["toward_ai"]))


if __name__ == "__main__":
    unittest.main()