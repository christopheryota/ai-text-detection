import unittest

import numpy as np

from src.model_benchmark import (
    evaluate_scores,
    grouped_probability_diagnostics,
    select_risk_thresholds,
)


class ModelBenchmarkTests(unittest.TestCase):
    def test_separated_scores_meet_risk_targets_without_abstaining(self):
        labels = np.array([0] * 100 + [1] * 100)
        scores = np.array([-2.0 - index / 100 for index in range(100)]
                          + [2.0 + index / 100 for index in range(100)])
        thresholds = select_risk_thresholds(labels, scores, 0, 1, minimum_verdicts=20)
        report = evaluate_scores(labels, scores, 0, 1, thresholds)["risk_controlled"]

        self.assertTrue(thresholds["targets_met"])
        self.assertLessEqual(thresholds["validation_human_false_positive_upper_bound"], 0.05)
        self.assertLessEqual(thresholds["validation_ai_false_negative_upper_bound"], 0.10)
        self.assertEqual(report["coverage"], 1.0)
        self.assertEqual(report["human_false_positive_rate"], 0.0)
        self.assertEqual(report["ai_recall"], 1.0)

    def test_overlapping_scores_do_not_claim_risk_target(self):
        labels = np.array([0] * 80 + [1] * 80)
        class_scores = np.linspace(-1.0, 1.0, 80)
        scores = np.concatenate([class_scores, class_scores])
        thresholds = select_risk_thresholds(labels, scores, 0, 1, minimum_verdicts=20)
        self.assertFalse(thresholds["targets_met"])

    def test_test_metrics_preserve_abstentions_and_both_class_errors(self):
        labels = np.array([0] * 100 + [1] * 100)
        scores = np.array([-2.0] * 80 + [0.0] * 20 + [0.0] * 15 + [2.0] * 85)
        thresholds = {
            "human_score_at_or_below": -1.0,
            "ai_score_at_or_above": 1.0,
            "targets_met": True,
        }
        result = evaluate_scores(labels, scores, 0, 1, thresholds)["risk_controlled"]
        self.assertEqual(result["verdict_counts"], {"human": 80, "ai_generated": 85, "inconclusive": 35})
        self.assertEqual(result["human_false_positive_count"], 0)
        self.assertEqual(result["ai_generated_labelled_human_count"], 0)
        self.assertEqual(result["inconclusive_fraction"], 0.175)

    def test_grouped_calibration_reports_single_class_strata_without_auc_assumptions(self):
        result = grouped_probability_diagnostics(
            labels=[0, 0, 1, 1],
            probabilities=[0.1, 0.2, 0.8, 0.9],
            groups=["human_only", "human_only", "ai_only", "ai_only"],
            positive_label=1,
        )
        self.assertEqual(result["human_only"]["positive_support"], 0)
        self.assertEqual(result["ai_only"]["positive_support"], 2)
        for group in result.values():
            self.assertIn("brier_score", group["diagnostics"])
            self.assertIn("log_loss", group["diagnostics"])
            self.assertIn("expected_calibration_error", group["diagnostics"])


if __name__ == "__main__":
    unittest.main()