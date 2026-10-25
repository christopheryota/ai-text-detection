import unittest

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.text_analysis import build_text_analysis
from src.verity_analysis import explain_prediction


class TextAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts = [
            "personal diary human voice varied details and specific memories",
            "human journal writing includes personal observations and details",
            "my human account describes ordinary days with varied expressions",
            "machine generated response follows structured repeated instructions",
            "AI generated writing uses an automated structured response format",
            "generated assistant text repeats a formal ordered response structure",
        ]
        labels = [0, 0, 0, 1, 1, 1]
        cls.model = Pipeline([
            ("features", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(random_state=42)),
        ]).fit(texts, labels)
        cls.ai_label = 1
        cls.ai_text = "machine generated assistant response follows structured instructions"
        cls.human_text = "my personal human diary describes varied ordinary memories"
        cls.thresholds = {
            "human_score_at_or_below": -0.1,
            "ai_score_at_or_above": 0.1,
            "target_human_false_positive_rate": 0.05,
        }

    def analyze(self, text, thresholds=None, word_count=100, risk_gate=True):
        explanation = explain_prediction(self.model, text, self.ai_label)
        analysis = build_text_analysis(
            explanation,
            word_count,
            self.thresholds if thresholds is None else thresholds,
            risk_gate,
        )
        return explanation, analysis

    def test_ai_and_human_texts_report_score_sign_direction(self):
        _, ai_analysis = self.analyze(self.ai_text)
        _, human_analysis = self.analyze(self.human_text)
        self.assertGreater(ai_analysis["direction_score"], 0)
        self.assertEqual(ai_analysis["direction_key"], "ai_associated")
        self.assertLess(human_analysis["direction_score"], 0)
        self.assertEqual(human_analysis["direction_key"], "human_associated")

    def test_borderline_score_keeps_direction_but_reports_weak_strength(self):
        explanation = explain_prediction(self.model, self.ai_text, self.ai_label)
        score = explanation["decision_score"]
        thresholds = {
            "human_score_at_or_below": score - 0.01,
            "ai_score_at_or_above": score + 0.01,
        }
        _, analysis = self.analyze(self.ai_text, thresholds=thresholds)
        self.assertEqual(analysis["strength"], "weak")
        self.assertEqual(analysis["direction_key"], "ai_associated")
        self.assertEqual(analysis["risk_assessment"], "No conservative decision cutoff crossed")

    def test_crossed_cutoff_is_moderate_only_when_gate_passes(self):
        _, passed = self.analyze(self.ai_text)
        _, failed = self.analyze(self.ai_text, risk_gate=False)
        self.assertEqual(passed["strength"], "moderate")
        self.assertEqual(failed["strength"], "weak")
        self.assertIn("research context only", failed["overall_assessment"])

    def test_short_and_long_text_limitations_are_explicit(self):
        _, short = self.analyze(self.human_text, word_count=55)
        _, long = self.analyze(self.human_text, word_count=220)
        self.assertTrue(any("short relative" in item for item in short["limitations"]))
        self.assertTrue(any("length band" in item for item in long["limitations"]))

    def test_explanations_use_measured_values_and_not_authorship_claims(self):
        explanation, analysis = self.analyze(self.ai_text)
        item = explanation["all_feature_contributions"][0]
        sentence = next(
            evidence["plain_english"]
            for evidence in analysis["ranked_evidence"]
            if evidence["feature_name"] == item["feature_name"]
        )
        self.assertIn(f"{item['tfidf_value']:.6g}", sentence)
        self.assertIn(f"{item['ai_oriented_coefficient']:+.6g}", sentence)
        self.assertIn(f"{item['contribution']:+.6g}", sentence)
        self.assertIn("not evidence that a person or AI authored", sentence)
        self.assertTrue(all(
            left["absolute_contribution"] >= right["absolute_contribution"]
            for left, right in zip(analysis["ranked_evidence"], analysis["ranked_evidence"][1:])
        ))

    def test_probability_is_never_renderable_from_uncalibrated_analysis(self):
        _, analysis = self.analyze(self.ai_text)
        self.assertFalse(analysis["probability_display_enabled"])
        self.assertIsNone(analysis["probability_estimate"])
        self.assertIn("not a probability", analysis["contribution_definition"])

    def test_both_full_and_bounded_technical_evidence_reconstruct_score(self):
        explanation = explain_prediction(self.model, self.ai_text, self.ai_label)
        analysis = build_text_analysis(explanation, 100, self.thresholds, True)
        self.assertLessEqual(abs(analysis["reconstruction_error"]), 1e-8)
        self.assertLessEqual(
            abs(analysis["technical_reconstructed_score"] - analysis["direction_score"]),
            1e-8,
        )
        self.assertLessEqual(len(analysis["technical_evidence"]), 100)

    def test_invalid_or_overlapping_cutoffs_are_rejected(self):
        explanation = explain_prediction(self.model, self.ai_text, self.ai_label)
        with self.assertRaisesRegex(ValueError, "overlap"):
            build_text_analysis(
                explanation,
                100,
                {"human_score_at_or_below": 1.0, "ai_score_at_or_above": -1.0},
                True,
            )


if __name__ == "__main__":
    unittest.main()