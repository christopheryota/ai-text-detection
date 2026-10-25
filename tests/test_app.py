import unittest
from unittest.mock import patch

import app


SHORT_TEXT = (
    "Yesterday I walked home after rain and noticed wet leaves shining beside "
    "the old library. A neighbor waved across the quiet street."
)
LONG_PARAGRAPH = (
    "This detailed personal account describes an ordinary afternoon, a walk "
    "through a familiar neighborhood, a brief conversation with a neighbor, "
    "and small observations about the weather and nearby buildings. "
)


class AppAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.client = app.app.test_client()

    def test_short_input_receives_direction_and_short_text_limitation(self):
        response = self.client.post("/", data={"text": SHORT_TEXT})
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("EVIDENCE DIRECTION", body)
        self.assertIn("short relative to the accepted input range", body)
        self.assertIn("Evidence direction score", body)

    def test_long_input_receives_analysis_and_length_limitation(self):
        text = LONG_PARAGRAPH * 18
        response = self.client.post("/", data={"text": text})
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("EVIDENCE DIRECTION", body)
        self.assertIn("does not establish calibrated reliability by text-length band", body)
        self.assertIn("Calibration by held-out length band", body)
        self.assertIn("Calibration by held-out sub-source", body)
        self.assertIn("Calibration by dataset model metadata", body)
        self.assertIn("Raw output reliability diagram", body)
        self.assertIn("Raw log loss", body)

    def test_unavailable_score_does_not_force_a_direction(self):
        text = LONG_PARAGRAPH * 3
        with patch.object(app, "ai_decision_scores", return_value=[float("nan")]):
            response = self.client.post("/", data={"text": text})
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("No direction is reported", body)
        self.assertNotIn("EVIDENCE DIRECTION", body)

    def test_no_authorship_probability_is_rendered(self):
        response = self.client.post("/", data={"text": LONG_PARAGRAPH * 3})
        body = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("not a probability", body)
        self.assertNotIn("AI-generated probability", body)
        self.assertNotIn("% probability", body)

    def test_empty_and_oversized_inputs_keep_existing_handling(self):
        empty = self.client.post("/", data={"text": "  "})
        oversized = self.client.post("/", data={"text": "x" * 30_001})
        self.assertEqual(empty.status_code, 200)
        self.assertIn("Please enter some text", empty.get_data(as_text=True))
        self.assertEqual(oversized.status_code, 200)
        self.assertIn("30,000-character", oversized.get_data(as_text=True))

    def test_evaluation_endpoint_is_preserved(self):
        response = self.client.get("/evaluation")
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["selection"]["selected_candidate"], app.SELECTED_MODEL_NAME)
        self.assertEqual(payload["same_test_set_comparison"]["selected"]["risk_controlled"]["human_false_positive_rate"],
                         app.SELECTED_TEST_METRICS["human_false_positive_rate"])
        self.assertIsNotNone(payload["stratified_analysis"])
        self.assertEqual(payload["stratified_analysis"]["seed"], payload["seed"])


if __name__ == "__main__":
    unittest.main()