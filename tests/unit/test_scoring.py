import unittest
from chronoseek.scoring import (
    STRICT_IOU_THRESHOLD,
    PURE_IOU_SCORING,
    best_iou,
    calculate_iou,
    interval_quality_score,
    passes_strict_iou,
    score_response,
)
from chronoseek.protocol_models import VideoSearchResult


class TestScoring(unittest.TestCase):

    def test_iou_calculation(self):
        """Test IoU logic for various overlap scenarios"""

        # 1. Perfect Match (IoU = 1.0)
        self.assertAlmostEqual(calculate_iou(10, 20, 10, 20), 1.0)

        # 2. No Overlap (IoU = 0.0)
        self.assertAlmostEqual(calculate_iou(0, 10, 20, 30), 0.0)

        # 3. Partial Overlap (Half overlap)
        # Pred: 0-10 (len 10), GT: 5-15 (len 10)
        # Intersection: 5-10 (len 5)
        # Union: 10 + 10 - 5 = 15
        # IoU: 5/15 = 0.333...
        self.assertAlmostEqual(calculate_iou(0, 10, 5, 15), 1 / 3)

        # 4. Containment (Pred inside GT)
        # Pred: 10-15 (len 5), GT: 0-20 (len 20)
        # Intersection: 5
        # Union: 20
        # IoU: 5/20 = 0.25
        self.assertAlmostEqual(calculate_iou(10, 15, 0, 20), 0.25)

    def test_scoring_rules(self):
        """Test continuous shape-aware interval scoring rules"""

        gt = (10.0, 20.0)

        # Case A: High IoU, center-aligned, slightly trimmed boundaries.
        pred_pass = [VideoSearchResult(start=11.0, end=19.0, confidence=0.9)]
        self.assertGreater(score_response(pred_pass, gt, 0.1), 0.75)
        self.assertLess(score_response(pred_pass, gt, 0.1), 0.8)

        # Case B: Low IoU is dampened further by center and boundary misses.
        pred_fail = [VideoSearchResult(start=0.0, end=12.0, confidence=0.9)]
        self.assertLess(score_response(pred_fail, gt, 0.1), 0.1)

        # Case C: Multiple predictions, take max quality
        preds_mixed = [
            VideoSearchResult(start=0.0, end=5.0, confidence=0.5),  # IoU 0
            VideoSearchResult(start=10.0, end=20.0, confidence=0.8),  # IoU 1.0
        ]
        self.assertEqual(score_response(preds_mixed, gt, 0.1), 1.0)

        # Case D: Empty predictions
        self.assertEqual(score_response([], gt, 0.1), 0.0)

    def test_scoring_accepts_multiple_ground_truths(self):
        preds = [VideoSearchResult(start=30.0, end=40.0, confidence=0.9)]
        ground_truths = [(10.0, 20.0), (31.0, 39.0)]

        self.assertGreater(score_response(preds, ground_truths, 0.1), 0.75)
        self.assertLess(score_response(preds, ground_truths, 0.1), 0.8)
        self.assertAlmostEqual(best_iou(preds, ground_truths), 0.8)

    def test_pure_iou_scoring_config_keeps_legacy_score(self):
        preds = [VideoSearchResult(start=11.0, end=19.0, confidence=0.9)]

        self.assertAlmostEqual(
            score_response(
                preds,
                (10.0, 20.0),
                0.1,
                interval_scoring_config=PURE_IOU_SCORING,
            ),
            0.8,
        )

    def test_shape_penalties_lower_broad_and_center_missed_intervals(self):
        gt = (10.0, 20.0)
        perfect = VideoSearchResult(start=10.0, end=20.0, confidence=0.9)
        broad_centered = VideoSearchResult(start=5.0, end=25.0, confidence=0.9)
        center_missed = VideoSearchResult(start=5.0, end=15.0, confidence=0.9)

        self.assertEqual(interval_quality_score(perfect, gt), 1.0)
        self.assertLess(
            interval_quality_score(broad_centered, gt),
            calculate_iou(5.0, 25.0, 10.0, 20.0),
        )
        self.assertLess(
            interval_quality_score(center_missed, gt),
            interval_quality_score(broad_centered, gt),
        )

    def test_absent_canary_scoring_rewards_only_empty_predictions(self):
        pred = [VideoSearchResult(start=10.0, end=20.0, confidence=0.9)]

        self.assertEqual(
            score_response([], [], 0.1, expects_empty_response=True),
            1.0,
        )
        self.assertEqual(
            score_response(pred, [], 0.1, expects_empty_response=True),
            0.0,
        )
        self.assertEqual(score_response([], [], 0.1), 0.0)

    def test_strict_threshold_helper(self):
        self.assertTrue(passes_strict_iou(STRICT_IOU_THRESHOLD))
        self.assertFalse(passes_strict_iou(STRICT_IOU_THRESHOLD - 0.01))


if __name__ == "__main__":
    unittest.main()
