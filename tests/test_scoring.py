import unittest
from chronoseek.scoring import calculate_iou, score_response
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
        """Test strict binary scoring rules"""

        gt = (10.0, 20.0)

        # Case A: High IoU (>0.5) -> Score 1.0
        # Pred: 11-19 (IoU should be high)
        # Int: 8, Union: 10. IoU=0.8
        pred_pass = [VideoSearchResult(start=11.0, end=19.0, confidence=0.9)]
        self.assertEqual(score_response(pred_pass, gt, 0.1), 1.0)

        # Case B: Low IoU (<0.5) -> Score 0.0
        # Pred: 0-12
        # Int: 2 (10-12), Union: 20 (0-20). IoU=0.1
        pred_fail = [VideoSearchResult(start=0.0, end=12.0, confidence=0.9)]
        self.assertEqual(score_response(pred_fail, gt, 0.1), 0.0)

        # Case C: Multiple predictions, take max IoU
        preds_mixed = [
            VideoSearchResult(start=0.0, end=5.0, confidence=0.5),  # IoU 0
            VideoSearchResult(start=10.0, end=20.0, confidence=0.8),  # IoU 1.0
        ]
        self.assertEqual(score_response(preds_mixed, gt, 0.1), 1.0)

        # Case D: Empty predictions
        self.assertEqual(score_response([], gt, 0.1), 0.0)


if __name__ == "__main__":
    unittest.main()
