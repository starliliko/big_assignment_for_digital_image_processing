import unittest

import numpy as np

from metrics import ImageMetrics


class ImageMetricsTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.full((32, 32, 3), 120, dtype=np.uint8)
        self.changed = self.reference.copy()
        self.changed[8:24, 8:24] = 150

    def test_identical_images_have_perfect_reference_scores(self):
        self.assertEqual(ImageMetrics.mse(self.reference, self.reference), 0)
        self.assertEqual(ImageMetrics.mae(self.reference, self.reference), 0)
        self.assertEqual(ImageMetrics.rmse(self.reference, self.reference), 0)
        self.assertEqual(
            ImageMetrics.psnr(self.reference, self.reference), float("inf")
        )
        self.assertAlmostEqual(
            ImageMetrics.ssim(self.reference, self.reference), 1.0, places=5
        )

    def test_all_metrics_returns_reference_and_no_reference_metrics(self):
        metrics = ImageMetrics.calculate_all_metrics(
            self.reference, self.changed
        )
        expected = {
            "PSNR",
            "SSIM",
            "MSE",
            "MAE",
            "RMSE",
            "熵",
            "对比度",
            "平均梯度",
            "锐度",
            "色彩丰富度",
        }
        self.assertEqual(set(metrics), expected)
        self.assertTrue(all(np.isfinite(value) for value in metrics.values()))
