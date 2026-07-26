import unittest

import cv2
import numpy as np

from algorithms import DehazeAlgorithms, DerainAlgorithms


class AlgorithmSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x = np.linspace(20, 220, 32, dtype=np.uint8)
        gradient = np.tile(x, (32, 1))
        cls.color_image = cv2.merge(
            (gradient, np.flipud(gradient), np.fliplr(gradient))
        )

    def assert_valid_result(self, result):
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, self.color_image.shape)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.isfinite(result).all())

    def test_dehaze_algorithms_return_valid_images(self):
        algorithms = (
            DehazeAlgorithms.histogram_equalization,
            DehazeAlgorithms.clahe,
            DehazeAlgorithms.dark_channel_prior_adaptive,
            DehazeAlgorithms.gamma_correction,
        )
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm.__name__):
                self.assert_valid_result(algorithm(self.color_image.copy()))

    def test_derain_algorithms_return_valid_images(self):
        algorithms = (
            DerainAlgorithms.median_filter,
            DerainAlgorithms.bilateral_filter,
            DerainAlgorithms.guided_filter_derain,
            DerainAlgorithms.morphological_derain,
            DerainAlgorithms.low_rank_derain,
            DerainAlgorithms.dsc_derain,
        )
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm.__name__):
                self.assert_valid_result(algorithm(self.color_image.copy()))
