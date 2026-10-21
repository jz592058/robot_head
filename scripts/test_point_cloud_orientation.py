"""Offline orientation regression; no device connection or servo commands."""

import unittest

import depthai as dai
import numpy as np

from focused_person_follow import PointCloudForViewer


class PointCloudOrientationTest(unittest.TestCase):
    def test_physical_top_stays_above_bottom_in_viewer(self):
        # Inverted sensor: the top of an upright subject is BELOW the raw
        # principal point, so optical Y is positive. The viewer uses Y-up.
        points = np.array([[0.2, 0.4, 2.0], [0.2, -0.4, 2.0]], dtype=np.float32)
        colors = np.array([[255, 0, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)
        message = dai.PointCloudData()
        message.setPointsRGB(points, colors)
        message.setSize(2, 1)

        result = PointCloudForViewer.process(None, message)
        actual, actual_colors = result.getPointsRGB()

        self.assertGreater(actual[0, 1], actual[1, 1])
        np.testing.assert_array_equal(actual[:, 2], points[:, 2])
        np.testing.assert_array_equal(actual_colors, colors)
        self.assertAlmostEqual(result.getMinY(), -0.4)


if __name__ == "__main__":
    unittest.main()
