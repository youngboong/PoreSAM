"""Check physical scaling, image coordinates and brightness on known masks."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from analyze_candidates import measure_masks


class ExtraMetricsTests(unittest.TestCase):
    def test_rectangle_and_brightness(self):
        mask = np.zeros((12, 16), bool); mask[3:5, 4:8] = True
        gray = np.zeros(mask.shape, np.uint8); gray[3:5, 4:8] = [10, 20, 30, 40]
        row = measure_masks([(1, mask)], mask.shape, .5, gray=gray)[0].iloc[0]
        expected = dict(area_um2=2, area_box_ratio=1, angle_deg=0, solidity=1, convexity=1,
            rectangle_left_um=2, rectangle_right_um=4, rectangle_top_um=1.5, rectangle_bottom_um=2.5,
            centroid_x_um=2.75, centroid_y_um=1.75, brightness_min=10, brightness_max=40,
            brightness_mean=25, brightness_std=np.std([10, 20, 30, 40]), integral_density=200,
            mass_center_x_um=3, mass_center_y_um=1.75)
        for key, value in expected.items(): self.assertAlmostEqual(row[key], value, msg=key)
        self.assertTrue(row.complete)

    def test_concave_boundary_and_undefined_centers(self):
        mask = np.zeros((12, 16), bool); mask[:8, :2] = True; mask[6:8, :8] = True
        row = measure_masks([(1, mask)], mask.shape, 1, gray=np.zeros(mask.shape))[0].iloc[0]
        self.assertFalse(row.complete)
        self.assertLess(row.solidity, 1)
        self.assertLess(row.convexity, 1)
        self.assertEqual(row.area_box_ratio, 28/64)
        self.assertIsNone(row.mass_center_x_um)
        self.assertIsNone(row.mass_center_y_um)
        single = np.zeros_like(mask); single[5, 5] = True
        point = measure_masks([(2, single)], mask.shape, 1)[0].iloc[0]
        self.assertIsNone(point.angle_deg)
        self.assertIsNone(point.brightness_mean)

    def test_scale_changes_geometry_but_not_brightness(self):
        mask = np.zeros((12, 16), bool); mask[2:8, 4:6] = True
        gray = np.full(mask.shape, 30, np.uint8)
        a = measure_masks([(1, mask)], mask.shape, 1, gray=gray)[0].iloc[0]
        b = measure_masks([(1, mask)], mask.shape, 2, gray=gray)[0].iloc[0]
        self.assertEqual(a.angle_deg, 90)
        self.assertEqual(b.area_um2, a.area_um2*4)
        self.assertEqual(b.perimeter_um, a.perimeter_um*2)
        self.assertEqual(b.mass_center_x_um, a.mass_center_x_um*2)
        self.assertEqual(b.integral_density, a.integral_density)
        self.assertEqual(b.solidity, a.solidity)
        with self.assertRaises(ValueError):
            measure_masks([(1, mask)], mask.shape, 1, gray=gray[:-1])


if __name__ == '__main__': unittest.main()
