"""Small numerical checks for units, overlaps, edge policy, and empty results."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import unittest
import numpy as np
from analyze_candidates import measure_masks, histogram_table


class MeasurementChecks(unittest.TestCase):
    def test_overlap_units_edges_and_count_conservation(self):
        a = np.zeros((12,16), dtype=bool)
        b = a.copy()
        a[2:6,2:6] = True
        b[0:4,4:8] = True
        frame, stats, union, counts, complete = measure_masks([(1,a),(2,b)], a.shape, .5)
        self.assertEqual(stats["analyzed_area_um2"], 48)
        self.assertEqual(stats["candidate_count"], 2)
        self.assertEqual(stats["complete_candidate_count"], 1)
        self.assertEqual(stats["union_candidate_area_um2"], 7)
        self.assertEqual(stats["overlap_excess_area_um2"], 1)
        self.assertAlmostEqual(frame.iloc[0].equivalent_diameter_um, 4/np.sqrt(np.pi))
        self.assertEqual(counts.sum(), 2)
        self.assertEqual(complete.sum(), 1)
        self.assertEqual(histogram_table(frame.equivalent_diameter_um)["count"].sum(), 2)

    def test_empty(self):
        frame, stats, union, counts, complete = measure_masks([], (12,16), .5)
        self.assertEqual(stats["candidate_count"], 0)
        self.assertEqual(stats["candidate_union_area_percent"], 0)
        self.assertIsNone(stats["complete_equivalent_diameter_um_median"])
        self.assertEqual(len(histogram_table(np.array([]))), 0)


if __name__ == "__main__":
    unittest.main()
