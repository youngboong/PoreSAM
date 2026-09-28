"""Saving masks must not require full measurements; explicit refresh is revision-safe."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pore_editor import Editor, measure_masks


class DeferredMeasurementsTests(unittest.TestCase):
    def test_delete_refresh_cache_and_undo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / 'baseline'
            baseline.mkdir()
            a = np.zeros((80, 80), bool); a[10:25, 10:25] = True
            b = np.zeros_like(a); b[:12, 50:65] = True
            masks = {1: a, 2: b}
            report = dict(image='unused.png', scale=dict(um_per_pixel=.5))
            (baseline / 'report.json').write_text(json.dumps(report))
            np.savez_compressed(baseline / 'entrance_candidates.npz', candidate_1=a, candidate_2=b)
            editor = Editor(root / 'edits', root / 'projects', device='cpu')
            state = dict(dataset='test', baseline=baseline, gray=np.zeros_like(a, dtype=np.uint8),
                         report=report, revision=0, masks=masks, history=[], next_id=3,
                         annotations={str(i): dict(source='automatic') for i in masks})
            with patch('pore_editor.measure_masks', wraps=measure_masks) as measure:
                original = editor.measurements(state)
                self.assertEqual(measure.call_count, 1)
                saved = editor.mutate(state, dict(revision=0, target_ids=[1]), 'delete')
                self.assertEqual(measure.call_count, 1)
                self.assertIsNone(saved['stats'])
                self.assertEqual([r['candidate_id'] for r in saved['candidates']], [2])
                self.assertTrue(saved['candidates'][0]['touches_image_edge'])
                refreshed = editor.measurements(state)
                self.assertEqual(measure.call_count, 2)
                self.assertEqual(refreshed['stats']['candidate_count'], 1)
                self.assertIs(editor.measurements(state), refreshed)
                self.assertEqual(measure.call_count, 2)
                self.assertEqual(original['stats']['candidate_count'], 2)
                editor.mutate(state, dict(revision=1), 'undo')
                self.assertEqual(measure.call_count, 2)
                self.assertEqual(editor.measurements(state)['stats']['candidate_count'], 2)
                self.assertEqual(measure.call_count, 3)
                editor.mutate(state, dict(revision=2, target_ids=[1, 2]), 'delete')
                self.assertEqual(measure.call_count, 3)
                empty = editor.measurements(state)
                self.assertEqual(empty['candidates'], [])
                self.assertEqual(empty['stats']['candidate_union_area_percent'], 0)


if __name__ == '__main__':
    unittest.main()
