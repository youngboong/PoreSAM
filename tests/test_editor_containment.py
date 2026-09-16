"""Containment decisions and mutations, independent of SAM model output."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import copy
import unittest

import numpy as np

from pore_editor import Editor, contained_candidates


def rectangle(x, y, width, height):
    mask = np.zeros((80, 80), bool)
    mask[y:y+height, x:x+width] = True
    return mask


class MemoryEditor(Editor):
    def save(self, state, masks, annotations, history, next_id, action):
        state.update(masks=masks, annotations=annotations, history=history,
                     next_id=next_id, revision=state['revision']+1, action=action)
        return state


class ContainmentTests(unittest.TestCase):
    def test_coverage_boundary_and_direction(self):
        old = rectangle(10, 10, 10, 10)
        large = rectangle(10, 10, 11, 10)
        large[10, 10:15] = False  # 95/100 covered, 105px new mask
        self.assertEqual(contained_candidates(large, {1: old}), [1])
        large[10, 15] = False  # 94/100 covered
        self.assertEqual(contained_candidates(large, {1: old}), [])
        self.assertEqual(contained_candidates(old, {1: old}), [])
        self.assertEqual(contained_candidates(rectangle(11, 11, 8, 8), {1: old}), [])

    def state_for(self, new):
        masks = {1: rectangle(10, 10, 5, 5), 2: rectangle(20, 10, 5, 5),
                 3: rectangle(28, 10, 10, 10), 4: rectangle(60, 60, 5, 5)}
        return dict(revision=0, masks=masks, next_id=5, history=[],
                    gray=np.full((80,80),128,np.uint8), report={'scale':{'um_per_pixel':1}},
                    annotations={str(i): {'source': 'automatic'} for i in masks},
                    preview=dict(token='test', masks=[new], source='manual_polygon', prompts={}))

    def test_multiple_contained_partial_and_unrelated(self):
        new = rectangle(9, 9, 22, 17)
        state = self.state_for(new)
        original = copy.deepcopy(state['masks'])
        editor=MemoryEditor()
        with self.assertRaises(ValueError): editor.mutate(state, dict(revision=0, token='test'), 'apply')
        trimmed=editor.trim_preview_overlap(state,dict(revision=0,token='test'))
        self.assertEqual(trimmed['choices'][0]['contained_ids'],[1,2])
        self.assertEqual(trimmed['trimmed_overlap_pixels'],30)
        self.assertEqual(state['revision'],0)
        editor.mutate(state, dict(revision=0, token=trimmed['token']), 'apply')
        self.assertEqual(set(state['masks']), {3, 4, 5})
        self.assertEqual(state['action']['replaced_candidate_ids'], [1, 2])
        self.assertEqual(set(state['annotations']), {'3', '4', '5'})
        self.assertTrue(np.array_equal(state['masks'][3], original[3]))
        self.assertTrue(np.array_equal(state['masks'][4], original[4]))
        self.assertTrue(np.array_equal(state['masks'][5], new & ~original[3]))
        self.assertFalse((state['masks'][5]&state['masks'][3]).any())

    def test_explicit_target_and_other_contained(self):
        state = self.state_for(rectangle(9, 9, 22, 17))
        editor=MemoryEditor()
        trimmed=editor.trim_preview_overlap(state,dict(revision=0,token='test',target_id=4))
        editor.mutate(state, dict(revision=0, token=trimmed['token'], target_id=4), 'apply')
        self.assertEqual(set(state['masks']), {3, 4})
        self.assertEqual(state['action']['replaced_candidate_ids'], [1, 2, 4])
        self.assertEqual(state['next_id'], 5)

    def test_trim_splits_choices_and_rejects_stale_requests(self):
        new=rectangle(5,5,40,40)
        state=self.state_for(new)
        state['masks']={3:rectangle(20,0,3,60)}
        before=state['masks'][3].copy()
        editor=MemoryEditor()
        for bad in [dict(revision=1,token='test'),dict(revision=0,token='old')]:
            with self.assertRaises(ValueError): editor.trim_preview_overlap(state,bad)
        result=editor.trim_preview_overlap(state,dict(revision=0,token='test'))
        self.assertEqual(result['split_count'],2)
        self.assertTrue(np.array_equal(state['masks'][3],before))
        self.assertTrue(all(not (m&before).any() for m in state['preview']['masks']))

    def test_duplicate_rejection_does_not_remove_anything(self):
        state = self.state_for(rectangle(10, 10, 5, 5))
        before = copy.deepcopy(state)
        with self.assertRaises(ValueError):
            MemoryEditor().mutate(state, dict(revision=0, token='test'), 'apply')
        self.assertEqual(state['revision'], 0)
        self.assertEqual(state['annotations'], before['annotations'])
        for i in before['masks']:
            self.assertTrue(np.array_equal(state['masks'][i], before['masks'][i]))


if __name__ == '__main__':
    unittest.main()
