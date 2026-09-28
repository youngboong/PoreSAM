"""Behavioral safeguards for image-only nested pore resolution."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'fine_tuning'))
from nested_candidates import prepare_candidates, resolve_nested


class NestedSelectionTests(unittest.TestCase):
    def scene(self, gap=140):
        y, x = np.mgrid[:160, :200]
        parent = ((x-100)/80)**2 + ((y-80)/65)**2 < 1
        holes = [(x-65)**2+(y-80)**2 < 27**2, (x-135)**2+(y-80)**2 < 27**2]
        gray = np.full(parent.shape, 160, np.uint8)
        gray[parent] = gap
        for hole in holes:
            gray[hole] = 20
        raw = [dict(segmentation=h, predicted_iou=.95, stability_score=.98) for h in holes]
        labels = parent.astype(np.uint16)*7
        return gray, labels, raw

    def test_broad_bright_surface_splits_and_preserves_inputs(self):
        gray, labels, raw = self.scene()
        original = labels.copy()
        candidates = prepare_candidates(raw, gray)
        out, log = resolve_nested(labels, candidates, gray)
        self.assertTrue(log[0]['replaced'])
        self.assertEqual(len(np.unique(out[out>0])), 2)
        self.assertTrue(np.array_equal(labels, original))
        self.assertFalse(np.any((out>0) & (labels==0)))
        self.assertEqual(np.count_nonzero(out), sum(r['segmentation'].sum() for r in raw))

    def test_dark_interior_with_bright_fibers_stays_whole(self):
        gray, labels, raw = self.scene(gap=45)
        # Thin bright strands cross the gap; their presence alone must not split.
        gray[35:125, 98:103] = 180
        gray[45:49, 50:150] = 180
        out, log = resolve_nested(labels, prepare_candidates(raw, gray), gray)
        self.assertFalse(log[0]['replaced'])
        self.assertTrue(np.array_equal(out, labels))

    def test_duplicates_do_not_create_multiple_children(self):
        gray, labels, raw = self.scene()
        candidates = prepare_candidates([raw[0], raw[0]], gray)
        self.assertEqual(len(candidates), 1)
        out, log = resolve_nested(labels, candidates, gray)
        self.assertTrue(np.array_equal(out, labels))

    def test_other_parent_is_unchanged_and_selection_is_deterministic(self):
        gray, labels, raw = self.scene()
        labels[1:8, 1:8] = 2
        candidates = prepare_candidates(raw, gray)
        out, log = resolve_nested(labels, candidates, gray)
        out2, log2 = resolve_nested(labels, candidates, gray)
        self.assertTrue(np.all(out[labels==2]==2))
        self.assertTrue(np.array_equal(out, out2))
        self.assertEqual(log, log2)

    def test_bad_shapes_are_rejected(self):
        gray, labels, raw = self.scene()
        with self.assertRaises(ValueError):
            resolve_nested(labels[:-1], [], gray)
        raw[0]['segmentation'] = np.zeros((2, 2), bool)
        with self.assertRaises(ValueError):
            prepare_candidates(raw, gray)

    def test_selection_entry_points_replace_after_base_selection(self):
        from compare_methods16 import select
        gray, labels, children = self.scene()
        image = np.full((320,400),160,np.uint8)
        image[:160,:200] = gray
        raw = [dict(segmentation=labels>0,predicted_iou=.95,stability_score=.98)] + children
        pool=[]
        for item in raw:
            mask=np.zeros(image.shape,bool);mask[:160,:200]=item['segmentation']
            flat=mask.ravel(order='F').astype(np.uint8)
            cuts=np.r_[0,np.flatnonzero(np.diff(flat))+1,len(flat)]
            counts=np.diff(cuts).tolist()
            if flat[0]:counts.insert(0,0)
            y,x=np.where(mask)
            pool.append(dict(segmentation=dict(size=list(image.shape),counts=counts),
                             predicted_iou=item['predicted_iou'],stability_score=item['stability_score'],
                             box=[int(x.min()),int(y.min()),int(x.max()),int(y.max())]))
        for base in ('relaxed','two_stage'):
            original=select(pool,image,100,8,'cpu',base)
            nested=select(pool,image,100,8,'cpu',base+'_nested')
            self.assertEqual(len(original['masks']),1)
            self.assertEqual(len(nested['masks']),2)
            self.assertTrue(nested['nested_decisions'][0]['replaced'])
        with self.assertRaises(ValueError):
            select(pool,image,100,8,'cpu','unknown')

    def test_metrics_reward_two_correct_instances_and_handle_gapped_ids(self):
        from evaluate_nested import metrics
        truth=np.array([[0,1,1,0,7,7]],np.uint16)
        merged=np.array([[0,100,100,0,100,100]],np.uint32)
        split=np.array([[0,900,900,0,501,501]],np.uint32)
        self.assertEqual(metrics(truth,merged)['iou'],1.)
        self.assertAlmostEqual(metrics(truth,merged)['instance_f1'],2/3)
        self.assertEqual(metrics(truth,split)['instance_f1'],1.)
        self.assertEqual(metrics(truth,np.zeros_like(truth))['instance_f1'],0.)


if __name__ == '__main__':
    unittest.main()
