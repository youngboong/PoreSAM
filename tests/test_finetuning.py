"""Guard against optimistic matching and incorrect prompt coordinates."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fine_tuning'))
from train import prompts
from evaluate_automatic import measurements
from leave_one_out import area_metrics


class FineTuningTests(unittest.TestCase):
    def test_area_errors_do_not_cancel_in_overlap_score(self):
        truth=np.zeros((10,10),bool);truth[:5,:]=True
        pred=truth.copy();pred[0,:]=False;pred[5,:]=True
        score=area_metrics(truth,pred)
        self.assertEqual(score['relative_area_error_pct'],0)
        self.assertAlmostEqual(score['iou'],40/60)
        self.assertEqual(score['missed_area_pct'],20)
        self.assertEqual(score['extra_area_pct'],20)
        self.assertEqual(score['area_recall'],.8)

    def test_area_metrics_ignore_instance_numbers(self):
        truth=np.ones((10,10),np.uint16)
        pred=truth.copy();pred[:5]=2
        self.assertEqual(area_metrics(truth,pred)['iou'],1)
        score=area_metrics(truth,np.zeros_like(truth))
        self.assertEqual(score['iou'],0)
        self.assertEqual(score['missed_area_pct'],100)

    def test_duplicate_predictions_are_false_positives(self):
        labels=np.zeros((20,30),np.uint16)
        labels[2:7,3:9]=1;labels[12:18,20:28]=2
        scores=measurements(labels,[labels==1,labels==1,labels==2])
        self.assertEqual(scores['true_positives_iou50'],2)
        self.assertEqual(scores['false_positives'],1)
        self.assertEqual(scores['false_negatives'],0)

    def test_merging_two_objects_does_not_count_as_two_matches(self):
        labels=np.zeros((20,30),np.uint16)
        labels[2:7,3:9]=1;labels[12:17,20:26]=2
        scores=measurements(labels,[labels>0])
        self.assertEqual(scores['true_positives_iou50'],1)
        self.assertEqual(scores['false_negatives'],1)
        self.assertEqual(scores['union_iou'],1)

    def test_empty_detection_is_not_a_perfect_result(self):
        labels=np.ones((20,30),np.uint16)
        scores=measurements(labels,[])
        self.assertEqual(scores['recall'],0)
        self.assertEqual(scores['f1'],0)

    def test_prompts_use_xy_and_keep_entire_target_inside_box(self):
        mask=np.zeros((100,200),bool);mask[20:40,120:180]=True
        coords,_=prompts(mask[None],'point',np.random.default_rng(5))
        x,y=coords[0,0]*np.array([200,100])/1024
        self.assertTrue(mask[round(y),round(x)])
        coords,labels=prompts(mask[None],'box',np.random.default_rng(5))
        corners=coords[0]*np.array([200,100])/1024
        self.assertTrue(np.all(corners[0]<=[120,20]))
        self.assertTrue(np.all(corners[1]>=[180,40]))
        np.testing.assert_array_equal(labels,[[2,3]])


if __name__=='__main__':unittest.main()
