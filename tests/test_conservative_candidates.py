import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fine_tuning'))
from conservative_candidates import supplement


class SupplementTests(unittest.TestCase):
    def example(self):
        gray=np.full((80,100),180,np.uint8)
        old=np.zeros(gray.shape,bool);old[10:30,10:30]=True;gray[old]=30
        new=np.zeros(gray.shape,bool);new[40:65,60:85]=True;gray[new]=30
        candidate=dict(segmentation=new,predicted_iou=.75,stability_score=.9)
        return gray,old,new,candidate

    def test_supported_new_region_and_old_pixels_preserved(self):
        gray,old,new,candidate=self.example()
        masks,log,_=supplement([old],[candidate],gray)
        self.assertEqual(len(masks),2)
        np.testing.assert_array_equal(masks[0],old)
        np.testing.assert_array_equal(masks[1],new)
        self.assertFalse(np.any(masks[0]&masks[1]))
        self.assertTrue(log[0]['accepted'])

    def test_flat_region_and_duplicate_are_not_added(self):
        gray,old,new,candidate=self.example()
        duplicate=dict(candidate,segmentation=old)
        gray[new]=180
        masks,log,_=supplement([old],[candidate,duplicate],gray)
        self.assertEqual(len(masks),1)
        self.assertEqual([r['reason'] for r in log],['contrast','existing_overlap'])

    def test_competing_duplicates_remain_exclusive(self):
        gray,old,new,candidate=self.example()
        masks,log,_=supplement([old],[candidate,dict(candidate)],gray)
        self.assertEqual(len(masks),2)
        self.assertEqual(sum(r['accepted'] for r in log),1)


if __name__=='__main__':unittest.main()
