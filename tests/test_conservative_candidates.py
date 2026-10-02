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

    def parent_scene(self):
        gray=np.full((120,160),180,np.uint8)
        parent=np.zeros(gray.shape,bool);parent[25:65,40:80]=True
        child=np.zeros(gray.shape,bool);child[35:45,50:60]=True
        gray[parent]=30
        return gray,parent,child,dict(segmentation=parent,predicted_iou=.75,stability_score=.9)

    def test_contained_child_replaced_without_hole(self):
        gray,parent,child,candidate=self.parent_scene()
        masks,log,settings=supplement([child],[candidate],gray)
        self.assertEqual(len(masks),1)
        np.testing.assert_array_equal(masks[0],parent)
        self.assertEqual(log[0]['reason'],'replaced_contained')
        self.assertEqual(settings['replaced_masks'],1)

    def test_large_overlap_is_allowed_only_for_containment(self):
        gray,parent,child,candidate=self.parent_scene()
        child=parent.copy();child[25:35]=False
        masks,log,_=supplement([child],[candidate],gray)
        np.testing.assert_array_equal(masks[0],parent)
        partial=np.roll(parent,20,axis=1)
        masks,log,_=supplement([partial],[candidate],gray)
        np.testing.assert_array_equal(masks[0],partial)
        self.assertEqual(log[0]['reason'],'existing_overlap')

    def test_weak_parent_does_not_delete_child(self):
        gray,parent,child,candidate=self.parent_scene()
        gray[parent]=180;gray[child]=30
        masks,log,_=supplement([child],[candidate],gray)
        np.testing.assert_array_equal(masks[0],child)
        self.assertFalse(log[0]['accepted'])

    def test_child_selected_first_is_replaced_by_parent(self):
        gray,parent,child,candidate=self.parent_scene()
        small=dict(candidate,segmentation=child,predicted_iou=.9)
        # Both candidates must have independent rim evidence to enter selection.
        gray[child]=0
        masks,log,_=supplement([],[small,candidate],gray)
        self.assertEqual(len(masks),1)
        np.testing.assert_array_equal(masks[0],parent)

    def test_two_children_replaced_as_one_parent(self):
        gray,parent,child,candidate=self.parent_scene()
        second=np.roll(child,15,axis=0)
        masks,log,_=supplement([child,second],[candidate],gray)
        self.assertEqual(len(masks),1)
        np.testing.assert_array_equal(masks[0],parent)
        self.assertEqual(len(log[0]['replaced_mask_indices']),2)

    def test_replacement_cannot_erase_child_after_partial_overlap_trim(self):
        # A thin unrelated neighbour bisects the proposed parent.
        gray,parent,child,candidate=self.parent_scene()
        neighbour=np.zeros(gray.shape,bool);neighbour[20:70,59:61]=True
        child=np.zeros(gray.shape,bool);child[35:45,65:75]=True
        masks,log,_=supplement([child,neighbour],[candidate],gray,settings={'min_new_component_fraction':.4})
        self.assertFalse(log[0]['accepted'])
        self.assertEqual(log[0]['reason'],'containment_lost_after_trim')
        np.testing.assert_array_equal(masks[0],child)
        np.testing.assert_array_equal(masks[1],neighbour)


if __name__=='__main__':unittest.main()
