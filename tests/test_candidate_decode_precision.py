"""CPU float32 replay must not inherit the GPU bfloat16 score cutoff."""
import sys
from pathlib import Path
import unittest
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fine_tuning'))
from diagnose_candidates import decode_candidates


class DecodePrecisionTests(unittest.TestCase):
    def test_cpu_and_bfloat16_thresholds_are_explicit(self):
        pool=[dict(segmentation=dict(size=[2,2],counts=[0,4]),box=[0,0,1,1],
                   predicted_iou=.8005,stability_score=.95)]
        cpu=decode_candidates(pool,.8,.92,device='cpu',score_dtype=torch.float32)
        rounded=decode_candidates(pool,.8,.92,device='cpu',score_dtype=torch.bfloat16)
        self.assertEqual(len(cpu),1)
        self.assertEqual(len(rounded),0)
        np.testing.assert_array_equal(cpu[0]['segmentation'],np.ones((2,2),bool))

    def test_nms_removes_duplicate_boxes(self):
        item=dict(segmentation=dict(size=[2,2],counts=[0,4]),box=[0,0,1,1],stability_score=.99)
        pool=[dict(item,predicted_iou=.95),dict(item,predicted_iou=.9)]
        result=decode_candidates(pool,.7,.85,device='cpu',score_dtype=torch.float32)
        self.assertEqual([r['pool_id'] for r in result],[0])


if __name__=='__main__':unittest.main()
