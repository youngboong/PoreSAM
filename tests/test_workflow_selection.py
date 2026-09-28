"""Exercise the application analysis path, including reuse of pre-NMS hypotheses."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from pore_editor import Editor
from analysis_timing import AnalysisTiming


class WorkflowSelectionTests(unittest.TestCase):
    def test_nested_selection_and_cached_pool(self):
        y, x = np.mgrid[:320, :400]
        parent = ((x-100)/80)**2 + ((y-80)/65)**2 < 1
        children = [(x-65)**2+(y-80)**2 < 27**2, (x-135)**2+(y-80)**2 < 27**2]
        gray = np.full(parent.shape, 160, np.uint8)
        gray[parent] = 140
        for mask in children: gray[mask] = 20
        pool = []
        for mask in [parent, *children]:
            flat = mask.ravel(order='F').astype(np.uint8)
            cuts = np.r_[0, np.flatnonzero(np.diff(flat))+1, len(flat)]
            counts = np.diff(cuts).tolist()
            if flat[0]: counts.insert(0, 0)
            yy, xx = np.where(mask)
            pool.append(dict(segmentation=dict(size=list(gray.shape), counts=counts),
                             box=[int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
                             predicted_iou=.95, stability_score=.98))

        class Generator:
            def __init__(self, model, **settings):
                assert settings['pred_iou_thresh'] == .7
                assert settings['output_mode'] == 'uncompressed_rle'
                self.pool = pool
            def generate(self, image): return []

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            editor = Editor(root/'edits', root/'projects', device='cpu')
            buffer = io.BytesIO(); Image.fromarray(gray).save(buffer, format='PNG')
            project = editor.workflow.upload('nested.png', buffer.getvalue())
            workflow = editor.workflow
            config = workflow.validate_config(project, dict(analysis_bottom=320, scale_um=10,
                scale_pixels=100, min_area_pixels=100, min_contrast=8, points_per_side=16,
                scale_confirmed=True, preprocessing_mode='none'))
            for number in [1, 2, 3]:
                if number == 3:
                    # An otherwise identical result from different weights must
                    # never supply cached candidates to the newly selected model.
                    for old in (workflow.root/project['id']/'runs').glob('*/report.json'):
                        prior=json.loads(old.read_text());prior['checkpoint_sha256']='different-weights'
                        old.write_text(json.dumps(prior))
                run = f'run_{number:04d}'
                folder = workflow.root/project['id']/'runs'/run
                folder.mkdir(parents=True)
                workflow.jobs[run] = dict(status='running', progress=1)
                workflow.job_timings[run] = AnalysisTiming()
                workflow.running = True
                with patch('project_workflow.build_sam2', return_value=object()) as build, patch('project_workflow.TraceGenerator', Generator):
                    workflow.run(run, workflow.projects[project['id']], config, run, number)
                    self.assertEqual(workflow.jobs[run]['status'], 'complete', workflow.jobs[run])
                    self.assertEqual(build.call_count, 0 if number == 2 else 1)
                report = json.loads((folder/'report.json').read_text())
                self.assertEqual(report['selection_settings']['method'], 'two_stage_nested')
                self.assertEqual(len(report['checkpoint_sha256']), 64)
                self.assertEqual(report['entrance_candidate_count'], 2)
                self.assertTrue(report['nested_decisions'][0]['replaced'])
                self.assertEqual(len(json.loads((folder/'sam_raw/pool.json').read_text())), 3)


if __name__ == '__main__':
    unittest.main()
