"""Real CPU checkpoint smoke check, writing only isolated verification outputs."""
import io
import json
import time
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from pore_editor import Editor, ROOT


def main():
    folder = ROOT/'outputs/ui_checks'/('cpu_'+time.strftime('%Y%m%d_%H%M%S'))
    folder.mkdir(parents=True)
    yy,xx = np.mgrid[:320,:320]
    gray = np.full((320,320),190,np.uint8)
    for x,y in [(80,80),(230,80),(80,230),(230,230)]:
        gray[(xx-x)**2+(yy-y)**2<32**2] = 35
    stream = io.BytesIO()
    Image.fromarray(gray).save(stream,format='PNG')
    with patch('torch.cuda.is_available',return_value=False), \
         patch('torch.cuda.empty_cache',side_effect=AssertionError('CPU requested CUDA cache')):
        editor = Editor(folder/'edits',folder/'projects',device='auto')
        assert editor.device == 'cpu'
        project = editor.workflow.upload('cpu_fixture.png',stream.getvalue())
        config = dict(project['config'],analysis_bottom=320,scale_um=10,scale_pixels=100,
                      scale_confirmed=True,points_per_side=16,min_area_pixels=20,min_contrast=0)
        started = time.monotonic()
        job = editor.workflow.start(project['id'],config)
        last = None
        while True:
            status = editor.workflow.status(job['job_id'])
            if status['progress'] != last:
                last = status['progress'];print('Automatic analysis:',last,flush=True)
            if status['status'] != 'running':break
            if time.monotonic()-started>900:raise TimeoutError('CPU smoke analysis exceeded 15 minutes')
            time.sleep(.2)
        assert status['status']=='complete',status
        automatic_seconds = time.monotonic()-started
        state = editor.state(status['dataset'])
        assert state['report']['inference_device']=='cpu'
        assert not (state['baseline']/'measurements').exists()
        assert not editor.response(state)['report_ready']
        payload = dict(revision=0,box=[43,43,117,117],points=[],labels=[])
        started = time.monotonic()
        first = editor.preview(state,payload)
        first_seconds = time.monotonic()-started
        assert first['choices']
        assert next(editor.predictor.model.parameters()).device.type=='cpu'
        with patch.object(editor.predictor,'set_image',wraps=editor.predictor.set_image) as encode:
            started = time.monotonic()
            second = editor.preview(state,dict(payload,box=[190,43,269,117]))
            repeat_seconds = time.monotonic()-started
            assert second['choices'] and encode.call_count==0
        result = dict(device=editor.device,threads=torch.get_num_threads(),shape=list(gray.shape),
                      points_per_side=16,automatic_seconds=automatic_seconds,
                      first_prompt_seconds=first_seconds,repeated_prompt_seconds=repeat_seconds,
                      candidate_count=status['candidate_count'],embedding_reused=True,
                      note='Synthetic smoke check; not a representative speed or accuracy benchmark.')
        (folder/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2),flush=True)
        print(folder,flush=True)


if __name__=='__main__':main()
