"""Compare a frozen implementation against ROI processing with identical SAM outputs."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import automate_pores as optimized
from pore_editor import Editor, ROOT


def signature(state):
    return {str(i): hashlib.sha256(m.tobytes()).hexdigest() for i,m in state['masks'].items()}


def check_regions(reference):
    rng=np.random.default_rng(917)
    for trial in range(100):
        shape=(100,140);gray=rng.integers(0,256,shape,dtype=np.uint8)
        masks={}
        for i in range(12):
            x,y=rng.integers(0,140),rng.integers(0,100)
            mask=np.zeros(shape,bool);mask[y:min(100,y+int(rng.integers(2,30))),x:min(140,x+int(rng.integers(2,30)))]=True
            masks[i]=mask
        candidate=np.zeros(shape,bool)
        for k in rng.choice(12,3,replace=False):candidate|=masks[k]
        if trial%2:gray[candidate]=30
        a=reference.reconcile(candidate,masks,gray,10,8)
        b=optimized.reconcile(candidate,masks,gray,10,8)
        assert a[1:]==b[1:],(trial,a[1:],b[1:])
        assert (a[0] is None)==(b[0] is None),trial
        if a[0] is not None:assert np.array_equal(a[0],b[0]),trial
        state=dict(masks=masks,annotations={str(i):dict(source='automated_sam') for i in range(0,12,2)})
        assert reference.consolidation_boxes(state)==optimized.consolidation_boxes(state),trial


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('automate_reference',args.reference)
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    check_regions(reference)
    folder=args.snapshot;report=json.loads((folder/'report.json').read_text());meta=json.loads((folder/'edit.json').read_text())
    gray=np.asarray(Image.open(report['image']).convert('L'))[:report['analysis_bottom_exclusive']]
    with np.load(folder/'entrance_candidates.npz') as archive:
        masks={int(k.rsplit('_',1)[1]):archive[k] for k in archive.files}
    base=dict(dataset='performance-test',report=report,gray=gray,
              sam_gray=np.asarray(Image.open(report['sam_input_image']).convert('L')),
              masks=masks,annotations=meta['annotations'],next_id=meta['next_id'],revision=0,preview=None)
    run=ROOT/'outputs/performance'/time.strftime('automate_roi_%Y%m%d_%H%M%S');run.mkdir(parents=True)
    editor=Editor(run/'edits',run/'projects',device='cpu')
    cache={};preview_elapsed=[0.]
    def preview(state,payload):
        key=json.dumps(payload,sort_keys=True)
        started=time.perf_counter()
        if key not in cache:
            result=editor.preview(state,payload)
            cache[key]=(copy.deepcopy(state['preview']),copy.deepcopy(result))
        else:
            cached,result=cache[key];state['preview']=copy.deepcopy(cached);result=copy.deepcopy(result)
        preview_elapsed[0]+=time.perf_counter()-started
        return result
    adapter=SimpleNamespace(check_revision=editor.check_revision,preview=preview)
    def execute(module):
        state=copy.deepcopy(base);preview_elapsed[0]=0.;started=time.perf_counter();trace=[]
        proposals=module.propose_boxes(state)['boxes']
        for box in proposals:
            response=module.add_candidate(adapter,state,dict(box,revision=0),staged=True)
            trace.append(dict(box=box,response=response,signature=signature(state)))
        groups=module.consolidation_boxes(state)
        for box in groups:
            response=module.add_candidate(adapter,state,dict(box,revision=0),staged=True)
            trace.append(dict(box=box,response=response,signature=signature(state)))
        elapsed=time.perf_counter()-started
        return state,trace,dict(elapsed_seconds=elapsed,preview_seconds=preview_elapsed[0],
                               postprocess_seconds=elapsed-preview_elapsed[0],proposals=len(proposals),groups=len(groups))
    fast,fast_trace,fast_timing=execute(optimized)
    print('optimized',fast_timing,flush=True)
    size=len(cache)
    old,old_trace,old_timing=execute(reference)
    assert len(cache)==size,'Prompt sequence changed'
    assert fast_trace==old_trace,'Intermediate decisions or masks differ'
    assert fast['annotations']==old['annotations'] and fast['next_id']==old['next_id']
    result=dict(reference=old_timing,optimized=fast_timing,pixel_identical=True,intermediate_steps_identical=True,
                random_cases=100,count=len(fast['masks']),postprocess_speedup=old_timing['postprocess_seconds']/fast_timing['postprocess_seconds'])
    np.savez_compressed(run/'expected_masks.npz',**{f'candidate_{i}':m for i,m in fast['masks'].items()})
    (run/'verification.json').write_text(json.dumps(result,indent=2))
    print(run,result,flush=True)


if __name__=='__main__':main()
