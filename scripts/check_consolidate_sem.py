"""Reproduce the 72/94/96 split using a read-only snapshot and real CPU SAM."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from pore_editor import Editor,ROOT
from automate_pores import consolidation_boxes,add_candidate
from segment_first_pass import overlay


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--ids',type=int,nargs='+',default=[72,94,96]);args=parser.parse_args()
    folder=args.snapshot;report=json.loads((folder/'report.json').read_text());meta=json.loads((folder/'edit.json').read_text())
    gray=np.asarray(Image.open(report['image']).convert('L'))[:report['analysis_bottom_exclusive']]
    with np.load(folder/'entrance_candidates.npz') as data:masks={int(k.rsplit('_',1)[1]):data[k] for k in data.files}
    state=dict(dataset='consolidate-test',report=report,gray=gray,sam_gray=np.asarray(Image.open(report['sam_input_image']).convert('L')),
               masks=masks,annotations=meta['annotations'],next_id=meta['next_id'],revision=0,preview=None)
    groups=consolidation_boxes(state);print('groups',groups,flush=True);group=next(g for g in groups if set(args.ids)<=set(g['candidate_ids']))
    run=ROOT/'outputs/merge_debug'/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
    editor=Editor(run/'edits',run/'projects',device='cpu')
    response=add_candidate(editor,state,dict(group,revision=0),staged=True)
    print(response,group,flush=True)
    assert response.get('merged'),response
    keep=min(args.ids)
    assert keep in state['masks'] and all(i not in state['masks'] for i in args.ids if i!=keep)
    assert set(args.ids)<=set(state['annotations'][str(keep)]['merged_from'])
    assert not (np.sum(list(state['masks'].values()),axis=0)>1).any()
    overlay(gray,[state['masks'][keep]],True).save(run/'merged_pore.png')
    (run/'verification.json').write_text(json.dumps(dict(response=response,group=group,annotation=state['annotations'][str(keep)],count_before=len(masks),count_after=len(state['masks'])),indent=2))
    print(run,flush=True)


if __name__=='__main__':main()
