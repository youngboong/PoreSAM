"""Read-only SEM box investigation; all previews are written to a separate folder."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from pore_editor import Editor,ROOT
from box_pores import preview_box
from automate_pores import propose_boxes
from automate_pores import reconcile
from segment_first_pass import overlay


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--box',type=int,nargs=4,required=True);p.add_argument('--automate-only',action='store_true');p.add_argument('--original',action='store_true');args=p.parse_args()
    run=ROOT/'outputs/box_debug'/time.strftime('%Y%m%d_%H%M%S');run.mkdir(parents=True)
    report=json.loads(args.report.read_text(encoding='utf-8'))
    gray=np.asarray(Image.open(report['image']).convert('L'))[:report['analysis_bottom_exclusive']]
    sam_gray=np.asarray(Image.open(report.get('sam_input_image',report['image'])).convert('L'))[:gray.shape[0]]
    if args.original:sam_gray=gray
    masks_path=args.report.parent/'entrance_candidates.npz'
    with np.load(masks_path) as archive:masks={int(k.rsplit('_',1)[1]):archive[k] for k in archive.files}
    state=dict(dataset='sem-box-check',report=report,gray=gray,sam_gray=sam_gray,masks=masks,revision=0,preview=None)
    editor=Editor(run/'edits',run/'projects',device='cpu')
    payload=dict(revision=0,box=args.box,max_candidates=32)
    first=editor.preview(state,payload)
    overlay(gray,[state['preview']['masks'][0]],True).save(run/'single_box_prompt.png')
    if args.automate_only:
        details=[]
        for i,(mask,choice) in enumerate(zip(state['preview']['masks'],first['choices'])):
            reconciled,ids,review=reconcile(mask,masks,sam_gray,100,8)
            details.append(dict(choice=choice['score'],area=int(mask.sum()),overlap=choice['overlap_percent'],merged=ids,review=review,accepted=reconciled is not None))
            overlay(gray,[mask],True).save(run/f'choice_{i}.png')
        (run/'verification.json').write_text(json.dumps(details,indent=2));print(run,details,flush=True);return
    result=preview_box(editor,state,payload)
    found=[np.unpackbits(state['queued_previews'][r['token']]['masks'][0],count=gray.size).reshape(gray.shape).astype(bool) for r in result['regions']]
    overlay(gray,found,True).save(run/'multiple_preview.png')
    original=Image.fromarray(gray).convert('RGB');ImageDraw.Draw(original).rectangle(args.box,outline='yellow',width=3);original.save(run/'box.png')
    proposed=propose_boxes(state)
    proposal_image=Image.fromarray(gray).convert('RGB');draw=ImageDraw.Draw(proposal_image)
    for i,item in enumerate(proposed['boxes'],1):draw.rectangle(item['box'],outline='cyan',width=2);draw.text(item['box'][:2],str(i),fill='yellow')
    proposal_image.save(run/'automate_boxes.png')
    summary=dict(box=args.box,candidates=len(found),fallback=result['fallback'],points_per_side=result.get('points_per_side'),proposed=proposed)
    (run/'verification.json').write_text(json.dumps(summary,indent=2));print(run,summary,flush=True)


if __name__=='__main__':main()
