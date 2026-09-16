"""Trace a saved pair through joint-box SAM and overlap reconciliation, read-only."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from pore_editor import Editor
from automate_pores import consolidation_boxes, reconcile, dark_shared_boundaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--ids', type=int, nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    report = json.loads((args.snapshot/'report.json').read_text())
    metadata = json.loads((args.snapshot/'edit.json').read_text())
    with np.load(args.snapshot/'entrance_candidates.npz') as archive:
        masks = {int(k.rsplit('_',1)[1]):archive[k] for k in archive.files}
    gray = np.asarray(Image.open(report['image']).convert('L'))[:report['analysis_bottom_exclusive']]
    sam_gray = np.asarray(Image.open(report['sam_input_image']).convert('L'))
    state = dict(dataset='pair-diagnostic', report=report, gray=gray, sam_gray=sam_gray,
                 masks=masks, annotations=metadata['annotations'], revision=0, preview=None,
                 next_id=metadata['next_id'])
    groups = [g for g in consolidation_boxes(state,limit=1000) if set(args.ids)<=set(g['candidate_ids'])]
    union = np.logical_or.reduce([masks[i] for i in args.ids])
    yy, xx = np.nonzero(union)
    tight = [max(0,int(xx.min())-8), max(0,int(yy.min())-8),
             min(gray.shape[1]-1,int(xx.max())+8),min(gray.shape[0]-1,int(yy.max())+8)]
    boxes = [('group',g['box']) for g in groups]+[('pair',tight)]
    editor = Editor(args.output/'edits',args.output/'projects',device='cpu')
    records = []
    for name, box in boxes:
        choices = editor.preview(state,dict(revision=0,box=box,points=[],labels=[],fill_holes=True))
        for number,(mask,choice) in enumerate(zip(state['preview']['masks'], choices['choices'])):
            area = int(mask.sum())
            overlaps = {str(i):dict(pixels=int((old&mask).sum()),old_fraction=float((old&mask).sum()/old.sum()),new_fraction=float((old&mask).sum()/max(1,area)))
                        for i,old in masks.items() if (old&mask).any()}
            settings=report.get('selection_settings',{})
            accepted,ids,review=reconcile(mask,masks,sam_gray,settings.get('min_area_pixels',100),settings.get('min_contrast',8))
            others=np.logical_or.reduce([m for i,m in masks.items() if i not in args.ids])
            pair_mask=mask & ~others
            pair_accepted,pair_ids,pair_review=reconcile(pair_mask,masks,sam_gray,settings.get('min_area_pixels',100),settings.get('min_contrast',8))
            record=dict(box_kind=name,box=box,option=number,score=choice['score'],area=area,
                        accepted=accepted is not None,merged_ids=ids,ambiguous=review,overlaps=overlaps,
                        pair_only_accepted=pair_accepted is not None,pair_only_ids=pair_ids,pair_only_ambiguous=pair_review)
            if pair_accepted is not None:
                rgb=np.repeat(gray[:,:,None],3,axis=2)
                rgb[pair_accepted]=(rgb[pair_accepted]*.55+np.array([0,220,255])*.45).astype(np.uint8)
                Image.fromarray(rgb).save(args.output/f'{name}_{number}_pair_only.png')
            records.append(record)
            rgb=np.repeat(gray[:,:,None],3,axis=2)
            rgb[mask]=(rgb[mask]*.55+np.array([0,220,255])*.45).astype(np.uint8)
            Image.fromarray(rgb).save(args.output/f'{name}_{number}.png')
            print(record,flush=True)
    (args.output/'verification.json').write_text(json.dumps(dict(ids=args.ids,groups=groups,choices=records),indent=2))


if __name__=='__main__':main()
