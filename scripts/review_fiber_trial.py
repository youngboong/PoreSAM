"""Post-run review using pre-existing user edits; never generates SAM prompts."""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from fiber_pores import large_consensus
from segment_first_pass import overlay
from trial_fiber_structure import render
from trial_pore_methods import ROOT,overlap_metrics,reference_masks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('folder',type=Path);args=parser.parse_args()
    out=args.folder
    report=json.loads((out/'report.json').read_text())
    source=ROOT/'outputs/manual_edits/image_dffc9b43ff372f12__run_0002/revision_0002/entrance_candidates.npz'
    with np.load(source) as data:large_refs=[data['candidate_30'].copy(),data['candidate_31'].copy()]
    report['user_reference_source']=dict(file=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),candidate_ids=[30,31],
                                         role='post-run comparison only; pre-existing user-accepted masks, not a complete ground truth')
    report['review_filter']=dict(area_fraction=[.015,.12],consensus_iou=.55,chosen_after_exploratory_run=True,
                                 status='alternative candidates; no automatic acceptance or measurement update')
    np.savez_compressed(out/'user_large_references.npz',**{f'candidate_{i}':m for i,m in zip([30,31],large_refs)})
    for name,item in report['results'].items():
        folder=out/name
        gray=np.asarray(Image.open(folder/'images/original.png'))
        with np.load(folder/'masks/sam_simplified.npz') as data:raw=[data[k].copy() for k in data.files]
        pool=[m for m in raw if .015<=m.mean()<=.12]
        representatives,support=large_consensus(raw)
        refs=reference_masks(name,gray.shape) if name=='GF_1kx_1_BSE' else []
        for key,masks in [('large_pool',pool),('large_review',representatives)]:
            np.savez_compressed(folder/'masks'/f'{key}.npz',**{f'candidate_{i+1}':m for i,m in enumerate(masks)})
            overlay(gray,masks,numbered=True).save(folder/'images'/f'{key}.png')
            item['methods'][key]=overlap_metrics(refs,masks)
        (folder/'representative_support.json').write_text(json.dumps(support,indent=2),encoding='utf-8')
        for key,method in item['methods'].items():
            if name=='GF_1kx_1_BSE':
                with np.load(folder/'masks'/f'{key}.npz') as data:masks=[data[k].copy() for k in data.files]
                method['user_large_references']=overlap_metrics(large_refs,masks)
        previews=folder/'images/large_candidates';previews.mkdir(exist_ok=True)
        for i,mask in enumerate(pool):
            rgba=np.zeros((*mask.shape,4),np.uint8);rgba[mask]=[35,195,235,70]
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgba,contours,-1,(35,220,255,255),2)
            Image.fromarray(rgba).save(previews/f'{i+1}.png')
        if name=='GF_1kx_1_BSE':
            matches=item['methods']['large_pool']['user_large_references']['references']
            best=[pool[r['best_candidate_id']-1] for r in matches]
            comparison=Image.new('RGB',(gray.shape[1]*2,gray.shape[0]))
            comparison.paste(overlay(gray,large_refs,numbered=True),(0,0));comparison.paste(overlay(gray,best,numbered=True),(gray.shape[1],0))
            comparison.save(folder/'images/user_large_comparison.png')
        # Analysis source files remain untouched by inference and review.
        assert hashlib.sha256((Path(item['baseline'])/'masks/entrance_candidates.npz').read_bytes()).hexdigest()==item['baseline_sha256']
        print(name,len(pool),flush=True)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    render(out,report['results'])
    print(json.dumps(report['results']['GF_1kx_1_BSE']['methods']['large_pool']['user_large_references']),flush=True)


if __name__=='__main__':main()
