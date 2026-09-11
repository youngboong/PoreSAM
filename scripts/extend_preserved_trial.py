"""Append an image-designed boundary-preservation condition to a finished ablation."""
import json
import sys
import hashlib
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from fiber_pores import normalized_image,prompts,select_prediction,deduplicate
from segment_first_pass import overlay
from trial_preprocessing_ablation import ROOT,SOURCE,CONDITIONS,input_image,load_masks,overlap_metrics,render


def main():
    out=Path(sys.argv[1]);report=json.loads((out/'report.json').read_text())
    assert report['status']=='completed'
    plan=json.loads((out/'plan.json').read_text());plan['conditions']=CONDITIONS
    plan['boundary_preservation']=dict(foreground_otsu_multiplier=.8,core_min_radius=6,protect_distance=8,
        purpose='retain original pixels around sufficiently thick bright structures; not a depth classification',
        chosen_using='input-image visual inspection, not reference masks')
    model=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(ROOT/'checkpoints/sam2.1_hiera_small.pt'),device='cuda',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model);pools={};grays={}
    for name in report['results']:
        folder=out/name/'preserved';folder.mkdir();(folder/'candidates').mkdir()
        gray=np.asarray(Image.open(SOURCE/name/'images/original.png'));grays[name]=gray
        pixels=input_image(gray,CONDITIONS['preserved'])
        Image.fromarray(pixels).save(folder/'input.png');Image.fromarray(pixels[128:448,128:448]).save(folder/'crop.png')
        Image.fromarray(cv2.subtract(normalized_image(gray),pixels)[128:448,128:448]).save(folder/'removed_crop.png')
        proposals=load_masks(SOURCE/name/'masks/prompts.npz');selected=[];audit=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            predictor.set_image(cv2.cvtColor(pixels,cv2.COLOR_GRAY2RGB))
            for i,proposal in enumerate(proposals):
                box,points=prompts(proposal)
                masks,scores,_=predictor.predict(box=box,point_coords=points,point_labels=np.ones(len(points),np.int32),multimask_output=True)
                pick=select_prediction(masks,scores,proposal,gray)
                if pick:
                    mask,metadata=pick;selected.append(mask);audit.append(dict(proposal=i,**metadata))
                if i%40==0:print(name,'preserved',i+1,'/',len(proposals),flush=True)
        selected=deduplicate(selected);large=[m for m in selected if .015<=m.mean()<=.12];pools[name]=large
        for filename,masks in [('all_masks',selected),('large_masks',large)]:np.savez_compressed(folder/(filename+'.npz'),**{f'candidate_{i+1}':m for i,m in enumerate(masks)})
        overlay(gray,large,numbered=True).save(folder/'result.png')
        for i,m in enumerate(large):overlay(gray,[m]).save(folder/'candidates'/f'{i+1}.png')
        (folder/'selection.json').write_text(json.dumps(audit,indent=2))
        report['results'][name]['methods']['preserved']=dict(all_count=len(selected),large_count=len(large),reused=False)
    ref_file=ROOT/'outputs/manual_edits/image_dffc9b43ff372f12__run_0002/revision_0002/entrance_candidates.npz'
    with np.load(ref_file) as data:refs=[data['candidate_30'].copy(),data['candidate_31'].copy()]
    for name,large in pools.items():
        metrics=overlap_metrics(refs if name=='GF_1kx_1_BSE' else [],large)
        report['results'][name]['methods']['preserved']['evaluation']=metrics
        if name=='GF_1kx_1_BSE':
            gray=grays[name];best=[large[r['best_candidate_id']-1] for r in metrics['references'] if r['best_candidate_id']]
            comparison=Image.new('RGB',(gray.shape[1]*2,gray.shape[0]));comparison.paste(overlay(gray,refs,numbered=True),(0,0));comparison.paste(overlay(gray,best,numbered=True),(gray.shape[1],0));comparison.save(out/name/'preserved/matches.png')
        print(name,json.dumps(metrics),flush=True)
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in report['protected_hashes'].items())
    render(out,report)
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
