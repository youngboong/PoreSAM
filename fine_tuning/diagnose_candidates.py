"""Exploratory filter ablations for folds 7/8; never modifies app defaults."""
import json
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
from torchvision.ops import batched_nms
from sam2.build_sam import build_sam2
from sam2.utils.amg import rle_to_mask
from train import ROOT, CONFIG, dump, save_csv
from leave_one_out import area_metrics
sys.path.insert(0,str(ROOT/'scripts'))
from segment_first_pass import ProgressGenerator, candidates_from_masks


from pore_selection import TraceGenerator, decode_candidates


def union(masks,shape):
    result=np.zeros(shape,bool)
    for mask in masks:result|=mask
    return result


def main():
    torch.set_num_threads(4)
    run=ROOT/'fine_tuning/runs/loo16_area_20260918_162623'
    plan=json.loads((run/'plan.json').read_text())
    records=json.loads((run/'dataset/manifest.json').read_text())['records']
    out=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else ROOT/'fine_tuning/runs'/('diagnosis_7_8_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=True)
    print(f'RUN {out}',flush=True)
    rows=[]
    for fold in [7,8]:
        record=next(r for r in records if r['name']==plan['folds'][fold-1]['validation'][0])
        image=np.array(Image.open(run/'dataset'/record['image']).convert('RGB'))
        gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY);truth=np.array(Image.open(run/'dataset'/record['labels']))
        gt_stats=[]
        for i in np.unique(truth):
            if not i:continue
            mask=truth==i;ring=ndi.binary_dilation(mask,iterations=4)&~mask
            gt_stats.append(dict(id=int(i),area_pixels=int(mask.sum()),contrast=float(gray[ring].mean()-gray[mask].mean())))
        save_csv(out/f'fold_{fold}_reference_geometry.csv',gt_stats)
        for name,checkpoint in [('baseline',Path(plan['base_checkpoint'])),('finetuned',Path(plan['model_directory'])/f'fold_{fold}.pt')]:
            print(f'GENERATE fold {fold} {name}',flush=True)
            model=generator=None
            pool_path=out/f'fold_{fold}_{name}_pool.json'
            if pool_path.exists():pool=json.loads(pool_path.read_text())
            else:
                model=build_sam2(CONFIG,str(checkpoint),device='cuda',apply_postprocessing=False)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    generator=TraceGenerator(model,points_per_side=48,points_per_batch=4,pred_iou_thresh=.7,
                        stability_score_thresh=.85,crop_n_layers=0,min_mask_region_area=0,output_mode='uncompressed_rle')
                    generator.generate(image)
                pool=generator.pool
                dump(pool_path,pool)
            raw_default=None
            for variant,iou,stability,contrast in [('default',.8,.92,2),('score_070',.7,.92,2),('stability_085',.8,.85,2),('both_relaxed',.7,.85,2),('no_contrast',.8,.92,0)]:
                raw=decode_candidates(pool,iou,stability)
                selected=candidates_from_masks(raw,gray,min_contrast=contrast,min_area=100)
                pred=union([c['mask'] for c in selected],truth.shape)
                m=area_metrics(truth,pred)
                rows.append(dict(fold=fold,image=record['name'],model=name,variant=variant,raw_candidates=len(raw),selected=len(selected),**m))
                Image.fromarray(pred.astype(np.uint8)*255).save(out/f'fold_{fold}_{name}_{variant}.png')
                print(json.dumps(rows[-1]),flush=True)
                if variant=='default':raw_default=raw;default_pred=pred
            # Trace the ordinary application filters, with stage-level area metrics.
            stages={'sam_after_nms':union([c['segmentation'] for c in raw_default],truth.shape)}
            counts=Counter();kept=[];candidate_rows=[]
            for i,c in enumerate(raw_default):
                mask=c['segmentation'];components,n=ndi.label(mask)
                sizes=np.bincount(components.ravel());sizes[0]=0
                if not n:reason='empty';filled=mask;contrast=0.
                else:
                    largest=components==sizes.argmax();filled=ndi.binary_fill_holes(largest)
                    ring=ndi.binary_dilation(filled,iterations=4)&~filled
                    contrast=float(gray[ring].mean()-gray[filled].mean()) if ring.any() else 0.
                    if largest.sum()<.9*mask.sum():reason='disconnected'
                    elif filled.sum()<100:reason='too_small'
                    elif filled.sum()>.2*gray.size:reason='too_large'
                    elif contrast<2:reason='contrast'
                    else:reason='passed';kept.append(filled)
                counts[reason]+=1
                candidate_rows.append(dict(raw_id=i,reason=reason,area=int(filled.sum()),contrast=contrast,
                    predicted_iou=c['predicted_iou'],stability=c['stability_score'],
                    reference_overlap_pixels=int(np.count_nonzero(filled & (truth>0))),
                    unique_missed_overlap_pixels=int(np.count_nonzero(filled & (truth>0) & ~default_pred))))
            stages['app_before_overlap']=union(kept,truth.shape)
            stages['app_final']=default_pred
            diag=dict(rejection_counts=dict(counts),stage_areas={k:area_metrics(truth,v) for k,v in stages.items()})
            old=np.array(Image.open(run/f'fold_{fold}/area_evaluation/{name}_instances.tif'))>0
            diag['default_differs_from_saved_pixels']=int(np.count_nonzero(old!=default_pred))
            dump(out/f'fold_{fold}_{name}_stages.json',diag)
            save_csv(out/f'fold_{fold}_{name}_candidate_reasons.csv',candidate_rows)
            # Best achievable reference overlap among returned masks is diagnostic only, not an automatic score.
            best=[]
            for i in np.unique(truth):
                if not i:continue
                mask=truth==i
                options=[]
                for c in raw_default:
                    pred=c['segmentation'];intersection=np.count_nonzero(mask & pred)
                    options.append(intersection/np.count_nonzero(mask | pred))
                best.append(dict(id=int(i),area_pixels=int(mask.sum()),best_raw_iou=max(options,default=0.),
                                 final_coverage=float(default_pred[mask].mean())))
            save_csv(out/f'fold_{fold}_{name}_best_candidates.csv',best)
            del model,generator,pool,raw_default;torch.cuda.empty_cache()
            save_csv(out/'ablation_metrics.csv',rows)
    dump(out/'result.json',dict(results=rows,notes='Diagnostic use of already examined folds. Threshold ablations are not independent validation or production recommendations. Original settings and models unchanged.'))
    print(f'COMPLETE {out}',flush=True)


if __name__=='__main__':main()
