"""Compare original and tuned checkpoints with identical app candidate filters."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from segment_first_pass import ProgressGenerator, candidates_from_masks
from sam2.build_sam import build_sam2
from train import CONFIG, dump, sha


def measurements(truth, predictions):
    ids=np.unique(truth);ids=ids[ids>0]
    counts=np.bincount(truth.ravel())
    ious=np.zeros((len(predictions),len(ids)))
    union_mask=np.zeros_like(truth,dtype=bool)
    for row,prediction in enumerate(predictions):
        intersection=np.bincount(truth[prediction],minlength=len(counts))[ids]
        ious[row]=intersection/(int(prediction.sum())+counts[ids]-intersection)
        union_mask|=prediction
    matched=[]
    if ious.size:
        a,b=linear_sum_assignment(-((ious>=.5)*1000+ious))
        matched=[float(ious[i,j]) for i,j in zip(a,b) if ious[i,j]>=.5]
    tp=len(matched);fp=len(predictions)-tp;fn=len(ids)-tp
    return dict(reference_count=len(ids),predicted_count=len(predictions),true_positives_iou50=tp,false_positives=fp,false_negatives=fn,
                precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),
                union_iou=float((union_mask & (truth>0)).sum()/max(1,(union_mask | (truth>0)).sum())),
                reference_area_fraction=float((truth>0).mean()),predicted_area_fraction=float(union_mask.mean()))


def overlay(image,masks):
    view=image.copy()
    for mask in masks:
        view[mask]=np.rint(.60*view[mask]+.40*np.array([255,215,0])).astype(np.uint8)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(view,contours,-1,(255,215,0),1)
    return view


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--split',choices=['test','validation'],default='test')
    args=parser.parse_args()
    config=json.loads((args.run/'config.json').read_text())
    dataset=Path(config['arguments']['dataset'])
    manifest=json.loads((args.run/'dataset_manifest.json').read_text())
    test=[r for r in manifest['records'] if r['split']==args.split]
    out=args.run/'automatic_evaluation'
    out.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    settings=dict(points_per_side=48,points_per_batch=4,pred_iou_thresh=.8,stability_score_thresh=.92,crop_n_layers=0,min_mask_region_area=0)
    results=[];start=time.monotonic()
    for model_name,checkpoint in [('baseline',Path(config['arguments']['checkpoint'])),('finetuned',args.checkpoint or args.run/'checkpoints/best.pt')]:
        model=build_sam2(CONFIG,str(checkpoint),device='cuda',apply_postprocessing=False)
        for record in test:
            image=np.array(Image.open(dataset/record['image']).convert('RGB'))
            truth=np.array(Image.open(dataset/record['labels']))
            gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
            print(f"Automatic {model_name}: {record['name']}",flush=True)
            with torch.inference_mode(), torch.autocast('cuda',dtype=torch.bfloat16):
                generator=ProgressGenerator(model,**settings)
                raw=generator.generate(image)
            # Both models use the same pre-existing image-specific detection thresholds.
            contrast=record['source_preprocessing'].get('min_contrast',8)
            area=record['source_preprocessing'].get('min_area_pixels',100)
            selected=candidates_from_masks(raw,gray,min_contrast=contrast,min_area=area)
            masks=[c['mask'] for c in selected]
            labels=np.zeros_like(truth,np.uint16)
            for i,mask in enumerate(masks,1):
                assert not np.any(labels[mask]);labels[mask]=i
            Image.fromarray(labels).save(out/f"{record['name']}_{model_name}_instances.tif")
            result=dict(image=record['name'],model=model_name,raw_candidates=len(raw),min_contrast=contrast,min_area_pixels=area,**measurements(truth,masks))
            results.append(result)
            print(json.dumps(result),flush=True)
            del generator,raw,selected
        del model;torch.cuda.empty_cache()
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'fine_tuning/runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for record in test:
        image=np.array(Image.open(dataset/record['image']).convert('RGB'))
        truth=np.array(Image.open(dataset/record['labels']))
        fig,axes=plt.subplots(2,2,figsize=(16,12),layout='constrained')
        axes[0,0].imshow(image);axes[0,0].set_title('Input (brightness normalization ON)' if config.get('normalize_enabled') else 'Original (no background removal)')
        axes[0,1].imshow(overlay(image,[truth==i for i in np.unique(truth) if i]));axes[0,1].set_title(f"Reviewed reference: {record['instances']} pores")
        for ax,model_name in zip(axes[1],['baseline','finetuned']):
            labels=np.array(Image.open(out/f"{record['name']}_{model_name}_instances.tif"))
            metric=next(r for r in results if r['model']==model_name and r['image']==record['name'])
            ax.imshow(overlay(image,[labels==i for i in np.unique(labels) if i]))
            ax.set_title(f"{model_name.title()}: {metric['predicted_count']} pores | precision {metric['precision']:.3f}, recall {metric['recall']:.3f}")
        for ax in axes.flat:ax.axis('off')
        fig.suptitle(record['name']+' | Automatic analysis, identical settings',fontsize=16)
        fig.savefig(out/f"{record['name']}_comparison.png",dpi=140);plt.close(fig)
    report=dict(settings=settings,inference_precision='bfloat16 autocast',results=results,seconds=time.monotonic()-start,
                input_processing=manifest.get('input_processing'),
                notes='Held-out test, no Automate or user prompts. Identical prefilters and no background removal. IoU >= 0.5 one-to-one matching. Test not used to select the checkpoint.')
    dump(out/'results.json',report)
    rows=[]
    for r in results:
        rows.append(f"<tr><td>{r['image']}</td><td>{r['model']}</td><td>{r['predicted_count']}</td><td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['f1']:.3f}</td></tr>")
    plots=''.join(f'<h2>{r["name"]}</h2><a href="{r["name"]}_comparison.png"><img src="{r["name"]}_comparison.png" style="width:100%"></a>' for r in test)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>PoreSAM fine-tuning comparison</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:16px}td,th{padding:8px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>Fine-tuning: held-out automatic analysis</h1><p>Same settings, no background removal, no Automate. Reference labels were manually reviewed. '+str(len(test))+' held-out images; results do not establish general performance.</p><table><tr><th>Image</th><th>Model</th><th>Count</th><th>Precision</th><th>Recall</th><th>F1</th></tr>'+''.join(rows)+'</table>'+plots,encoding='utf-8')
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
