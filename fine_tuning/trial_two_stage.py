"""Five predeclared held-out models: strict, relaxed, and guarded additions."""
from datetime import datetime
import argparse
from pathlib import Path
import json
import os
import shutil
import sys
import time
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from train import ROOT,CONFIG,dump,sha,save_csv
from diagnose_candidates import TraceGenerator,decode_candidates,union
from leave_one_out import area_metrics
from conservative_candidates import supplement,DEFAULTS
sys.path.insert(0,str(ROOT/'scripts'))
from segment_first_pass import candidates_from_masks


def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,default=ROOT/'fine_tuning/runs/loo16_area_20260918_162623')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    source=args.source_run.resolve()
    source_plan=json.loads((source/'plan.json').read_text())
    records=json.loads((source/'dataset/manifest.json').read_text())['records']
    folds=[7,8,6,12,15]
    out=args.out.resolve() if args.out else ROOT/'fine_tuning/runs'/('two_stage_5_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    plan=dict(folds=folds,source_run=str(source),settings=DEFAULTS,
              protocol='Same normalized images and held-out fine-tuned weights from the source run. No additional training during this comparison. Strict masks preserved; lower-quality candidates considered with reference-free contrast, rim and overlap checks. No Automate or new box prompts.',
              limitations='Exploratory development trial; two difficult images and three regression checks selected before running. No generalization claim. Application unchanged.')
    dump(out/'plan.json',plan);(out/'source').mkdir()
    for name in ['trial_two_stage.py','conservative_candidates.py','diagnose_candidates.py','leave_one_out.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    print(f'RUN {out}',flush=True);results=[]
    for fold in folds:
        folder=out/f'fold_{fold}';folder.mkdir();start=time.monotonic()
        name=source_plan['folds'][fold-1]['validation'][0]
        record=next(r for r in records if r['name']==name)
        config=json.loads((source/f'fold_{fold}/config.json').read_text())
        assert name not in config['train_images'] and len(config['train_images'])==15 and config['validation_images']==[name]
        checkpoint=Path(source_plan['model_directory'])/f'fold_{fold}.pt'
        expected=json.loads((source/f'fold_{fold}/model_verification.json').read_text())['checkpoint_sha256']
        assert sha(checkpoint)==expected
        assert sha(source/'dataset'/record['image'])==record['image_sha256']
        image=np.array(Image.open(source/'dataset'/record['image']).convert('RGB'));gray=image[:,:,0]
        old_path=source/f'fold_{fold}/area_evaluation/finetuned_instances.tif'
        old=np.array(Image.open(old_path)) if old_path.exists() else None
        cached=ROOT/f'fine_tuning/runs/diagnosis_7_8_20260918_174329/fold_{fold}_finetuned_pool.json'
        if source==(ROOT/'fine_tuning/runs/loo16_area_20260918_162623').resolve() and cached.exists():
            pool=json.loads(cached.read_text());pool_source=str(cached)
        else:
            model=build_sam2(CONFIG,str(checkpoint),device='cuda',apply_postprocessing=False)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                generator=TraceGenerator(model,points_per_side=48,points_per_batch=4,pred_iou_thresh=.7,
                    stability_score_thresh=.85,crop_n_layers=0,min_mask_region_area=0,output_mode='uncompressed_rle')
                generator.generate(image)
            pool=generator.pool;dump(folder/'pool.json',pool);pool_source=str(folder/'pool.json')
            del model,generator;torch.cuda.empty_cache()
        strict_raw=decode_candidates(pool,.8,.92)
        min_contrast=record['source_preprocessing'].get('min_contrast',8)
        min_area=record['source_preprocessing'].get('min_area_pixels',100)
        strict=[c['mask'] for c in candidates_from_masks(strict_raw,gray,min_contrast=min_contrast,min_area=min_area)]
        strict_union=union(strict,gray.shape)
        if old is not None:assert np.array_equal(strict_union,old>0),'Strict replay must match saved inference.'
        relaxed_raw=decode_candidates(pool,.7,.85)
        relaxed=[c['mask'] for c in candidates_from_masks(relaxed_raw,gray,min_contrast=min_contrast,min_area=min_area)]
        guarded,decisions,settings=supplement(strict,relaxed_raw,gray,min_area,min_contrast)
        variants={'strict':strict,'relaxed':relaxed,'two_stage':guarded}
        # Ground truth is loaded only after inference/selection for all variants.
        assert sha(source/'dataset'/record['labels'])==record['labels_sha256']
        truth=np.array(Image.open(source/'dataset'/record['labels']))
        local=[];maps={}
        for variant,masks in variants.items():
            labels=np.zeros(gray.shape,np.uint16)
            for i,mask in enumerate(masks,1):
                assert not np.any(labels[mask]);labels[mask]=i
            maps[variant]=labels>0;Image.fromarray(labels).save(folder/f'{variant}_instances.tif')
            row=dict(fold=fold,image=name,variant=variant,**area_metrics(truth,labels));local.append(row);results.append(row)
            print(json.dumps(row),flush=True)
        dump(folder/'decisions.json',decisions)
        dump(folder/'provenance.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=expected,held_out=name,
             train_images=config['train_images'],strict_replay_exact=True if old is not None else None,pool_source=pool_source,pool_sha256=sha(Path(pool_source)),settings=settings,
             seconds=time.monotonic()-start,reference_used_for_selection=False))
        assert sha(checkpoint)==expected
        plot(folder,image,truth,maps,local)
        save_csv(out/'metrics.csv',results);write_report(out,results,plan)
        print(f'COMPLETED {fold}: {name}',flush=True)
    print(f'COMPLETE {out}/index.html',flush=True)


def plot(folder,image,truth,maps,metrics):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'fine_tuning/runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def overlay(mask):
        view=image.copy();view[mask]=(.6*view[mask]+.4*np.array([255,215,0])).astype(np.uint8);return view
    def errors(mask):
        view=(image*.55).astype(np.uint8);view[(truth>0)&~mask]=[0,220,255];view[(truth==0)&mask]=[255,0,170];return view
    fig,axes=plt.subplots(2,4,figsize=(22,9),layout='constrained')
    axes[0,0].imshow(overlay(truth>0));axes[0,0].set_title('Reference')
    axes[1,0].imshow(image);axes[1,0].set_title('Normalized input')
    for col,(variant,title) in enumerate([('strict','Current'),('relaxed','Relaxed only'),('two_stage','Two-stage')],1):
        metric=next(r for r in metrics if r['variant']==variant)
        axes[0,col].imshow(overlay(maps[variant]));axes[0,col].set_title(f'{title} | IoU {metric["iou"]:.3f}')
        view=errors(maps[variant]);Image.fromarray(view).save(folder/f'{variant}_errors.png')
        axes[1,col].imshow(view);axes[1,col].set_title(f'Missed {metric["missed_area_pct"]:.1f}% | extra {metric["extra_area_pct"]:.1f}%')
    for ax in axes.flat:ax.axis('off')
    fig.suptitle(metrics[0]['image']+' | Held-out fine-tuned model | Cyan: missed; magenta: extra',fontsize=16)
    fig.savefig(folder/'comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(16,7),layout='constrained')
    for ax,variant,title in zip(axes,['strict','two_stage'],['Current','Two-stage']):
        ax.imshow(errors(maps[variant]));ax.set_title(title);ax.axis('off')
    fig.suptitle(metrics[0]['image']+' | Cyan: reference missed; magenta: outside reference',fontsize=15)
    fig.savefig(folder/'area_errors_comparison.png',dpi=150);plt.close(fig)


def write_report(out,rows,plan):
    metrics=['iou','dice','missed_area_pct','extra_area_pct','relative_area_error_pct']
    means={variant:{key:float(np.mean([r[key] for r in rows if r['variant']==variant])) for key in metrics} for variant in ['strict','relaxed','two_stage']}
    dump(out/'summary.json',dict(completed_images=len(rows)//3,macro_image_means=means,protocol=plan['protocol'],limitations=plan['limitations']))
    table=''.join(f'<tr><td>{r["image"]}</td><td>{r["variant"]}</td><td>{r["iou"]:.3f}</td><td>{r["missed_area_pct"]:.2f}%</td><td>{r["extra_area_pct"]:.2f}%</td><td>{r["relative_area_error_pct"]:.2f}%</td></tr>' for r in rows)
    pictures=''.join(f'<h2>{r["image"]}</h2><a href="fold_{r["fold"]}/comparison.png"><img src="fold_{r["fold"]}/comparison.png"></a><p><a href="fold_{r["fold"]}/area_errors_comparison.png">Two-color error comparison</a></p>' for r in rows if r['variant']=='two_stage')
    page='<!doctype html><meta charset="utf-8"><title>Two-stage pore candidate trial</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:16px}td,th{padding:8px;border-bottom:1px solid #ddd}img{width:100%}</style><h1>Two-stage candidate trial: five held-out fine-tuned models</h1><p>'+plan['protocol']+'</p><p>Current: score .80/stability .92. Relaxed only: .70/.85 with existing pore filtering. Two-stage: retain Current masks; consider relaxed candidates with ring contrast at least max(5, image setting), inward/outward boundary support at least 60%, overlap at most 20%, and a connected retained region at least 90% of new area. Fixed settings for all five images; reference labels used only for evaluation.</p><p>IoU measures occupied-area overlap. Missed and extra percentages divide by reference pore area. Total area error is abs(predicted - reference) / reference; it can hide cancellation.</p><table><tr><th>Image</th><th>Variant</th><th>Area IoU</th><th>Missed area</th><th>Extra area</th><th>Total area error</th></tr>'+table+'</table>'+pictures+'<p>'+plan['limitations']+'</p>'
    means_table='<h2>Equal-weight image means</h2><table><tr><th>Variant</th><th>IoU</th><th>Missed area</th><th>Extra area</th><th>Total area error</th></tr>'+''.join(f'<tr><td>{v}</td><td>{m["iou"]:.3f}</td><td>{m["missed_area_pct"]:.2f}%</td><td>{m["extra_area_pct"]:.2f}%</td><td>{m["relative_area_error_pct"]:.2f}%</td></tr>' for v,m in means.items())+'</table><p>Averages can hide regressions; review each image below. Candidate additions preserve all strict pixels, so existing false positives cannot be corrected by this pass.</p><h2>Per-image results</h2>'
    page=page.replace('<table>',means_table+'<table>',1)
    (out/'index.html').write_text(page,encoding='utf-8')


if __name__=='__main__':main()
