"""Fixed relaxed-only vs two-stage area comparison with separately timed paths."""
import argparse
from contextlib import nullcontext
import csv
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from train import ROOT,CONFIG,dump,sha,save_csv
from diagnose_candidates import TraceGenerator,decode_candidates
from leave_one_out import area_metrics
from conservative_candidates import supplement,DEFAULTS
sys.path.insert(0,str(ROOT/'scripts'))
from segment_first_pass import candidates_from_masks


from pore_selection import label_map, select


def sync(device):
    if device=='cuda':torch.cuda.synchronize()


def plot(folder,image,truth,maps,metrics):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'fine_tuning/runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def overlay(mask):
        result=image.copy();result[mask]=(.6*result[mask]+.4*np.array([255,215,0])).astype(np.uint8)
        return result
    def errors(mask):
        result=(image*.55).astype(np.uint8)
        result[(truth>0)&~mask]=[0,220,255];result[(truth==0)&mask]=[255,0,170]
        return result
    fig,axes=plt.subplots(2,3,figsize=(18,9),layout='constrained')
    axes[0,0].imshow(overlay(truth>0));axes[0,0].set_title('Reference')
    axes[1,0].imshow(image);axes[1,0].set_title('Normalized SEM input (footer excluded)')
    for col,variant in enumerate(['relaxed','two_stage'],1):
        row=next(r for r in metrics if r['variant']==variant)
        title='Relaxed only' if variant=='relaxed' else 'Two-stage'
        axes[0,col].imshow(overlay(maps[variant]));axes[0,col].set_title(f'{title} | area IoU {row["iou"]:.3f}')
        axes[1,col].imshow(errors(maps[variant]));axes[1,col].set_title(f'Missed {row["missed_area_pct"]:.1f}% | extra {row["extra_area_pct"]:.1f}%')
    for ax in axes.flat:ax.axis('off')
    fig.suptitle(metrics[0]['image']+' | Cyan: missed reference; magenta: outside reference')
    fig.savefig(folder/'comparison.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(16,7),layout='constrained')
    for ax,v,title in zip(axes,['relaxed','two_stage'],['Relaxed only','Two-stage']):
        ax.imshow(errors(maps[v]));ax.set_title(title);ax.axis('off')
    fig.suptitle(metrics[0]['image']+' | Cyan: missed; magenta: extra')
    fig.savefig(folder/'area_errors_comparison.png',dpi=130);plt.close(fig)
    for v in maps:
        Image.fromarray(errors(maps[v])).save(folder/f'{v}_errors.png')


def report(out,plan):
    rows=[];timings=[]
    for fold in plan['folds']:
        file=out/f'fold_{fold}/result.json'
        if not file.exists():continue
        result=json.loads(file.read_text());rows.extend(result['metrics']);timings.append(result['timing'])
    if not rows:return
    metrics=['iou','dice','missed_area_pct','extra_area_pct','relative_area_error_pct','area_fraction_error_pp']
    means={v:{k:float(np.mean([r[k] for r in rows if r['variant']==v])) for k in metrics} for v in ['strict','relaxed','two_stage']}
    groups={}
    for group in ['PI100','PI300','PI35','top']:
        subset=[r for r in rows if r['image'].startswith(group+'_')]
        if subset:groups[group]={v:{k:float(np.mean([r[k] for r in subset if r['variant']==v])) for k in metrics} for v in ['relaxed','two_stage']}
    wins={k:dict(relaxed=0,two_stage=0,ties=0) for k in metrics}
    for fold in sorted({r['fold'] for r in rows}):
        pair={r['variant']:r for r in rows if r['fold']==fold}
        for k in metrics:
            delta=pair['two_stage'][k]-pair['relaxed'][k]
            winner='ties' if abs(delta)<1e-10 else ('two_stage' if (delta>0)==(k in ['iou','dice']) else 'relaxed')
            wins[k][winner]+=1
    fields=['sam_generation_seconds','relaxed_post_seconds','two_stage_post_seconds','relaxed_analysis_seconds','two_stage_analysis_seconds','extra_two_stage_seconds']
    time_summary={k:dict(mean=float(np.mean([t[k] for t in timings])),median=float(np.median([t[k] for t in timings])),min=min(t[k] for t in timings),max=max(t[k] for t in timings)) for k in fields}
    summary=dict(completed_images=len(timings),expected_images=len(plan['folds']),macro_image_means=means,
        paired_image_wins=wins,image_group_means=groups,timing=time_summary,device=plan['device'],protocol=plan,app_model_replaced=False)
    dump(out/'summary.json',summary);save_csv(out/'metrics.csv',rows)
    save_csv(out/'timing.csv',[{k:v for k,v in t.items() if not isinstance(v,(list,dict))} for t in timings])
    title=f'Relaxed only vs Two-stage: {len(timings)}/{len(plan["folds"])} held-out images ({plan["device"]})'
    html=f'<!doctype html><meta charset="utf-8"><title>{title}</title><style>body{{font:16px system-ui;max-width:1450px;margin:32px auto;padding:16px}}td,th{{padding:8px;border-bottom:1px solid #ddd}}img{{width:100%}}a{{color:#1262a4}}</style><h1>{title}</h1>'
    html+='<p>16 footer-corrected images; independent 15-train/1-held-out fine-tuned models. Area IoU is the primary measure. SAM quality thresholds and added boundary rules are fixed across images. Existing per-image minimum-area and contrast settings are preserved identically for both methods; reference masks never enter candidate selection. No Automate, model replacement, or additional test-dependent threshold tuning.</p>'
    html+='<p>Relaxed only: .70 score/.85 stability plus existing pore filters. Two-stage: preserve strict .80/.92 detections, then add rim-supported candidates from the same relaxed pool. Strict is shown as a diagnostic reference. All methods use the same image and model.</p>'
    html+='<h2>Equal-weight image means</h2><table><tr><th>Method</th><th>Area IoU</th><th>Dice</th><th>Missed %</th><th>Extra %</th><th>Total-area error %</th></tr>'
    for v,m in means.items():html+=f'<tr><td>{v}</td><td>{m["iou"]:.4f}</td><td>{m["dice"]:.4f}</td><td>{m["missed_area_pct"]:.2f}</td><td>{m["extra_area_pct"]:.2f}</td><td>{m["relative_area_error_pct"]:.2f}</td></tr>'
    html+='</table><p>Missed, extra and total-area error divide by reference pore area. Total-area error can cancel simultaneous missing and extra pixels. These metrics do not score pore count.</p>'
    html+=f'<p>Area-IoU wins: Relaxed {wins["iou"]["relaxed"]}; Two-stage {wins["iou"]["two_stage"]}; ties {wins["iou"]["ties"]}.</p>'
    html+='<h2>Image-family means</h2><table><tr><th>Family</th><th>Relaxed IoU</th><th>Two-stage IoU</th><th>Relaxed area error %</th><th>Two-stage area error %</th></tr>'
    for g,m in groups.items():html+=f'<tr><td>{g}</td><td>{m["relaxed"]["iou"]:.4f}</td><td>{m["two_stage"]["iou"]:.4f}</td><td>{m["relaxed"]["relative_area_error_pct"]:.2f}</td><td>{m["two_stage"]["relative_area_error_pct"]:.2f}</td></tr>'
    html+='</table><p>Families describe filenames; they are not independent specimen groups. Family summaries are descriptive, not a rule for selecting a method using the test result.</p>'
    html+='<h2>Measured analysis time</h2><p>'+plan['timing_notes']+'</p><table><tr><th>Stage (seconds/image)</th><th>Mean</th><th>Median</th><th>Min</th><th>Max</th></tr>'
    for k,m in time_summary.items():html+=f'<tr><td>{k}</td><td>{m["mean"]:.2f}</td><td>{m["median"]:.2f}</td><td>{m["min"]:.2f}</td><td>{m["max"]:.2f}</td></tr>'
    html+='</table><p>Timings are pipeline measurements on this machine, not a cross-machine guarantee. GPU runs do not estimate CPU inference time. Check the separate CPU report when present.</p><h2>Per-image comparison</h2><table><tr><th>Image</th><th>Relaxed IoU</th><th>Two-stage IoU</th><th>Difference</th><th>Relaxed seconds</th><th>Two-stage seconds</th></tr>'
    for t in timings:
        pair={r['variant']:r for r in rows if r['fold']==t['fold']}
        a=pair['relaxed']['iou'];b=pair['two_stage']['iou']
        html+=f'<tr><td><a href="fold_{t["fold"]}/comparison.png">{t["image"]}</a></td><td>{a:.4f}</td><td>{b:.4f}</td><td>{b-a:+.4f}</td><td>{t["relaxed_analysis_seconds"]:.2f}</td><td>{t["two_stage_analysis_seconds"]:.2f}</td></tr>'
    html+='</table><p>'+plan['limitations']+'</p>'
    for t in timings:html+=f'<h2>{t["image"]}</h2><a href="fold_{t["fold"]}/comparison.png"><img loading="lazy" src="fold_{t["fold"]}/comparison.png"></a><p><a href="fold_{t["fold"]}/area_errors_comparison.png">Missed / extra area comparison</a></p>'
    (out/'index.html').write_text(html,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--folds',type=int,nargs='+',default=list(range(1,17)))
    parser.add_argument('--repeats',type=int,default=1)
    args=parser.parse_args();source=args.run.resolve();device=args.device
    torch.set_num_threads(8 if device=='cpu' else 4);cv2.setNumThreads(4)
    training=json.loads((source/'plan.json').read_text())
    data=Path(training['dataset']);assert sha(data/'manifest.json')==training['dataset_manifest_sha256']
    records=json.loads((data/'manifest.json').read_text())['records']
    out=source/('methods_cpu' if device=='cpu' else 'methods_gpu');out.mkdir(exist_ok=True)
    plan=dict(source_run=str(source),folds=args.folds,device=device,points_per_side=48,
        points_per_batch=8 if device=='cpu' else 4,torch_threads=torch.get_num_threads(),opencv_threads=cv2.getNumThreads(),
        dtype='float32' if device=='cpu' else 'bfloat16',post_repeats=args.repeats,settings=DEFAULTS,
        torch=str(torch.__version__),processor=platform.processor(),gpu=torch.cuda.get_device_name() if device=='cuda' else None,
        timing_notes='Shared SAM candidate generation measured once per image, including encoder and automatic-mask output conversion. Both method paths timed independently, alternating order by fold/repetition; Two-stage includes strict selection, relaxed decoding and extra candidate checks. Post time is median across '+str(args.repeats)+' repetition(s). Per-method analysis time = shared SAM time + its post time. Excludes model loading, training, serialization, metrics and plots. No concurrent training or other experiment benchmarks.',
        limitations='Exploratory previously examined image-wise LOO; related acquisitions may be in training. CPU subset is preselected folds 1, 7, 15 (dense PI, difficult PI300, complex top) rather than all 16. CPU float32 and GPU bfloat16 may yield different masks; their metrics are kept separate.')
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==plan,'Resume configuration mismatch'
    else:dump(out/'plan.json',plan)
    (out/'source').mkdir(exist_ok=True)
    for name in ['compare_methods16.py','diagnose_candidates.py','conservative_candidates.py']:
        shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    print(f'EVALUATION {out}',flush=True)
    for fold in args.folds:
        folder=out/f'fold_{fold}';folder.mkdir(exist_ok=True)
        if (folder/'result.json').exists():report(out,plan);continue
        name=training['folds'][fold-1]['validation'][0]
        r=next(r for r in records if r['name']==name)
        config=json.loads((source/f'fold_{fold}/config.json').read_text())
        assert len(config['train_images'])==15 and name not in config['train_images'] and config['validation_images']==[name]
        checkpoint=Path(training['model_directory'])/f'fold_{fold}.pt'
        expected=json.loads((source/f'fold_{fold}/model_verification.json').read_text())['checkpoint_sha256']
        assert sha(checkpoint)==expected and sha(data/r['image'])==r['image_sha256']
        image=np.array(Image.open(data/r['image']).convert('RGB'));gray=image[:,:,0]
        assert gray.shape==(768,1024)
        print(f'GENERATE fold {fold}: {name} ({device})',flush=True)
        model=build_sam2(CONFIG,str(checkpoint),device=device,apply_postprocessing=False)
        with torch.inference_mode(),(torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else nullcontext()):
            generator=TraceGenerator(model,points_per_side=48,points_per_batch=plan['points_per_batch'],
                pred_iou_thresh=.7,stability_score_thresh=.85,crop_n_layers=0,min_mask_region_area=0,output_mode='uncompressed_rle')
            sync(device);start=time.perf_counter();generator.generate(image);sync(device)
            sam_seconds=time.perf_counter()-start;pool=generator.pool
        dump(folder/'pool.json',pool)
        del model,generator;gc.collect()
        if device=='cuda':torch.cuda.empty_cache()
        times={v:[] for v in ['relaxed','two_stage']};outputs={};fingerprints={}
        for repeat in range(args.repeats):
            order=['relaxed','two_stage'] if (fold+repeat)%2 else ['two_stage','relaxed']
            for variant in order:
                print(f'SELECT fold {fold}: {variant}, repetition {repeat+1}',flush=True)
                sync(device);start=time.perf_counter()
                result=select(pool,gray,r['source_preprocessing'].get('min_area_pixels',100),r['source_preprocessing'].get('min_contrast',8),device,variant)
                sync(device);times[variant].append(time.perf_counter()-start)
                labels=label_map(result['masks'],gray.shape)
                if variant in fingerprints:assert np.array_equal(labels,outputs[variant]),'Nondeterministic repeated selection'
                fingerprints[variant]=True;outputs[variant]=labels
                if variant=='two_stage':
                    outputs['strict']=label_map(result['strict'],gray.shape)
                    dump(folder/'decisions.json',result['decisions'])
                del result
        # Test labels are accessed only after both predictions are finalized.
        assert sha(data/r['labels'])==r['labels_sha256']
        truth=np.array(Image.open(data/r['labels']))
        assert np.array_equal(outputs['two_stage'][outputs['strict']>0],outputs['strict'][outputs['strict']>0])
        local=[]
        for v in ['strict','relaxed','two_stage']:
            Image.fromarray(outputs[v]).save(folder/f'{v}_instances.tif')
            local.append(dict(fold=fold,image=name,variant=v,**area_metrics(truth,outputs[v])))
        timing=dict(fold=fold,image=name,sam_generation_seconds=sam_seconds,
            relaxed_post_seconds=float(np.median(times['relaxed'])),two_stage_post_seconds=float(np.median(times['two_stage'])),raw_repetitions=times)
        for v in ['relaxed','two_stage']:timing[v+'_analysis_seconds']=sam_seconds+timing[v+'_post_seconds']
        timing['extra_two_stage_seconds']=timing['two_stage_post_seconds']-timing['relaxed_post_seconds']
        assert sha(checkpoint)==expected
        plot(folder,image,truth,{v:m>0 for v,m in outputs.items()},local)
        dump(folder/'result.json',dict(metrics=local,timing=timing,checkpoint=str(checkpoint),checkpoint_sha256=expected,
             held_out=name,train_images=config['train_images'],reference_used_for_selection=False,pool_sha256=sha(folder/'pool.json'),
             image_settings=dict(min_area=r['source_preprocessing'].get('min_area_pixels',100),min_contrast=r['source_preprocessing'].get('min_contrast',8))))
        report(out,plan)
        print(f'COMPLETE fold {fold}: '+json.dumps(dict(iou={r['variant']:r['iou'] for r in local},timing=timing)),flush=True)
        del pool,outputs;gc.collect()
    print(f'COMPLETE {out}/index.html',flush=True)


if __name__=='__main__':main()
