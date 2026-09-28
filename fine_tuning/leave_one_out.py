"""16 image-wise 15/1 experiments with normalized inputs and area-first reports."""
import argparse
import copy
from datetime import datetime
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from train import ROOT, CONFIG, dump, sha, save_csv

sys.path.insert(0, str(ROOT/'scripts'))
from pore_preprocessing import prepare_adjustable
from segment_first_pass import ProgressGenerator, candidates_from_masks


METRICS = {
    'iou': ('Area IoU', 'up', 'Intersection / union of predicted and reference pore pixels.'),
    'dice': ('Area Dice', 'up', '2 x intersection / (predicted area + reference area).'),
    'area_precision': ('Area precision', 'up', 'Correctly filled pore pixels / all predicted pore pixels.'),
    'area_recall': ('Area recall', 'up', 'Correctly filled pore pixels / all reference pore pixels.'),
    'missed_area_pct': ('Missed area (%)', 'down', 'Missed pore pixels / reference pore area x 100.'),
    'extra_area_pct': ('Extra area (%)', 'down', 'Predicted pixels outside reference / reference pore area x 100.'),
    'relative_area_error_pct': ('Total area error (%)', 'down', 'Absolute difference of total areas / reference pore area x 100. Under/over-filling can cancel.'),
    'area_fraction_error_pp': ('Area fraction error (pp)', 'down', 'Absolute difference in pore fraction of the image, in percentage points.'),
}


def area_metrics(reference, prediction):
    truth=np.asarray(reference)>0; pred=np.asarray(prediction)>0
    tp=int(np.count_nonzero(truth & pred)); fn=int(np.count_nonzero(truth & ~pred))
    fp=int(np.count_nonzero(~truth & pred)); a=tp+fn; b=tp+fp
    assert a>0, 'Reference pore area must be nonzero.'
    return dict(intersection_pixels=tp, missed_pixels=fn, extra_pixels=fp,
                reference_pixels=a, predicted_pixels=b, image_pixels=int(truth.size),
                iou=tp/(tp+fp+fn), dice=2*tp/(a+b), area_precision=tp/b if b else 0.,
                area_recall=tp/a, missed_area_pct=100*fn/a, extra_area_pct=100*fp/a,
                relative_area_error_pct=100*abs(b-a)/a, signed_area_error_pct=100*(b-a)/a,
                reference_area_fraction_pct=100*a/truth.size, predicted_area_fraction_pct=100*b/truth.size,
                area_fraction_error_pp=100*abs(b-a)/truth.size)


def prepare(out):
    sources=[ROOT/'fine_tuning/datasets/260917_train15_v1', ROOT/'fine_tuning/datasets/external_PI100_5kx-5_20260918']
    records=[]; seen=set(); originals=set(); dataset=out/'dataset'; dataset.mkdir()
    for source in sources:
        manifest=json.loads((source/'manifest.json').read_text())
        for original in manifest['records']:
            r=copy.deepcopy(original)
            assert r['image_sha256'] not in seen and r['original_file_sha256'] not in originals
            seen.add(r['image_sha256']); originals.add(r['original_file_sha256'])
            assert sha(source/r['image'])==r['image_sha256'] and sha(source/r['labels'])==r['labels_sha256']
            rgb=np.array(Image.open(source/r['image']).convert('RGB'))
            gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
            pixels,meta=prepare_adjustable(gray,dict(normalize_enabled=True,background_strength=0,blur_method='none',blur_strength=2))
            folder=dataset/r['name'];folder.mkdir()
            Image.fromarray(pixels).save(dataset/r['image'])
            Image.fromarray(rgb).save(folder/'original.png')
            shutil.copy2(source/r['labels'],dataset/r['labels'])
            shutil.copy2(source/r['name']/'source_masks.npz',folder/'source_masks.npz')
            r.update(split='unassigned',prepared_source=str(source),evaluation_preprocessing=meta,
                     before_normalization_image_sha256=r['image_sha256'],image_sha256=sha(dataset/r['image']))
            records.append(r)
    assert len(records)==16 and len({r['name'] for r in records})==16
    records.sort(key=lambda r:r['name'])
    assert len({r['image_sha256'] for r in records})==16, 'Duplicate normalized input pixels.'
    dump(dataset/'manifest.json', dict(version=3,records=records,
         input_processing='App brightness normalization (2nd/98th percentile) on ROI for both training and evaluation. No background removal or blur.',
         label_review='Initial 14 were confirmed manually reviewed; later two are user-supplied edited references. Source provenance retained per record.',
         split_notes='One complete image held out in each fold; related acquisitions are NOT grouped. Exploratory image-wise leave-one-out.'))
    return dataset,records


def verify_model(plan, number):
    folder=Path(plan['run'])/f'fold_{number}'
    config=json.loads((folder/'config.json').read_text())
    held=plan['folds'][number-1]['validation']
    assert config['validation_images']==held and len(config['train_images'])==15
    assert not set(held)&set(config['train_images'])
    base=torch.load(plan['base_checkpoint'],map_location='cpu',weights_only=True)['model']
    checkpoint=Path(plan['model_directory'])/f'fold_{number}.pt'
    obj=torch.load(checkpoint,map_location='cpu',weights_only=True);tuned=obj['model']
    assert obj['finetuning']['train_images']==config['train_images']
    assert base.keys()==tuned.keys()
    changed=[];allowed=set(config['trainable_names'])
    for name,tensor in tuned.items():
        assert tensor.shape==base[name].shape and torch.isfinite(tensor).all(),name
        if not torch.equal(tensor,base[name]):
            assert name in allowed,name
            changed.append(name)
    assert changed
    dump(folder/'model_verification.json',dict(checkpoint_sha256=sha(checkpoint),changed_tensor_count=len(changed),
         frozen_tensors_unchanged=True,held_out=held,train_images=config['train_images']))


def evaluate_fold(plan_path, number):
    plan=json.loads(plan_path.read_text());torch.set_num_threads(4)
    assert sha(Path(plan['dataset'])/'manifest.json')==plan['dataset_manifest_sha256']
    assert sha(plan['base_checkpoint'])==plan['base_checkpoint_sha256']
    verify_model(plan,number)
    dataset=Path(plan['dataset']);folder=plan_path.parent/f'fold_{number}'
    out=folder/'area_evaluation';out.mkdir()
    records=json.loads((dataset/'manifest.json').read_text())['records']
    r=next(r for r in records if r['name']==plan['folds'][number-1]['validation'][0])
    image=np.array(Image.open(dataset/r['image']).convert('RGB'))
    truth=np.array(Image.open(dataset/r['labels']))
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    settings=plan['automatic_settings'];results={};label_maps={};start=time.monotonic()
    for name,checkpoint in [('baseline',Path(plan['base_checkpoint'])),('finetuned',Path(plan['model_directory'])/f'fold_{number}.pt')]:
        model=build_sam2(CONFIG,str(checkpoint),device='cuda',apply_postprocessing=False)
        print(f'EVALUATE {number}/16 {r["name"]} {name}',flush=True)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            generator=ProgressGenerator(model,**settings)
            raw=generator.generate(image)
        selected=candidates_from_masks(raw,gray,min_contrast=r['source_preprocessing'].get('min_contrast',8),
                                       min_area=r['source_preprocessing'].get('min_area_pixels',100))
        labels=np.zeros_like(truth,np.uint16)
        for i,c in enumerate(selected,1):
            assert not np.any(labels[c['mask']]);labels[c['mask']]=i
        Image.fromarray(labels).save(out/f'{name}_instances.tif')
        label_maps[name]=labels
        results[name]=area_metrics(truth,labels)
        # Area coverage for every reference pore, without selecting only easy matches.
        coverage=[]
        for i in np.unique(truth):
            if not i:continue
            mask=truth==i;total=int(mask.sum());covered=int(np.count_nonzero(mask & (labels>0)))
            coverage.append(dict(reference_id=int(i),original_id=r['label_to_original_id'][str(i)],
                                 reference_area_pixels=total,covered_pixels=covered,missed_pixels=total-covered,
                                 area_recall=covered/total))
        save_csv(out/f'{name}_reference_area_coverage.csv',coverage)
        print(json.dumps(dict(fold=number,model=name,**results[name])),flush=True)
        del model,generator,raw,selected;torch.cuda.empty_cache()
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'fine_tuning/runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    def tint(mask):
        view=image.copy();view[mask]=np.rint(.6*view[mask]+.4*np.array([255,215,0])).astype(np.uint8)
        return view
    def errors(labels):
        view=(image*.55).astype(np.uint8)
        view[(truth>0)&(labels==0)]=[0,220,255]
        view[(truth==0)&(labels>0)]=[255,0,170]
        return view
    fig,axes=plt.subplots(2,3,figsize=(19,10),layout='constrained')
    axes[0,0].imshow(tint(truth>0));axes[0,0].set_title(f'Reference | pore area {100*np.mean(truth>0):.2f}%')
    axes[1,0].imshow(image);axes[1,0].set_title('Normalized input')
    for col,name in [(1,'baseline'),(2,'finetuned')]:
        m=results[name];axes[0,col].imshow(tint(label_maps[name]>0))
        axes[0,col].set_title(f'{name.title()} | IoU {m["iou"]:.3f}, Dice {m["dice"]:.3f}\nPore area {m["predicted_area_fraction_pct"]:.2f}% | area error {m["relative_area_error_pct"]:.2f}%')
        axes[1,col].imshow(errors(label_maps[name]))
        axes[1,col].set_title(f'Missed area {m["missed_area_pct"]:.2f}% | extra area {m["extra_area_pct"]:.2f}%\nCyan: missed; magenta: extra (both / reference area)')
    for ax in axes.flat:ax.axis('off')
    fig.suptitle(f'{r["name"]} | 15 train / 1 held out | Area comparison',fontsize=17)
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    # Standalone error maps retain native resolution; the combined view has a legend.
    for name in ['baseline','finetuned']:
        Image.fromarray(errors(label_maps[name])).save(out/f'{name}_area_errors.png')
    fig,axes=plt.subplots(1,2,figsize=(16,7),layout='constrained')
    for ax,name in zip(axes,['baseline','finetuned']):
        ax.imshow(errors(label_maps[name]));ax.axis('off')
        ax.set_title('Original SAM' if name=='baseline' else 'Fine-tuned SAM',fontsize=16)
    fig.suptitle(r['name']+' | Differences from reference\nCyan: reference pore missed | Magenta: predicted pore outside reference',fontsize=16)
    fig.savefig(out/'area_errors_comparison.png',dpi=150);plt.close(fig)
    result=dict(image=r['name'],fold=number,results=results,seconds=time.monotonic()-start,
                settings=settings,normalization=r['evaluation_preprocessing'],
                min_contrast=r['source_preprocessing'].get('min_contrast',8),min_area_pixels=r['source_preprocessing'].get('min_area_pixels',100))
    dump(out/'results.json',result)
    rows=''.join(f'<tr><td>{title}</td><td>{results["baseline"][key]:.4f}</td><td>{results["finetuned"][key]:.4f}</td></tr>' for key,(title,_,_) in METRICS.items())
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><style>body{font:16px system-ui;margin:32px}td,th{padding:8px}img{width:100%}</style>'
         +f'<h1>{html.escape(r["name"])}: area comparison</h1><p>Normalized training and evaluation. No Automate. Cyan: missed area; magenta: extra area.</p>'
         +'<table><tr><th>Metric</th><th>Original SAM</th><th>Fine-tuned SAM</th></tr>'+rows+'</table><a href="comparison.png"><img src="comparison.png"></a><h2>Differences from reference</h2><a href="area_errors_comparison.png"><img src="area_errors_comparison.png"></a><p>Native resolution: <a href="baseline_area_errors.png">Original SAM errors</a> | <a href="finetuned_area_errors.png">Fine-tuned SAM errors</a></p><p><a href="../../index.html">All images and metric definitions</a></p>',encoding='utf-8')


def report(out,plan):
    completed=[];csv=[]
    for fold in plan['folds']:
        path=out/f'fold_{fold["fold"]}'/'area_evaluation/results.json'
        if path.exists():
            r=json.loads(path.read_text());completed.append(r)
            csv.extend(dict(image=r['image'],fold=r['fold'],model=name,**m) for name,m in r['results'].items())
    summary={}
    for name in ['baseline','finetuned']:
        subset=[r for r in csv if r['model']==name]
        if subset:
            summary[name]={key:float(np.mean([r[key] for r in subset])) for key in METRICS}
    if csv:save_csv(out/'area_metrics.csv',csv)
    wins={key:sum(r['results']['finetuned'][key]>r['results']['baseline'][key] if direction=='up' else r['results']['finetuned'][key]<r['results']['baseline'][key] for r in completed) for key,(_,direction,_) in METRICS.items()}
    dump(out/'summary.json',dict(completed_images=len(completed),total_images=16,macro_image_means=summary,finetuned_wins=wins,
         metric_definitions=METRICS,protocol=plan['protocol'],limitations=plan['limitations']))
    table=''.join(f'<tr><td>{title} ({direction})</td><td>{summary["baseline"][key]:.4f}</td><td>{summary["finetuned"][key]:.4f}</td><td>{wins[key]}/{len(completed)}</td></tr>' for key,(title,direction,_) in METRICS.items()) if summary else ''
    links=''.join(f'<tr><td><a href="fold_{r["fold"]}/area_evaluation/index.html">{html.escape(r["image"])}</a></td><td>{r["results"]["baseline"]["iou"]:.3f}</td><td>{r["results"]["finetuned"]["iou"]:.3f}</td><td>{r["results"]["baseline"]["relative_area_error_pct"]:.2f}%</td><td>{r["results"]["finetuned"]["relative_area_error_pct"]:.2f}%</td><td><a href="fold_{r["fold"]}/area_evaluation/comparison.png">Image</a></td></tr>' for r in completed)
    definitions=''.join(f'<li><b>{title}:</b> {definition}</li>' for title,_,definition in METRICS.values())
    page=f'<!doctype html><meta charset="utf-8"><title>PoreSAM area evaluation</title><style>body{{font:16px system-ui;max-width:1300px;margin:32px auto;padding:16px;color:#17212f}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:left}}table{{border-collapse:collapse;width:100%}}a{{color:#1265b0}}</style><h1>PoreSAM: 16-image leave-one-out area evaluation</h1><p>{len(completed)}/16 images completed. Each model starts from original SAM, trains on 15 images for 300 fixed updates and tests on the excluded image. Brightness normalization for training and evaluation; no background removal, blur or Automate.</p><h2>Mean per-image metrics</h2><p>Each image has equal weight. Area precision/recall refer to pixels, not numbers of pores.</p><table><tr><th>Metric</th><th>Original SAM</th><th>Fine-tuned SAM</th><th>Fine-tuned wins</th></tr>{table}</table><h2>Image comparisons</h2><table><tr><th>Image</th><th>Original IoU</th><th>Fine-tuned IoU</th><th>Original area error</th><th>Fine-tuned area error</th><th>Comparison</th></tr>{links}</table><h2>Metric definitions</h2><ul>{definitions}</ul><p>Metrics use the union of all pore masks. They measure occupied area and do not penalize a split/merge if the occupied pixels stay identical. Per-reference coverage CSVs include every reference pore, but do not estimate individual predicted pore size or diameter accuracy.</p><h2>Scope</h2><p>{plan["limitations"]}</p><p>No best fold is selected as a production model. All 16 checkpoints are separate; the app model and experimental registry remain unchanged. <a href="area_metrics.csv">Metrics CSV</a></p>'
    (out/'index.html').write_text(page,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--resume',type=Path)
    p.add_argument('--evaluate-plan',type=Path);p.add_argument('--fold',type=int);args=p.parse_args()
    if args.evaluate_plan:evaluate_fold(args.evaluate_plan,args.fold);return
    if args.resume:out=args.resume.resolve().parent;plan=json.loads(args.resume.read_text())
    else:
        name='loo16_area_'+datetime.now().strftime('%Y%m%d_%H%M%S')
        out=ROOT/'fine_tuning/runs'/name;out.mkdir(parents=True)
        dataset,records=prepare(out);base=ROOT/'checkpoints/sam2.1_hiera_small.pt'
        models=ROOT/'fine_tuning/models'/name;models.mkdir(parents=True)
        plan=dict(run=str(out),dataset=str(dataset),dataset_manifest_sha256=sha(dataset/'manifest.json'),
                  base_checkpoint=str(base),base_checkpoint_sha256=sha(base),model_directory=str(models),
                  seed=260917,steps=300,batch=4,lr=1e-5,prompted_evaluation=False,
                  folds=[dict(fold=i+1,validation=[r['name']]) for i,r in enumerate(records)],
                  automatic_settings=dict(points_per_side=48,points_per_batch=4,pred_iou_thresh=.8,stability_score_thresh=.92,crop_n_layers=0,min_mask_region_area=0),
                  protocol='Image-wise leave-one-out: 16 independently initialized 15/1 models. Fixed training budget and analysis thresholds. Normalized train and test inputs. Area-first automatic evaluation.',
                  limitations='Exploratory evaluation on an already examined collection. Similar captures from the same material/magnification may be in training and test; this is not specimen-grouped validation or independent external testing. The held-out image never enters optimization or checkpoint selection. Normalization is now applied during training too, so differences from earlier raw-input training are not attributable solely to fold membership.')
        dump(out/'plan.json',plan);(out/'source').mkdir()
        for script in Path(__file__).parent.glob('*.py'):shutil.copy2(script,out/'source'/script.name)
        for name in ['pore_preprocessing.py','segment_first_pass.py']:shutil.copy2(ROOT/'scripts'/name,out/'source'/name)
    print(f'RUN {out}',flush=True);report(out,plan)
    for fold in plan['folds']:
        number=fold['fold'];folder=out/f'fold_{number}'
        if not (folder/'result.json').exists():
            subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('cross_validate.py')),'--worker-plan',str(out/'plan.json'),'--fold',str(number)],check=True)
        if not (folder/'area_evaluation/index.html').exists():
            subprocess.run([sys.executable,'-u',__file__,'--evaluate-plan',str(out/'plan.json'),'--fold',str(number)],check=True)
        report(out,plan)
        print(f'COMPLETED {number}/16: {fold["validation"][0]}',flush=True)
    assert sha(plan['base_checkpoint'])==plan['base_checkpoint_sha256']
    print(f'COMPLETE {out}/index.html',flush=True)


if __name__=='__main__':main()
