"""Evaluate a new prepared reference against the original and all-images model."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from train import ROOT, CONFIG, Features, evaluate, dump, sha
sys.path.insert(0,str(ROOT/'scripts'))
from pore_preprocessing import prepare_adjustable


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--training-run',type=Path,help='Explicit refit run; defaults to the registered experiment.')
    parser.add_argument('--normalize',action='store_true',help='Apply the app 2nd/98th percentile brightness normalization; no background removal or blur.')
    args=parser.parse_args();dataset=args.dataset.resolve()
    if args.training_run:
        training_run=args.training_run.resolve()
        checkpoint=Path(json.loads((training_run/'final_all_images/result.json').read_text())['checkpoint'])
    else:
        registry=json.loads((ROOT/'fine_tuning/models/latest_experiment.json').read_text())
        if registry.get('eligible_for_new_evaluation') is False:
            raise SystemExit('Registered model has a data-quality notice; see fine_tuning/DATA_QUALITY_NOTICE.md. No clean all-images refit is registered. Use --training-run only for an explicit historical comparison.')
        training_run=ROOT/'fine_tuning'/registry['run']
        checkpoint=ROOT/'fine_tuning'/registry['checkpoint']
    plan=json.loads((training_run/'plan.json').read_text())
    baseline=Path(plan['base_checkpoint'])
    assert sha(baseline)==plan['base_checkpoint_sha256']
    original_data=Path(plan['dataset'])
    previous=json.loads((original_data/'manifest.json').read_text())
    manifest=json.loads((dataset/'manifest.json').read_text())
    new=manifest['records'];assert new
    for record in new:
        assert record['original_file_sha256'] not in {r['original_file_sha256'] for r in previous['records']},'This original file was used in training.'
        image=np.array(Image.open(dataset/record['image']).convert('RGB'))
        assert not any(np.array_equal(image,np.array(Image.open(original_data/r['image']).convert('RGB'))) for r in previous['records']),'Duplicate training image pixels.'
        assert sha(dataset/record['image'])==record['image_sha256']
        assert sha(dataset/record['labels'])==record['labels_sha256']
        record['split']='test'
    out=ROOT/'fine_tuning/runs'/('new_image_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True);(out/'evaluation').mkdir();(out/'source').mkdir()
    for filename in ['evaluate_new_image.py','evaluate_automatic.py','train.py']:
        shutil.copy2(Path(__file__).with_name(filename),out/'source'/filename)
    for filename in ['pore_preprocessing.py','segment_first_pass.py']:
        shutil.copy2(ROOT/'scripts'/filename,out/'source'/filename)
    reference_dataset=dataset
    if args.normalize:
        dataset=out/'normalized_dataset';dataset.mkdir()
        for record in new:
            original_image=reference_dataset/record['image']
            gray=cv2.cvtColor(np.array(Image.open(original_image).convert('RGB')),cv2.COLOR_RGB2GRAY)
            pixels,meta=prepare_adjustable(gray,dict(normalize_enabled=True,background_strength=0,blur_method='none',blur_strength=2))
            target=dataset/record['image'];target.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(pixels).save(target)
            shutil.copy2(reference_dataset/record['labels'],dataset/record['labels'])
            record['before_normalization_image_sha256']=record['image_sha256']
            record['image_sha256']=sha(target)
            record['evaluation_preprocessing']=meta
        manifest['input_processing']='App 2nd/98th percentile brightness normalization on the cropped ROI; no background removal or blur.'
        dump(dataset/'manifest.json',manifest)
    preprocessing=manifest['input_processing']
    note='New image files and pixels were absent from training. Similar PI100/magnification images were in training; independence of physical specimens is unknown. No training or threshold tuning uses this reference.'
    config=dict(arguments=dict(dataset=str(dataset),checkpoint=str(baseline)),finetuned_checkpoint=str(checkpoint),
                baseline_sha256=sha(baseline),finetuned_sha256=sha(checkpoint),training_run=str(training_run),notes=note,
                label_review=manifest['label_review'],app_model_replaced=False,
                reference_dataset=str(reference_dataset),evaluation_preprocessing=preprocessing,
                normalize_enabled=args.normalize)
    dump(out/'config.json',config);dump(out/'dataset_manifest.json',manifest)
    print(f'RUN {out}',flush=True)
    subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('evaluate_automatic.py')),'--run',str(out),'--checkpoint',str(checkpoint)],check=True)
    # Also compare prompted boundaries separately from automatic localization and filtering.
    torch.set_num_threads(4);prompted={}
    for name,weights in [('baseline',baseline),('finetuned',checkpoint)]:
        model=build_sam2(CONFIG,str(weights),device='cuda',apply_postprocessing=False)
        features=Features(model,dataset,out,shared_root=out/'prompt_features')
        prompted[name]=evaluate(model,features,new,dataset,out/'evaluation',name+'_prompted',4)
        del features,model;torch.cuda.empty_cache()
    automatic=json.loads((out/'automatic_evaluation/results.json').read_text())
    assert sha(baseline)==config['baseline_sha256'] and sha(checkpoint)==config['finetuned_sha256']
    result=dict(automatic=automatic,prompted=prompted,notes=note,app_model_replaced=False,evaluation_preprocessing=preprocessing)
    dump(out/'result.json',result)
    rows=[]
    for r in automatic['results']:
        rows.append(f"<tr><td>{r['model']}</td><td>{r['reference_count']}</td><td>{r['predicted_count']}</td><td>{r['true_positives_iou50']}</td><td>{r['false_negatives']}</td><td>{r['false_positives']}</td><td>{r['precision']:.3f}</td><td>{r['recall']:.3f}</td><td>{r['f1']:.3f}</td></tr>")
    prompt_rows=''.join(f"<tr><td>{name}</td><td>{s['point']['mean_image_iou']:.3f}</td><td>{s['box']['mean_image_iou']:.3f}</td></tr>" for name,s in prompted.items())
    area_rows=''.join(f"<tr><td>{r['model']}</td><td>{r['union_iou']:.3f}</td><td>{100*r['reference_area_fraction']:.2f}%</td><td>{100*r['predicted_area_fraction']:.2f}%</td></tr>" for r in automatic['results'])
    area_section='<h2>Total pore area (ignoring instance IDs)</h2><table><tr><th>Model</th><th>Union IoU</th><th>Reference area fraction</th><th>Predicted area fraction</th></tr>'+area_rows+'</table><p>Union IoU measures the combined pore region, so it does not penalize splitting or merging in the same way as instance F1.</p>'
    images=''.join(f'<h2>{r["name"]}</h2><a href="automatic_evaluation/{r["name"]}_comparison.png"><img style="width:100%" src="automatic_evaluation/{r["name"]}_comparison.png"></a>' for r in new)
    page='<!doctype html><meta charset="utf-8"><title>PoreSAM new-image comparison</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:16px}td,th{padding:9px;border-bottom:1px solid #ddd}table{border-collapse:collapse}</style><h1>Original SAM vs fine-tuned SAM: new image</h1><p>User-provided reference. Original checkpoint vs all-'+str(len(previous['records']))+'-images refit, selected before seeing this evaluation. Same 48-by-48 grid, area/contrast filters and CUDA precision; no background removal, extra blur or Automate.</p><table><tr><th>Model</th><th>Reference</th><th>Detected</th><th>Matched</th><th>Missed</th><th>False positives</th><th>Precision</th><th>Recall</th><th>F1</th></tr>'+''.join(rows)+'</table><p>Matches require IoU >= 0.5 and are one-to-one. A merged prediction cannot count as two correct pores.</p><h2>Prompted boundary comparison</h2><table><tr><th>Model</th><th>Point IoU</th><th>Box IoU</th></tr>'+prompt_rows+'</table><p>Prompted metrics use identical reference-derived points/boxes, so they measure boundaries when the location is already provided.</p>'+images+'<p>'+note+'</p>'
    page=page.replace('<h2>Prompted boundary comparison</h2>',area_section+'<h2>Prompted boundary comparison</h2>')
    page=page.replace('<table>',f'<p>Evaluation input: {preprocessing} Model weights are unchanged; no retraining.</p><table>',1)
    (out/'index.html').write_text(page,encoding='utf-8')
    subpage=out/'automatic_evaluation/index.html'
    subpage.write_text(subpage.read_text(encoding='utf-8').replace('Reference labels were manually reviewed.','User-provided edited labels are used as reference.'),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True);print(out/'index.html',flush=True)


if __name__=='__main__':main()
