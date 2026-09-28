"""Remove confirmed SEM information panels, then retrain the same five 15/1 models."""
import copy
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
from train import ROOT,dump,sha
from leave_one_out import verify_model
sys.path.insert(0,str(ROOT/'scripts'))
from pore_preprocessing import prepare_adjustable


def write_clean_report(out):
    page='''<!doctype html><meta charset="utf-8"><title>Footer-corrected retraining</title>
<style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:20px}img{max-width:100%}li{margin:12px 0}</style>
<h1>Footer-corrected retraining</h1>
<p>16 clean images, 2,018 reference pores. Five 15/1 models retrained independently from original SAM: folds 7, 8, 6, 12 and 15. Each test image was excluded from its model's training. The other eleven historical folds have not been retrained.</p>
<p>Three footer-only masks removed: top_2_B IDs 135 and 136; top_3_A ID 89. All three top-image information panels excluded before brightness normalization. Valid SEM-region label pixels are unchanged. The app model is unchanged.</p>
<p><a href="two_stage_evaluation/index.html">Open five-image area comparison report and error maps</a></p>
<p>This report compares candidate-selection methods using the corrected fine-tuned models. It is not a baseline-versus-fine-tuned comparison. Two-stage additions improve mean IoU, but still over-segment top_2_B; total-area error does not improve on average. Footer correction alone does not resolve that detection problem.</p>
<ul><li><a href="footer_audit.json">Exact crop and removed-label audit</a></li>
<li><a href="dataset_verification.json">Dataset pixel verification</a></li>
<li><a href="verification.json">Checkpoint, metrics and error-map verification</a></li></ul>
<h2>Removed labels</h2><p>Magenta: removed footer-only masks. Cyan line: verified SEM image boundary at y=768.</p>
'''
    for name in ['top_2_B_1kx-1','top_3_A_10kx-1']:
        page+=f'<h3>{name}</h3><a href="{name}_footer_audit.png"><img src="{name}_footer_audit.png"></a>'
    (out/'index.html').write_text(page,encoding='utf-8')


def main():
    torch.set_num_threads(4)
    previous=ROOT/'fine_tuning/runs/loo16_area_20260918_162623'
    original=json.loads((previous/'dataset/manifest.json').read_text())
    old_plan=json.loads((previous/'plan.json').read_text())
    name='footer_clean_5_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT/'fine_tuning/runs'/name;out.mkdir(parents=True)
    dataset=out/'dataset';dataset.mkdir()
    records=[];audit=[]
    for old in original['records']:
        r=copy.deepcopy(old);folder=dataset/r['name'];folder.mkdir()
        raw_path=previous/'dataset'/r['name']/'original.png'
        raw=np.array(Image.open(raw_path).convert('RGB'))
        labels=np.array(Image.open(previous/'dataset'/r['labels']))
        assert sha(previous/'dataset'/r['labels'])==r['labels_sha256']
        # All 16 originals have a visually confirmed SEM image extent of 768 rows.
        bottom=768
        assert raw.shape[1]==1024 and labels.shape==raw.shape[:2]
        assert raw.shape[0]==(888 if r['name'].startswith('top_') else 768)
        cropped=labels[:bottom].copy()
        kept={int(i) for i in np.unique(cropped) if i}
        removed=[dict(label=int(i),original_id=r['label_to_original_id'][str(i)],pixels=int(np.count_nonzero(labels==i)))
                 for i in np.unique(labels) if i and int(i) not in kept]
        truncated=[int(i) for i in np.unique(labels[bottom:]) if i and int(i) in kept]
        row=dict(image=r['name'],old_shape=list(labels.shape),new_shape=list(cropped.shape),
                 footer_start_y=bottom,removed=removed,truncated_labels=truncated,
                 removed_footer_pixels=int(np.count_nonzero(labels[bottom:])),input_footer_rows_removed=raw.shape[0]-bottom)
        audit.append(row)
        gray=cv2.cvtColor(raw[:bottom],cv2.COLOR_RGB2GRAY)
        pixels,meta=prepare_adjustable(gray,dict(normalize_enabled=True,background_strength=0,blur_method='none',blur_strength=2))
        Image.fromarray(raw[:bottom]).save(folder/'original.png')
        Image.fromarray(pixels).save(dataset/r['image']);Image.fromarray(cropped).save(dataset/r['labels'])
        with np.load(previous/'dataset'/r['name']/'source_masks.npz',allow_pickle=False) as old_masks:
            masks={k:old_masks[k][:bottom].copy() for k in old_masks.files if np.any(old_masks[k][:bottom])}
        assert len(masks)==len(kept)
        np.savez_compressed(folder/'source_masks.npz',**masks)
        previous_shape=r['shape'];r.update(shape=[768,1024],instances=len(kept),label_to_original_id={str(i):r['label_to_original_id'][str(i)] for i in sorted(kept)},
             image_sha256=sha(dataset/r['image']),labels_sha256=sha(dataset/r['labels']),source_masks_sha256=sha(folder/'source_masks.npz'),
             source_masks=str(folder/'source_masks.npz'),evaluation_preprocessing=meta,
             footer_correction=row,original_export_masks=str(previous/'dataset'/r['name']/'source_masks.npz'))
        r['source_preprocessing']['analysis_bottom']=bottom
        if previous_shape==r['shape']:
            assert np.array_equal(pixels,np.array(Image.open(previous/'dataset'/old['image'])))
            assert np.array_equal(cropped,labels)
        records.append(r)
    assert sum(len(a['removed']) for a in audit)==3
    assert sum(r['instances'] for r in records)==2018
    assert all(not a['truncated_labels'] for a in audit)
    manifest=dict(version=4,records=records,label_review='Previously reviewed reference labels with three confirmed footer-only masks removed; valid SEM-region masks preserved.',
        input_processing='Crop raw imported image to confirmed SEM region y=[0,768), THEN app brightness normalization (2nd/98th percentile). No information panel, blur or background removal.',
        split_notes='Same image-wise 15/1 splits; five requested folds retrained. Related acquisitions may appear in training and evaluation.',
        supersedes=str(previous/'dataset/manifest.json'),footer_audit=audit)
    dump(dataset/'manifest.json',manifest);dump(out/'footer_audit.json',audit)
    models=ROOT/'fine_tuning/models'/name;models.mkdir(parents=True)
    plan=copy.deepcopy(old_plan)
    plan.update(run=str(out),dataset=str(dataset),dataset_manifest_sha256=sha(dataset/'manifest.json'),model_directory=str(models),
        selected_folds=[7,8,6,12,15],protocol='Footer-corrected data; same five 15/1 folds, independent original-checkpoint initialization, 300 updates, normalized train/evaluation inputs.',
        limitations='Corrects a confirmed information-panel contamination bug. Historical contaminated results are retained but not valid evidence of SEM-only performance. Only five selected folds are retrained in this run.')
    dump(out/'plan.json',plan);(out/'source').mkdir()
    for script in Path(__file__).parent.glob('*.py'):shutil.copy2(script,out/'source'/script.name)
    print(f'RUN {out}',flush=True);print(json.dumps(audit,indent=2),flush=True)
    for fold in plan['selected_folds']:
        subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('cross_validate.py')),
            '--worker-plan',str(out/'plan.json'),'--fold',str(fold)],check=True)
        verify_model(plan,fold)
        print(f'RETRAINED fold {fold}; excluded {plan["folds"][fold-1]["validation"]}',flush=True)
    subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('trial_two_stage.py')),
        '--source-run',str(out),'--out',str(out/'two_stage_evaluation')],check=True)
    subprocess.run([sys.executable,str(Path(__file__).with_name('verify_footer_clean.py')),'--run',str(out)],check=True)
    write_clean_report(out)
    dump(ROOT/'fine_tuning/models/latest_clean_experiment.json',dict(run=str(out.relative_to(ROOT/'fine_tuning')),dataset=str(dataset),
         model_directory=str(models),folds=plan['selected_folds'],status='five_held_out_models_no_full_data_refit',app_model_replaced=False))
    print(f'COMPLETE {out}/index.html',flush=True)


if __name__=='__main__':main()
