"""Five-fold grouped cross-validation, fixed training budget, separate models."""
import argparse
from collections import Counter
import copy
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from train import ROOT, CONFIG, Features, decode, prompts, flip, loss_fn, evaluate, save_checkpoint, save_csv, dump, sha


def worker(plan_path, number):
    plan=json.loads(plan_path.read_text());dataset=Path(plan['dataset']);base=Path(plan['base_checkpoint'])
    out=plan_path.parent/('final_all_images' if number==0 else f'fold_{number}')
    out.mkdir();(out/'evaluation').mkdir()
    manifest=json.loads((dataset/'manifest.json').read_text())
    held=set(plan['folds'][number-1]['validation']) if number else set()
    for record in manifest['records']:record['split']='validation' if record['name'] in held else 'train'
    training=[r for r in manifest['records'] if r['split']=='train']
    validation=[r for r in manifest['records'] if r['split']=='validation']
    for record in manifest['records']:
        assert sha(dataset/record['image'])==record['image_sha256']
        assert sha(dataset/record['labels'])==record['labels_sha256']
    assert sha(base)==plan['base_checkpoint_sha256']
    seed=plan['seed'];torch.manual_seed(seed);np.random.seed(seed);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    model=build_sam2(CONFIG,str(base),device='cuda',apply_postprocessing=False)
    model.requires_grad_(False)
    for name,p in model.sam_mask_decoder.named_parameters():
        if not name.startswith(('conv_s0.','conv_s1.','pred_obj_score_head.','obj_score_token.')):p.requires_grad_(True)
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    metadata=dict(fold=number,protocol='Fixed 300 updates; validation never selects a checkpoint or stopping time.',
                  arguments=dict(dataset=str(dataset),checkpoint=str(base)),base_checkpoint_sha256=sha(base),
                  trainable_names=list(trainable),steps=plan['steps'],lr=plan['lr'],batch=plan['batch'],seed=seed,
                  train_images=[r['name'] for r in training],validation_images=sorted(held),
                  all_images_refit=number==0,preprocessing=manifest['input_processing'],torch=str(torch.__version__))
    dump(out/'config.json',metadata);dump(out/'dataset_manifest.json',manifest)
    cache=Features(model,dataset,out,shared_root=plan_path.parent/'features')
    baseline=evaluate(model,cache,validation,dataset,out/'evaluation','baseline_validation',plan['batch']) if validation and plan.get('prompted_evaluation',True) else None
    optimizer=torch.optim.AdamW(trainable.values(),lr=plan['lr'],weight_decay=.01)
    rng=np.random.default_rng(seed);order=[];history=[];sampled=Counter();start=time.monotonic()
    for step in range(1,plan['steps']+1):
        model.eval();model.sam_mask_decoder.train()
        if (step-1)%5==0:
            if not order:order=rng.permutation(len(training)).tolist()
            record=training[order.pop()];variant=int(rng.integers(4))
            label_map=flip(np.array(Image.open(dataset/record['labels'])),variant)
            instance_ids=np.unique(label_map);instance_ids=instance_ids[instance_ids>0]
            feat=cache.get(record,variant)
        ids=rng.choice(instance_ids,size=min(plan['batch'],len(instance_ids)),replace=False)
        masks=np.stack([label_map==i for i in ids]);mode='point' if rng.random()<.5 else 'box'
        coords,labels=prompts(masks,mode,rng,training=True)
        optimizer.zero_grad(set_to_none=True)
        logits,scores=decode(model,feat,coords,labels)
        loss=loss_fn(logits,scores,torch.as_tensor(masks,device='cuda'))
        assert torch.isfinite(loss)
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(trainable.values(),1.,error_if_nonfinite=True);optimizer.step()
        sampled[record['name']]+=len(ids)
        history.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm),image=record['name'],mode=mode))
        if step%50==0:print(f'Fold {number}: step {step}/{plan["steps"]}, loss {float(loss.detach()):.4f}',flush=True)
    assert set(sampled)=={r['name'] for r in training}
    model_path=Path(plan['model_directory'])/('all_images.pt' if number==0 else f'fold_{number}.pt')
    save_checkpoint(model_path,model,metadata)
    save_csv(out/'training.csv',history)
    tuned=evaluate(model,cache,validation,dataset,out/'evaluation','finetuned_validation',plan['batch']) if validation and plan.get('prompted_evaluation',True) else None
    result=dict(fold=number,baseline=baseline,finetuned=tuned,checkpoint=str(model_path),checkpoint_sha256=sha(model_path),
                train_images=len(training),validation_images=len(validation),sampled_prompts=dict(sampled),seconds=time.monotonic()-start,
                status='experimental',app_model_replaced=False)
    dump(out/'result.json',result)
    print(json.dumps(result),flush=True)


def summarize(out, plan):
    rows=[];auto=[];folds=[]
    for fold in plan['folds']:
        folder=out/f"fold_{fold['fold']}"
        folds.append(json.loads((folder/'result.json').read_text()))
        for name in ['baseline','finetuned']:
            with (folder/'evaluation'/f'{name}_validation.csv').open() as f:
                for row in csv.DictReader(f):
                    rows.append(dict(fold=fold['fold'],model=name,**row))
        auto.extend(dict(fold=fold['fold'],**r) for r in json.loads((folder/'automatic_evaluation/results.json').read_text())['results'])
    save_csv(out/'out_of_fold_predictions.csv',rows);save_csv(out/'automatic_metrics.csv',auto)
    summary={}
    for name in ['baseline','finetuned']:
        summary[name]={}
        for mode in ['point','box']:
            subset=[r for r in rows if r['model']==name and r['mode']==mode]
            names=sorted({r['image'] for r in subset})
            per_image=[np.mean([float(r['iou']) for r in subset if r['image']==image]) for image in names]
            fold_means=[f['baseline' if name=='baseline' else 'finetuned'][mode]['mean_image_iou'] for f in folds]
            summary[name][mode]=dict(image_mean_iou=float(np.mean(per_image)),fold_mean_iou=float(np.mean(fold_means)),
                                     fold_std_iou=float(np.std(fold_means,ddof=1)),instances=len(subset),images=len(names))
            assert len(names)==14 and len(subset)==1772
        subset=[r for r in auto if r['model']==name]
        tp=sum(r['true_positives_iou50'] for r in subset);fp=sum(r['false_positives'] for r in subset);fn=sum(r['false_negatives'] for r in subset)
        summary[name]['automatic']=dict(precision=tp/max(1,tp+fp),recall=tp/max(1,tp+fn),f1=2*tp/max(1,2*tp+fp+fn),
                                         matched=tp,false_positives=fp,missed=fn,reference_count=tp+fn,
                                         image_mean_f1=float(np.mean([r['f1'] for r in subset])))
    report=dict(protocol=plan['protocol'],fold_sizes=[len(f['validation']) for f in plan['folds']],summary=summary,
                limitations=plan['limitations'],models=plan['model_directory'],app_model_replaced=False)
    dump(out/'summary.json',report)
    table=''
    for name in ['baseline','finetuned']:
        s=summary[name]
        table+=f'<tr><td>{name}</td><td>{s["point"]["image_mean_iou"]:.3f}</td><td>{s["box"]["image_mean_iou"]:.3f}</td><td>{s["automatic"]["precision"]:.3f}</td><td>{s["automatic"]["recall"]:.3f}</td><td>{s["automatic"]["f1"]:.3f}</td></tr>'
    links=''.join(f'<li><a href="fold_{f["fold"]}/automatic_evaluation/index.html">Fold {f["fold"]}: {", ".join(f["validation"])}</a></li>' for f in plan['folds'])
    html='<!doctype html><meta charset="utf-8"><title>PoreSAM 5-fold evaluation</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}td,th{padding:12px;border-bottom:1px solid #ccc}table{border-collapse:collapse}</style><h1>PoreSAM: 5-fold evaluation</h1><p>14 images, 1,772 reviewed instances. Grouped folds, 300 fixed updates per fold. Each held-out image is evaluated once. No background removal or blur.</p><table><tr><th>Model</th><th>Point IoU</th><th>Box IoU</th><th>Automatic precision</th><th>Automatic recall</th><th>Automatic F1</th></tr>'+table+'</table><p>Prompt IoU is averaged per image. Automatic metrics pool one-to-one matches at IoU 0.5 with identical app filters; Automate is not run.</p><h2>Image comparisons</h2><ul>'+links+'</ul><h2>Scope</h2><p>'+plan['limitations']+'</p><p>The all-images checkpoint is refit on all 14 images and has no independent test score. The application model has not been replaced.</p>'
    (out/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker-plan',type=Path);parser.add_argument('--fold',type=int)
    parser.add_argument('--resume-plan',type=Path)
    args=parser.parse_args()
    if args.worker_plan:
        worker(args.worker_plan,args.fold);return
    if args.resume_plan:
        plan=json.loads(args.resume_plan.read_text());out=args.resume_plan.parent
        finish(out,plan)
        return
    dataset=ROOT/'fine_tuning/datasets/260917_v1';base=ROOT/'checkpoints/sam2.1_hiera_small.pt'
    manifest=json.loads((dataset/'manifest.json').read_text())
    groups={}
    for record in manifest['records']:groups.setdefault(record['group'],[]).append(record['name'])
    assigned=[[] for _ in range(5)]
    for group,names in sorted(groups.items(),key=lambda item:(-len(item[1]),item[0])):
        index=min(range(5),key=lambda i:(len(assigned[i]),i));assigned[index].extend(names)
    assert len({name for fold in assigned for name in fold})==14
    for names in groups.values():assert sum(bool(set(names)&set(fold)) for fold in assigned)==1
    name='260917_cv5_'+datetime.now().strftime('%Y%m%d_%H%M%S')
    out=ROOT/'fine_tuning/runs'/name;out.mkdir(parents=True)
    models=ROOT/'fine_tuning/models'/name;models.mkdir(parents=True)
    plan=dict(dataset=str(dataset),base_checkpoint=str(base),base_checkpoint_sha256=sha(base),
              dataset_manifest_sha256=sha(dataset/'manifest.json'),model_directory=str(models),seed=260917,
              steps=300,batch=4,lr=1e-5,protocol='5 grouped folds, fixed 300 updates from original checkpoint independently, no validation-selected checkpoints.',
              limitations='Exploratory cross-validation on 14 related images after an initial pilot on the same collection. Filename acquisition groups are kept together; physical specimen independence is unknown. Fold SD is variability, not a confidence interval. No independent external test set.',
              folds=[dict(fold=i+1,validation=sorted(names)) for i,names in enumerate(assigned)])
    dump(out/'plan.json',plan)
    for script in Path(__file__).parent.glob('*.py'):
        (out/'source').mkdir(exist_ok=True);shutil.copy2(script,out/'source'/script.name)
    print(f'CROSS VALIDATION {out}\n'+json.dumps(plan['folds'],indent=2),flush=True)
    finish(out,plan)


def finish(out,plan):
    out=out.resolve()
    models=Path(plan['model_directory'])
    assert sha(Path(plan['base_checkpoint']))==plan['base_checkpoint_sha256']
    for number in range(1,6):
        if not (out/f'fold_{number}'/'result.json').exists():
            subprocess.run([sys.executable,'-u',__file__,'--worker-plan',str(out/'plan.json'),'--fold',str(number)],check=True)
        if not (out/f'fold_{number}'/'automatic_evaluation/index.html').exists():
            subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('evaluate_automatic.py')),'--run',str(out/f'fold_{number}'),
                            '--checkpoint',str(models/f'fold_{number}.pt'),'--split','validation'],check=True)
    summarize(out,plan)
    if not (out/'final_all_images/result.json').exists():
        subprocess.run([sys.executable,'-u',__file__,'--worker-plan',str(out/'plan.json'),'--fold','0'],check=True)
    dump(ROOT/'fine_tuning/models/latest_experiment.json',dict(run=str(out.relative_to(ROOT/'fine_tuning')),checkpoint=str((models/'all_images.pt').relative_to(ROOT/'fine_tuning')),
                                                            status='experimental_all_images_refit',app_model_replaced=False))
    print(f'COMPLETE {out}/index.html',flush=True)


if __name__=='__main__':main()
