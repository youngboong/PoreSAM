"""Small-data SAM 2.1 decoder fine-tuning with grouped held-out evaluation.

Uses installed Meta SAM2 modules directly; does not change the application's model.
"""
import argparse
from collections import OrderedDict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'configs/sam2.1/sam2.1_hiera_s.yaml'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def save_csv(path, rows):
    with Path(path).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def flip(a, variant):
    if variant & 1:a=np.flip(a,axis=1)
    if variant & 2:a=np.flip(a,axis=0)
    return a.copy()


def prompts(masks, mode, rng, training=False):
    coords=[];labels=[]
    h,w=masks.shape[-2:]
    for mask in masks:
        ys,xs=np.where(mask)
        if mode=='point':
            if training:
                # Interior points avoid annotation boundary ambiguity.
                interior=cv2.erode(mask.astype(np.uint8),np.ones((3,3),np.uint8))
                py,px=np.where(interior)
                if not len(px):py,px=ys,xs
                pick=rng.integers(len(px));x,y=px[pick],py[pick]
            else:
                y,x=np.unravel_index(cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3).argmax(),mask.shape)
            coords.append([[x,y]]);labels.append([1])
        else:
            x0,y0,x1,y1=float(xs.min()),float(ys.min()),float(xs.max()+1),float(ys.max()+1)
            # Outward expansion leaves the whole annotated target inside the box.
            jitter=rng.uniform(0,.1,4)*np.array([x1-x0,y1-y0,x1-x0,y1-y0])
            coords.append([[max(0,x0-jitter[0]),max(0,y0-jitter[1])],[min(w,x1+jitter[2]),min(h,y1+jitter[3])]])
            labels.append([2,3])
    coords=np.asarray(coords,np.float32)
    coords[...,0]*=1024/w;coords[...,1]*=1024/h
    return coords,np.asarray(labels,np.int64)


class Features:
    def __init__(self, model, dataset, run, shared_root=None):
        self.model=model;self.dataset=dataset;self.root=shared_root or run/'features';self.root.mkdir(parents=True,exist_ok=True)
        self.predictor=SAM2ImagePredictor(model)
        self.memory=OrderedDict()

    def get(self, record, variant=0):
        key=f"{record['name']}_{variant}"
        if key in self.memory:
            self.memory.move_to_end(key)
            return self.memory[key]
        path=self.root/(key+'.pt')
        if path.exists():
            features=torch.load(path,map_location='cpu',weights_only=True)
        else:
            image=flip(np.array(Image.open(self.dataset/record['image']).convert('RGB')),variant)
            with torch.no_grad():
                self.predictor.set_image(image)
            features={'image_embed':self.predictor._features['image_embed'].detach().cpu(),
                      'high_res_feats':[v.detach().cpu() for v in self.predictor._features['high_res_feats']]}
            torch.save(features,path)
            self.predictor.reset_predictor()
        self.memory[key]=features
        while len(self.memory)>2:self.memory.popitem(last=False)
        return features


def decode(model, features, coords, labels):
    device=next(model.parameters()).device
    with torch.no_grad():
        sparse,dense=model.sam_prompt_encoder(points=(torch.as_tensor(coords,device=device),torch.as_tensor(labels,device=device)),boxes=None,masks=None)
        pe=model.sam_prompt_encoder.get_dense_pe()
    masks,scores,_,_=model.sam_mask_decoder.predict_masks(
        image_embeddings=features['image_embed'].to(device), image_pe=pe,
        sparse_prompt_embeddings=sparse,dense_prompt_embeddings=dense,
        repeat_image=len(coords)>1,high_res_features=[v.to(device) for v in features['high_res_feats']])
    return masks,scores


def loss_fn(logits,scores,target):
    target=F.interpolate(target[:,None].float(),size=logits.shape[-2:],mode='nearest').expand_as(logits)
    bce=F.binary_cross_entropy_with_logits(logits,target,reduction='none').mean((-2,-1))
    prob=logits.sigmoid()
    dice=1-(2*(prob*target).sum((-2,-1))+1)/(prob.sum((-2,-1))+target.sum((-2,-1))+1)
    per_mask=bce+dice
    # Train the single-mask token and the best of three ambiguous mask tokens.
    segment=.5*per_mask[:,0].mean()+.5*per_mask[:,1:].min(1).values.mean()
    with torch.no_grad():
        binary=logits>0;truth=target>0
        iou=(binary & truth).sum((-2,-1))/(binary | truth).sum((-2,-1)).clamp_min(1)
    return segment+.1*F.mse_loss(scores,iou)


@torch.no_grad()
def evaluate(model, features, records, dataset, out, stage, batch):
    model.eval();rows=[]
    for record in records:
        labels=np.array(Image.open(dataset/record['labels']))
        ids=np.unique(labels);ids=ids[ids>0]
        feat=features.get(record)
        for mode in ['point','box']:
            rng=np.random.default_rng(1701)
            for start in range(0,len(ids),batch):
                selected=ids[start:start+batch]
                truth=np.stack([labels==i for i in selected])
                coords,prompt_labels=prompts(truth,mode,rng)
                logits,scores=decode(model,feat,coords,prompt_labels)
                # Match the app's multi-mask choice: highest predicted score, not oracle IoU.
                choice=scores[:,1:].argmax(1)+1
                chosen=logits[torch.arange(len(choice),device=logits.device),choice][:,None]
                predictions=(F.interpolate(chosen,size=labels.shape,mode='bilinear',align_corners=False)[:,0]>0).cpu().numpy()
                for idx,(candidate_id,pred,gt) in enumerate(zip(selected,predictions,truth)):
                    intersection=int((pred & gt).sum());union=int((pred | gt).sum())
                    rows.append(dict(image=record['name'],split=record['split'],mode=mode,id=int(candidate_id),
                                     iou=intersection/max(1,union),dice=2*intersection/max(1,int(pred.sum()+gt.sum())),
                                     relative_area_error=abs(int(pred.sum())-int(gt.sum()))/max(1,int(gt.sum())),
                                     predicted_score=float(scores[idx,choice[idx]])))
        print(f"{stage}: {record['name']} ({len(ids)} pores)",flush=True)
    save_csv(out/(stage+'.csv'),rows)
    summary={}
    for mode in ['point','box']:
        subset=[r for r in rows if r['mode']==mode]
        per_image=[np.mean([r['iou'] for r in subset if r['image']==name]) for name in sorted({r['image'] for r in subset})]
        summary[mode]=dict(mean_image_iou=float(np.mean(per_image)),mean_instance_iou=float(np.mean([r['iou'] for r in subset])),
                           mean_dice=float(np.mean([r['dice'] for r in subset])),iou_at_least_50=float(np.mean([r['iou']>=.5 for r in subset])),
                           instances=len(subset),images=len(per_image))
    summary['selection_score']=float(np.mean([summary[m]['mean_image_iou'] for m in ['point','box']]))
    dump(out/(stage+'.json'),summary)
    return summary


def save_checkpoint(path,model,metadata):
    temporary=path.with_suffix('.partial')
    torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'finetuning':metadata},temporary)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=ROOT/'fine_tuning/datasets/260917_v1')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/sam2.1_hiera_small.pt')
    parser.add_argument('--steps',type=int,default=300)
    parser.add_argument('--batch',type=int,default=4)
    parser.add_argument('--eval-every',type=int,default=50)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--seed',type=int,default=260917)
    args=parser.parse_args()
    assert args.steps>0 and args.batch>0 and args.eval_every>0
    assert torch.cuda.is_available(), 'This training command requires the GPU environment; the CPU app is unaffected.'
    torch.set_num_threads(4);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    torch.backends.cudnn.benchmark=False
    run=ROOT/'fine_tuning/runs'/('260917_decoder_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    run.mkdir(parents=True);(run/'evaluation').mkdir();(run/'checkpoints').mkdir()
    manifest=json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    for rec in manifest['records']:
        assert sha(args.dataset/rec['image'])==rec['image_sha256']
        assert sha(args.dataset/rec['labels'])==rec['labels_sha256']
    groups={s:[r for r in manifest['records'] if r['split']==s] for s in ['train','validation','test']}
    assert all(groups.values())
    metadata=dict(run=str(run),config=CONFIG,base_checkpoint_sha256=sha(args.checkpoint),dataset_manifest_sha256=sha(args.dataset/'manifest.json'),
                  arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},torch=str(torch.__version__),
                  gpu=torch.cuda.get_device_name(),preprocessing=manifest['input_processing'],label_review=manifest['label_review'],
                  objective='Prompt-conditioned instance masks; this evaluation is not automatic detection recall.',
                  augmentation='Horizontal/vertical flips of training images only; random interior points and expanded boxes.',
                  split_notes=manifest['split_notes'])
    model=build_sam2(CONFIG,str(args.checkpoint),device='cuda',apply_postprocessing=False)
    model.requires_grad_(False)
    # The high-resolution projection layers run in the cached image encoder path.
    for name,parameter in model.sam_mask_decoder.named_parameters():
        if not name.startswith(('conv_s0.','conv_s1.','pred_obj_score_head.','obj_score_token.')):
            parameter.requires_grad_(True)
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    metadata['trainable_parameters']=sum(p.numel() for p in trainable.values())
    metadata['trainable_names']=list(trainable)
    dump(run/'config.json',metadata);dump(run/'dataset_manifest.json',manifest)
    print(f'RUN {run}\nTrainable parameters: {metadata["trainable_parameters"]}',flush=True)
    features=Features(model,args.dataset,run)
    baseline=evaluate(model,features,groups['validation'],args.dataset,run/'evaluation','baseline_validation',args.batch)
    optimizer=torch.optim.AdamW(trainable.values(),lr=args.lr,weight_decay=.01)
    rng=np.random.default_rng(args.seed);history=[];best_score=-1.;best_step=None
    start_time=time.monotonic();first_parameters={n:p.detach().cpu().clone() for n,p in trainable.items()}
    # Use several batches per image so feature transfer does not dominate CPU RAM usage.
    for step in range(1,args.steps+1):
        model.eval();model.sam_mask_decoder.train()
        if (step-1)%5==0:
            record=groups['train'][int(rng.integers(len(groups['train'])))];variant=int(rng.integers(4))
            label_map=flip(np.array(Image.open(args.dataset/record['labels'])),variant)
            instance_ids=np.unique(label_map);instance_ids=instance_ids[instance_ids>0]
            feat=features.get(record,variant)
        chosen=rng.choice(instance_ids,size=min(args.batch,len(instance_ids)),replace=False)
        truth=np.stack([label_map==i for i in chosen]);mode='point' if rng.random()<.5 else 'box'
        coords,labels=prompts(truth,mode,rng,training=True)
        optimizer.zero_grad(set_to_none=True)
        logits,scores=decode(model,feat,coords,labels)
        loss=loss_fn(logits,scores,torch.as_tensor(truth,device='cuda'))
        assert torch.isfinite(loss),'Non-finite training loss'
        loss.backward()
        grad_norm=torch.nn.utils.clip_grad_norm_(trainable.values(),1.,error_if_nonfinite=True)
        optimizer.step()
        history.append(dict(step=step,loss=float(loss.detach()),grad_norm=float(grad_norm),image=record['name'],mode=mode,seconds=time.monotonic()-start_time))
        if step%10==0 or step==1:
            print(f'Step {step}/{args.steps}, loss={float(loss.detach()):.4f}, elapsed={time.monotonic()-start_time:.1f}s',flush=True)
        if step%args.eval_every==0 or step==args.steps:
            validation=evaluate(model,features,groups['validation'],args.dataset,run/'evaluation',f'validation_{step:04d}',args.batch)
            if validation['selection_score']>best_score:
                best_score=validation['selection_score'];best_step=step
                save_checkpoint(run/'checkpoints/best.pt',model,{**metadata,'step':step,'validation':validation})
            save_csv(run/'training.csv',history)
            print(f'Validation score {validation["selection_score"]:.4f}; baseline {baseline["selection_score"]:.4f}',flush=True)
    changed=[n for n,p in trainable.items() if not torch.equal(first_parameters[n],p.detach().cpu())]
    assert changed,'No weights changed'
    save_checkpoint(run/'checkpoints/last.pt',model,{**metadata,'step':args.steps})
    best=torch.load(run/'checkpoints/best.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(best['model'],strict=True)
    tuned=evaluate(model,features,groups['test'],args.dataset,run/'evaluation','finetuned_test',args.batch)
    original=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    model.load_state_dict(original['model'],strict=True)
    baseline_test=evaluate(model,features,groups['test'],args.dataset,run/'evaluation','baseline_test',args.batch)
    assert sha(args.checkpoint)==metadata['base_checkpoint_sha256'],'Base checkpoint changed'
    report=dict(run=str(run),best_step=best_step,baseline_validation=baseline,best_validation_score=best_score,
                baseline_test=baseline_test,finetuned_test=tuned,changed_parameter_tensors=len(changed),
                seconds=time.monotonic()-start_time,max_gpu_memory_bytes=torch.cuda.max_memory_allocated(),
                limitation='Prompted segmentation on two held-out images, not proof of improved automatic pore detection or specimen generalization.',
                app_model_replaced=False)
    dump(run/'result.json',report)
    registry=ROOT/'fine_tuning/models';registry.mkdir(exist_ok=True)
    dump(registry/'latest_experiment.json',dict(run=str(run.relative_to(ROOT/'fine_tuning')),checkpoint=str((run/'checkpoints/best.pt').relative_to(ROOT/'fine_tuning')),status='experimental',app_model_replaced=False))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
