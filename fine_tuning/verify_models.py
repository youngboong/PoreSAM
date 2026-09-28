"""Check that only declared decoder weights changed and the refit runs on CPU."""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from train import CONFIG, dump, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True)
    args=parser.parse_args();plan=json.loads((args.run/'plan.json').read_text())
    # Correct the pilot's fixed caption on any report generated before the CV caption update.
    for fold in plan['folds']:
        page=args.run/f"fold_{fold['fold']}"/'automatic_evaluation/index.html'
        text=page.read_text(encoding='utf-8')
        page.write_text(text.replace('Two test images;',f"{len(fold['validation'])} held-out images;"),encoding='utf-8')
    base=Path(plan['base_checkpoint']);assert sha(base)==plan['base_checkpoint_sha256']
    original=torch.load(base,map_location='cpu',weights_only=True)['model']
    checked=[]
    for path in sorted(Path(plan['model_directory']).glob('*.pt')):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
        state=checkpoint['model'];allowed=set(checkpoint['finetuning']['trainable_names'])
        assert state.keys()==original.keys()
        changes=[]
        for key,value in state.items():
            assert value.shape==original[key].shape
            if value.is_floating_point():assert torch.isfinite(value).all(), key
            if not torch.equal(value,original[key]):changes.append(key)
        assert changes and set(changes)<=allowed,(path,changes)
        checked.append(dict(file=path.name,sha256=sha(path),changed_tensors=len(changes),frozen_tensors_unchanged=True))
        del checkpoint,state
    assert len(checked)==6
    del original
    torch.set_num_threads(4)
    final=Path(plan['model_directory'])/'all_images.pt'
    model=build_sam2(CONFIG,str(final),device='cpu',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model)
    data=Path(plan['dataset']);manifest=json.loads((data/'manifest.json').read_text())
    record=manifest['records'][0];image=np.array(Image.open(data/record['image']).convert('RGB'))
    labels=np.array(Image.open(data/record['labels']));y,x=np.where(labels==1)
    with torch.inference_mode():
        predictor.set_image(image)
        masks,scores,_=predictor.predict(box=np.array([x.min(),y.min(),x.max()+1,y.max()+1]),multimask_output=True)
    assert masks.shape==(3,*labels.shape) and np.isfinite(scores).all()
    assert next(model.parameters()).device.type=='cpu'
    result=dict(checkpoints=checked,cpu_inference=True,cpu_mask_shape=list(masks.shape),base_checkpoint_unchanged=True,app_model_replaced=False)
    dump(args.run/'model_verification.json',result)
    summary=json.loads((args.run/'summary.json').read_text())['summary']
    current=summary['baseline'];tuned=summary['finetuned']
    recommendation='Keep the original model as the application default; use this checkpoint for further experiments.'
    if tuned['automatic']['f1']>current['automatic']['f1'] and all(tuned[m]['image_mean_iou']>=current[m]['image_mean_iou'] for m in ['point','box']):
        recommendation='Candidate for further validation on new images; not automatically promoted into the application.'
    rows=[]
    for name,s in [('Original',current),('Fine-tuned (out of fold)',tuned)]:
        rows.append(f"| {name} | {s['point']['image_mean_iou']:.4f} | {s['box']['image_mean_iou']:.4f} | {s['automatic']['precision']:.4f} | {s['automatic']['recall']:.4f} | {s['automatic']['f1']:.4f} |")
    card='\n'.join([
        '# PoreSAM experimental decoder fine-tuning','',
        'Base: SAM 2.1 Hiera Small. Image encoder and prompt encoder frozen; mask decoder fine-tuned.','',
        'Data: 14 manually reviewed SEM images, 1,772 instance masks. No additional normalization, background removal or blur.','',
        'Training: 300 optimizer updates per model, AdamW, learning rate 1e-5, batch 4, seed 260917.','',
        '`fold_1.pt` through `fold_5.pt` are independent grouped cross-validation models. `all_images.pt` is a separate refit using all 14 images. The all-images model has no independent test score.','',
        '| Model | Point IoU | Box IoU | Automatic precision | Automatic recall | Automatic F1 |',
        '|---|---:|---:|---:|---:|---:|',*rows,'',
        'Prompted IoU is averaged per image. Automatic metrics pool one-to-one instance matches at IoU >= 0.5. Automatic evaluation uses identical app filters and no Automate; GPU autocast is used for both models.','',
        '**Interpretation:** '+recommendation,'',plan['limitations'],'',
        'All six checkpoints passed tensor integrity and frozen-weight checks. The all-images model loaded and predicted masks on CPU. The original app checkpoint was unchanged.','',
        '## Load','',
        '```python','from sam2.build_sam import build_sam2',
        'model = build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml",',
        '                   "<model-folder>/all_images.pt", device="cpu")','```','',
        'Run metadata: '+str(args.run.resolve()),'Base SHA-256: '+plan['base_checkpoint_sha256'],''])
    (Path(plan['model_directory'])/'README.md').write_text(card,encoding='utf-8')
    (args.run/'MODEL_CARD.md').write_text(card,encoding='utf-8')
    result['recommendation']=recommendation
    dump(args.run/'model_verification.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
