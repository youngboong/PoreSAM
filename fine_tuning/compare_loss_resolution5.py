"""Paired 300-update, five held-out folds: change segmentation loss resolution only."""
from pathlib import Path
import sys,json,time,shutil,subprocess,gc,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch,cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'fine_tuning'));sys.path.insert(0,str(ROOT/'scripts'))
from train import CONFIG,sha,dump,save_csv
from sam2.build_sam import build_sam2
from pore_selection import TraceGenerator,select,label_map
from evaluate_nested import metrics,mask_overlay
from diagnose_boundary_post import match,bmetrics,boundary
PREV=ROOT/'fine_tuning/runs/steps300_vs500_5_20260923_125241'
KEYS=['low_resolution','native_resolution']

LOSS='''
def loss_fn(logits,scores,target):
    import torch.nn.functional as F
    # Keep score-head supervision identical to the control condition.
    low_target=F.interpolate(target[:,None].float(),size=logits.shape[-2:],mode='nearest').expand_as(logits)
    native_logits=F.interpolate(logits,size=target.shape[-2:],mode='bilinear',align_corners=False)
    native_target=target[:,None].float().expand_as(native_logits)
    bce=F.binary_cross_entropy_with_logits(native_logits,native_target,reduction='none').mean((-2,-1))
    prob=native_logits.sigmoid()
    dice=1-(2*(prob*native_target).sum((-2,-1))+1)/(prob.sum((-2,-1))+native_target.sum((-2,-1))+1)
    per_mask=bce+dice
    segment=.5*per_mask[:,0].mean()+.5*per_mask[:,1:].min(1).values.mean()
    with torch.no_grad():
        binary=logits>0;truth=low_target>0
        iou=(binary & truth).sum((-2,-1))/(binary | truth).sum((-2,-1)).clamp_min(1)
    return segment+.1*F.mse_loss(scores,iou)

'''

def render(folder,gray,truth,maps):
    h,w=gray.shape;font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
    panels=[('Reviewed reference',truth),('Original loss (300)',maps[KEYS[0]]),('Native-resolution loss (300)',maps[KEYS[1]])]
    canvas=Image.new('RGB',(3*(w+16)+16,h+70),'white');draw=ImageDraw.Draw(canvas)
    for i,(name,lab) in enumerate(panels):
        x=16+i*(w+16);draw.text((x,15),name,font=font,fill='black');canvas.paste(Image.fromarray(mask_overlay(gray,lab)),(x,54))
    canvas.save(folder/'overview.png')
    sheet=Image.new('RGB',(3*272+16,4*306+60),'white');draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for j,name in enumerate(['SEM + reference','Original loss','Native-resolution loss']):draw.text((16+j*272,12),name,font=font,fill='black')
    gb=boundary(truth>0);base=np.repeat(gray[:,:,None],3,axis=2)
    for row,(cx,cy) in enumerate([(256,192),(768,192),(256,576),(768,576)]):
        for col,key in enumerate([None]+KEYS):
            im=base.copy();im[gb]=[0,220,255]
            if key:im[boundary(maps[key]>0)]=[255,40,80]
            x=16+col*272;y=50+row*306;sheet.paste(Image.fromarray(im[cy-128:cy+128,cx-128:cx+128]),(x,y));draw.text((x,y+262),f'Center {cx},{cy}; native pixels',font=font,fill='black')
    sheet.save(folder/'boundary_crops.png')

def report(out,results):
    rows=[x for r in results for x in r['metrics']];save_csv(out/'metrics.csv',rows)
    fields=['iou','instance_f1','bf1_2','common_bf1_2','common_assd_px']
    means={k:{f:float(np.mean([r[f] for r in rows if r['variant']==k])) for f in fields} for k in KEYS}
    scope='Five previously inspected image-wise 15:1 splits, one seed. Related acquisitions may cross folds. Common-pore metrics exclude image-edge pores and condition on detection at IoU >= .5 in both conditions; image means are equally weighted.'
    summary=dict(completed=len(results),expected=5,means=means,common_pores=sum(r['common_count'] for r in results),scope=scope,app_model_replaced=False)
    dump(out/'summary.json',summary)
    page='<!doctype html><meta charset="utf-8"><title>Loss resolution comparison</title><style>body{font:16px system-ui;line-height:1.6;max-width:1400px;margin:32px auto;padding:20px}td,th{padding:10px;border-bottom:1px solid #ddd}img{width:100%}</style><h1>Segmentation loss resolution: paired 300-update comparison</h1><p>Control: nearest-downsampled reference at decoder output resolution. Treatment: bilinear-upsampled logits at native reference resolution. BCE + Dice weights, score supervision, seed, sampled pores, flips, prompts, optimizer, frozen layers and two-stage nested inference are held constant. No boundary loss added. Control checkpoints and predictions are reused from the earlier 300-update run.</p>'
    page+='<p>'+scope+'</p><p>Boundary F1 tolerance: 2 pixels (higher is better); symmetric mean boundary distance in pixels (lower is better). Cyan: reference; red: prediction. Fixed quadrant crops.</p><table><tr><th>Loss</th><th>Area IoU %</th><th>Instance F1 %</th><th>Union boundary F1 %</th><th>Common boundary F1 %</th><th>Common distance px</th></tr>'
    for k in KEYS:page+='<tr><td>'+k+'</td>'+''.join(f'<td>{means[k][f]*(1 if f=="common_assd_px" else 100):.3f}</td>' for f in fields)+'</tr>'
    page+='</table><h2>Per-image paired results</h2><table><tr><th>Image</th><th>Common pores</th><th>IoU old / new %</th><th>Boundary F1 old / new %</th><th>Distance old / new px</th></tr>'
    for r in results:
        a,b=r['metrics'];page+=f'<tr><td>{html.escape(r["image"])}</td><td>{r["common_count"]}</td><td>{a["iou"]*100:.2f} / {b["iou"]*100:.2f}</td><td>{a["common_bf1_2"]*100:.2f} / {b["common_bf1_2"]*100:.2f}</td><td>{a["common_assd_px"]:.3f} / {b["common_assd_px"]:.3f}</td></tr>'
    page+='</table>'
    for r in results:page+=f'<h2>{html.escape(r["image"])}</h2><a href="fold_{r["fold"]}/overview.png"><img src="fold_{r["fold"]}/overview.png"></a><a href="fold_{r["fold"]}/boundary_crops.png"><img src="fold_{r["fold"]}/boundary_crops.png"></a>'
    (out/'index.html').write_text(page,encoding='utf-8')

def setup():
    out=ROOT/'fine_tuning/runs'/('loss_resolution5_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir();(out/'source').mkdir()
    plan=json.loads((PREV/'plan.json').read_text());plan.update(run=str(out),steps=300,model_directory=str(ROOT/'fine_tuning/models'/out.name),control_run=str(PREV),intervention='Native-resolution BCE+Dice; score-head loss unchanged',primary_metric='Common-pore boundary F1 at 2px; equal image means',secondary_metrics=['common_assd_px','iou','instance_f1'],control_default_sha256=sha(ROOT/'checkpoints/default_model.json'))
    Path(plan['model_directory']).mkdir()
    for folder,file in [('fine_tuning','train.py'),('fine_tuning','evaluate_nested.py'),('scripts','pore_selection.py'),('scripts','pore_conservative.py'),('scripts','pore_nested.py'),('scripts','segment_first_pass.py')]:
        assert sha(ROOT/folder/file)==sha(PREV/'source'/file),file+' changed since control'
        shutil.copy2(ROOT/folder/file,out/'source'/file)
    shutil.copy2(__file__,out/'source'/Path(__file__).name)
    worker=(PREV/'source/worker.py').read_text()
    worker=worker.replace('def worker(plan_path, number):',LOSS+'\ndef worker(plan_path, number):')
    worker=worker.replace("shared_root=plan_path.parent/'features'","shared_root=Path(plan['control_run'])/'features'")
    worker=worker.replace('Paired snapshots at 300 and 500 updates; no held-out selection or early stopping.','Native-resolution segmentation loss, 300 updates; unchanged score supervision; no held-out selection.')
    worker=worker.replace("        if step==300:\n            save_checkpoint(Path(plan['model_directory'])/f'fold_{number}_step300.pt',model,dict(metadata,steps=300))\n",'')
    (out/'source/worker.py').write_text(worker,encoding='utf-8');dump(out/'plan.json',plan)
    (ROOT/'outputs/loss_resolution_active.txt').write_text(str(out))
    return out

def main(out):
    plan=json.loads((out/'plan.json').read_text());dataset=Path(plan['dataset']);assert sha(dataset/'manifest.json')==plan['dataset_manifest_sha256']
    records=json.loads((dataset/'manifest.json').read_text())['records'];results=[];torch.set_num_threads(4);cv2.setNumThreads(4)
    for f in plan['selected_folds']:
        folder=out/f'fold_{f}';record=records[f-1];name=record['name'];assert plan['folds'][f-1]['validation']==[name]
        if (folder/'paired_result.json').exists():results.append(json.loads((folder/'paired_result.json').read_text()));report(out,results);continue
        if not (folder/'result.json').exists():
            assert not folder.exists(),'Inspect incomplete training before resuming'
            print(f'TRAIN fold {f}: {name}',flush=True);subprocess.run([sys.executable,'-u',str(out/'source/worker.py'),str(out/'plan.json'),str(f)],check=True)
        log=json.loads((folder/'sample_log.json').read_text());oldlog=json.loads((PREV/f'fold_{f}/sample_log.json').read_text())
        assert len(log)==300 and log==oldlog[:300]
        config=json.loads((folder/'config.json').read_text());assert len(config['train_images'])==15 and name not in config['train_images'] and config['validation_images']==[name]
        print(f'PAIRED SAMPLING VERIFIED fold {f}',flush=True)
        image=np.array(Image.open(dataset/record['image']).convert('RGB'));gray=image[:,:,0]
        control=PREV/f'fold_{f}/step300_instances.tif';shutil.copy2(control,folder/'low_resolution.tif')
        checkpoint=Path(plan['model_directory'])/f'fold_{f}.pt'
        if not (folder/'native_resolution.tif').exists():
            print(f'INFERENCE fold {f}',flush=True);model=build_sam2(CONFIG,str(checkpoint),device='cuda',apply_postprocessing=False)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                gen=TraceGenerator(model,points_per_side=48,points_per_batch=4,pred_iou_thresh=.7,stability_score_thresh=.85,crop_n_layers=0,min_mask_region_area=0,output_mode='uncompressed_rle');gen.generate(image)
            settings=record['source_preprocessing'];selected=select(gen.pool,gray,settings.get('min_area_pixels',100),settings.get('min_contrast',8),'cuda','two_stage_nested')
            Image.fromarray(label_map(selected['masks'],gray.shape)).save(folder/'native_resolution.tif')
            dump(folder/'selection_decisions.json',{k:v for k,v in selected.items() if k not in ['masks','strict']})
            del model,gen,selected;gc.collect();torch.cuda.empty_cache()
        truth=np.array(Image.open(dataset/record['labels']));maps={k:np.array(Image.open(folder/(k+'.tif'))) for k in KEYS};matches={k:match(truth,m) for k,m in maps.items()}
        common=set(matches[KEYS[0]])&set(matches[KEYS[1]]);common-=set(np.unique(np.concatenate([truth[0],truth[-1],truth[:,0],truth[:,-1]])))
        rows=[];details=[]
        for k,m in maps.items():
            per=[]
            for i in sorted(common):
                bm=bmetrics(truth==i,m==matches[k][i]);per.append(bm);details.append(dict(variant=k,reference_id=i,**bm))
            row=dict(fold=f,image=name,variant=k,**metrics(truth,m),**bmetrics(truth>0,m>0),common_count=len(common),**{'common_'+field:float(np.mean([x[field] for x in per])) for field in ['bf1_1','bf1_2','bf1_4','assd_px']});rows.append(row)
        oldmetric=json.loads((PREV/f'fold_{f}/step300_metrics.json').read_text());assert abs(rows[0]['iou']-oldmetric['iou'])<1e-12
        save_csv(folder/'common_pore_boundaries.csv',details);render(folder,gray,truth,maps)
        result=dict(fold=f,image=name,common_count=len(common),common_ids=sorted(common),metrics=rows,sampling_identical=True,control_mask_sha256=sha(control),checkpoint_sha256=sha(checkpoint))
        dump(folder/'paired_result.json',result);results.append(result);report(out,results)
        print(f'COMPLETE FOLD {f}: boundary F1 {rows[0]["common_bf1_2"]:.4f} -> {rows[1]["common_bf1_2"]:.4f}; IoU {rows[0]["iou"]:.4f} -> {rows[1]["iou"]:.4f}',flush=True)
    assert sha(ROOT/'checkpoints/default_model.json')==plan['control_default_sha256']
    dump(out/'verification.json',dict(completed_folds=len(results),sampling_identical=True,held_out_excluded=True,control_metrics_reproduced=True,app_default_unchanged=True))
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else setup())
