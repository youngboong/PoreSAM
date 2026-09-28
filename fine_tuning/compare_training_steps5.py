"""Paired 300 vs 500 update experiment on five preselected 15:1 splits."""
import sys,json,subprocess,time,gc,hashlib,csv,html
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch,cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'fine_tuning'));sys.path.insert(0,str(ROOT/'scripts'))
from sam2.build_sam import build_sam2
from train import CONFIG,sha,dump,save_csv
from pore_selection import TraceGenerator,select,label_map
from evaluate_nested import metrics,mask_overlay

def report(out,results):
 rows=[m for r in results for m in r['metrics']]
 if not rows:return
 fields=['iou','dice','instance_f1','missed_area_pct','extra_area_pct','relative_area_error_pct']
 means={str(step):{k:float(np.mean([r[k] for r in rows if r['steps']==step])) for k in fields} for step in [300,500]}
 delta=[r['metrics'][1]['iou']-r['metrics'][0]['iou'] for r in results]
 summary=dict(completed=len(results),expected=5,macro_image_means=means,mean_iou_change_pp=100*float(np.mean(delta)),wins500=sum(d>1e-9 for d in delta),wins300=sum(d< -1e-9 for d in delta),ties=sum(abs(d)<=1e-9 for d in delta),app_model_replaced=False,limitations='Five previously inspected held-out images; one seed; image-wise 15:1 splits may contain related acquisitions. Not independent specimen-level validation.')
 dump(out/'summary.json',summary);save_csv(out/'metrics.csv',rows)
 page='<!doctype html><meta charset="utf-8"><title>300 vs 500 updates</title><style>body{font:16px system-ui;line-height:1.6;max-width:1500px;margin:32px auto;padding:20px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}img{width:100%}a{color:#1262a4}</style><h1>300 vs 500 updates - five 15:1 held-out splits</h1><p>Each fold starts from base SAM. Snapshots at 300 and 500 updates share one uninterrupted training trajectory and AdamW state. Same normalized inputs, seed, prompts recipe and two-stage nested selection. No held-out masks used for inference or checkpoint selection.</p>'
 page+='<h2>Equal-weight image means</h2><table><tr><th>Updates</th><th>Area IoU (%)</th><th>Dice (%)</th><th>Instance F1 (%)</th><th>Missed (%)</th><th>Extra (%)</th><th>Area error (%)</th></tr>'
 for step in ['300','500']:
  m=means[step];page+=f'<tr><td>{step}</td>'+''.join(f'<td>{m[k]*(100 if k in ["iou","dice","instance_f1"] else 1):.2f}</td>' for k in fields)+'</tr>'
 page+='</table><p>Missed, extra and total-area error are divided by reference pore area. Instance F1 uses one-to-one IoU >= 0.5 matching.</p>'
 page+=f'<p>Completed: {len(results)}/5; mean IoU change: {summary["mean_iou_change_pp"]:+.2f} percentage points. 500 wins: {summary["wins500"]}; 300 wins: {summary["wins300"]}; ties: {summary["ties"]}.</p>'
 page+='<table><tr><th>Held-out image</th><th>300 IoU (%)</th><th>500 IoU (%)</th><th>Change (pp)</th></tr>'
 for r in results:
  a,b=r['metrics'];page+=f'<tr><td>{html.escape(r["image"])}</td><td>{100*a["iou"]:.2f}</td><td>{100*b["iou"]:.2f}</td><td>{100*(b["iou"]-a["iou"]):+.2f}</td></tr>'
 page+='</table><p>'+summary['limitations']+'</p>'
 for r in results:page+=f'<h2>{html.escape(r["image"])}</h2><a href="fold_{r["fold"]}/comparison.png"><img src="fold_{r["fold"]}/comparison.png"></a>'
 (out/'index.html').write_text(page,encoding='utf-8')

def main(out):
 plan=json.loads((out/'plan.json').read_text());dataset=Path(plan['dataset']);assert sha(dataset/'manifest.json')==plan['dataset_manifest_sha256']
 records=json.loads((dataset/'manifest.json').read_text())['records'];modeldir=Path(plan['model_directory'])
 original_default=sha(ROOT/'checkpoints/default_model.json');torch.set_num_threads(4);cv2.setNumThreads(4);results=[]
 for fold in plan['selected_folds']:
  folder=out/f'fold_{fold}';name=plan['folds'][fold-1]['validation'][0];record=next(r for r in records if r['name']==name)
  if (folder/'paired_result.json').exists():results.append(json.loads((folder/'paired_result.json').read_text()));report(out,results);continue
  if not (folder/'result.json').exists():
   assert not folder.exists(),'Incomplete training directory; inspect before resuming.'
   print(f'TRAIN fold {fold}: {name} held out',flush=True)
   subprocess.run([sys.executable,'-u',str(out/'source/worker.py'),str(out/'plan.json'),str(fold)],check=True)
  config=json.loads((folder/'config.json').read_text());assert len(config['train_images'])==15 and name not in config['train_images'] and config['validation_images']==[name]
  logs=json.loads((folder/'sample_log.json').read_text());assert len(logs)==500 and all(x['image']!=name for x in logs)
  coverage={str(step):{n:len({p for x in logs[:step] if x['image']==n for p in x['ids']}) for n in config['train_images']} for step in [300,500]};dump(folder/'sampling_coverage.json',coverage)
  image=np.array(Image.open(dataset/record['image']).convert('RGB'));gray=image[:,:,0];maps={};rows=[]
  for step in [300,500]:
   checkpoint=modeldir/(f'fold_{fold}_step300.pt' if step==300 else f'fold_{fold}.pt');saved=folder/f'step{step}_instances.tif';evaluation=folder/f'step{step}_metrics.json'
   if saved.exists() and evaluation.exists():maps[step]=np.array(Image.open(saved));rows.append(json.loads(evaluation.read_text()));continue
   print(f'EVALUATE fold {fold}, step {step}: {name}',flush=True);start=time.monotonic()
   model=build_sam2(CONFIG,str(checkpoint),device='cuda',apply_postprocessing=False)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    gen=TraceGenerator(model,points_per_side=48,points_per_batch=4,pred_iou_thresh=.7,stability_score_thresh=.85,crop_n_layers=0,min_mask_region_area=0,output_mode='uncompressed_rle');gen.generate(image)
   settings=record['source_preprocessing'];prediction=select(gen.pool,gray,settings.get('min_area_pixels',100),settings.get('min_contrast',8),'cuda','two_stage_nested')
   labels=label_map(prediction['masks'],gray.shape);Image.fromarray(labels).save(saved);maps[step]=labels
   # Reference is loaded only after automatic selection is complete.
   truth=np.array(Image.open(dataset/record['labels']));row=dict(fold=fold,image=name,steps=step,**metrics(truth,labels),seconds=time.monotonic()-start,checkpoint_sha256=sha(checkpoint));dump(evaluation,row);rows.append(row)
   print(f'EVALUATED fold {fold} step {step}: IoU {row["iou"]:.5f}',flush=True)
   del model,gen,prediction;gc.collect();torch.cuda.empty_cache()
  truth=np.array(Image.open(dataset/record['labels']));h,w=gray.shape;canvas=Image.new('RGB',(4*w+100,h+140),'white');draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',28)
  draw.text((20,10),name+' - held out (15 train / 1 evaluation)',font=font,fill='black')
  for i,(title,im) in enumerate([('Normalized SEM',image),('Reviewed reference',mask_overlay(gray,truth)),(f'300 updates | IoU {rows[0]["iou"]:.3f}',mask_overlay(gray,maps[300])),(f'500 updates | IoU {rows[1]["iou"]:.3f}',mask_overlay(gray,maps[500]))]):
   x=20+i*(w+20);draw.text((x,66),title,font=font,fill='black');canvas.paste(Image.fromarray(im),(x,120))
  canvas.save(folder/'comparison.png')
  result=dict(fold=fold,image=name,metrics=rows,sampling_unique_totals={s:sum(c.values()) for s,c in coverage.items()},train_images=config['train_images'],held_out=name);dump(folder/'paired_result.json',result);results.append(result);report(out,results)
  print(f'COMPLETE FOLD {fold}; {len(results)}/5',flush=True)
 assert sha(ROOT/'checkpoints/default_model.json')==original_default
 dump(out/'verification.json',dict(completed_folds=len(results),same_trajectory=True,held_out_excluded=True,app_default_unchanged=True))
 print('COMPLETE '+str(out),flush=True)
if __name__=='__main__':main(Path(sys.argv[1]))
