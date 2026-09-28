"""Diagnose model vs selection effects on boundaries; deployed-model visual audit."""
from pathlib import Path
import sys,json,time,gc,csv,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import torch,cv2
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'fine_tuning'))
from sam2.build_sam import build_sam2
from train import CONFIG,dump,sha,save_csv
from pore_selection import TraceGenerator,select,decode_candidates,label_map
from segment_first_pass import candidates_from_masks
from evaluate_nested import metrics,mask_overlay
OUT=ROOT/'fine_tuning/runs'/('boundary_model_post_'+time.strftime('%Y%m%d_%H%M%S'))
DATA=ROOT/'fine_tuning/runs/clean16_methods_20260921_110028/dataset'
STAGES=['strict','two_stage','two_stage_nested'];KEYS=[f'{m}_{s}' for m in ['base','tuned'] for s in STAGES]
MAIN=['base_strict','base_two_stage_nested','tuned_strict','tuned_two_stage_nested']
def boundary(mask):
 b=mask & ~ndi.binary_erosion(mask);b[[0,-1],:]=False;b[:,[0,-1]]=False;return b

def bmetrics(a,b):
 ba,bb=boundary(a),boundary(b)
 if not ba.any() or not bb.any():return dict(bf1_1=0.,bf1_2=0.,bf1_4=0.,assd_px=None)
 da=ndi.distance_transform_edt(~ba)[bb];db=ndi.distance_transform_edt(~bb)[ba]
 result={'assd_px':float((da.mean()+db.mean())/2)}
 for t in [1,2,4]:
  p=float(np.mean(da<=t));r=float(np.mean(db<=t));result[f'bf1_{t}']=2*p*r/(p+r) if p+r else 0.
 return result

def match(gt,pred):
 ids,a=np.unique(gt,return_inverse=True);pids,b=np.unique(pred,return_inverse=True)
 joint=np.bincount(a.ravel()*len(pids)+b.ravel(),minlength=len(ids)*len(pids)).reshape(len(ids),len(pids));rr=np.flatnonzero(ids>0);cc=np.flatnonzero(pids>0);inter=joint[np.ix_(rr,cc)];iou=inter/np.maximum(1,joint.sum(1)[rr,None]+joint.sum(0)[None,cc]-inter)
 result={}
 if iou.size:
  x,y=linear_sum_assignment(-((iou>=.5)*(min(iou.shape)+1)+iou))
  for i,j in zip(x,y):
   if iou[i,j]>=.5:result[int(ids[rr[i]])]=int(pids[cc[j]])
 return result

def render(folder,gray,truth,maps):
 h,w=gray.shape;font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',24)
 panels=[('Reference',truth)]+[(k,maps[k]) for k in MAIN]
 canvas=Image.new('RGB',(5*(w+16)+16,h+70),'white');d=ImageDraw.Draw(canvas)
 for i,(name,labs) in enumerate(panels):x=16+i*(w+16);d.text((x,15),name,font=font,fill='black');canvas.paste(Image.fromarray(mask_overlay(gray,labs)),(x,54))
 canvas.save(folder/'overview.png')
 # Predetermined spatial crops, no selection using metric improvements.
 sheet=Image.new('RGB',(5*272+16,4*306+60),'white');draw=ImageDraw.Draw(sheet);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
 for j,name in enumerate(['SEM + reference']+MAIN):draw.text((16+j*272,12),name,font=small,fill='black')
 for row,(cx,cy) in enumerate([(256,192),(768,192),(256,576),(768,576)]):
  sl=np.s_[cy-128:cy+128,cx-128:cx+128];base=np.repeat(gray[:,:,None],3,axis=2);gb=boundary(truth>0)
  for col,key in enumerate([None]+MAIN):
   image=base.copy();image[gb]=[0,220,255]
   if key:image[boundary(maps[key]>0)]=[255,40,80]
   x=16+col*272;y=50+row*306;sheet.paste(Image.fromarray(image[sl]),(x,y));draw.text((x,y+262),f'Center {cx},{cy}; native pixels',font=small,fill='black')
 sheet.save(folder/'boundary_crops.png')

def report(results):
 rows=[x for r in results for x in r['metrics']];save_csv(OUT/'metrics.csv',rows)
 fields=['iou','bf1_2','common_bf1_2','common_assd_px','common_area_bias_pct']
 means={k:{f:float(np.mean([r[f] for r in rows if r['variant']==k and r[f] is not None])) for f in fields} for k in KEYS}
 summary=dict(completed=len(results),means=means,common_pores=sum(r['common_count'] for r in results),scope='Training-set diagnosis of the deployed model, not held-out generalization. Common-pore analysis conditions on IoU>=0.5 detection in all four primary conditions and excludes image-edge pores.',primary_comparison=MAIN)
 dump(OUT/'summary.json',summary)
 page='<!doctype html><meta charset="utf-8"><title>Boundary diagnosis</title><style>body{font:16px system-ui;max-width:1600px;margin:30px auto;padding:20px}td,th{padding:9px;border-bottom:1px solid #ddd}img{width:100%}</style><h1>Boundary diagnosis: model x postprocessing</h1><p>Base SAM and deployed 300-update model; identical normalization-only inputs, thresholds, area/contrast filters. Strict: .80/.92; two-stage: add .70/.85 rim-supported candidates; nested: additional parent/child replacement.</p><p>Boundary F1 uses pixel-center Euclidean distance tolerance of 2 pixels (also recorded at 1 and 4). ASSD is the equally weighted mean of the two directed mean boundary distances. Common-pore metrics match the same interior reference pores in all four primary conditions at IoU >= .5. Smaller distance and absolute area bias are better. Image means are equally weighted.</p><p>'+summary['scope']+'</p><table><tr><th>Variant</th><th>Area IoU %</th><th>Union boundary F1 %</th><th>Common pore boundary F1 %</th><th>Common distance px</th><th>Common signed area bias %</th></tr>'
 for k,m in means.items():page+=f'<tr><td>{k}</td>'+''.join(f'<td>{m[f]*(100 if f in ["iou","bf1_2","common_bf1_2"] else 1):.3f}</td>' for f in fields)+'</tr>'
 page+='</table><p>Union boundary metrics also respond to added/missed objects; use common-pore metrics for boundary shape. Cyan: reference boundary; red: predicted boundary. Crop locations fixed to four image quadrants before evaluation.</p>'
 for r in results:page+=f'<h2>{html.escape(r["image"])} - common pores {r["common_count"]}</h2><a href="fold_{r["fold"]}/overview.png"><img src="fold_{r["fold"]}/overview.png"></a><a href="fold_{r["fold"]}/boundary_crops.png"><img src="fold_{r["fold"]}/boundary_crops.png"></a>'
 (OUT/'index.html').write_text(page,encoding='utf-8')

def main():
 OUT.mkdir();(OUT/'source').mkdir();(ROOT/'outputs/boundary_diagnosis_active.txt').write_text(str(OUT))
 records=json.loads((DATA/'manifest.json').read_text())['records'];folds=[1,7,10,15,16]
 deployed=json.loads((ROOT/'checkpoints/default_model.json').read_text());paths={'base':ROOT/'checkpoints/sam2.1_hiera_small.pt','tuned':ROOT/'checkpoints'/deployed['checkpoint']}
 import shutil
 for file in [Path(__file__),ROOT/'scripts/pore_selection.py',ROOT/'scripts/pore_nested.py',ROOT/'scripts/pore_conservative.py',ROOT/'scripts/segment_first_pass.py']:shutil.copy2(file,OUT/'source'/file.name)
 dump(OUT/'plan.json',dict(folds=folds,models={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},settings=dict(points_per_side=48,points_per_batch=4,device='cuda',selection_stages=STAGES),reference_used_in_selection=False,app_model_changed=False))
 torch.set_num_threads(4);cv2.setNumThreads(4)
 for modelname,checkpoint in paths.items():
  model=build_sam2(CONFIG,str(checkpoint),device='cuda',apply_postprocessing=False)
  for f in folds:
   r=records[f-1];folder=OUT/f'fold_{f}';folder.mkdir(exist_ok=True);image=np.array(Image.open(DATA/r['image']).convert('RGB'));gray=image[:,:,0];settings=r['source_preprocessing'];area=settings.get('min_area_pixels',100);contrast=settings.get('min_contrast',8)
   print(f'INFERENCE {modelname} fold {f} {r["name"]}',flush=True)
   with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
    gen=TraceGenerator(model,points_per_side=48,points_per_batch=4,pred_iou_thresh=.7,stability_score_thresh=.85,crop_n_layers=0,min_mask_region_area=0,output_mode='uncompressed_rle');gen.generate(image)
   dump(folder/f'{modelname}_pool.json',gen.pool)
   strict=candidates_from_masks(decode_candidates(gen.pool,.8,.92,device='cuda'),gray,min_area=area,min_contrast=contrast)
   Image.fromarray(label_map([c['mask'] for c in strict],gray.shape)).save(folder/f'{modelname}_strict.tif')
   for stage in ['two_stage','two_stage_nested']:
    selected=select(gen.pool,gray,area,contrast,'cuda',stage);Image.fromarray(label_map(selected['masks'],gray.shape)).save(folder/f'{modelname}_{stage}.tif')
    dump(folder/f'{modelname}_{stage}_decisions.json',{k:v for k,v in selected.items() if k not in ['masks','strict']})
   print(f'DONE {modelname} fold {f}',flush=True);del gen,strict,selected;gc.collect();torch.cuda.empty_cache()
  del model;gc.collect();torch.cuda.empty_cache()
 results=[]
 for f in folds:
  r=records[f-1];folder=OUT/f'fold_{f}';gray=np.array(Image.open(DATA/r['image']));truth=np.array(Image.open(DATA/r['labels']));maps={k:np.array(Image.open(folder/f'{k}.tif')) for k in KEYS};matches={k:match(truth,m) for k,m in maps.items()};common=set.intersection(*(set(matches[k]) for k in MAIN));edges=set(np.unique(np.concatenate([truth[0],truth[-1],truth[:,0],truth[:,-1]])));common-=edges
  rows=[];details=[]
  for k,m in maps.items():
   valid=[i for i in sorted(common) if i in matches[k]];per=[]
   for i in valid:
    a=truth==i;b=m==matches[k][i];bm=bmetrics(a,b);bm['area_bias_pct']=100*(int(b.sum())-int(a.sum()))/int(a.sum());per.append(bm);details.append(dict(variant=k,reference_id=i,**bm))
   row=dict(fold=f,image=r['name'],variant=k,**metrics(truth,m),**bmetrics(truth>0,m>0),common_count=len(valid),**{'common_'+field:float(np.mean([x[field] for x in per])) if per else None for field in ['bf1_1','bf1_2','bf1_4','assd_px','area_bias_pct']});rows.append(row)
  save_csv(folder/'common_pore_boundaries.csv',details);render(folder,gray,truth,maps)
  result=dict(fold=f,image=r['name'],common_count=len(common),common_ids=sorted(common),metrics=rows);dump(folder/'result.json',result);results.append(result);report(results)
 print('COMPLETE '+str(OUT),flush=True)
if __name__=='__main__':main()
