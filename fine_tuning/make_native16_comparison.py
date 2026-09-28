"""All sixteen images: base vs native-loss fine tuning, identical nested selection."""
from pathlib import Path
import sys,json,time,shutil,subprocess,gc,zipfile,html
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import torch,cv2
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'fine_tuning'));sys.path.insert(0,str(ROOT/'scripts'))
from train import sha,dump,CONFIG,save_csv
from sam2.build_sam import build_sam2
from pore_selection import TraceGenerator,select,label_map
from evaluate_nested import metrics,mask_overlay
from diagnose_boundary_post import boundary
OLD=ROOT/'fine_tuning/runs/boundary_model_post_20260923_132646'
NEW=ROOT/'fine_tuning/runs/loss_resolution5_20260923_135551'
PIEXTRA=ROOT/'fine_tuning/runs/pi5_extra_20260923_141827'
PICACHE=ROOT/'outputs/pi5_base_vs_native_finetuned_20260923'
FOLDS=list(range(1,17))
MISSING=[3,4,5,6,8,11,12,13,14]

def report(out,results):
    dump(out/'results.json',results)
    page='<!doctype html><meta charset="utf-8"><title>16 images: base vs fine-tuned</title><style>body{font:16px system-ui;max-width:1700px;margin:24px auto;padding:20px;line-height:1.6;background:#f5f6f8}article{background:white;padding:20px;margin:24px 0;border-radius:12px}img{width:100%}a{color:#125ab5}button{padding:8px 14px;cursor:pointer}small{color:#555}</style><h1>16 images: Original / Base SAM / Fine-tuned SAM</h1><p>Both predictions use normalization-only input and identical two-stage nested selection. Fine-tuned: native-resolution segmentation loss, 300 updates, 15:1 image-wise holdout. Each displayed image is excluded from its model training; related acquisitions may remain in training. The original SEM is used as the display background, with its footer preserved. No application model changed.</p><p>Click an image for its full-resolution file. Boundary-only versions are available under each image.</p><p><a href="comparison16_png.zip">Download all PNG images</a></p>'
    for r in results:
        fn='images/'+r['file'];bn=fn.replace('.png','_boundaries.png');page+=f'<article><h2>{html.escape(r["image"])}</h2><p><a href="{fn}">Full-resolution filled view</a> &nbsp; <a href="{bn}">Boundary-only view</a></p><a href="{fn}"><img loading="lazy" src="{fn}"></a><details><summary>Show boundary-only comparison</summary><a href="{bn}"><img loading="lazy" src="{bn}"></a></details></article>'
    page+=f'<p>Completed: {len(results)} / 16</p>'
    (out/'index.html').write_text(page,encoding='utf-8')

def run(out):
    out.mkdir(exist_ok=True);(out/'images').mkdir(exist_ok=True);(out/'masks').mkdir(exist_ok=True)
    if not (out/'plan.json').exists():
        extra=ROOT/'fine_tuning/runs'/('native16_extra_'+time.strftime('%Y%m%d_%H%M%S'));extra.mkdir();(extra/'source').mkdir()
        plan=json.loads((NEW/'plan.json').read_text());plan.update(run=str(extra),selected_folds=MISSING,model_directory=str(ROOT/'fine_tuning/models'/extra.name),checkpoint_steps=[300],purpose='Complete 16-image held-out visual comparison')
        Path(plan['model_directory']).mkdir();shutil.copy2(NEW/'source/worker.py',extra/'source/worker.py');dump(extra/'plan.json',plan)
        shutil.copy2(__file__,extra/'source'/Path(__file__).name)
        dump(out/'plan.json',dict(extra_run=str(extra),folds=FOLDS,selection='two_stage_nested',training_steps=300,loss='native-resolution segmentation; original low-resolution score target',default_sha256=sha(ROOT/'checkpoints/default_model.json')))
    task=json.loads((out/'plan.json').read_text());extra=Path(task['extra_run']);plan=json.loads((extra/'plan.json').read_text());dataset=Path(plan['dataset']);records=json.loads((dataset/'manifest.json').read_text())['records']
    torch.set_num_threads(4);cv2.setNumThreads(4)
    results=[]
    for f in FOLDS:
        record=records[f-1];name=record['name'];source=NEW if f in [1,7,10,15,16] else (PIEXTRA if f in [2,9] else extra)
        if not (source/f'fold_{f}/result.json').exists():
            print(f'TRAIN additional fold {f}: {name}',flush=True)
            subprocess.run([sys.executable,'-u',str(extra/'source/worker.py'),str(extra/'plan.json'),str(f)],check=True)
        config=json.loads((source/f'fold_{f}/config.json').read_text());assert name not in config['train_images'] and config['validation_images']==[name] and config['steps']==300
        image=np.array(Image.open(dataset/record['image']).convert('RGB'));gray=image[:,:,0];maps={};paths={}
        modeldir=Path(json.loads((source/'plan.json').read_text())['model_directory'])
        for key,checkpoint in [('base',Path(plan['base_checkpoint'])),('tuned',modeldir/f'fold_{f}.pt')]:
            saved=out/'masks'/f'{f:02d}_{key}.tif';paths[key]=dict(checkpoint=str(checkpoint),sha256=sha(checkpoint))
            if not saved.exists():
                cached=(OLD/f'fold_{f}/base_two_stage_nested.tif') if key=='base' else (NEW/f'fold_{f}/native_resolution.tif')
                if f in [1,2,7,9,10]:shutil.copy2(PICACHE/'masks'/f'{f:02d}_{key}.tif',saved)
                elif f in [15,16]:shutil.copy2(cached,saved)
                else:
                    print(f'INFERENCE {name}: {key}',flush=True);model=build_sam2(CONFIG,str(checkpoint),device='cuda',apply_postprocessing=False)
                    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                        gen=TraceGenerator(model,points_per_side=48,points_per_batch=4,pred_iou_thresh=.7,stability_score_thresh=.85,crop_n_layers=0,min_mask_region_area=0,output_mode='uncompressed_rle');gen.generate(image)
                    settings=record['source_preprocessing'];chosen=select(gen.pool,gray,settings.get('min_area_pixels',100),settings.get('min_contrast',8),'cuda','two_stage_nested')
                    Image.fromarray(label_map(chosen['masks'],gray.shape)).save(saved)
                    del model,gen,chosen;gc.collect();torch.cuda.empty_cache()
            maps[key]=np.array(Image.open(saved));assert maps[key].shape==gray.shape
        truth=np.array(Image.open(dataset/record['labels']));rows={key:metrics(truth,m) for key,m in maps.items()}
        # Display the original imported SEM, not the contrast-normalized inference image.
        original_path=Path(record['project']).parent/'input/normalized.png'
        original=np.array(Image.open(original_path).convert('RGB'));h,w=original.shape[:2];assert w==gray.shape[1] and h>=gray.shape[0]
        panels=[original]
        for key in ['base','tuned']:
            im=original.copy();im[:gray.shape[0]]=mask_overlay(original[:gray.shape[0],:,0],maps[key]);panels.append(im)
        font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',27);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',21)
        canvas=Image.new('RGB',(3*w+80,h+135),'white');d=ImageDraw.Draw(canvas);d.text((20,12),name+' | Same two-stage nested postprocessing',font=font,fill='black')
        titles=['Original SEM','Base SAM','Fine-tuned: native loss, 300 updates']
        for i,(title,im) in enumerate(zip(titles,panels)):
            x=20+i*(w+20);d.text((x,56),title,font=small,fill='black');canvas.paste(Image.fromarray(im),(x,90))
        d.text((20,h+101),'Fine-tuned model: this image excluded from training (15 train / 1 held out). Yellow: predicted pore area.',font=small,fill='black')
        filename=f'{f:02d}_{name}.png';canvas.save(out/'images'/filename)
        # Boundary-only image avoids obscuring the SEM texture with a fill.
        for i,key in enumerate(['base','tuned'],start=1):
            im=original.copy();roi=im[:gray.shape[0]];roi[boundary(maps[key]>0)]=[255,190,0];canvas.paste(Image.fromarray(im),(20+i*(w+20),90))
        d.rectangle((0,h+95,canvas.width,canvas.height),fill='white');d.text((20,h+101),'Boundary-only view. Same masks as the filled comparison; no smoothing or geometric resizing.',font=small,fill='black')
        canvas.save(out/'images'/filename.replace('.png','_boundaries.png'))
        result=dict(fold=f,image=name,file=filename,metrics=rows,models=paths,original_display=str(original_path),inference_image=str(dataset/record['image']),held_out=True)
        results.append(result);dump(out/'masks'/f'{f:02d}_result.json',result);report(out,results);print(f'COMPLETE IMAGE {name} ({len(results)}/16)',flush=True)
    report(out,results)
    with zipfile.ZipFile(out/'comparison16_png.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted((out/'images').glob('*.png')):z.write(path,'images/'+path.name)
    assert sha(ROOT/'checkpoints/default_model.json')==task['default_sha256']
    dump(out/'verification.json',dict(completed_images=len(results),held_out_excluded=True,postprocessing_identical=True,original_footer_preserved=True,app_unchanged=True))
    print('COMPLETE '+str(out),flush=True)

if __name__=='__main__':run(ROOT/'outputs/comparison16_base_vs_native_finetuned_20260923')
