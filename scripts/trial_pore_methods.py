"""Exploratory shared-parameter PI/GF comparison; references are NOT ground truth."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
import torch
from sam2.build_sam import build_sam2

from segment_first_pass import ProgressGenerator, candidates_from_masks, overlay
from result_paths import read_artifact

ROOT=Path(__file__).resolve().parents[1]
NAMES=['PI35_5kx-4_bse','GF_1kx_1_BSE']
# Approximate polygons interpreted from the original images BEFORE comparing results.
# They describe desired projected outer regions including visible background structure.
# No reference coordinates are given to any automatic method.
REFERENCES={
    'PI35_5kx-4_bse':[
        dict(id='P1',note='central large pore',polygon=[[418,98],[433,84],[461,87],[490,101],[515,118],[540,125],[564,129],[588,143],[606,168],[619,191],[617,223],[601,254],[581,278],[554,289],[522,287],[492,278],[461,264],[439,246],[426,221],[419,189],[418,150]]),
        dict(id='P2',note='left central pore',polygon=[[275,169],[302,151],[336,140],[365,143],[383,157],[394,178],[400,209],[401,238],[392,267],[375,286],[347,288],[318,276],[294,260],[278,236],[269,210],[268,189]]),
        dict(id='P3',note='lower right large pore',polygon=[[702,481],[726,477],[750,481],[775,489],[801,501],[831,517],[852,540],[867,569],[875,600],[867,620],[847,631],[818,632],[790,626],[764,612],[740,595],[720,574],[707,548],[700,518]])],
    'GF_1kx_1_BSE':[
        dict(id='G1',note='upper left opening below broad foreground strut; provisional depth interpretation',polygon=[[100,73],[131,79],[160,88],[191,99],[214,111],[228,130],[231,141],[219,153],[181,151],[147,146],[116,139],[102,123]]),
        dict(id='G2',note='lower right opening enclosed by broad struts; background fibers included',polygon=[[604,532],[632,530],[676,535],[719,540],[759,544],[796,551],[785,562],[762,578],[737,597],[712,616],[690,624],[667,619],[646,605],[629,585],[615,560]]),
        dict(id='G3',evaluation=False,note='REJECTED during enlarged visual review: polygon covers foreground solid strut, not pore; retained for audit, excluded from all overlap scores',polygon=[[456,541],[478,532],[494,548],[506,582],[515,610],[518,636],[510,660],[499,676],[482,662],[469,640],[454,616],[444,589],[443,566]])]
}
METHODS={'baseline':'기존 SAM + 밝기 조건','normalized_sam':'밝기 정규화 + SAM',
         'coarse_components':'굵은 경계로 닫힌 영역','coarse_sam':'가는 구조 약화 + SAM'}
SETTINGS=dict(points_per_side=48,points_per_batch=8,pred_iou_thresh=.8,stability_score_thresh=.92,crop_n_layers=0,min_mask_region_area=0)


def normalize(gray):
    lo,hi=np.percentile(gray,[2,98])
    return np.clip((gray.astype(float)-lo)*255/max(hi-lo,1),0,255).astype(np.uint8)


def coarse_image(gray):
    # Shared radius relative to image dimensions, not inferred physical depth.
    radius=max(2,round(min(gray.shape)*.009))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
    opened=cv2.morphologyEx(normalize(gray),cv2.MORPH_OPEN,kernel)
    return cv2.GaussianBlur(opened,(0,0),1.5),radius


def coarse_components(coarse):
    _,solid=cv2.threshold(coarse,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    solid=cv2.morphologyEx(solid,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    labels,n=ndi.label(solid==0)
    masks=[]
    for i in range(1,n+1):
        m=ndi.binary_fill_holes(labels==i)
        if 100<=m.sum()<=coarse.size*.12:
            masks.append(m)
    masks.sort(key=lambda m:tuple(np.argwhere(m).mean(axis=0)))
    return masks,solid


def reference_masks(name,shape):
    masks=[]
    for item in REFERENCES[name]:
        if not item.get('evaluation',True): continue
        mask=np.zeros(shape,np.uint8)
        cv2.fillPoly(mask,[np.asarray(item['polygon'],np.int32)],1)
        masks.append(mask.astype(bool))
    return masks


def reference_overlay(gray,name):
    pixels=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    for item in REFERENCES[name]:
        pts=np.asarray(item['polygon'],np.int32)
        color=(40,230,255) if item.get('evaluation',True) else (255,85,85)
        cv2.polylines(pixels,[pts],True,color,2)
        x,y=pts.mean(axis=0).astype(int)
        cv2.putText(pixels,item['id'],(x-10,y),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,0,0),4)
        cv2.putText(pixels,item['id'],(x-10,y),cv2.FONT_HERSHEY_SIMPLEX,.6,color,1)
    return Image.fromarray(pixels)


def overlap_metrics(refs,masks):
    scores=np.zeros((len(refs),len(masks)))
    for r,ref in enumerate(refs):
        for c,mask in enumerate(masks): scores[r,c]=(ref&mask).sum()/(ref|mask).sum()
    rows=[]
    for r,ref in enumerate(refs):
        best=int(scores[r].argmax()) if len(masks) else None
        mask=masks[best] if best is not None else np.zeros(ref.shape,bool)
        rows.append(dict(reference_index=r,best_candidate_id=best+1 if best is not None else None,
                         best_iou=float(scores[r,best]) if best is not None else 0.,
                         reference_coverage=float((ref&mask).sum()/ref.sum()),
                         candidate_outside_reference=float((mask&~ref).sum()/mask.sum()) if mask.any() else None))
    matched=sum(scores[r,c]>=.5 for r,c in zip(*linear_sum_assignment(-scores))) if len(masks) else 0
    return dict(candidates=len(masks),reference_count=len(refs),unique_reference_matches_at_iou_05=int(matched),references=rows)


def brightness_audit(baseline,gray):
    """Reselect fixed original SAM masks: this does NOT rerun SAM on dark images."""
    metadata=json.loads(read_artifact(baseline,'raw_metadata.json').read_text())
    with np.load(read_artifact(baseline,'raw_masks.npz')) as data:
        raw=[dict(item,segmentation=data[f'mask_{i}']) for i,item in enumerate(metadata)]
    variants={'original':gray,'gain_0.6':(gray.astype(float)*.6).astype(np.uint8),
              'offset_minus_25':np.clip(gray.astype(float)-25,0,255).astype(np.uint8)}
    rows=[];reference={}
    for variant,pixels in variants.items():
        for method,input_image in [('fixed_8',pixels),('normalized_then_8',normalize(pixels))]:
            ids={item['raw_id'] for item in candidates_from_masks(raw,input_image)}
            if variant=='original':reference[method]=ids
            origin=reference[method]
            rows.append(dict(variant=variant,method=method,candidates=len(ids),
                             same_raw_id_jaccard=len(ids&origin)/len(ids|origin) if ids|origin else 1.))
    return dict(scope='only selection stability with fixed original SAM masks; NOT end-to-end segmentation robustness',rows=rows)


def review_pool(gray,baseline,proposals,path):
    extra=[m for m in proposals if max([(m&old).sum()/(m|old).sum() for old in baseline],default=0)<.5]
    pixels=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    for masks,color in [(baseline,(60,190,255)),(extra,(255,175,45))]:
        for mask in masks:
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(pixels,contours,-1,color,2)
    Image.fromarray(pixels).save(path)
    return extra


def comparison_panel(gray,name,methods,path):
    # Crops display every method in exactly the same coordinates.
    tiles=[]
    images={method:np.asarray(overlay(gray,masks,numbered=False)) for method,masks in methods.items()}
    reference=np.asarray(reference_overlay(gray,name))
    for item in REFERENCES[name]:
        pts=np.asarray(item['polygon'])
        x0,y0=np.maximum(pts.min(axis=0)-25,0)
        x1,y1=np.minimum(pts.max(axis=0)+25,[gray.shape[1],gray.shape[0]])
        row=[]
        for label,pixels in [(('Provisional ' if item.get('evaluation',True) else 'EXCLUDED: solid ')+item['id'],reference)]+list(images.items()):
            crop=Image.fromarray(pixels[y0:y1,x0:x1]).resize((260,260))
            tile=Image.new('RGB',(260,290),'#193239');tile.paste(crop,(0,30))
            ImageDraw.Draw(tile).text((7,8),label,fill='white');row.append(tile)
        tiles.append(row)
    canvas=Image.new('RGB',(260*5,290*len(tiles)))
    for r,row in enumerate(tiles):
        for c,tile in enumerate(row):canvas.paste(tile,(260*c,290*r))
    canvas.save(path)


def write_page(out,results):
    sections=[]
    for name,result in results.items():
        reference_ids=[r['id'] for r in REFERENCES[name] if r.get('evaluation',True)]
        nrefs=len(reference_ids)
        rows=[]
        for method,metrics in result['methods'].items():
            ious=' / '.join(f"{m['best_iou']:.2f}" for m in metrics['references'])
            rows.append(f"<tr><td>{METHODS[method]}</td><td>{metrics['candidates']}</td><td>{ious}</td><td>{metrics['unique_reference_matches_at_iou_05']} / {nrefs}</td></tr>")
        buttons=''.join(f'<option value="{m}">{label}</option>' for m,label in METHODS.items())
        audit_rows=''.join(f"<tr><td>{r['variant']}</td><td>{r['method']}</td><td>{r['candidates']}</td><td>{r['same_raw_id_jaccard']:.2f}</td></tr>" for r in result['brightness_audit']['rows'])
        sections.append(f'''<section><h2>{name}</h2><p>청록 외곽은 사람이 확인하기 전의 임시 기준입니다.</p>
        <table><tr><th>방식</th><th>전체 후보 수</th><th>{' / '.join(reference_ids)}과 최고 IoU</th><th>IoU ≥ 0.5 일대일 대응</th></tr>{''.join(rows)}</table>
        <p><label>전체 결과 선택 <select onchange="document.getElementById('{name}-result').src='{name}/images/'+this.value+'.png'">{buttons}</select></label></p>
        <div class="pair"><img src="{name}/images/reference.png" alt="원본과 임시 외곽"><img id="{name}-result" src="{name}/images/baseline.png" alt="선택한 방법 결과"></div>
        <details open><summary>대표 영역 확대 비교</summary><img src="{name}/images/comparison.png" alt="같은 영역의 네 가지 결과"></details>
        <details><summary>전처리 영상과 굵은 경계</summary><div class="pair"><img src="{name}/images/coarse_input.png"><img src="{name}/images/solid_boundary.png"></div></details>
        <details><summary>기존 결과를 보존한 보완 검토 후보 {result['supplemental_review_count']}개</summary><p>파랑: 기존 후보. 주황: 새 방식의 후보 중 기존 후보와 IoU 0.5 미만인 영역입니다.
        겹침·오검출을 정리하기 전의 검토용 후보군이며, 합쳐서 pore 개수나 면적 분포를 계산하지 않았습니다.</p><img src="{name}/images/review_pool.png"></details>
        <details><summary>후보 선택 조건의 밝기 민감도</summary><p>기존 SAM 마스크를 고정하고 후보를 다시 선택한 검사입니다. SAM 자체의 밝기 민감도를 검증한 것은 아닙니다.
        gain_0.6은 밝기값 60%, offset_minus_25는 25 감소 후 0에서 잘림입니다. Jaccard 1은 해당 조건의 원본과 선택한 마스크 ID가 모두 같다는 뜻입니다.</p><table><tr><th>밝기 변화</th><th>선택 조건</th><th>후보 수</th><th>ID Jaccard</th></tr>{audit_rows}</table></details></section>''')
    page=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>PI · GF 방식 비교 실험</title>
    <style>body{{font:15px 'Malgun Gothic',sans-serif;background:#edf3f3;color:#203e43;max-width:1400px;margin:25px auto;padding:0 16px;line-height:1.7}}section{{background:white;padding:20px;margin:20px 0;border-radius:10px}}img{{max-width:100%;height:auto}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;border-bottom:1px solid #dbe4e5;padding:9px}}aside{{background:#fff0d4;padding:15px}}select{{padding:8px}}a{{color:#076d70}}details{{margin:18px 0}}summary{{cursor:pointer}}@media(max-width:700px){{.pair{{grid-template-columns:1fr}}table{{font-size:11px}}}}</style>
    <h1>PI · GF 방식 비교 실험</h1><aside>검증 전 실험입니다. 외곽 6개는 assistant가 원본을 보고 그린 근사 기준이며 사용자 정답이 아닙니다.
    GF는 앞뒤 골격 해석도 불확실합니다. IoU는 이 임시 외곽과의 겹침이며 정확도·누락률·오검출률로 해석하지 않습니다.
    기준 외 영역은 평가하지 않았습니다. 후보 수 증가는 성능 개선의 증거가 아닙니다.</aside>
    <p><strong>기준 검토:</strong> 초안 G3는 확대 확인에서 pore가 아닌 골격 표면으로 확인되어 모든 방식의 평가에서 제외했습니다.
    원래 좌표와 빨간 외곽을 남겼으며 새 기준으로 교체하지 않았습니다. 평가에 사용한 임시 외곽은 PI 3개, GF 2개입니다.</p>
    <p>PI와 GF에 같은 설정을 적용했습니다. 임시 외곽은 자동 분할의 입력으로 사용하지 않았습니다.
    굵은 경계 방식은 가는 밝은 구조를 약화시키는 영상 처리이며 실제 깊이를 판별하는 모델이 아닙니다.</p>
    <p><strong>판단:</strong> 이번 두 이미지에서는 가는 구조 약화 + SAM이 GF의 큰 pore 일부를 새로 잡았지만 PI의 대표 pore 하나를 놓쳤습니다.
    공통 기본값으로 교체하기에는 부족합니다. 기존 결과를 보존하고 추가 후보를 검토하는 용도로만 비교합니다.</p>
    <p><a href="report.json">조건과 겹침 지표 JSON</a> · <a href="references.json">검토할 임시 외곽 좌표</a></p>{''.join(sections)}</html>'''
    (out/'index.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--reuse',action='store_true');args=parser.parse_args()
    out=args.output or ROOT/'outputs/generalization_trial'/time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=True)
    (out/'references.json').write_text(json.dumps(dict(status='assistant provisional; user validation required',input_to_algorithms=False,images=REFERENCES),indent=2),encoding='utf-8')
    (out/'plan.json').write_text(json.dumps(dict(datasets=NAMES,methods=METHODS,settings=SETTINGS,normalization_percentiles=[2,98],opening_radius_fraction=.009,closing_diameter=11,reference_iou_threshold=.5,scope='exploratory on 2 images; no tuning against reference polygons'),indent=2,ensure_ascii=False),encoding='utf-8')
    results={};model=None
    for name in NAMES:
        baseline=ROOT/'outputs'/f'{name}_first_pass'
        report=json.loads((baseline/'report.json').read_text())
        gray=np.asarray(Image.open(ROOT/report['image']).convert('L'))[:report['analysis_bottom_exclusive']]
        folder=out/name;(folder/'images').mkdir(parents=True,exist_ok=True);(folder/'masks').mkdir(exist_ok=True)
        with np.load(read_artifact(baseline,'entrance_candidates.npz')) as data:
            methods={'baseline':[data[k].copy() for k in sorted(data.files,key=lambda k:int(k.split('_')[-1]))]}
        coarse,radius=coarse_image(gray)
        methods['coarse_components'],solid=coarse_components(coarse)
        Image.fromarray(coarse).save(folder/'images/coarse_input.png');Image.fromarray(solid).save(folder/'images/solid_boundary.png')
        reference_overlay(gray,name).save(folder/'images/reference.png')
        for method,input_image in [('normalized_sam',normalize(gray)),('coarse_sam',coarse)]:
            saved=folder/'masks'/f'{method}.npz'
            if args.reuse and saved.exists():
                with np.load(saved) as data: methods[method]=[data[k].copy() for k in data.files]
                continue
            print(f'{name}: {method}',flush=True)
            if model is None:model=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(ROOT/'checkpoints/sam2.1_hiera_small.pt'),device='cuda',apply_postprocessing=False)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                raw=ProgressGenerator(model,**SETTINGS).generate(cv2.cvtColor(input_image,cv2.COLOR_GRAY2RGB))
            # Evaluate contrast on each method's actual processing image.
            selected=candidates_from_masks(raw,input_image)
            methods[method]=[item['mask'] for item in selected]
            np.savez_compressed(saved,**{f'candidate_{i+1}':m for i,m in enumerate(methods[method])})
            (folder/f'{method}_metadata.json').write_text(json.dumps(dict(raw_count=len(raw),selected_count=len(selected))),encoding='utf-8')
        methods={m:methods[m] for m in METHODS}
        for method,masks in methods.items():
            overlay(gray,masks,numbered=True).save(folder/'images'/f'{method}.png')
            np.savez_compressed(folder/'masks'/f'{method}.npz',**{f'candidate_{i+1}':m for i,m in enumerate(masks)})
        refs=reference_masks(name,gray.shape)
        extra=review_pool(gray,methods['baseline'],methods['coarse_sam'],folder/'images/review_pool.png')
        np.savez_compressed(folder/'masks/supplemental_review.npz',**{f'candidate_{i+1}':m for i,m in enumerate(extra)})
        results[name]=dict(methods={m:overlap_metrics(refs,masks) for m,masks in methods.items()},opening_radius_pixels=radius,
                           supplemental_review_count=len(extra),brightness_audit=brightness_audit(baseline,gray),
                           baseline_sha256=hashlib.sha256(read_artifact(baseline,'entrance_candidates.npz').read_bytes()).hexdigest())
        comparison_panel(gray,name,methods,folder/'images/comparison.png')
        print(json.dumps({m:dict(count=v['candidates'],ious=[round(r['best_iou'],3) for r in v['references']]) for m,v in results[name]['methods'].items()}),flush=True)
    result=dict(status='exploratory; provisional references, not ground truth',results=results,output=str(out))
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8');write_page(out,results)
    print(str(out/'index.html'),flush=True)


if __name__=='__main__':main()
