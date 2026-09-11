"""Controlled GF comparison: opening diameter and Gaussian blur varied separately."""
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from fiber_pores import normalized_image, prompts, select_prediction, deduplicate
from segment_first_pass import overlay
from trial_pore_methods import ROOT, overlap_metrics
from trial_repaired_struts import load_masks

SOURCE=ROOT/'outputs/generalization_trial/fiber_20260910_202011'
CONDITIONS={
    'normalized':dict(label='밝기 정규화만',diameter=1,sigma=0,reuse='sam_points'),
    'open7':dict(label='가는 선 약하게 제거 · blur 없음',diameter=7,sigma=0),
    'open15':dict(label='가는 선 중간 제거 · blur 없음',diameter=15,sigma=0),
    'open15_blur':dict(label='가는 선 중간 제거 · blur 2',diameter=15,sigma=2),
    'open29':dict(label='기존 제거 강도 · blur 없음',diameter=29,sigma=0),
    'previous':dict(label='기존 제거 강도 · blur 2',diameter=29,sigma=2,reuse='sam_simplified'),
    'preserved':dict(label='골격 주변 원본 보존 · blur 없음',diameter=15,sigma=0,preserve=True,core_radius=6,protect_distance=8,otsu_multiplier=.8),
}


def input_image(gray,condition):
    pixels=normalized_image(gray)
    original=pixels.copy()
    size=condition['diameter']
    if size>1:pixels=cv2.morphologyEx(pixels,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size)))
    if condition['sigma']:pixels=cv2.GaussianBlur(pixels,(0,0),condition['sigma'])
    if condition.get('preserve'):
        threshold,_=cv2.threshold(original,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        core=ndi.distance_transform_edt(original>threshold*condition['otsu_multiplier'])>=condition['core_radius']
        protected=ndi.distance_transform_edt(~core)<=condition['protect_distance']
        pixels=np.where(protected,original,pixels).astype(np.uint8)
    return pixels


def render(out,report):
    sections=[]
    for name,item in report['results'].items():
        rows=[]
        for key,condition in CONDITIONS.items():
            method=item['methods'].get(key)
            if method:
                score=' / '.join(f"{r['best_iou']:.3f}" for r in method.get('evaluation',{}).get('references',[])) or '평가 전'
                if name!='GF_1kx_1_BSE':score='정답 기준 없음'
                rows.append(f'<tr><td>{condition["label"]}</td><td>{condition["diameter"]} px</td><td>{method["large_count"]}</td><td>{score}</td></tr>')
        options=''.join(f'<option value="{key}" {"selected" if key=="open15_blur" else ""}>{condition["label"]}</option>' for key,condition in CONDITIONS.items())
        matches='<details><summary>사용자가 저장한 큰 pore 두 개와 비교</summary><p>왼쪽: 사용자 수정 · 오른쪽: 평가 기준과 가장 겹치는 후보. 이 그림의 후보 선택에만 기준 외곽을 사용했으며 자동 최종 선택이 아닙니다.</p><img data-matches></details>' if name=='GF_1kx_1_BSE' and report['status']=='completed' else ''
        sections.append(f'''<section data-name="{name}"><h2>{name}</h2><select data-method>{options}</select> <select data-candidate><option value="">전체 후보</option></select><div class="grid"><figure><figcaption>정규화 원본</figcaption><img src="{name}/normalized/input.png"></figure><figure><figcaption>SAM 입력 · 클릭하면 원본 크기로 열립니다</figcaption><a data-input-link target="_blank"><img data-input></a></figure><figure><figcaption>SAM 결과 · 후보 하나씩 확인 가능</figcaption><img data-result></figure></div><details open><summary>경계 확대 비교</summary><div class="grid zoom"><figure><figcaption>정규화 원본</figcaption><img src="{name}/normalized/crop.png"></figure><figure><figcaption>선택한 전처리</figcaption><img data-crop></figure><figure><figcaption>제거된 밝은 구조</figcaption><img data-removed></figure></div></details><table><tr><th>조건</th><th>제거 필터 크기</th><th>큰 후보 수</th><th>큰 pore 2개 · 최고 IoU</th></tr>{''.join(rows)}</table>{matches}</section>''')
    counts={n:{k:m['large_count'] for k,m in x['methods'].items()} for n,x in report['results'].items()}
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>선 제거 강도와 blur 분리 비교</title><style>body{font:15px 'Malgun Gothic',sans-serif;line-height:1.65;background:#eef3f3;color:#173e45;max-width:1650px;margin:24px auto;padding:0 20px}section{background:white;padding:20px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img{display:block;max-width:100%;width:100%}.zoom img{image-rendering:auto}select{padding:10px;margin:8px 0}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}.notice{background:#fff0d5;padding:15px}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style><h1>선 제거 강도와 blur를 따로 비교</h1><p>같은 SAM 2.1 Small, 같은 자동 박스·포함 점, 같은 후보 필터를 사용했습니다. 경계 연결 처리는 넣지 않았습니다.</p><p class="notice">실험 결과입니다. 겹치는 후보는 같은 pore의 경계 대안일 수 있습니다. 최고 IoU는 저장된 큰 pore 두 개에 대한 후보 존재 여부를 보는 지표이며, 전체 정확도나 자동 선택 성능이 아닙니다. 기본 분석과 측정값에는 반영하지 않았습니다.</p><p><a href="report.json">평가 결과</a> · <a href="plan.json">실험 조건</a></p>'''+''.join(sections)
    page+='''<script>const counts='''+json.dumps(counts)+''';document.querySelectorAll('section').forEach(section=>{const name=section.dataset.name,method=section.querySelector('[data-method]'),candidate=section.querySelector('[data-candidate]');function show(){const root=name+'/'+method.value+'/';section.querySelector('[data-result]').src=counts[name][method.value]===undefined?root+'input.png':root+(candidate.value?'candidates/'+candidate.value+'.png':'result.png')}function change(){const root=name+'/'+method.value+'/';candidate.replaceChildren(new Option('전체 후보',''));for(let i=1;i<=(counts[name][method.value]||0);i++)candidate.add(new Option('후보 '+i,i));section.querySelector('[data-input]').src=root+'input.png';section.querySelector('[data-input-link]').href=root+'input.png';section.querySelector('[data-crop]').src=root+'crop.png';section.querySelector('[data-removed]').src=root+'removed_crop.png';const matches=section.querySelector('[data-matches]');if(matches)matches.src=root+'matches.png';show()}method.onchange=change;candidate.onchange=show;change()});</script></html>'''
    (out/'index.html').write_text(page,encoding='utf-8')


def main():
    out=ROOT/'outputs/generalization_trial'/('unblur_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir()
    protected=list(SOURCE.glob('*/masks/*.npz'))
    ref_file=ROOT/'outputs/manual_edits/image_dffc9b43ff372f12__run_0002/revision_0002/entrance_candidates.npz'
    protected.append(ref_file)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    plan=dict(conditions=CONDITIONS,source=str(SOURCE),model='SAM 2.1 Small',normalization_percentiles=[2,98],
              all_prompts_fixed=True,connections=False,postprocessing_unchanged=True,
              large_area_fraction=[.015,.12],reference_usage='post-inference evaluation only',
              display_crop_xyxy=[128,128,448,448],crop_usage='display only, never inference or scoring')
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    report=dict(status='running',results={},protected_hashes=hashes)
    arrays={};pools={}
    for name in ['GF_1kx_1_BSE','GF_1kx_1']:
        gray=np.asarray(Image.open(SOURCE/name/'images/original.png'))
        arrays[name]=gray;pools[name]={};report['results'][name]=dict(methods={})
        for key,condition in CONDITIONS.items():
            folder=out/name/key;folder.mkdir(parents=True);(folder/'candidates').mkdir()
            pixels=input_image(gray,condition)
            Image.fromarray(pixels).save(folder/'input.png')
            Image.fromarray(pixels[128:448,128:448]).save(folder/'crop.png')
            removed=cv2.subtract(normalized_image(gray),pixels)
            Image.fromarray(removed[128:448,128:448]).save(folder/'removed_crop.png')
        assert np.array_equal(input_image(gray,CONDITIONS['previous']),np.asarray(Image.open(SOURCE/name/'images/simplified.png')))
    render(out,report)
    print('PREVIEW',out/'index.html',flush=True)
    model=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(ROOT/'checkpoints/sam2.1_hiera_small.pt'),device='cuda',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model)
    for name,gray in arrays.items():
        proposals=load_masks(SOURCE/name/'masks/prompts.npz')
        prompt_data=[prompts(m) for m in proposals]
        (out/name/'fixed_prompts.json').write_text(json.dumps([dict(box=b.tolist(),points=p.tolist()) for b,p in prompt_data],indent=2))
        for key,condition in CONDITIONS.items():
            folder=out/name/key;pixels=input_image(gray,condition);audit=[]
            print(name,key,'start',flush=True)
            if 'reuse' in condition:
                selected=load_masks(SOURCE/name/'masks'/(condition['reuse']+'.npz'))
            else:
                selected=[]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    predictor.set_image(cv2.cvtColor(pixels,cv2.COLOR_GRAY2RGB))
                    for i,(proposal,(box,points)) in enumerate(zip(proposals,prompt_data)):
                        masks,scores,_=predictor.predict(box=box,point_coords=points,point_labels=np.ones(len(points),np.int32),multimask_output=True)
                        result=select_prediction(masks,scores,proposal,gray)
                        if result:
                            mask,metadata=result;selected.append(mask);audit.append(dict(proposal=i,**metadata))
                        if i%40==0:print(key,i+1,'/',len(proposals),flush=True)
                selected=deduplicate(selected)
            large=[m for m in selected if .015<=m.mean()<=.12]
            pools[name][key]=large
            np.savez_compressed(folder/'all_masks.npz',**{f'candidate_{i+1}':m for i,m in enumerate(selected)})
            np.savez_compressed(folder/'large_masks.npz',**{f'candidate_{i+1}':m for i,m in enumerate(large)})
            (folder/'selection.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
            overlay(gray,large,numbered=True).save(folder/'result.png')
            for i,m in enumerate(large):overlay(gray,[m]).save(folder/'candidates'/f'{i+1}.png')
            report['results'][name]['methods'][key]=dict(all_count=len(selected),large_count=len(large),reused='reuse' in condition)
            render(out,report)
            print(name,key,'done',len(large),flush=True)
    with np.load(ref_file) as data:refs=[data['candidate_30'].copy(),data['candidate_31'].copy()]
    for name,item in report['results'].items():
        for key,method in item['methods'].items():
            large=pools[name][key]
            method['evaluation']=overlap_metrics(refs if name=='GF_1kx_1_BSE' else [],large)
            if name=='GF_1kx_1_BSE':
                best=[large[row['best_candidate_id']-1] for row in method['evaluation']['references'] if row['best_candidate_id']]
                gray=arrays[name];comparison=Image.new('RGB',(gray.shape[1]*2,gray.shape[0]))
                comparison.paste(overlay(gray,refs,numbered=True),(0,0));comparison.paste(overlay(gray,best,numbered=True),(gray.shape[1],0))
                comparison.save(out/name/key/'matches.png')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    report['status']='completed';render(out,report)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report['results']['GF_1kx_1_BSE']['methods']),flush=True)
    print('COMPLETE',out/'index.html',flush=True)


if __name__=='__main__':main()
