"""Compare repaired SAM inputs without altering live analysis or user revisions."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from fiber_pores import deduplicate, prompts, select_prediction
from repair_struts import repair
from segment_first_pass import overlay
from trial_pore_methods import ROOT, overlap_metrics

PREVIOUS=ROOT/'outputs/generalization_trial/fiber_20260910_202011'
LABELS={'previous':'기존 · 가는 선 제거만','closed':'가까운 틈 메우기 → SAM',
        'connected':'방향·원본 확인 후 연결 → SAM','connected_regions':'연결 후 영역도 다시 탐색 → SAM'}


def proposals_from_image(pixels):
    threshold,_=cv2.threshold(pixels,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    solid=cv2.morphologyEx((pixels>threshold*.8).astype(np.uint8),cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
    void=solid==0;distance=ndi.distance_transform_edt(void);masks=[]
    seeds=peak_local_max(distance,min_distance=max(12,round(min(pixels.shape)*.035)),threshold_abs=8.4,exclude_border=False)
    standard=np.zeros(pixels.shape,np.int32)
    for i,(y,x) in enumerate(seeds):standard[y,x]=i+1
    prominent,_=ndi.label(h_maxima(distance,min(pixels.shape)*.02)&(distance>8.4))
    for markers in [standard,prominent]:
        regions=watershed(-distance,markers,mask=void,watershed_line=True)
        for label in range(1,int(markers.max())+1):
            mask=ndi.binary_fill_holes(regions==label)
            if .002<=mask.mean()<=.4:masks.append(mask)
    return deduplicate(masks)


def load_masks(path):
    with np.load(path,allow_pickle=False) as data:return [data[k].copy() for k in data.files]


def render(out,report):
    sections=[]
    for name,item in report['results'].items():
        rows=[]
        for method,data in item['methods'].items():
            refs=data['evaluation']['references']
            values=' / '.join(f"{r['best_iou']:.3f}" for r in refs) or '평가 기준 없음'
            outside=' / '.join(f"{100*r['candidate_outside_reference']:.1f}%" for r in refs) or '—'
            rows.append(f'<tr><td>{LABELS[method]}</td><td>{data["large_candidates"]}</td><td>{values}</td><td>{outside}</td></tr>')
        options=''.join(f'<option value="{k}">{v}</option>' for k,v in LABELS.items())
        refs_block='<h3>저장된 큰 pore 두 개와 비교</h3><p>왼쪽은 사용자 수정, 오른쪽은 각 기준과 가장 겹치는 후보입니다. 평가용으로 고른 결과이며 자동 최종 선택이 아닙니다.</p><img data-view="matches" src="'+name+'/previous/matches.png">' if name=='GF_1kx_1_BSE' else ''
        sections.append(f'''<section data-name="{name}"><h2>{name}</h2><p>원본에서 확인하며 연결한 곳: {item['repair']['accepted_connections']}곳. 분홍색은 연결 경로이며, 실제 전면 골격인지 확인이 필요합니다.</p><details><summary>연결 위치 확인</summary><img src="{name}/connections.png"></details><label>비교 방식 <select data-method>{options}</select></label> <label>결과 표시 <select data-candidate><option value="">전체 후보</option></select></label><p>후보 수에는 같은 pore의 경계 대안이 포함됩니다. 비교 시 하나씩 선택할 수 있습니다.</p><div class="grid"><figure><figcaption>원본</figcaption><img src="{name}/original.png"></figure><figure><figcaption>SAM에 전달한 이미지</figcaption><img data-view="input" src="{name}/previous/input.png"></figure><figure><figcaption>SAM 후보</figcaption><img data-view="result" src="{name}/previous/result.png"></figure></div><table><tr><th>방식</th><th>큰 후보 수</th><th>큰 pore 2개 · 최고 IoU</th><th>선택 후보 중 기준 바깥 면적</th></tr>{''.join(rows)}</table>{refs_block}</section>''')
    data=json.dumps({n:{k:v['large_candidates'] for k,v in r['methods'].items()} for n,r in report['results'].items()})
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>끊긴 골격 연결 · SAM 비교</title><style>body{font:15px 'Malgun Gothic',sans-serif;line-height:1.65;background:#edf3f3;color:#173e45;margin:25px auto;max-width:1700px;padding:0 18px}section{padding:20px;margin:20px 0;background:white;border-radius:12px}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}figure{margin:0}img{max-width:100%;display:block}select{padding:9px}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}.notice{padding:15px;background:#fff0d0}@media(max-width:900px){.grid{grid-template-columns:1fr}}</style><h1>끊긴 골격을 연결하면 큰 pore가 좋아질까?</h1><p>가는 선 제거만 한 입력, 가까운 틈을 메운 입력, 원본의 구조·방향을 확인해 연결한 입력을 비교했습니다.</p><p class="notice">기존 분석과 측정값에 적용하지 않은 실험입니다. 앞의 세 방식은 같은 자동 박스·포함 점을 사용해 입력 이미지의 영향을 비교합니다. 마지막 방식은 연결된 이미지에서 영역도 다시 찾습니다. 수동 기준은 SAM 실행 후 평가에만 사용했습니다. 두 영역의 최고 IoU는 전체 검출 정확도가 아닙니다.</p><p><a href="report.json">평가 결과</a> · <a href="plan.json">실험 조건</a></p>'''+''.join(sections)
    page+='''<script>const counts='''+data+''';document.querySelectorAll('section').forEach(section=>{const name=section.dataset.name,method=section.querySelector('[data-method]'),candidate=section.querySelector('[data-candidate]');function show(){const root=name+'/'+method.value+'/';section.querySelector('[data-view="result"]').src=root+(candidate.value?'candidates/'+candidate.value+'.png':'result.png')}function change(){candidate.replaceChildren(new Option('전체 후보',''));for(let i=1;i<=counts[name][method.value];i++)candidate.add(new Option('후보 '+i,i));section.querySelector('[data-view="input"]').src=name+'/'+method.value+'/input.png';const matches=section.querySelector('[data-view="matches"]');if(matches)matches.src=name+'/'+method.value+'/matches.png';show()}method.onchange=change;candidate.onchange=show;change()});</script></html>'''
    (out/'index.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    out=args.output or ROOT/'outputs/generalization_trial'/('strut_repair_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=False)
    refs_path=ROOT/'outputs/manual_edits/image_dffc9b43ff372f12__run_0002/revision_0002/entrance_candidates.npz'
    protected=[refs_path]+list(PREVIOUS.glob('*/masks/*.npz'))
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    plan=dict(model='SAM 2.1 Small',source_trial=str(PREVIOUS),closing_diameter=29,
              bridge_max_distance_fraction=.13,bridge_min_alignment=.4,bridge_min_original_support=.8,
              bridge_support_opening_diameter=7,bridge_corridor_halfwidth=12,bridge_width_range=[3,8],
              fixed_prompts=['closed','connected'],regenerated_prompts=['connected_regions'],
              evaluation_only_reference=str(refs_path),reference_prompting=False,large_area_fraction=[.015,.12],
              sources=['https://scikit-image.org/docs/stable/api/skimage.graph.html#skimage.graph.route_through_array',
                       'https://scikit-image.org/docs/stable/auto_examples/applications/plot_morphology.html'])
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    model=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(ROOT/'checkpoints/sam2.1_hiera_small.pt'),device='cuda',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model);results={};evaluated={}
    for name in ['GF_1kx_1_BSE','GF_1kx_1']:
        source=PREVIOUS/name;folder=out/name;folder.mkdir()
        gray=np.asarray(Image.open(source/'images/original.png'))
        images,repair_meta=repair(gray)
        Image.fromarray(gray).save(folder/'original.png')
        connections=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB);connections[images['bridges']>0]=[255,60,170]
        Image.fromarray(connections).save(folder/'connections.png')
        (folder/'connections.json').write_text(json.dumps(repair_meta,indent=2),encoding='utf-8')
        fixed=load_masks(source/'masks/prompts.npz')
        methods={'previous':load_masks(source/'masks/sam_simplified.npz')}
        results[name]=dict(repair={k:v for k,v in repair_meta.items() if k!='connections'},methods={})
        evaluated[name]={}
        for method in LABELS:
            target=folder/method;target.mkdir();(target/'candidates').mkdir()
            pixels=images[{'previous':'simplified','closed':'closed','connected':'connected','connected_regions':'connected'}[method]]
            Image.fromarray(pixels).save(target/'input.png')
            audit=[]
            if method!='previous':
                proposals=proposals_from_image(pixels) if method=='connected_regions' else fixed
                np.savez_compressed(target/'prompts.npz',**{f'proposal_{i}':m for i,m in enumerate(proposals)})
                print(name,method,len(proposals),'prompts',flush=True)
                selected=[]
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    predictor.set_image(cv2.cvtColor(pixels,cv2.COLOR_GRAY2RGB))
                    for i,proposal in enumerate(proposals):
                        box,points=prompts(proposal)
                        masks,scores,_=predictor.predict(box=box,point_coords=points,point_labels=np.ones(len(points),np.int32),multimask_output=True)
                        pick=select_prediction(masks,scores,proposal,gray)
                        if pick:
                            mask,metadata=pick;selected.append(mask)
                            audit.append(dict(proposal=i,box=box.tolist(),points=points.tolist(),**metadata))
                        if i%40==0:print(method,i+1,'/',len(proposals),flush=True)
                methods[method]=deduplicate(selected)
            masks=methods[method];large=[m for m in masks if .015<=m.mean()<=.12]
            evaluated[name][method]=large
            np.savez_compressed(target/'all_masks.npz',**{f'candidate_{i+1}':m for i,m in enumerate(masks)})
            np.savez_compressed(target/'large_masks.npz',**{f'candidate_{i+1}':m for i,m in enumerate(large)})
            (target/'prompts.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
            overlay(gray,large,numbered=True).save(target/'result.png')
            for i,m in enumerate(large):overlay(gray,[m]).save(target/'candidates'/f'{i+1}.png')
            results[name]['methods'][method]=dict(all_candidates=len(masks),large_candidates=len(large))
            print(name,method,'large',len(large),flush=True)
    # No reference mask is read until ALL image processing and inference have finished.
    with np.load(refs_path) as data:refs=[data['candidate_30'].copy(),data['candidate_31'].copy()]
    for name,item in results.items():
        gray=np.asarray(Image.open(out/name/'original.png'))
        for method,metrics in item['methods'].items():
            masks=evaluated[name][method]
            metrics['evaluation']=overlap_metrics(refs if name=='GF_1kx_1_BSE' else [],masks)
            if name=='GF_1kx_1_BSE':
                ids=[r['best_candidate_id'] for r in metrics['evaluation']['references']]
                best=[masks[i-1] for i in ids if i]
                comparison=Image.new('RGB',(gray.shape[1]*2,gray.shape[0]))
                comparison.paste(overlay(gray,refs,numbered=True),(0,0));comparison.paste(overlay(gray,best,numbered=True),(gray.shape[1],0))
                comparison.save(out/name/method/'matches.png')
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
    report=dict(status='exploratory_completed',protected_hashes=hashes,results=results)
    render(out,report)
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(results['GF_1kx_1_BSE']['methods']),flush=True)
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
