"""GF-specific exploratory methods; all prompts are generated from image pixels."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

from fiber_pores import strut_proposals,normalized_image,prompts,select_prediction,deduplicate
from segment_first_pass import overlay
from trial_pore_methods import ROOT,reference_masks,overlap_metrics

LABELS={'baseline':'현재 자동 분석','regions':'굵은 골격 사이 영역','merged_regions':'큰 공간 중심으로 영역 찾기',
        'sam_box':'영역 박스 + 원본 SAM','sam_points':'영역 박스·내부 점 + 원본 SAM','sam_simplified':'영역 박스·내부 점 + 단순화 영상 SAM',
        'large_pool':'큰 pore 추가 후보 · 하나씩 검토','large_review':'겹친 후보의 대표만 선택 · 비교용'}
DATASETS={
    'GF_1kx_1_BSE':ROOT/'outputs/projects/image_dffc9b43ff372f12/runs/run_0004',
    'GF_1kx_1':ROOT/'outputs/projects/image_5d8e6a257fb13137/runs/run_0001'}


def render(out,results):
    sections=[]
    for name,item in results.items():
        options=''.join(f'<option value="{key}" {"selected" if key=="large_pool" else ""}>{LABELS[key]}</option>' for key in item['methods'])
        rows=[]
        for key,data in item['methods'].items():
            overlaps=' / '.join(f"{row['best_iou']:.3f}" for row in data.get('references',[])) or '임시 기준 없음'
            large=' / '.join(f"{row['best_iou']:.3f}" for row in data.get('user_large_references',{}).get('references',[])) or '비교 기준 없음'
            rows.append(f'<tr><td>{LABELS[key]}</td><td>{data["candidates"]}</td><td>{overlaps}</td><td>{large}</td></tr>')
        count=item['methods'].get('large_pool',{}).get('candidates',0)
        choices='<option value="">전체 후보</option>'+''.join(f'<option value="{i}">후보 {i}</option>' for i in range(1,count+1))
        initial='large_pool' if count else 'baseline'
        examples='<h3>사용자가 추가한 큰 pore와 비교</h3><p>왼쪽: 저장된 사용자 수정 / 오른쪽: 자동 생성 후보 중 겹침이 가장 큰 것. 이 두 후보를 골라 보여주는 데만 사용자 외곽을 사용했습니다. 자동 대표 선택 결과가 아닙니다.</p><img src="'+name+'/images/user_large_comparison.png">' if name=='GF_1kx_1_BSE' and count else ''
        sections.append(f'''<section><h2>{name}</h2><select id="{name}-method" onchange="chooseMethod('{name}')">{options}</select><select id="{name}-candidate" aria-label="큰 pore 후보 하나씩 보기" onchange="chooseCandidate('{name}')">{choices}</select><p>추가 후보는 서로 겹칠 수 있습니다. 후보 하나씩 보기로 외곽을 확인하세요. 후보 수는 최종 pore 개수가 아닙니다.</p><div class="pair"><img src="{name}/images/original.png" alt="원본"><div class="stack"><img id="{name}-result" src="{name}/images/{initial}.png" alt="선택한 방식"><img id="{name}-overlay" class="mask" hidden alt="선택한 후보 윤곽"></div></div><table><tr><th>방식</th><th>후보 수</th><th>이전 임시 외곽 2개</th><th>사용자 큰 pore 2개 · 최고 IoU</th></tr>{''.join(rows)}</table>{examples}<details><summary>큰 공간을 찾을 때 사용한 영상</summary><img src="{name}/images/simplified.png"></details></section>''')
    document='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>GF 구조별 pore 탐색 비교</title><style>body{font:15px 'Malgun Gothic',sans-serif;max-width:1500px;margin:25px auto;padding:0 20px;color:#25424a;background:#eef3f3;line-height:1.7}section{background:white;padding:20px;margin:20px 0;border-radius:12px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{max-width:100%;height:auto}td,th{padding:8px 15px;text-align:left;border-bottom:1px solid #ddd}select{padding:10px;margin:10px 0}aside{padding:15px;background:#fff0d5}@media(max-width:750px){.pair{grid-template-columns:1fr}}</style><h1>GF 구조별 pore 탐색 비교</h1><p>굵은 골격을 추린 뒤 빈 공간을 나누고, 그 영역에서 박스와 포함 점을 자동 생성해 SAM에 전달했습니다.</p><aside>탐색용 결과입니다. 큰 후보가 실제 pore라는 뜻은 아닙니다. GF BSE의 이전 assistant 임시 외곽 2개는 평가에만 사용했으며 사용자 정답이 아닙니다. 다른 GF 영상은 원본과 나란히 비교합니다. 기존 분석·측정값은 보존했습니다.</aside><p><a href="report.json">실행 결과</a> · <a href="plan.json">실행 조건</a></p>'''+''.join(sections)+'</html>'
    document=document.replace('</style>','.stack{position:relative}.stack img{display:block;width:100%}.stack .mask{position:absolute;top:0;left:0}.stack img[hidden]{display:none}</style>')
    document=document.replace('</html>','''<script>
function chooseMethod(name){const method=document.getElementById(name+'-method').value;document.getElementById(name+'-result').src=name+'/images/'+method+'.png';document.getElementById(name+'-overlay').hidden=true;const select=document.getElementById(name+'-candidate');select.disabled=method!=='large_pool';select.value=''}
function chooseCandidate(name){const value=document.getElementById(name+'-candidate').value;const overlay=document.getElementById(name+'-overlay');overlay.hidden=!value;document.getElementById(name+'-result').src=name+'/images/'+(value?'original':'large_pool')+'.png';if(value)overlay.src=name+'/images/large_candidates/'+value+'.png'}
</script></html>''')
    (out/'index.html').write_text(document,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    out=args.output or ROOT/'outputs/generalization_trial'/('fiber_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=True)
    plan=dict(radius_fractions=[.009,.018,.03],normalization=[2,98],region_area_fractions=[.002,.4],
              sam_model='SAM 2.1 Small',positive_points=3,box_padding_pixels=4,reference_coordinates_used_for_prompts=False,
              selection='original-image nonnegative ring contrast; proposal IoU >= .2; rank .7 IoU + .3 SAM score; no reference-based selection',
              sources=['https://scikit-image.org/docs/stable/api/skimage.segmentation','https://github.com/facebookresearch/sam2/blob/main/sam2/sam2_image_predictor.py'])
    (out/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    model=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(ROOT/'checkpoints/sam2.1_hiera_small.pt'),device='cuda',apply_postprocessing=False)
    predictor=SAM2ImagePredictor(model);results={}
    for name,baseline in DATASETS.items():
        report=json.loads((baseline/'report.json').read_text())
        gray=np.asarray(Image.open(report['image']).convert('L'))[:report['analysis_bottom_exclusive']]
        folder=out/name;(folder/'images').mkdir(parents=True);(folder/'masks').mkdir()
        Image.fromarray(gray).save(folder/'images/original.png')
        with np.load(baseline/'masks/entrance_candidates.npz') as data: previous=[data[k].copy() for k in data.files]
        proposals=[];merged=[]
        for fraction in plan['radius_fractions']:
            masks,_,_,_=strut_proposals(gray,fraction);proposals.extend(masks)
            masks,_,_,_=strut_proposals(gray,fraction,prominent=True);merged.extend(masks)
        proposals=deduplicate(proposals);merged=deduplicate(merged)
        all_proposals=deduplicate(proposals+merged)
        _,simplified,_,_=strut_proposals(gray,.018)
        Image.fromarray(simplified).save(folder/'images/simplified.png')
        methods=dict(baseline=previous,regions=proposals,merged_regions=merged)
        audit=[]
        for method,pixels,with_points in [('sam_box',normalized_image(gray),False),('sam_points',normalized_image(gray),True),('sam_simplified',simplified,True)]:
            print(f'{name}: {method}, {len(all_proposals)} boxes',flush=True)
            chosen=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                predictor.set_image(cv2.cvtColor(pixels,cv2.COLOR_GRAY2RGB))
                for i,proposal in enumerate(all_proposals):
                    box,points=prompts(proposal)
                    masks,scores,_=predictor.predict(box=box,point_coords=points if with_points else None,
                                                    point_labels=np.ones(len(points),np.int32) if with_points else None,multimask_output=True)
                    pick=select_prediction(masks,scores,proposal,gray)
                    if pick:
                        mask,meta=pick;chosen.append(mask)
                        audit.append(dict(method=method,proposal=i,box=box.tolist(),points=points.tolist() if with_points else [],**meta))
                    if (i+1)%40==0:print(f'  {i+1}/{len(all_proposals)}',flush=True)
            methods[method]=deduplicate(chosen)
        for key,masks in methods.items():
            overlay(gray,masks,numbered=True).save(folder/'images'/f'{key}.png')
            np.savez_compressed(folder/'masks'/f'{key}.npz',**{f'candidate_{i+1}':m for i,m in enumerate(masks)})
        np.savez_compressed(folder/'masks/prompts.npz',**{f'proposal_{i}':m for i,m in enumerate(all_proposals)})
        (folder/'prompt_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
        refs=reference_masks(name,gray.shape) if name=='GF_1kx_1_BSE' else []
        results[name]=dict(baseline=str(baseline),baseline_sha256=hashlib.sha256((baseline/'masks/entrance_candidates.npz').read_bytes()).hexdigest(),
                           methods={key:overlap_metrics(refs,masks) for key,masks in methods.items()})
        print(json.dumps(results[name]['methods']),flush=True)
    (out/'report.json').write_text(json.dumps(dict(status='exploratory',results=results),indent=2),encoding='utf-8')
    render(out,results);print(str(out/'index.html'),flush=True)


if __name__=='__main__':main()
