"""Fixed shared-strength comparison, reusing the previous provisional references."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw
import torch
from sam2.build_sam import build_sam2

from trial_pore_methods import ROOT,NAMES,REFERENCES,SETTINGS,reference_masks,reference_overlay,overlap_metrics
from pore_preprocessing import preprocessing_config,prepare_image
from segment_first_pass import candidates_from_masks,overlay,ProgressGenerator
from result_paths import read_artifact

PREVIOUS=ROOT/'outputs/generalization_trial/20260910_183110'
METHODS={'medium':'이전 coarse · 중간','weak':'약하게 · 경계 보존','strong':'강하게 · 가는 구조 제거','detail':'중간 + 원본 20%'}


def panels(gray,name,methods,path):
    refs=[r for r in REFERENCES[name] if r.get('evaluation',True)]
    images={'reference':np.asarray(reference_overlay(gray,name)),**{k:np.asarray(overlay(gray,v)) for k,v in methods.items()}}
    canvas=Image.new('RGB',(250*len(images),280*len(refs)),'#17383c')
    draw=ImageDraw.Draw(canvas)
    for r,item in enumerate(refs):
        points=np.asarray(item['polygon']);x0,y0=np.maximum(points.min(axis=0)-25,0);x1,y1=np.minimum(points.max(axis=0)+25,[gray.shape[1],gray.shape[0]])
        for c,(key,pixels) in enumerate(images.items()):
            crop=Image.fromarray(pixels[y0:y1,x0:x1]);crop.thumbnail((246,246))
            canvas.paste(crop,(c*250+(250-crop.width)//2,r*280+30+(246-crop.height)//2))
            draw.text((c*250+7,r*280+7),item['id']+' | '+key,fill='white')
    canvas.save(path)


def page(out,results):
    parts=[]
    for name,result in results.items():
        rows=''.join('<tr><td>'+METHODS[k]+'</td><td>'+str(v['candidates'])+'</td><td>'+' / '.join(f"{r['best_iou']:.2f}" for r in v['references'])+'</td></tr>' for k,v in result['methods'].items())
        options=''.join(f'<option value="{k}">{label}</option>' for k,label in METHODS.items())
        parts.append(f'''<section><h2>{name}</h2><table><tr><th>방식</th><th>전체 후보 수</th><th>임시 기준별 최고 IoU</th></tr>{rows}</table><p><select onchange="document.getElementById('{name}-mask').src='{name}/images/'+this.value+'.png';document.getElementById('{name}-input').src='{name}/images/'+this.value+'_input.png'">{options}</select></p><div class="pair"><img src="{name}/images/reference.png" alt="원본과 임시 기준"><img id="{name}-mask" src="{name}/images/medium.png" alt="선택한 전처리의 후보"></div><details><summary>실제로 SAM에 넣은 영상</summary><img id="{name}-input" src="{name}/images/medium_input.png"></details><h3>같은 위치에서 비교</h3><img src="{name}/images/comparison.png"></section>''')
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Coarse SAM 개선 비교</title><style>body{{font:15px 'Malgun Gothic',sans-serif;max-width:1350px;margin:25px auto;padding:0 18px;color:#25424a;background:#edf3f3;line-height:1.7}}section{{padding:20px;background:white;border-radius:10px;margin:20px 0}}img{{max-width:100%;height:auto}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:10px;border-bottom:1px solid #dbe5e6;text-align:left}}select{{padding:8px}}aside{{background:#fff0d4;padding:15px}}a{{color:#087373}}@media(max-width:700px){{.pair{{grid-template-columns:1fr}}}}</style><h1>Coarse SAM 개선 비교</h1><p>정규화는 유지하고 가는 구조 약화 강도와 원본 밝기 정보 혼합을 비교했습니다. PI와 GF에 같은 설정을 사용했습니다.</p><aside>검토 전 실험입니다. 이전과 동일한 assistant 임시 외곽(PI 3개, GF 2개)을 평가에만 사용했습니다. 제외된 G3는 사용하지 않았습니다. 전처리 강도는 깊이 판정이 아니며, 새 후보·겹침 지표가 실제 정확도를 보장하지 않습니다.</aside><p><a href="report.json">결과 JSON</a> · <a href="plan.json">실행 조건</a></p>{''.join(parts)}</html>'''
    text=text.replace('<aside>','<p><strong>이번 비교에서는 강도 조절과 원본 혼합만으로 GF의 누락이 해결되지 않았습니다. 이전 coarse 중간 설정을 비교 기준으로 유지합니다.</strong></p><aside>')
    (out/'index.html').write_text(text,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--reuse',action='store_true');args=parser.parse_args()
    out=args.output or ROOT/'outputs/generalization_trial'/('coarse_v2_'+time.strftime('%Y%m%d_%H%M%S'))
    out.mkdir(parents=True,exist_ok=True)
    configs={key:preprocessing_config('coarse',key) for key in METHODS}
    (out/'plan.json').write_text(json.dumps(dict(settings=SETTINGS,preprocessing=configs,references_from=str(PREVIOUS/'references.json'),reference_polygons_used_as_prompts=False,fixed_before_this_run=True),indent=2),encoding='utf-8')
    results={};model=None
    for name in NAMES:
        baseline=ROOT/'outputs'/f'{name}_first_pass'
        report=json.loads((baseline/'report.json').read_text())
        gray=np.asarray(Image.open(ROOT/report['image']).convert('L'))[:report['analysis_bottom_exclusive']]
        folder=out/name;(folder/'images').mkdir(parents=True,exist_ok=True);(folder/'masks').mkdir(exist_ok=True)
        reference_overlay(gray,name).save(folder/'images/reference.png')
        methods={}
        for key,config in configs.items():
            pixels,metadata=prepare_image(gray,config)
            Image.fromarray(pixels).save(folder/'images'/f'{key}_input.png')
            saved=folder/'masks'/f'{key}.npz'
            cached=PREVIOUS/name/'masks/coarse_sam.npz' if key=='medium' else saved
            if key=='medium' or (args.reuse and saved.exists()):
                with np.load(cached) as data:methods[key]=[data[k].copy() for k in data.files]
            else:
                print(f'{name}: {key}',flush=True)
                if model is None:model=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(ROOT/'checkpoints/sam2.1_hiera_small.pt'),device='cuda',apply_postprocessing=False)
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    raw=ProgressGenerator(model,**SETTINGS).generate(cv2.cvtColor(pixels,cv2.COLOR_GRAY2RGB))
                selected=candidates_from_masks(raw,pixels)
                methods[key]=[item['mask'] for item in selected]
                metadata.update(raw_count=len(raw),selected_count=len(selected))
                # Preserve raw masks to allow later threshold changes without rerunning SAM.
                (folder/'sam_raw'/key).mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/'sam_raw'/key/'raw_masks.npz',**{f'mask_{i}':a['segmentation'] for i,a in enumerate(raw)})
                (folder/'sam_raw'/key/'raw_metadata.json').write_text(json.dumps([{k:v for k,v in a.items() if k!='segmentation'} for a in raw]),encoding='utf-8')
            np.savez_compressed(saved,**{f'candidate_{i+1}':m for i,m in enumerate(methods[key])})
            overlay(gray,methods[key],numbered=True).save(folder/'images'/f'{key}.png')
            (folder/f'{key}_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        results[name]=dict(methods={k:overlap_metrics(reference_masks(name,gray.shape),v) for k,v in methods.items()},baseline_sha256=hashlib.sha256(read_artifact(baseline,'entrance_candidates.npz').read_bytes()).hexdigest())
        panels(gray,name,methods,folder/'images/comparison.png')
        print(json.dumps({k:dict(count=v['candidates'],ious=[round(r['best_iou'],3) for r in v['references']]) for k,v in results[name]['methods'].items()}),flush=True)
    (out/'report.json').write_text(json.dumps(dict(status='exploratory; provisional references',results=results),indent=2),encoding='utf-8');page(out,results)
    print(str(out/'index.html'),flush=True)


if __name__=='__main__':main()
