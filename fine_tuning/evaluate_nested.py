"""Replay saved candidate pools through the experimental nested-pore resolver.

No training or SAM inference. CPU timings measure this added pass only, on pools
previously produced by GPU inference. References are loaded after selection.
"""
import argparse
import csv
from datetime import datetime
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import time

import cv2
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

from nested_candidates import DEFAULTS, prepare_candidates, resolve_nested

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ('relaxed', 'relaxed_nested', 'two_stage', 'two_stage_nested')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(truth, prediction):
    a, b = truth > 0, prediction > 0
    tp, fn, fp = (int(x.sum()) for x in (a & b, a & ~b, ~a & b))
    total = tp+fn
    if not total:
        raise ValueError('Reference is empty.')
    # Compact IDs before computing a contingency table; IDs may have gaps.
    tids, ti = np.unique(truth, return_inverse=True)
    pids, pi = np.unique(prediction, return_inverse=True)
    joint = np.bincount(ti.ravel()*len(pids)+pi.ravel(), minlength=len(tids)*len(pids)).reshape(len(tids),len(pids))
    tr, pr = np.flatnonzero(tids>0), np.flatnonzero(pids>0)
    inter = joint[np.ix_(tr,pr)]
    iou = inter / np.maximum(1, joint.sum(axis=1)[tr,None]+joint.sum(axis=0)[None,pr]-inter)
    matches = 0
    if iou.size:
        bonus = min(iou.shape)+1
        x,y=linear_sum_assignment(-((iou>=.5)*bonus+iou))
        matches=int(np.count_nonzero(iou[x,y]>=.5))
    return dict(iou=tp/(tp+fn+fp), dice=2*tp/(2*tp+fn+fp),
                missed_area_pct=100*fn/total, extra_area_pct=100*fp/total,
                relative_area_error_pct=100*abs(fp-fn)/total,
                reference_count=len(tr), predicted_count=len(pr), matched_iou50=matches,
                instance_f1=2*matches/max(1,len(tr)+len(pr)))


def mask_overlay(image, labels):
    result=np.repeat(image[:,:,None],3,axis=2)
    occupied=labels>0
    result[occupied]=np.rint(.60*result[occupied]+.40*np.array([255,215,0])).astype(np.uint8)
    for pid in np.unique(labels):
        if not pid: continue
        mask=(labels==pid).astype(np.uint8)
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result,contours,-1,(255,190,0),1)
    return result


def errors(image, truth, prediction):
    result=np.repeat((image*.55).astype(np.uint8)[:,:,None],3,axis=2)
    result[(truth>0)&(prediction==0)]=[0,220,255]
    result[(truth==0)&(prediction>0)]=[255,0,170]
    return result


def plot(folder,image,truth,maps,rows):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained')
    axes[0,0].imshow(mask_overlay(image,truth));axes[0,0].set_title('Reviewed reference')
    axes[1,0].imshow(image,cmap='gray',vmin=0,vmax=255);axes[1,0].set_title('Saved normalized SEM input')
    for col,v in enumerate(('two_stage','two_stage_nested'),1):
        m=next(r for r in rows if r['variant']==v)
        axes[0,col].imshow(mask_overlay(image,maps[v]));axes[0,col].set_title(f"{v} | area IoU {m['iou']:.3f} | F1 {m['instance_f1']:.3f}")
        axes[1,col].imshow(errors(image,truth,maps[v]));axes[1,col].set_title(f"Missed {m['missed_area_pct']:.1f}% | extra {m['extra_area_pct']:.1f}%")
    for ax in axes.flat:ax.axis('off')
    fig.suptitle(rows[0]['image']+' | gold fill: pore area; cyan: missed reference; magenta: outside reference')
    fig.savefig(folder/'comparison.png',dpi=130);plt.close(fig)
    if rows[0]['fold']==10:
        cases=[('A',324,111),('B',514,182),('C',506,488),('D',465,0)]
        fig,axes=plt.subplots(4,4,figsize=(13,13),layout='constrained')
        for row,(name,x,y) in enumerate(cases):
            sl=np.s_[y:y+280,x:x+280]
            panels=[np.repeat(image[:,:,None],3,axis=2),mask_overlay(image,truth),mask_overlay(image,maps['two_stage']),mask_overlay(image,maps['two_stage_nested'])]
            for col,panel in enumerate(panels):
                axes[row,col].imshow(panel[sl],interpolation='nearest')
                axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
                if row==0:axes[row,col].set_title(['SEM','Reviewed reference','Original Two-stage','Automatic nested selection'][col],fontsize=10)
            axes[row,0].set_ylabel(name)
            axes[row,0].plot([12,62],[262,262],color='white',lw=3)
            axes[row,0].text(12,252,'50 px',color='white',fontsize=8)
        fig.suptitle('PI35_5kx | gold fill: pore area; same 280 x 280 pixel crops')
        fig.savefig(folder/'cases.png',dpi=160);plt.close(fig)


def report(out, results):
    rows=[r for result in results for r in result['metrics']]
    with (out/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fields=('iou','instance_f1','missed_area_pct','extra_area_pct','relative_area_error_pct')
    means={v:{k:float(np.mean([r[k] for r in rows if r['variant']==v])) for k in fields} for v in VARIANTS}
    paired={}
    for base in ('relaxed','two_stage'):
        delta=[next(r['iou'] for r in x['metrics'] if r['variant']==base+'_nested')-next(r['iou'] for r in x['metrics'] if r['variant']==base) for x in results]
        paired[base]=dict(improved=sum(d>1e-10 for d in delta),worse=sum(d< -1e-10 for d in delta),unchanged=sum(abs(d)<=1e-10 for d in delta))
    summary=dict(completed_images=len(results),macro_image_means=means,area_iou_changes=paired,
                 mean_preparation_seconds=float(np.mean([r['prepare_seconds'] for r in results])),
                 mean_resolution_seconds={v:float(np.mean([r['resolution_seconds'][v] for r in results])) for v in ('relaxed','two_stage')},
                 reference_used_for_selection=False,app_defaults_changed=False,
                 evaluation_status='Development replay of previously inspected data, not an independent test.')
    dump(out/'summary.json',summary)
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Nested pore selection</title>
<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:16px;line-height:1.6}img{width:100%}td,th{padding:8px;border-bottom:1px solid #bbb}a{color:#075a9d}</style>
<h1>큰 pore / 내부 pore 자동 선택 — 실험 결과</h1>
<p>기존 Relaxed와 Two-stage 마스크 각각에 동일한 교체 단계를 적용했다. 내부 후보의 경계 근거, 후보 사이 밝은 영역의 폭과 밝기 일관성이 함께 충분할 때만 부모를 교체한다. 큰 후보와 내부 후보를 비교할 때 reference는 사용하지 않았다.</p>
<p>저장된 GPU SAM 후보를 재사용했다. 재학습·재추론은 없으며 기록된 CPU 시간은 추가 후보 준비와 교체 처리만 포함한다. 앱 기본값은 바꾸지 않았다. 이미 검토한 PI35를 포함한 개발 데이터 재평가이므로 독립 검증 성능이 아니다.</p>
<p>후보 경계는 밝기 변화로 검사하며, 밝은 영역은 두께를 가진 내부 핵심 영역에서 주변 표면 대비 밝은 픽셀 비율로 검사한다. 미세한 pore 후보가 없거나 경계가 부정확하면 이 단계만으로 복구하지 못한다. 면적 IoU와 함께 일대일 IoU≥0.5 instance F1을 표시한다.</p>
<h2>이미지별 동일 가중치 평균</h2><table><tr><th>Method</th><th>Area IoU</th><th>Instance F1</th><th>Missed %</th><th>Extra %</th><th>Area error %</th></tr>'''
    for v,m in means.items():
        page+=f'<tr><td>{v}</td>'+''.join(f'<td>{m[k]:.4f}</td>' for k in fields)+'</tr>'
    page+='</table><p>누락·추가·면적 오차의 분모는 reference pore 면적이다. 총면적 오차에서는 누락과 추가가 상쇄될 수 있다.</p>'
    page+='<h2>Two-stage 교체 전후</h2><table><tr><th>Image</th><th>Before IoU</th><th>After IoU</th><th>Difference</th><th>Replaced parents</th></tr>'
    for result in results:
        base=next(r for r in result['metrics'] if r['variant']=='two_stage')
        new=next(r for r in result['metrics'] if r['variant']=='two_stage_nested')
        fold=result['fold']
        page+=f'<tr><td><a href="fold_{fold}/comparison.png">{html.escape(result["image"])}</a></td><td>{base["iou"]:.4f}</td><td>{new["iou"]:.4f}</td><td>{new["iou"]-base["iou"]:+.4f}</td><td>{result["replaced_parents"]["two_stage"]}</td></tr>'
    page+='</table><p><a href="metrics.csv">모든 방법의 CSV</a> · <a href="summary.json">요약·추가 처리 시간</a> · <a href="plan.json">설정·코드 해시</a></p>'
    if any(r['fold']==10 for r in results):
        page+='<h2>PI35_5kx 사례 A–D</h2><p>앞선 진단의 같은 좌표다. 마지막 열은 reference로 고른 후보가 아닌 실제 자동 결과다. 국소 대비 보정 없이 동일한 280×280 입력 픽셀을 표시했다.</p><a href="fold_10/cases.png"><img src="fold_10/cases.png" alt="PI35 사례 A부터 D까지 원영상, reference, 기존 결과, 자동 교체 결과 비교"></a><p><a href="fold_10/two_stage_decisions.json">각 부모를 교체하거나 유지한 이유</a></p>'
    page+='</html>'
    (out/'index.html').write_text(page,encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'fine_tuning/runs/clean16_methods_20260921_110028')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--folds',type=int,nargs='+',default=[10]+[i for i in range(1,17) if i!=10])
    args=parser.parse_args()
    out=args.out or ROOT/'fine_tuning/runs'/('nested16_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=False)
    cv2.setNumThreads(4)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'fine_tuning/runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    source=args.run.resolve(); training=json.loads((source/'plan.json').read_text())
    data=Path(training['dataset'])
    if sha(data/'manifest.json') != training['dataset_manifest_sha256']:
        raise ValueError('Dataset manifest changed.')
    records=json.loads((data/'manifest.json').read_text())['records']
    (out/'source').mkdir()
    scripts=['nested_candidates.py','conservative_candidates.py','evaluate_nested.py']
    for name in scripts:shutil.copy2(Path(__file__).with_name(name),out/'source'/name)
    plan=dict(source_run=str(source),folds=args.folds,settings=DEFAULTS,
              source_hashes={name:sha(Path(__file__).with_name(name)) for name in scripts},
              reference_used_for_selection=False,initial_rule_development_image='PI35_5kx',
              timing='CPU replay only; no model loading, SAM inference, I/O, metric evaluation or plotting included.',
              opencv_threads=4,limitations='Previously examined collection; development evaluation. No per-image threshold tuning.')
    dump(out/'plan.json',plan)
    print(f'RUN {out}',flush=True)
    results=[]
    for fold in args.folds:
        f=source/f'methods_gpu/fold_{fold}'
        baseline=json.loads((f/'result.json').read_text())
        name=baseline['held_out'];record=next(r for r in records if r['name']==name)
        if sha(f/'pool.json') != baseline['pool_sha256'] or sha(data/record['image']) != record['image_sha256']:
            raise ValueError('Candidate pool or input image changed.')
        gray=np.array(Image.open(data/record['image']).convert('L'))
        raw=json.loads((f/'pool.json').read_text())
        settings=baseline['image_settings']
        start=time.perf_counter();candidates=prepare_candidates(raw,gray,settings['min_area'])
        prep_seconds=time.perf_counter()-start
        del raw
        folder=out/f'fold_{fold}';folder.mkdir()
        maps={};decisions={};seconds={}
        for v in ('relaxed','two_stage'):
            maps[v]=np.array(Image.open(f/f'{v}_instances.tif'))
            start=time.perf_counter()
            maps[v+'_nested'],decisions[v]=resolve_nested(maps[v],candidates,gray,settings['min_area'],settings['min_contrast'])
            seconds[v]=time.perf_counter()-start
            Image.fromarray(maps[v+'_nested']).save(folder/f'{v}_nested_instances.tif')
            dump(folder/f'{v}_decisions.json',decisions[v])
        # Finalize all predictions before accessing this image's reference.
        if sha(data/record['labels']) != record['labels_sha256']:
            raise ValueError('Reference labels changed.')
        truth=np.array(Image.open(data/record['labels']))
        rows=[dict(fold=fold,image=name,variant=v,**metrics(truth,maps[v])) for v in VARIANTS]
        result=dict(fold=fold,image=name,metrics=rows,prepare_seconds=prep_seconds,resolution_seconds=seconds,
                    prepared_candidates=len(candidates),replaced_parents={v:sum(d['replaced'] for d in decisions[v]) for v in decisions},
                    reference_used_for_selection=False,source_image_settings=settings,
                    input_sha256={str(p):sha(p) for p in [f/'pool.json',f/'relaxed_instances.tif',f/'two_stage_instances.tif',data/record['image'],data/record['labels']]})
        plot(folder,gray,truth,maps,rows)
        dump(folder/'result.json',result);results.append(result);report(out,results)
        print(json.dumps(dict(fold=fold,image=name,iou={r['variant']:round(r['iou'],4) for r in rows},replaced=result['replaced_parents'],prepare_seconds=round(prep_seconds,2))),flush=True)
    print(f'COMPLETE {out}/index.html',flush=True)


if __name__=='__main__':
    main()
