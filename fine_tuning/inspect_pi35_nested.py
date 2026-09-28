"""Read-only diagnosis of saved PI35 candidates; reference matching is diagnostic only."""
import csv
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'fine_tuning/runs/clean16_methods_20260921_110028'
OUT = ROOT / 'fine_tuning/runs/pi35_nested_diagnosis_20260921'


def decode(rle):
    counts = np.asarray(rle['counts'], dtype=np.int64)
    return np.repeat(np.arange(len(counts)) % 2, counts).reshape(rle['size'], order='F').astype(bool)


def main():
    OUT.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'fine_tuning/runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from conservative_candidates import rim_evidence

    source = RUN / 'methods_gpu/fold_10'
    data = RUN / 'dataset/PI35_5kx'
    gray = np.array(Image.open(data / 'image.png').convert('L'))
    truth = np.array(Image.open(data / 'instances.tif'))
    strict = np.array(Image.open(source / 'strict_instances.tif'))
    two = np.array(Image.open(source / 'two_stage_instances.tif'))
    pool = json.loads((source / 'pool.json').read_text())
    # Three observed disagreement cases and one large opening with internal fibers.
    cases = [('A', 19), ('B', 23), ('C', 40), ('D', int(strict[90, 610]))]
    records = []
    wanted = set()
    for name, pid in cases:
        parent = strict == pid
        y, x = np.where(parent)
        refs = [int(i) for i in np.unique(truth[parent]) if i and np.count_nonzero(parent & (truth == i)) >= 100]
        wanted.update(refs)
        cx, cy = int((x.min()+x.max())/2), int((y.min()+y.max())/2)
        # Fixed 280 x 280 px field of view in every row.
        x0, y0 = min(max(cx-140, 0), gray.shape[1]-280), min(max(cy-140, 0), gray.shape[0]-280)
        records.append(dict(case=name, parent_id=pid, reference_ids=refs, bbox=[x0,y0,x0+280,y0+280],
                            parent_area=int(parent.sum()), reference_background_in_parent=int(np.count_nonzero(parent & (truth==0))),
                            strict_parent_preserved_in_two_stage=bool(np.all(two[parent]>0))))

    sizes = np.bincount(truth.ravel())
    best = {i: dict(iou=-1.) for i in wanted}
    for index, item in enumerate(pool):
        mask = decode(item['segmentation'])
        area = int(mask.sum())
        overlap = np.bincount(truth[mask], minlength=len(sizes))
        for ref in wanted:
            iou = float(overlap[ref] / (area+sizes[ref]-overlap[ref]))
            if iou > best[ref]['iou']:
                best[ref] = dict(iou=iou, pool_id=index, score=item['predicted_iou'], stability=item['stability_score'], mask=mask.copy())

    smooth = cv2.GaussianBlur(gray.astype(np.float32), (0,0), 1.)
    rows=[]
    for case in records:
        parent = strict == case['parent_id']
        children = np.isin(truth,case['reference_ids']) & parent
        gap = parent & ~children
        # Reference-derived regions test a feature hypothesis, never select a prediction.
        gap_core = ndi.binary_erosion(gap, iterations=3)
        child_core = ndi.binary_erosion(children, iterations=3)
        case['reference_gap_core_pixels']=int(gap_core.sum())
        case['reference_gap_core_median_gray']=float(np.median(gray[gap_core])) if gap_core.any() else None
        case['reference_pore_core_median_gray']=float(np.median(gray[child_core])) if child_core.any() else None
        case['reference_gap_minus_pore_gray']=(case['reference_gap_core_median_gray']-case['reference_pore_core_median_gray']) if gap_core.any() and child_core.any() else None
        case['children']=[]
        for ref in case['reference_ids']:
            item=best[ref]
            evidence=rim_evidence(item['mask'],smooth,3.)
            row=dict(case=case['case'],reference_id=ref,reference_pixels=int(sizes[ref]),
                     **{k:v for k,v in item.items() if k!='mask'},
                     rim_support_at_5_gray=float(np.mean(evidence>=5.)) if evidence.size else 0.,
                     strict_score_stability_pass=bool(item['score']>float(np.float32(.80078125)) and item['stability']>=float(np.float32(.92))))
            rows.append(row);case['children'].append(row)

    fig,axes=plt.subplots(4,4,figsize=(13,13),layout='constrained')
    titles=['SEM input', 'Current strict outline', 'Reviewed reference outlines', 'Best pool matches with IoU >= 0.5\nDIAGNOSTIC ORACLE, not prediction']
    for row,case in enumerate(records):
        x0,y0,x1,y1=case['bbox'];sl=np.s_[y0:y1,x0:x1]
        for col,ax in enumerate(axes[row]):
            ax.imshow(gray[sl],cmap='gray',vmin=0,vmax=255,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(titles[col],fontsize=10)
        axes[row,0].set_ylabel(f"{case['case']}: parent {case['parent_id']}\n{len(case['reference_ids'])} reference pore(s)",fontsize=10)
        def outline(ax,mask,color):
            if mask[sl].any():ax.contour(mask[sl],levels=[.5],colors=[color],linewidths=1)
        outline(axes[row,1],strict==case['parent_id'],'#E69F00')
        for ref in case['reference_ids']:
            outline(axes[row,2],truth==ref,'#56B4E9')
            if best[ref]['iou'] >= .5:
                outline(axes[row,3],best[ref]['mask'],'#009E73')
        missing=sum(best[ref]['iou'] < .5 for ref in case['reference_ids'])
        if missing:
            axes[row,3].text(8,16,f'{missing} reference: no pool match >= 0.5',fontsize=8,color='white',bbox=dict(facecolor='black',alpha=.8,pad=2))
        axes[row,0].plot([12,62],[262,262],color='white',lw=3)
        axes[row,0].text(12,252,'50 px',color='white',fontsize=8)
    fig.suptitle('PI35_5kx | identical 280 x 280 pixel crops; no local contrast adjustment',fontsize=13)
    fig.savefig(OUT/'cases.png',dpi=160)
    plt.close(fig)

    with (OUT/'candidate_diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    source_files=[data/'image.png',data/'instances.tif',source/'strict_instances.tif',source/'two_stage_instances.tif',source/'pool.json']
    manifest=dict(source_run=str(RUN),image='PI35_5kx',fold=10,pool_count=len(pool),cases=records,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
                  limitations=['Cases chosen by visual inspection of disagreement; not a random sample.',
                               'Best candidate matching uses reference labels: availability diagnosis only, not automatic accuracy.',
                               'Gap brightness also uses reference labels and is not a validated classifier.',
                               'Saved pool already passed generator quality filters and within-batch NMS; it is not all SAM hypotheses.',
                               'Current masks and application code are unchanged.'],
                  display=dict(crop_pixels=[280,280],gray_limits=[0,255],local_contrast_adjustment=False,scale_bar='50 input pixels'))
    (OUT/'diagnosis.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    table=''.join(f"<tr><td>{r['case']}</td><td>{r['reference_id']}</td><td>{r['pool_id']}</td><td>{r['iou']:.3f}</td><td>{r['score']:.3f}</td><td>{r['stability']:.3f}</td><td>{r['rim_support_at_5_gray']:.2f}</td></tr>" for r in rows)
    html='''<!doctype html><html lang="ko"><meta charset="utf-8"><title>PI35 5kx 내부 pore 진단</title>
<style>body{font:16px system-ui;max-width:1350px;margin:32px auto;padding:16px;line-height:1.6}img{width:100%}td,th{padding:8px;border-bottom:1px solid #bbb}a{color:#075a9d}</style>
<h1>PI35_5kx: 큰 경계와 내부 pore</h1>
<p>이전 대화의 fold 10 문제를 저장된 입력·reference·SAM 후보로 확인했다. A–C는 여러 pore를 큰 마스크로 합친 사례, D는 내부 섬유가 보여도 reference에서 큰 pore 하나로 유지한 비교 사례다.</p>
<p>각 행은 동일한 280×280 입력 픽셀 범위다. 저장된 전체 영상의 밝기 정규화 결과를 그대로 표시했고 국소 대비 보정은 하지 않았다. 흰 막대는 입력 50픽셀이다.</p>
<img src="cases.png" alt="세 분할 사례와 큰 pore 유지 사례를 원영상, 현재 큰 경계, 검토된 reference, reference에 가장 가까운 저장 후보 순으로 비교">
<p><strong>마지막 열은 reference를 이용해 고른 최적 후보이므로 자동 분할 결과가 아니다.</strong> 내부 후보가 이미 존재하는지를 확인하기 위한 진단이며 IoU 0.5 이상인 후보만 표시했다. A–C의 작은 pore 10개 중 8개가 이 기준을 만족했다. 나머지 2개는 추가 후보 탐색 또는 경계 보정이 필요하다. 밝기 통계도 reference로 영역을 정의한 탐색 결과다.</p>
<h2>구별 규칙의 후보</h2><ol>
<li>포함된 작은 후보를 제거하기 전에 큰 후보와 함께 보존한다.</li>
<li>작은 후보 각각의 닫힌 경계에서 바깥이 안쪽보다 밝은지, 여러 방향에서 경계가 이어지는지 확인한다.</li>
<li>작은 후보 사이에 넓고 연속된 밝은 바탕이 있는지 검사한다. 밝은 바탕의 두께·연결성·바깥 표면과의 밝기 및 질감 유사도를 함께 본다.</li>
<li>얇은 섬유가 어두운 내부를 가로지르는 경우에는 밝은 픽셀만으로 분할하지 않는다. D를 큰 pore 유지 대조군으로 사용한다.</li>
<li>경계 근거와 밝은 바탕 근거가 모두 충분한 경우에만 부모를 자식들로 교체한다. 애매한 경우는 검토 대상으로 남긴다.</li></ol>
<p>아직 자동 판별 성능을 검증하거나 임계값을 확정한 단계가 아니다. PI35에서 규칙을 개발한 뒤에는 별도 이미지에서 과분할까지 확인해야 한다. 앱과 기존 예측은 변경하지 않았다.</p>
<h2>저장 후보의 reference 일치도</h2><p>IoU는 각 reference에 대한 최댓값이다. Rim 값은 후보 경계 양쪽 3픽셀에서 외측−내측 밝기 차이가 5 이상인 비율이다. SAM score는 이 의미의 pore가 맞다는 확률이 아니다.</p>
<table><tr><th>사례</th><th>Ref ID</th><th>Pool ID</th><th>Best IoU</th><th>SAM score</th><th>Stability</th><th>Rim</th></tr>'''+table+'''</table>
<p><a href="candidate_diagnostics.csv">후보 수치 CSV</a> · <a href="diagnosis.json">원본 경로·해시·사례 좌표·제한사항</a></p>
<p>그림 작성 절차: scientific-visualization skill의 원영상 보존·동일 표시 범위 원칙을 적용.</p></html>'''
    (OUT/'index.html').write_text(html,encoding='utf-8')
    print(json.dumps(records,indent=2))
    print(OUT)


if __name__=='__main__':
    main()
