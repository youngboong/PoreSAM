"""Export reviewable existing+additional pores with zero shared pixels, without applying edits."""
import copy
import hashlib
import json
import time

import cv2
import numpy as np
from PIL import Image

from pore_editor import Editor,ROOT,save_images
from pore_overlap import review_groups,overlap_pixels
from analyze_candidates import export_folder


def display(gray,masks,existing_ids):
    pixels=cv2.cvtColor(gray,cv2.COLOR_GRAY2RGB)
    for candidate_id,mask in masks.items():
        color=(77,184,234) if candidate_id in existing_ids else (255,194,60)
        pixels[mask]=(pixels[mask]*.78+np.asarray(color)*.22).astype(np.uint8)
        contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(pixels,contours,-1,color,2)
        y,x=np.unravel_index(cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3).argmax(),mask.shape)
        cv2.putText(pixels,str(candidate_id),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),3)
        cv2.putText(pixels,str(candidate_id),(int(x),int(y)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
    return Image.fromarray(pixels)


def main():
    out=ROOT/'outputs/generalization_trial'/('exclusive_'+time.strftime('%Y%m%d_%H%M%S'));out.mkdir()
    editor=Editor();results={};sections=[]
    for entry in editor.image_library():
        if entry['name'] not in ['GF_1kx_1_BSE.tif','GF_1kx_1.tif']:continue
        state=editor.state(entry['latest_dataset']);name=entry['name'].rsplit('.',1)[0]
        folder=out/name;folder.mkdir()
        before={i:m.copy() for i,m in state['masks'].items()}
        assert overlap_pixels(before.values())==0,'Resolve old overlaps in a separately reviewed revision first'
        raw=editor.large_search.masks(state,editor.large_search.key(state['gray']))
        groups,review=review_groups(raw,before,state['gray'],False)
        merged=dict(before);next_id=max(before,default=0)+1;added=[]
        for group in groups:
            merged[next_id]=group['mask'];added.append(dict(id=next_id,source_candidate_id=group['id'],alternative_ids=group['alternatives']))
            next_id+=1
        assert all(np.array_equal(merged[i],m) for i,m in before.items())
        assert overlap_pixels(merged.values())==0
        assert sum(int(m.sum()) for m in merged.values())==int(np.logical_or.reduce(list(merged.values())).sum())
        report=copy.deepcopy(state['report']);report.update(image=str((ROOT/report['image']).resolve()),entrance_candidate_count=len(merged),
            status='review only: existing pores preserved; unreviewed exclusive additions',source_dataset=state['dataset'],source_revision=state['revision'],
            overlap_pixels=0,existing_preserved_ids=sorted(before),unreviewed_additions=added,
            candidates=[dict(candidate_id=i,area=int(m.sum())) for i,m in merged.items()])
        (folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        np.savez_compressed(folder/'entrance_candidates.npz',**{f'candidate_{i}':m for i,m in merged.items()})
        save_images(folder,state['gray'],merged);export_folder(folder)
        original=Image.fromarray(state['gray']);original.save(folder/'images/original.png')
        old_image=display(state['gray'],before,set(before));new_image=display(state['gray'],merged,set(before))
        old_image.save(folder/'images/existing.png');new_image.save(folder/'images/existing_and_added.png')
        comparison=Image.new('RGB',(state['gray'].shape[1]*2,state['gray'].shape[0]))
        comparison.paste(old_image,(0,0));comparison.paste(new_image,(state['gray'].shape[1],0));comparison.save(folder/'images/preservation_comparison.png')
        result=dict(dataset=state['dataset'],revision=state['revision'],raw_additional_candidates=len(raw),existing_count=len(before),added_count=len(groups),combined_count=len(merged),
                    overlap_pixels=0,existing_masks_unchanged=True,review=review,added=added,
                    existing_mask_hashes={str(i):hashlib.sha256(m.tobytes()).hexdigest() for i,m in before.items()})
        results[name]=result;(folder/'review.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        sections.append(f'''<section><h2>{name}</h2><p>기존 {len(before)}개 그대로 유지 + 추가 검토 {len(groups)}개 = 총 {len(merged)}개 · 겹친 픽셀 0</p><p>파란색: 기존 pore / 노란색: 추가 검토 후보. 큰 후보를 우선 제안하며, 경계가 맞는지는 확인이 필요합니다.</p><div class="grid"><figure><figcaption>기존 분석</figcaption><a href="{name}/images/existing.png"><img src="{name}/images/existing.png"></a></figure><figure><figcaption>기존 + 겹치지 않는 추가 후보</figcaption><a href="{name}/images/existing_and_added.png"><img src="{name}/images/existing_and_added.png"></a></figure></div><p><a href="{name}/images/preservation_comparison.png">비교 이미지 원본 크기로 열기</a> · <a href="{name}/measurements/index.html">추가 후보 포함 잠정 통계</a> · <a href="{name}/review.json">보존·겹침 검증</a></p></section>''')
        print(name,json.dumps({k:result[k] for k in ['existing_count','added_count','combined_count','overlap_pixels']}),flush=True)
    (out/'index.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>기존 pore 보존 · 겹침 없는 추가 후보</title><style>body{font:15px 'Malgun Gothic',sans-serif;line-height:1.7;background:#eef3f3;color:#173e45;margin:24px auto;max-width:1500px;padding:0 20px}section{background:white;padding:20px;margin:20px 0;border-radius:12px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:15px}figure{margin:0}img{max-width:100%}.notice{padding:15px;background:#fff0d5}@media(max-width:850px){.grid{grid-template-columns:1fr}}</style><h1>기존 pore 보존 · 겹침 없는 추가 후보</h1><p class="notice">기존 마스크를 그대로 유지한 검토용 합본입니다. 사용자 수정 이력에는 적용하지 않았습니다. 노란 후보는 아직 정답으로 확정되지 않았으며 통계도 추가 후보를 포함한 잠정값입니다. 기존 pore와 겹치는 후보는 제외했습니다. 기존 pore를 더 큰 영역으로 바꾸려면 편집기에서 교체 후보를 선택하세요.</p>'''+''.join(sections)+'</html>',encoding='utf-8')
    (out/'report.json').write_text(json.dumps(dict(status='exclusive_review_completed',results=results),indent=2),encoding='utf-8')
    print(out/'index.html',flush=True)


if __name__=='__main__':main()
