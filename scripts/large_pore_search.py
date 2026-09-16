"""Background GF-style candidate discovery; applying a candidate remains an editor action."""
import hashlib
import json
from pathlib import Path
from app_paths import checkpoint_path
import secrets
import threading
import traceback

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam_runtime import release_device_cache

from fiber_pores import generate_large_pores
from pore_overlap import review_groups

VERSION='fiber-box-points-v2'


class LargePoreSearch:
    def __init__(self,editor,root):
        self.editor,self.root=editor,Path(root)
        self.jobs={};self.lock=threading.RLock()

    def key(self,gray,strength='medium'):
        return hashlib.sha256((VERSION+':'+strength).encode()+str(gray.shape).encode()+gray.tobytes()).hexdigest()

    def configuration(self,state,key):
        for strength in ['medium','strong']:
            if key==self.key(state['gray'],strength):
                return dict(strength=strength,radius_fraction=.009 if strength=='medium' else .018,sigma=2,percentiles=[2,98])
        raise ValueError('현재 이미지에서 큰 pore 탐색을 다시 실행해주세요.')

    def start(self,state,payload):
        self.editor.check_revision(state,payload)
        strength=payload.get('strength','medium')
        if strength not in ['medium','strong']:raise ValueError('구조 약화 강도를 다시 선택해주세요.')
        with self.editor.workflow.lock:
            if self.editor.workflow.running:raise ValueError('진행 중인 분석이 끝난 뒤 실행해주세요.')
            self.editor.workflow.running=True
        key=self.key(state['gray'],strength);job_id=secrets.token_hex(12)
        with self.lock:self.jobs[job_id]=dict(status='running',progress=1,message='큰 pore 탐색을 준비하고 있습니다.',key=key)
        threading.Thread(target=self.run,args=(job_id,key,state['gray'].copy(),strength),daemon=True).start()
        return dict(job_id=job_id)

    def status(self,job_id):
        with self.lock:
            if job_id not in self.jobs:raise ValueError('탐색을 다시 실행해주세요.')
            return dict(self.jobs[job_id])

    def progress(self,job_id,percent,message):
        with self.lock:self.jobs[job_id].update(progress=percent,message=message)

    def run(self,job_id,key,gray,strength):
        folder=self.root/key
        try:
            with self.editor.lock:
                folder.mkdir(parents=True,exist_ok=True)
                if (folder/'complete.json').is_file() and (folder/'masks.npz').is_file():
                    with np.load(folder/'masks.npz',allow_pickle=False) as data:count=len(data.files)
                else:
                    self.editor.predictor=None;self.editor.encoded_dataset=None
                    release_device_cache(self.editor.device)
                    model=build_sam2('configs/sam2.1/sam2.1_hiera_s.yaml',str(checkpoint_path()),device=self.editor.device,apply_postprocessing=False)
                    predictor=SAM2ImagePredictor(model)
                    try:
                        masks,audit=generate_large_pores(gray,predictor,lambda p,m:self.progress(job_id,p,m),strength=strength)
                    finally:
                        del predictor,model
                        release_device_cache(self.editor.device)
                    np.savez_compressed(folder/'masks.pending.npz',**{f'candidate_{i+1}':m for i,m in enumerate(masks)})
                    (folder/'masks.pending.npz').replace(folder/'masks.npz')
                    (folder/'prompts.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
                    count=len(masks)
                    (folder/'complete.json').write_text(json.dumps(dict(version=VERSION,key=key,shape=list(gray.shape),candidate_count=count,
                        preprocessing=self.configuration(dict(gray=gray),key))),encoding='utf-8')
            with self.editor.workflow.lock:
                self.editor.workflow.running=False
                with self.lock:self.jobs[job_id].update(status='complete',progress=100,message='큰 pore 후보를 찾았습니다.',candidate_count=count)
        except Exception as exc:
            folder.mkdir(parents=True,exist_ok=True)
            (folder/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
            with self.editor.workflow.lock:
                self.editor.workflow.running=False
                with self.lock:self.jobs[job_id].update(status='failed',message=str(exc))

    def masks(self,state,key):
        self.configuration(state,key)
        folder=self.root/key
        if not (folder/'complete.json').is_file():raise ValueError('큰 pore 탐색이 아직 완료되지 않았습니다.')
        with np.load(folder/'masks.npz',allow_pickle=False) as data:return [data[k].copy() for k in data.files]

    def overview(self,state,payload):
        from pore_editor import png_url
        from PIL import Image
        self.editor.check_revision(state,payload)
        masks=self.masks(state,payload.get('key'))
        groups,review=review_groups(masks,state['masks'],state['gray'],payload.get('allow_replacement') is True)
        rgba=np.zeros((*state['gray'].shape,4),np.uint8)
        replaced={i for group in groups for i in group['replacement_ids']}
        for candidate_id,mask in state['masks'].items():
            if candidate_id in replaced:continue
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgba,contours,-1,(77,184,234,255),1)
        for group in groups:
            mask=group['mask']
            contours,_=cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgba,contours,-1,(255,194,60,255),2)
        return dict(image=png_url(Image.fromarray(rgba)),review=review,candidates=[dict(id=g['id'],alternatives=len(g['alternatives']),replacement_ids=g['replacement_ids'],area_um2=float(g['mask'].sum()*state['report']['scale']['um_per_pixel']**2)) for g in groups])

    def pick(self,state,payload):
        self.editor.check_revision(state,payload)
        masks=self.masks(state,payload.get('key'))
        groups,_=review_groups(masks,state['masks'],state['gray'],payload.get('allow_replacement') is True)
        if payload.get('candidate_id') is not None:
            index=payload['candidate_id']
            group=next((g for g in groups if type(index) is int and g['id']==index),None)
        else:
            point=self.editor.coords([payload.get('point')],state['gray'].shape)[0]
            x,y=np.floor(point).astype(int)
            group=next((g for g in groups if g['mask'][y,x]),None)
        ids=[i-1 for i in group['alternatives']] if group else []
        if not ids:raise ValueError('이 위치에서 큰 pore 후보를 찾지 못했습니다. 다른 위치를 선택하거나 박스로 지정해주세요.')
        provenance=dict(payload,method=VERSION,preprocessing=self.configuration(state,payload['key']),pool_candidate_ids=[i+1 for i in ids],cache_folder=str((self.root/payload['key']).resolve()))
        result=self.editor.preview_masks(state,provenance,[masks[i] for i in ids],[None]*len(ids),'structure_sam')
        for choice,i in zip(result['choices'],ids):choice['pool_candidate_id']=i+1
        return result
