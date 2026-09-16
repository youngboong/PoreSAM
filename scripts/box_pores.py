"""Discover separate pores in a drawn box and queue each for individual review."""
import copy
import secrets
import cv2
import numpy as np
from scipy import ndimage as ndi
from sam_runtime import inference_context
from segment_first_pass import ProgressGenerator,candidates_from_masks


def separate_candidates(raw,gray,contrast,minimum):
    """Reject a broad parent only when substantial bright material separates its children."""
    candidates=[]
    for item in raw:
        candidates.extend(candidates_from_masks([item],gray,contrast,minimum,max_area_fraction=1.))
    rejected=set()
    for index,parent in enumerate(candidates):
        children=[];covered=np.zeros_like(parent['mask'])
        for child in sorted(candidates,key=lambda c:c['area'],reverse=True):
            if not parent['area']*.06<=child['area']<parent['area']*.8:continue
            if (child['mask']&parent['mask']).sum()<child['area']*.95:continue
            if (child['mask']&covered).sum()>child['area']*.1:continue
            children.append(child);covered|=child['mask']
        if len(children)<2 or (covered&parent['mask']).sum()<parent['area']*.25:continue
        gap=parent['mask']&~covered
        # Thin internal texture alone is insufficient to split a pore.
        inside=float(gray[covered].mean())
        bright=gap&(gray>=inside+max(contrast,15))
        bright=ndi.binary_opening(bright,iterations=2)
        if bright.sum()>=parent['area']*.06:rejected.add(index)
    retained=[dict(segmentation=c['mask'],predicted_iou=c['predicted_iou']) for i,c in enumerate(candidates) if i not in rejected]
    return candidates_from_masks(retained,gray,contrast,minimum,max_area_fraction=1.)


def preview_box(editor,state,payload):
    editor.check_revision(state,payload)
    bounds=np.asarray(payload.get('box'),dtype=np.float32)
    if bounds.shape!=(4,):raise ValueError('Draw a box first.')
    editor.coords(bounds.reshape(2,2),state['gray'].shape)
    if bounds[2]-bounds[0]<3 or bounds[3]-bounds[1]<3:raise ValueError('Draw a larger box.')
    limit=payload.get('max_candidates',32)
    if type(limit) is not int or not 1<=limit<=32:raise ValueError('Preview queue is full.')
    queue=state.setdefault('queued_previews',{})
    available=32-len(queue)+(payload.get('replace_queue_token') in queue)
    limit=min(limit,available)
    if limit<1:raise ValueError('Preview queue is full. Clear a region first.')
    bx0,by0=np.floor(bounds[:2]).astype(int);bx1,by1=np.ceil(bounds[2:]).astype(int)+1
    gray=state.get('sam_gray',state['gray'])
    # Retain surrounding structure for the image encoder, especially for a tight
    # box around one pore. Only objects predominantly inside the drawn box qualify.
    pad=max(32,int(np.ceil((160-min(bx1-bx0,by1-by0))/2)))
    x0,y0=max(0,bx0-pad),max(0,by0-pad)
    x1,y1=min(gray.shape[1],bx1+pad),min(gray.shape[0],by1+pad)
    crop=gray[y0:y1,x0:x1]
    settings=state['report'].get('selection_settings',{})
    minimum=settings.get('min_area_pixels',100)
    def select(raw):
        separate=[]
        for item in raw:
            components,count=ndi.label(item['segmentation'])
            sizes=np.bincount(components.ravel())
            for label in range(1,count+1):
                if sizes[label]>=minimum:
                    part=components==label
                    if part[by0-y0:by1-y0,bx0-x0:bx1-x0].sum()>=.8*sizes[label]:
                        separate.append(dict(item,segmentation=part))
        selected=separate_candidates(separate,crop,settings.get('min_contrast',8),minimum)
        return [item for item in selected if item['mask'][by0-y0:by1-y0,bx0-x0:bx1-x0].sum()>=.8*item['mask'].sum()]
    # Start with the existing fast box prompt and split disconnected objects.
    result=editor.preview(state,{**payload,'queue_preview':False})
    guided=state['preview']
    raw=[dict(segmentation=guided['masks'][0][y0:y1,x0:x1],predicted_iou=result['choices'][0]['score'])]
    found=select(raw)
    _,dark=cv2.threshold(gray[by0:by1,bx0:bx1],0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    remaining=(dark>0)&~ndi.binary_dilation(guided['masks'][0],iterations=2)[by0:by1,bx0:bx1]
    components,count=ndi.label(remaining);sizes=np.bincount(components.ravel())[1:]
    density=0
    if len(found)<=1 or (sizes>=minimum).any():
        # A broad box prompt can cover every dark region yet be a merged mask.
        # Independently search before accepting a single-pore interpretation.
        # Keep the full-image embedding used by later Include/Exclude edits intact.
        density=max(8,min(24,int(np.ceil(max(bx1-bx0,by1-by0)/20))))
        with inference_context(editor.device):
            generator=ProgressGenerator(editor.predictor.model,points_per_side=density,points_per_batch=8,
                                        pred_iou_thresh=.8,stability_score_thresh=.92,crop_n_layers=0,min_mask_region_area=0)
            raw+=generator.generate(cv2.cvtColor(crop,cv2.COLOR_GRAY2RGB))
        found=select(raw)
    # A single pore may occupy most of a hand-drawn box; the 20% full-image limit
    # must not be reinterpreted as 20% of this box.
    if not found:
        # Retain guided manual segmentation for low-contrast or unusual pores.
        queue.pop(payload.get('replace_queue_token'),None)
        queue[result['token']]=dict(masks=[np.packbits(mask) for mask in guided['masks']],
                                   scores=[choice['score'] for choice in result['choices']],prompts=guided['prompts'],source=guided['source'])
        return dict(regions=[dict(token=result['token'],input=copy.deepcopy(payload))],fallback=True)
    if len(found)>limit:
        raise ValueError(f'Found {len(found)} pores, but only {limit} preview slots are available. Use a smaller box or clear queued regions.')
    pending={};regions=[]
    for item in found:
        mask=np.zeros_like(gray,dtype=bool);mask[y0:y1,x0:x1]=item['mask']
        yy,xx=np.nonzero(mask)
        box=[max(0,int(xx.min())-5),max(0,int(yy.min())-5),min(gray.shape[1]-1,int(xx.max())+5),min(gray.shape[0]-1,int(yy.max())+5)]
        distance=cv2.distanceTransform(item['mask'].astype(np.uint8),cv2.DIST_L2,3)
        cy,cx=np.unravel_index(distance.argmax(),distance.shape)
        prompts=dict(dataset=state['dataset'],revision=state['revision'],box=box,points=[[int(cx+x0),int(cy+y0)]],labels=[1],
                     polygon=[],shape=None,fill_holes=True,target='',search_box=bounds.tolist())
        token=secrets.token_urlsafe(18)
        pending[token]=dict(masks=[np.packbits(mask)],scores=[item['predicted_iou']],prompts=prompts,source='prompted_sam')
        regions.append(dict(token=token,input=prompts))
    queue.pop(payload.get('replace_queue_token'),None);queue.update(pending)
    return dict(regions=regions,fallback=False,points_per_side=density)
