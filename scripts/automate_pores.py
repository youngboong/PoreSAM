"""Propose box prompts in dark regions not covered by existing pore masks."""
import cv2
import numpy as np
from scipy import ndimage as ndi


def propose_boxes(state, limit=16):
    gray=state.get('sam_gray',state['gray'])
    union=np.zeros(gray.shape,bool)
    for mask in state['masks'].values():union|=mask
    threshold,dark=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Mask out existing pores with a narrow margin to avoid tracing their rims.
    uncovered=(dark>0)&~ndi.binary_dilation(union,iterations=2)
    labels,count=ndi.label(uncovered)
    minimum=max(20,int(state['report'].get('selection_settings',{}).get('min_area_pixels',100)*.5))
    proposals=[];height,width=gray.shape
    for i,sl in enumerate(ndi.find_objects(labels),1):
        if sl is None:continue
        part=labels[sl]==i;area=int(part.sum())
        if not minimum<=area<=gray.size*.2:continue
        ys,xs=sl;w=xs.stop-xs.start;h=ys.stop-ys.start
        if min(w,h)<4:continue
        pad=max(4,int(max(w,h)*.2))
        box=[max(0,xs.start-pad),max(0,ys.start-pad),min(width-1,xs.stop+pad),min(height-1,ys.stop+pad)]
        proposals.append(dict(box=box,seed_area=area))
    proposals.sort(key=lambda p:p['seed_area'],reverse=True)
    selected=[]
    for item in proposals:
        a=item['box'];duplicate=False
        for other in selected:
            b=other['box'];overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            if overlap/min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))>.65:duplicate=True;break
        if not duplicate:selected.append(item)
        if len(selected)>=limit:break
    return dict(boxes=selected,threshold=float(threshold))


def add_candidate(editor,state,payload):
    """Add at most one nonoverlapping candidate from a proposed box."""
    import copy
    editor.check_revision(state,payload)
    result=editor.preview(state,dict(revision=state['revision'],box=payload.get('box'),points=[],labels=[],fill_holes=True))
    union=np.zeros(state['gray'].shape,bool)
    for mask in state['masks'].values():union|=mask
    settings=state['report'].get('selection_settings',{})
    minimum=settings.get('min_area_pixels',100);minimum_contrast=settings.get('min_contrast',8)
    gray=state.get('sam_gray',state['gray'])
    for mask,choice in zip(state['preview']['masks'],result['choices']):
        area=int(mask.sum())
        if choice['score']<.8 or not minimum<=area<=gray.size*.2:continue
        if (mask&union).sum()/area>.1:continue
        clipped=mask&~union
        components,count=ndi.label(clipped,structure=np.ones((3,3)))
        sizes=np.bincount(components.ravel());sizes[0]=0
        if not count or sizes.max()<minimum or sizes.max()<clipped.sum()*.9:continue
        clipped=components==sizes.argmax()
        ring=ndi.binary_dilation(clipped,iterations=4)&~clipped
        if not ring.any() or gray[ring].mean()-gray[clipped].mean()<minimum_contrast:continue
        masks=state['masks'].copy();candidate_id=state['next_id'];masks[candidate_id]=clipped
        annotations=copy.deepcopy(state['annotations'])
        annotations[str(candidate_id)]=dict(source='automated_sam',supplemental=True,review_status='unreviewed',box=payload['box'])
        response=editor.save(state,masks,annotations,state['history']+[state['revision']],candidate_id+1,dict(type='automate',candidate_id=candidate_id,box=payload['box']))
        return dict(added=True,state=response,candidate_id=candidate_id)
    state['preview']=None
    return dict(added=False)
