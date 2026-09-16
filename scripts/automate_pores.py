"""Propose box prompts in dark regions not covered by existing pore masks."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from operation_cancel import OperationCancelled, check_stop

MIN_SAM_SCORE = 0.7
MAX_PROPOSAL_BOXES = 32


def mask_bounds(mask):
    x,y,w,h=cv2.boundingRect(mask.view(np.uint8))
    return (x,y,x+w,y+h)


def intersects(a,b):
    return a[0]<b[2] and b[0]<a[2] and a[1]<b[3] and b[1]<a[3]


def region_slice(bounds,shape,padding=4):
    x0,y0,x1,y1=bounds
    return (slice(max(0,y0-padding),min(shape[0],y1+padding)),
            slice(max(0,x0-padding),min(shape[1],x1+padding)))


def propose_boxes(state, limit=MAX_PROPOSAL_BOXES):
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


def add_candidate(editor,state,payload,stop_event=None,staged=False):
    try:
        return _add_candidate(editor,state,payload,stop_event,staged)
    except OperationCancelled:
        state['preview']=None
        return dict(added=False,cancelled=True)


def consolidation_boxes(state,limit=16):
    """Revisit touching automatic additions, regardless of whether they needed trimming."""
    masks=state['masks'];annotations=state.get('annotations',{})
    bounds={i:mask_bounds(m) for i,m in masks.items()}
    seeds={i for i in masks if annotations.get(str(i),{}).get('source')=='automated_sam'}
    groups=[];seen=set()
    for seed in sorted(seeds):
        if seed in seen:continue
        group={seed};pending=[seed]
        while pending:
            current=pending.pop();roi=region_slice(bounds[current],masks[current].shape,2)
            near=ndi.binary_dilation(masks[current][roi],iterations=2)
            near_bounds=(roi[1].start,roi[0].start,roi[1].stop,roi[0].stop)
            for other in masks:
                if annotations.get(str(other),{}).get('source')=='manual_cut':continue
                if other in group or not intersects(near_bounds,bounds[other]) or not (near&masks[other][roi]).any():continue
                # Expand through automatic additions, not every baseline pore.
                if current in seeds or other in seeds:group.add(other);pending.append(other)
        seen|=group
        if len(group)<2:continue
        combined=np.logical_or.reduce([masks[i] for i in group])
        if combined.sum()>combined.size*.2:continue
        yy,xx=np.nonzero(combined);pad=8
        box=[max(0,int(xx.min())-pad),max(0,int(yy.min())-pad),min(combined.shape[1]-1,int(xx.max())+pad),min(combined.shape[0]-1,int(yy.max())+pad)]
        groups.append(dict(box=box,consolidation_only=True,candidate_ids=sorted(group)))
        if len(groups)>=limit:break
    return groups


def dark_shared_boundaries(mask,masks,ids,gray,ring,contrast):
    """A joint SAM hypothesis may contain internal texture brighter than its dark core."""
    if len(ids)<2 or any((mask&masks[i]).sum()<masks[i].sum()*.8 for i in ids):return False
    threshold=float(gray[ring].mean())-contrast
    connected={ids[0]}
    while True:
        additions=set()
        for i in connected:
            near=ndi.binary_dilation(masks[i],iterations=2)
            for j in set(ids)-connected:
                seam=(near&masks[j])|(ndi.binary_dilation(masks[j],iterations=2)&masks[i])
                if seam.sum()>=8 and gray[seam].mean()<threshold and (gray[seam]<threshold).mean()>=.5:additions.add(j)
        if not additions:return len(connected)==len(ids)
        connected|=additions


def reconcile(mask,masks,gray,minimum,contrast,bounds=None):
    """Run identical checks on candidate support plus the complete overlapping masks.

    Four pixels of context preserve the contrast ring. The area limit remains
    relative to the full image, and existing mask insertion order is preserved.
    """
    candidate_bounds=mask_bounds(mask)
    if candidate_bounds[0]==candidate_bounds[2]:return None,[],False
    bounds=bounds if bounds is not None else {i:mask_bounds(m) for i,m in masks.items()}
    nearby=[b for b in bounds.values() if intersects(candidate_bounds,b)]
    x0,y0,x1,y1=candidate_bounds
    for a,b,c,d in nearby:x0=min(x0,a);y0=min(y0,b);x1=max(x1,c);y1=max(y1,d)
    roi=region_slice((x0,y0,x1,y1),gray.shape)
    roi_bounds=(roi[1].start,roi[0].start,roi[1].stop,roi[0].stop)
    local={i:m[roi] for i,m in masks.items() if intersects(roi_bounds,bounds[i])}
    combined,ids,review=_reconcile_region(mask[roi],local,gray[roi],minimum,contrast,gray.size)
    if combined is None:return None,ids,review
    full=np.zeros_like(mask);full[roi]=combined
    return full,ids,review


def _reconcile_region(mask,masks,gray,minimum,contrast,field_area):
    """Conservative same-pore evidence; never connect separate regions by drawing a bridge."""
    area=int(mask.sum());ids=[];union=np.zeros_like(mask)
    for candidate_id,old in masks.items():
        union|=old
        overlap=int((mask&old).sum())
        if not overlap:continue
        old_area=int(old.sum())
        # Ignore a duplicate or a smaller fragment of an existing pore.
        if overlap/area>=.95 and area<=old_area:return None,[],False
        if overlap/old_area>=.8 or (overlap/old_area>=.5 and overlap/area>=.5):
            ids.append(candidate_id)
        elif overlap/area>.1:
            return None,[],True
    combined=mask.copy()
    for candidate_id in ids:combined|=masks[candidate_id]
    others=union.copy()
    for candidate_id in ids:others&=~masks[candidate_id]
    combined&=~others
    components,count=ndi.label(combined,structure=np.ones((3,3)))
    sizes=np.bincount(components.ravel());sizes[0]=0
    if not count or sizes.max()<minimum or sizes.max()<combined.sum()*.9:return None,[],bool(ids)
    combined=components==sizes.argmax()
    if combined.sum()>field_area*.2:return None,[],bool(ids)
    ring=ndi.binary_dilation(combined,iterations=4)&~combined
    if not ring.any() or gray[ring].mean()-gray[combined].mean()<contrast:return None,[],bool(ids)
    if ids:
        # Require each contributing region to lie in the same dark interior.
        # Bright separating fibers and one-pixel necks must not cause a merge.
        threshold=(float(np.median(gray[combined]))+float(np.median(gray[ring])))/2
        interior=ndi.binary_erosion(combined&(gray<threshold))
        labels,_=ndi.label(interior)
        parts=[mask]+[masks[i] for i in ids]
        common=None
        for part in parts:
            counts=np.bincount(labels[part].ravel());counts[0]=0
            eligible=set(np.flatnonzero(counts>=part.sum()*.7))
            common=eligible if common is None else common&eligible
        if not common and not dark_shared_boundaries(mask,masks,ids,gray,ring,contrast):return None,[],True
        previous=np.logical_or.reduce([masks[i] for i in ids])
        if len(ids)==1 and np.array_equal(combined,previous):return None,[],False
    return combined,ids,False


def trim_candidate(mask,masks,gray,minimum,contrast):
    """Keep the largest valid remainder without changing any existing pore."""
    union=np.zeros_like(mask)
    for old in masks.values():union|=old
    remainder=mask&~union
    labels,_=ndi.label(remainder,structure=np.ones((3,3)))
    sizes=np.bincount(labels.ravel());sizes[0]=0
    for label in np.argsort(sizes)[::-1]:
        area=int(sizes[label])
        if area<max(minimum,mask.sum()*.1):break
        if area>gray.size*.2:continue
        part=labels==label
        ring=ndi.binary_dilation(part,iterations=4)&~part
        if ring.any() and gray[ring].mean()-gray[part].mean()>=contrast:return part
    return None


def reconcile_touching_pair(mask,state,allowed_ids,gray,minimum,contrast,stop_event=None,bounds=None):
    """A rejected neighborhood must not veto an independently valid touching pair.

    Reuse the same SAM hypothesis and all reconciliation checks. Other existing
    masks are protected before evaluating each pair; no pixels or bridges are
    invented, and no pore IDs have special treatment.
    """
    masks=state['masks'];annotations=state.get('annotations',{})
    bounds=bounds if bounds is not None else {i:mask_bounds(m) for i,m in masks.items()}
    eligible=sorted(i for i in allowed_ids if i in masks
                    and annotations.get(str(i),{}).get('source')!='manual_cut'
                    and (mask&masks[i]).sum()>=masks[i].sum()*.8)
    union=np.zeros_like(mask)
    for old in masks.values():union|=old
    best=None;best_evidence=(-1.,-1)
    for index,i in enumerate(eligible):
        check_stop(stop_event)
        roi=region_slice(bounds[i],gray.shape,2)
        near=ndi.binary_dilation(masks[i][roi],iterations=2)
        near_bounds=(roi[1].start,roi[0].start,roi[1].stop,roi[0].stop)
        for j in eligible[index+1:]:
            check_stop(stop_event)
            if not intersects(near_bounds,bounds[j]) or not (near&masks[j][roi]).any():continue
            if not any(annotations.get(str(k),{}).get('source')=='automated_sam' for k in (i,j)):continue
            protected=union&~(masks[i]|masks[j])
            candidate=mask&~protected
            combined,ids,_=reconcile(candidate,masks,gray,minimum,contrast,bounds)
            if combined is None or set(ids)!={i,j}:continue
            # Do not drop an existing fragment while consolidating the pair.
            if ((masks[i]|masks[j])&~combined).any():continue
            # Prefer the pair most completely supported by the SAM hypothesis,
            # rather than favoring a larger unrelated neighbor merely by area.
            coverage=float(((mask&masks[i]).sum()/masks[i].sum()+(mask&masks[j]).sum()/masks[j].sum())/2)
            evidence=(coverage,int(combined.sum()))
            if evidence>best_evidence:best=(combined,ids);best_evidence=evidence
    return best if best is not None else (None,[])


def _add_candidate(editor,state,payload,stop_event,staged):
    """Add or merge one candidate, staging all changes until Automate completes."""
    import copy
    check_stop(stop_event)
    editor.check_revision(state,payload)
    result=editor.preview(state,dict(revision=state['revision'],box=payload.get('box'),points=[],labels=[],fill_holes=True))
    check_stop(stop_event)
    settings=state['report'].get('selection_settings',{})
    minimum=settings.get('min_area_pixels',100);minimum_contrast=settings.get('min_contrast',8)
    gray=state.get('sam_gray',state['gray'])
    choices=sorted(zip(state['preview']['masks'],result['choices']),key=lambda item:item[1]['score'],reverse=True)
    bounds={i:mask_bounds(m) for i,m in state['masks'].items()}
    protected=np.zeros(gray.shape,bool)
    for candidate_id,old in state['masks'].items():
        if state['annotations'].get(str(candidate_id),{}).get('source')=='manual_cut':protected|=old
    for mask,choice in choices:
        mask=mask&~protected
        area=int(mask.sum())
        if choice['score']<MIN_SAM_SCORE or not minimum<=area<=gray.size*.2:continue
        clipped,merged_ids,review=reconcile(mask,state['masks'],gray,minimum,minimum_contrast,bounds)
        if payload.get('consolidation_only') and (clipped is None or len(merged_ids)<2 or not set(merged_ids)<=set(payload['candidate_ids'])):
            clipped,merged_ids=reconcile_touching_pair(mask,state,payload['candidate_ids'],gray,minimum,minimum_contrast,stop_event,bounds)
            if clipped is None:continue
        resolution='merge' if merged_ids else 'add'
        if clipped is None:
            clipped=trim_candidate(mask,state['masks'],gray,minimum,minimum_contrast)
            merged_ids=[];resolution='trim'
        if clipped is None:continue
        masks=state['masks'].copy();candidate_id=min(merged_ids) if merged_ids else state['next_id']
        for old_id in merged_ids:masks.pop(old_id)
        masks[candidate_id]=clipped
        annotations=copy.deepcopy(state['annotations'])
        for old_id in merged_ids:annotations.pop(str(old_id),None)
        annotations[str(candidate_id)]=dict(source='automated_sam',supplemental=True,review_status='unreviewed',box=payload['box'],merged_from=merged_ids,overlap_resolution=resolution,sam_score=float(choice['score']))
        next_id=state['next_id'] if merged_ids else candidate_id+1
        check_stop(stop_event)
        if staged:
            state.update(masks=masks,annotations=annotations,next_id=next_id,preview=None)
            return dict(added=not bool(merged_ids),merged=bool(merged_ids),candidate_id=candidate_id)
        response=editor.save(state,masks,annotations,state['history']+[state['revision']],next_id,dict(type='automate',candidate_id=candidate_id,merged_from=merged_ids,box=payload['box']))
        return dict(added=not bool(merged_ids),merged=bool(merged_ids),state=response,candidate_id=candidate_id)
    state['preview']=None
    return dict(added=False)
