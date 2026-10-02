"""Experimental two-stage candidate selection, independent of reference masks."""
import cv2
import numpy as np
from scipy import ndimage as ndi

DEFAULTS=dict(min_score=.7,min_stability=.85,min_contrast=5.,min_rim_support=.60,
              containment_threshold=.95,max_overlap_fraction=.20,min_new_component_fraction=.90,rim_distance=3.,max_area_fraction=.20)


def rim_evidence(mask,gray,distance=3.):
    """Sample opposite sides of the candidate contour using its inward normal."""
    field=cv2.GaussianBlur(mask.astype(np.float32),(0,0),1.5)
    dx=cv2.Sobel(field,cv2.CV_32F,1,0,ksize=3)
    dy=cv2.Sobel(field,cv2.CV_32F,0,1,ksize=3)
    boundary=mask & ~ndi.binary_erosion(mask)
    y,x=np.where(boundary)
    valid=(x>=distance+1)&(x<gray.shape[1]-distance-1)&(y>=distance+1)&(y<gray.shape[0]-distance-1)
    x=x[valid];y=y[valid]
    if not len(x):return np.array([],np.float32)
    norm=np.hypot(dx[y,x],dy[y,x]);valid=norm>1e-5
    x=x[valid];y=y[valid];norm=norm[valid]
    if not len(x):return np.array([],np.float32)
    nx=dx[y,x]/norm;ny=dy[y,x]/norm
    inside=cv2.remap(gray,(x+distance*nx).astype(np.float32)[None],(y+distance*ny).astype(np.float32)[None],cv2.INTER_LINEAR)[0]
    outside=cv2.remap(gray,(x-distance*nx).astype(np.float32)[None],(y-distance*ny).astype(np.float32)[None],cv2.INTER_LINEAR)[0]
    return outside-inside


def supplement(strict_masks,raw_candidates,gray,min_area=100,min_contrast=2,settings=None):
    """Replace contained masks only with quality- and rim-supported parents.

    Partial overlaps remain subject to the conservative overlap and trimming
    checks. The subsequent nested pass can restore independently supported
    children separated by a broad bright surface.
    """
    config={**DEFAULTS,**(settings or {})}
    required_contrast=max(float(min_contrast),config['min_contrast'])
    masks=[np.asarray(m,bool).copy() for m in strict_masks]
    occupied=np.zeros(gray.shape,bool)
    for mask in masks:
        if np.any(mask&occupied):raise ValueError('Strict masks must be disjoint.')
        occupied|=mask
    original=occupied.copy();log=[];eligible=[];replaced_count=0
    def mask_info(mask):
        yy,xx=np.where(mask)
        return (len(yy),int(yy.min()),int(yy.max())+1,int(xx.min()),int(xx.max())+1) if len(yy) else (0,0,0,0,0)
    mask_infos=[mask_info(m) for m in masks]
    def replacement_context(candidate):
        area,y0,y1,x0,x1=mask_info(candidate)
        replaced=[]
        for i,(old_area,oy0,oy1,ox0,ox1) in enumerate(mask_infos):
            if not 0<old_area<area:continue
            sy0,sy1=max(y0,oy0),min(y1,oy1)
            sx0,sx1=max(x0,ox0),min(x1,ox1)
            if sy0>=sy1 or sx0>=sx1:continue
            region=np.s_[sy0:sy1,sx0:sx1]
            if np.count_nonzero(candidate[region]&masks[i][region])>=config['containment_threshold']*old_area:
                replaced.append(i)
        remaining=occupied.copy()
        for i in replaced:remaining[masks[i]]=False
        return replaced,remaining
    smooth=cv2.GaussianBlur(gray.astype(np.float32),(0,0),1.)
    for index,item in enumerate(raw_candidates):
        entry=dict(candidate=index,pool_id=item.get('pool_id',index),score=float(item['predicted_iou']),
                   stability=float(item['stability_score']),accepted=False)
        log.append(entry)
        if entry['score']<=float(np.float32(config['min_score'])) or entry['stability']<config['min_stability']:
            entry['reason']='quality';continue
        mask=np.asarray(item['segmentation'],bool)
        labels,n=ndi.label(mask)
        if not n:entry['reason']='empty';continue
        sizes=np.bincount(labels.ravel());sizes[0]=0;largest=labels==sizes.argmax()
        if largest.sum()<.9*mask.sum():entry['reason']='disconnected';continue
        mask=ndi.binary_fill_holes(largest);area=int(mask.sum());entry['area']=area
        if area<min_area or area>gray.size*config['max_area_fraction']:
            entry['reason']='area';continue
        ring=ndi.binary_dilation(mask,iterations=4)&~mask
        contrast=float(gray[ring].mean()-gray[mask].mean()) if ring.any() else 0.
        entry['contrast']=contrast
        if contrast<required_contrast:entry['reason']='contrast';continue
        contained,remaining=replacement_context(mask)
        overlap=float(np.count_nonzero(mask&remaining)/area);entry['initial_overlap']=overlap
        entry['initial_contained_count']=len(contained)
        if overlap>config['max_overlap_fraction']:entry['reason']='existing_overlap';continue
        evidence=rim_evidence(mask,smooth,config['rim_distance'])
        support=float(np.mean(evidence>=required_contrast)) if evidence.size else 0.
        entry['rim_support']=support
        if support<config['min_rim_support']:entry['reason']='weak_rim';continue
        rank=.5*entry['score']+.3*entry['stability']+.2*support
        eligible.append((rank,index,mask))
    for _,index,mask in sorted(eligible,key=lambda c:(-c[0],c[1])):
        entry=log[index];contained,remaining=replacement_context(mask)
        overlap=float(np.count_nonzero(mask&remaining)/mask.sum())
        if overlap>config['max_overlap_fraction']:entry['reason']='selected_overlap';continue
        novel=mask&~remaining;components,n=ndi.label(novel)
        if not n:entry['reason']='no_new_area';continue
        sizes=np.bincount(components.ravel());sizes[0]=0;new=components==sizes.argmax()
        if new.sum()<min_area or new.sum()<config['min_new_component_fraction']*novel.sum():
            entry['reason']='fragmented_after_trim';continue
        # Clipping against unrelated neighbours must not erase a replacement target.
        if any(np.count_nonzero(new&masks[i])<config['containment_threshold']*int(masks[i].sum()) for i in contained):
            entry['reason']='containment_lost_after_trim';continue
        entry['replaced_mask_indices']=contained
        entry['replaced_areas']=[int(masks[i].sum()) for i in contained]
        added_pixels=int(np.count_nonzero(new&~occupied))
        masks=[old for i,old in enumerate(masks) if i not in contained]
        mask_infos=[info for i,info in enumerate(mask_infos) if i not in contained]
        mask_infos.append(mask_info(new))
        masks.append(new);occupied=remaining|new;replaced_count+=len(contained)
        entry.update(accepted=True,reason='replaced_contained' if contained else 'added',added_pixels=added_pixels)
    assert sum(int(m.sum()) for m in masks)==int(occupied.sum())
    return masks,log,dict(**config,required_contrast=required_contrast,
                          added_masks=len(masks)-len(strict_masks),replaced_masks=replaced_count,removed_pixels=int(np.count_nonzero(original&~occupied)),added_pixels=int(np.count_nonzero(occupied&~original)))
