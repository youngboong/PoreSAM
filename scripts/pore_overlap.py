"""Exclusive pore ownership and conservative, existing-first candidate review."""
import numpy as np
from scipy import ndimage as ndi


def overlap_pixels(masks):
    masks=list(masks)
    if not masks:return 0
    count=np.zeros(masks[0].shape,np.uint16)
    for mask in masks:count+=mask
    return int(np.count_nonzero(count>1))


def exclusive_existing(masks,annotations=None):
    """Retain every unique pixel; shared pixels belong to an accepted/larger mask first."""
    if not masks:return {},dict(overlap_pixels=0,changed_ids=[],removed_ids=[])
    annotations=annotations or {}
    order=sorted(masks,key=lambda i:(annotations.get(str(i),{}).get('review_status')!='user_accepted',-int(masks[i].sum()),i))
    occupied=np.zeros(next(iter(masks.values())).shape,bool);result={};changed=[];removed=[]
    for i in order:
        mask=masks[i]&~occupied
        if not np.array_equal(mask,masks[i]):changed.append(i)
        if mask.any():result[i]=mask;occupied|=mask
        else:removed.append(i)
    return dict(sorted(result.items())),dict(overlap_pixels=overlap_pixels(masks.values()),changed_ids=changed,removed_ids=removed)


def review_groups(raw,existing,gray,allow_replacement=False):
    """Representatives never overlap. Alternatives appear only when a group is selected."""
    eligible=[];blocked=[]
    for index,mask in enumerate(raw):
        replacement=[];conflicts=[]
        for old_id,old in existing.items():
            intersection=int((mask&old).sum())
            if not intersection:continue
            if allow_replacement and old.sum()<mask.sum() and intersection>=.95*old.sum():replacement.append(old_id)
            else:conflicts.append(old_id)
        if conflicts:
            blocked.append(dict(candidate_id=index+1,conflicting_ids=conflicts));continue
        ring=ndi.binary_dilation(mask,iterations=4)&~mask
        contrast=float(gray[ring].mean()-gray[mask].mean()) if ring.any() else 0.
        eligible.append(dict(id=index+1,mask=mask,score=contrast,replacement_ids=replacement))
    # This is a large-pore supplement: offer the broadest eligible outer boundary first.
    # It is a review choice, not a claim that the largest boundary is ground truth.
    ranked=sorted(eligible,key=lambda item:(-int(item['mask'].sum()),-item['score'],item['id']))
    chosen=[];occupied=np.zeros(gray.shape,bool)
    for item in ranked:
        if (occupied&item['mask']).any():continue
        chosen.append(item);occupied|=item['mask']
    for item in chosen:
        others=occupied&~item['mask'];alternatives=[item['id']]
        for candidate in ranked:
            if candidate['id']==item['id'] or (candidate['mask']&others).any():continue
            intersection=int((item['mask']&candidate['mask']).sum())
            union=int((item['mask']|candidate['mask']).sum())
            if intersection/max(1,union)>=.5:alternatives.append(candidate['id'])
        item['alternatives']=alternatives
    return chosen,dict(raw_count=len(raw),eligible_count=len(eligible),group_count=len(chosen),blocked=blocked)
