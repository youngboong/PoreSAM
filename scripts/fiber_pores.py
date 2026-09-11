"""Experimental pore proposals for thick struts with superimposed fine fibers."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.morphology import h_maxima


def normalized_image(gray):
    lo,hi=np.percentile(gray,[2,98])
    return np.clip((gray.astype(float)-lo)*255/max(hi-lo,1),0,255).astype(np.uint8)


def strut_proposals(gray, radius_fraction=.018, prominent=False):
    normalized=normalized_image(gray)
    radius=max(2,round(min(gray.shape)*radius_fraction))
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(radius*2+1,)*2)
    simplified=cv2.GaussianBlur(cv2.morphologyEx(normalized,cv2.MORPH_OPEN,kernel),(0,0),2)
    threshold,_=cv2.threshold(simplified,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    solid=(simplified>threshold*.8).astype(np.uint8)
    solid=cv2.morphologyEx(solid,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7)))
    void=solid==0
    distance=ndi.distance_transform_edt(void)
    seeds=peak_local_max(distance,min_distance=max(12,round(min(gray.shape)*.035)),threshold_abs=max(5,radius*.6),exclude_border=False)
    markers=np.zeros(gray.shape,np.int32)
    for i,(y,x) in enumerate(seeds): markers[y,x]=i+1
    if prominent:
        markers,n=ndi.label(h_maxima(distance,min(gray.shape)*.02)&(distance>max(5,radius*.6)))
    else:n=len(seeds)
    regions=watershed(-distance,markers,mask=void,watershed_line=True)
    masks=[]
    for label in range(1,n+1):
        mask=ndi.binary_fill_holes(regions==label)
        if .002*gray.size<=mask.sum()<=.4*gray.size:masks.append(mask)
    return masks,simplified,solid,dict(radius_pixels=radius,threshold=float(threshold*.8),seed_count=n,prominent=prominent)


def mask_iou(a,b):
    return float((a&b).sum()/max(1,(a|b).sum()))


def prompts(mask):
    ys,xs=np.where(mask)
    box=np.array([max(0,xs.min()-4),max(0,ys.min()-4),min(mask.shape[1]-1,xs.max()+4),min(mask.shape[0]-1,ys.max()+4)],np.float32)
    distance=ndi.distance_transform_edt(mask)
    valid=distance>=max(2,distance.max()*.25)
    points=[]
    remaining=distance.copy()
    for _ in range(3):
        y,x=np.unravel_index(remaining.argmax(),mask.shape)
        if remaining[y,x]<=0: break
        points.append([int(x),int(y)])
        yy,xx=np.indices(mask.shape)
        spacing=np.sqrt((xx-x)**2+(yy-y)**2)
        remaining=np.minimum(remaining,spacing)*valid
    return box,np.asarray(points,np.float32)


def select_prediction(masks,scores,proposal,gray):
    ranked=[]
    for mask,score in zip(masks,scores):
        components,n=ndi.label(mask)
        if not n: continue
        counts=np.bincount(components.ravel());counts[0]=0
        largest=components==counts.argmax()
        if largest.sum()<.8*mask.sum(): continue
        filled=ndi.binary_fill_holes(largest)
        ring=ndi.binary_dilation(filled,iterations=4)&~filled
        contrast=float(gray[ring].mean()-gray[filled].mean()) if ring.any() else 0
        overlap=mask_iou(filled,proposal)
        if contrast<0 or overlap<.2 or not .001*gray.size<=filled.sum()<=.4*gray.size:continue
        rank=.7*overlap+.3*float(score)
        ranked.append((rank,filled,dict(sam_score=float(score),proposal_iou=overlap,contrast=contrast)))
    return max(ranked,key=lambda row:row[0])[1:] if ranked else None


def deduplicate(masks,threshold=.85):
    result=[];regions=[]
    for mask in sorted(masks,key=lambda m:int(m.sum()),reverse=True):
        ys,xs=np.where(mask)
        if not len(ys):continue
        box=(int(xs.min()),int(ys.min()),int(xs.max())+1,int(ys.max())+1);area=int(mask.sum())
        duplicate=False
        for other,(old,old_area) in zip(result,regions):
            x0,y0=max(box[0],old[0]),max(box[1],old[1]);x1,y1=min(box[2],old[2]),min(box[3],old[3])
            if x0>=x1 or y0>=y1:continue
            overlap=int((mask[y0:y1,x0:x1]&other[y0:y1,x0:x1]).sum())
            if overlap/(area+old_area-overlap)>threshold:duplicate=True;break
        if not duplicate:result.append(mask);regions.append((box,area))
    return result


def large_consensus(masks):
    """Select representative large proposals, keeping alternatives for review only."""
    pool=[m for m in masks if .015<=m.mean()<=.12]
    similarities=np.eye(len(pool))
    for i in range(len(pool)):
        for j in range(i): similarities[i,j]=similarities[j,i]=mask_iou(pool[i],pool[j])
    remaining=set(range(len(pool)));selected=[];support=[]
    while remaining:
        # Most mutually overlapping proposals first; never use evaluation references.
        ranked=[]
        for i in sorted(remaining):
            neighbors=[j for j in sorted(remaining) if similarities[i,j]>=.55]
            ranked.append((len(neighbors),float(similarities[i,neighbors].mean()),i,neighbors))
        count,agreement,i,neighbors=max(ranked,key=lambda r:(r[0],r[1],-r[2]))
        selected.append(pool[i]);support.append(dict(agreement_count=count,mean_overlap=agreement))
        remaining.difference_update(neighbors)
    return selected,support


def generate_large_pores(gray,predictor,progress=lambda percent,message:None,strength='medium'):
    """Same automatic proposal/point pipeline as the GF comparison, without references."""
    import torch
    if strength not in ['medium','strong']:raise ValueError('구조 약화 강도를 다시 선택해주세요.')
    input_fraction=.009 if strength=='medium' else .018
    proposals=[];merged=[];simplified=None
    for i,fraction in enumerate([.009,.018,.03]):
        progress(5+i*12,'굵은 골격 사이의 영역을 찾고 있습니다.')
        for prominent in [False,True]:
            masks,pixels,_,_=strut_proposals(gray,fraction,prominent)
            (merged if prominent else proposals).extend(masks)
            if fraction==input_fraction:simplified=pixels
    proposals=deduplicate(deduplicate(proposals)+deduplicate(merged))
    chosen=[];audit=[]
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        predictor.set_image(cv2.cvtColor(simplified,cv2.COLOR_GRAY2RGB))
        for i,proposal in enumerate(proposals):
            if i%5==0:progress(42+int(53*i/max(1,len(proposals))),'큰 pore의 윤곽을 확인하고 있습니다.')
            box,points=prompts(proposal)
            masks,scores,_=predictor.predict(box=box,point_coords=points,point_labels=np.ones(len(points),np.int32),multimask_output=True)
            pick=select_prediction(masks,scores,proposal,gray)
            if pick:
                mask,metadata=pick;chosen.append(mask)
                audit.append(dict(box=box.tolist(),points=points.tolist(),**metadata))
    # Keep alternatives: selecting a single representative reduced accuracy in the trial.
    return [m for m in deduplicate(chosen) if .015<=m.mean()<=.12],audit
