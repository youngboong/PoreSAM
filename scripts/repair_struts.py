"""Experimental, image-supported connections between interrupted foreground struts."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.graph import route_through_array
from skimage.morphology import skeletonize

from fiber_pores import normalized_image, strut_proposals


def endpoint_direction(skeleton, start, steps=18):
    current=tuple(start);seen={current}
    for _ in range(steps):
        y,x=current
        adjacent=[(yy,xx) for yy in range(max(0,y-1),min(skeleton.shape[0],y+2))
                  for xx in range(max(0,x-1),min(skeleton.shape[1],x+2))
                  if skeleton[yy,xx] and (yy,xx) not in seen]
        if len(adjacent)!=1:break
        current=adjacent[0];seen.add(current)
    vector=np.asarray(start)-current
    length=np.linalg.norm(vector)
    return vector/max(length,1),float(length)


def repair(gray):
    _,simplified,solid,metadata=strut_proposals(gray,.018)
    # Removing small isolated islands prevents their arbitrary skeleton tips acting as anchors.
    labels,n=ndi.label(solid)
    counts=np.bincount(labels.ravel());keep=counts>=max(100,gray.size*.0005);keep[0]=False
    core=keep[labels]
    skeleton=skeletonize(core)
    neighbors=ndi.convolve(skeleton.astype(np.uint8),np.ones((3,3),np.uint8),mode='constant')
    tips=np.argwhere(skeleton&(neighbors==2))
    directions=[endpoint_direction(skeleton,p) for p in tips]
    max_distance=round(min(gray.shape)*.13)
    original=normalized_image(gray)
    support=cv2.GaussianBlur(cv2.morphologyEx(original,cv2.MORPH_OPEN,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))),(0,0),1)
    strength=max(20,metadata['threshold']*.8)
    proposals=[]
    for i,j in sorted(cKDTree(tips).query_pairs(max_distance)) if len(tips) else []:
        a,b=tips[i],tips[j];delta=b-a;distance=np.linalg.norm(delta)
        if distance<8 or min(directions[i][1],directions[j][1])<5:continue
        alignment=min(float(directions[i][0]@(delta/distance)),float(directions[j][0]@(-delta/distance)))
        if alignment<.4:continue
        margin=12
        lo=np.maximum(np.minimum(a,b)-margin,0);hi=np.minimum(np.maximum(a,b)+margin+1,gray.shape)
        patch=support[lo[0]:hi[0],lo[1]:hi[1]].astype(float)
        yy,xx=np.indices(patch.shape);relative=np.stack([yy+lo[0]-a[0],xx+lo[1]-a[1]],axis=-1)
        perpendicular=np.abs(relative[:,:,0]*delta[1]-relative[:,:,1]*delta[0])/distance
        # Prefer short paths on surviving original-image ridges, in a narrow corridor.
        cost=1+5*(1-np.clip(patch/strength,0,1))+perpendicular/12
        cost[perpendicular>12]=np.inf
        path,_=route_through_array(cost,a-lo,b-lo,fully_connected=True,geometric=True)
        path=np.asarray(path)+lo
        values=support[path[:,0],path[:,1]]
        supported=float(np.mean(values>=strength))
        path_length=float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum())
        if supported<.8 or path_length>distance*1.35:continue
        # A connection must cross an actual gap; avoid simply repainting solid struts.
        missing=~core[path[:,0],path[:,1]]
        if missing.sum()<4:continue
        proposals.append((distance/(.5+alignment)* (2-supported),i,j,path,
                          dict(distance_pixels=float(distance),alignment=alignment,supported_fraction=supported,path_length=path_length)))
    result=simplified.copy();bridges=np.zeros(gray.shape,np.uint8);used=set();audit=[]
    widths=ndi.distance_transform_edt(core)
    for _,i,j,path,details in sorted(proposals,key=lambda item:item[0]):
        if i in used or j in used:continue
        width=int(np.clip(min(widths[tuple(tips[i])],widths[tuple(tips[j])]),3,8))
        ink=np.zeros(gray.shape,np.uint8)
        cv2.polylines(ink,[path[:,::-1].astype(np.int32)],False,255,width)
        level=max(float(simplified[tuple(tips[i])]),float(simplified[tuple(tips[j])]),metadata['threshold']*1.3)
        paint=cv2.GaussianBlur(ink.astype(float)/255,(0,0),1)*level
        result=np.maximum(result,paint.clip(0,255).astype(np.uint8));bridges|=ink
        used.update([i,j]);audit.append(dict(start_yx=tips[i].tolist(),end_yx=tips[j].tolist(),width_pixels=width,path_yx=path.tolist(),**details))
    closed=cv2.morphologyEx(simplified,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(29,29)))
    return dict(simplified=simplified,closed=closed,connected=result,bridges=bridges,support=support),dict(
        endpoint_count=len(tips),accepted_connections=len(audit),max_distance_pixels=max_distance,
        alignment_min=.4,support_min=.8,connections=audit)
