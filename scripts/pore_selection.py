"""Shared two-stage nested selection used by the desktop and evaluations."""
import numpy as np
import torch
from torchvision.ops import batched_nms
from sam2.utils.amg import rle_to_mask
from segment_first_pass import ProgressGenerator, candidates_from_masks
from pore_conservative import supplement


class TraceGenerator(ProgressGenerator):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.pool=[]

    def _process_batch(self,*args,**kwargs):
        data=super()._process_batch(*args,**kwargs)
        for i,rle in enumerate(data['rles']):
            self.pool.append(dict(segmentation=rle,predicted_iou=float(data['iou_preds'][i]),
                 stability_score=float(data['stability_score'][i]),box=data['boxes'][i].cpu().tolist()))
        return data


def decode_candidates(pool,iou,stability,nms_threshold=.7,device='cuda',score_dtype=torch.bfloat16):
    # The original comparison runs score filtering inside bfloat16 autocast.
    # Preserve threshold rounding before reconstructing NMS from serialized scores.
    iou=float(torch.tensor(iou,dtype=score_dtype))
    stability=float(torch.tensor(stability,dtype=torch.float32))
    indices=[i for i,c in enumerate(pool) if c['predicted_iou']>iou and c['stability_score']>=stability]
    if not indices:return []
    boxes=torch.tensor([pool[i]['box'] for i in indices],device=device,dtype=torch.float32)
    scores=torch.tensor([pool[i]['predicted_iou'] for i in indices],device=device)
    keep=batched_nms(boxes,scores,torch.zeros_like(scores),nms_threshold).cpu().tolist()
    return [dict(pool[indices[i]],segmentation=rle_to_mask(pool[indices[i]]['segmentation']),pool_id=indices[i]) for i in keep]



def label_map(masks,shape):
    result=np.zeros(shape,np.uint16)
    for i,mask in enumerate(masks,1):
        assert not np.any(result[mask]),'Overlapping output instances'
        result[mask]=i
    return result


def select(pool,gray,min_area,min_contrast,device,variant):
    if variant in ('relaxed_nested','two_stage_nested'):
        from pore_nested import prepare_candidates,resolve_nested,DEFAULTS as NESTED_DEFAULTS
        base=select(pool,gray,min_area,min_contrast,device,variant.removesuffix('_nested'))
        candidates=prepare_candidates(pool,gray,min_area)
        labels,decisions=resolve_nested(label_map(base['masks'],gray.shape),candidates,gray,min_area,min_contrast)
        return dict(base,masks=[labels==i for i in np.unique(labels) if i],
                    nested_decisions=decisions,nested_settings=dict(NESTED_DEFAULTS))
    if variant not in ('relaxed','two_stage'):
        raise ValueError(f'Unknown selection variant: {variant}')
    dtype=torch.bfloat16 if device=='cuda' else torch.float32
    def decode(iou,stability):
        return decode_candidates(pool,iou,stability,device=device,score_dtype=dtype)
    def normal(raw):
        return [c['mask'] for c in candidates_from_masks(raw,gray,min_contrast=min_contrast,min_area=min_area)]
    if variant=='relaxed':
        return dict(masks=normal(decode(.7,.85)))
    strict=normal(decode(.8,.92))
    masks,decisions,settings=supplement(strict,decode(.7,.85),gray,min_area,min_contrast)
    return dict(masks=masks,strict=strict,decisions=decisions,settings=settings)
