"""Local browser editor for existing PI candidate masks and prompted SAM additions."""
import argparse
import base64
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import mimetypes
from pathlib import Path
import secrets
import threading
import socket
import os
from urllib.parse import urlsplit, unquote

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
import torch

from analyze_candidates import measure_masks, export_folder, current_report_exists
from segment_first_pass import overlay
from result_paths import read_artifact
from project_workflow import ProjectWorkflow
from large_pore_search import LargePoreSearch
from pore_overlap import exclusive_existing,overlap_pixels
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam_runtime import configure_device, inference_context

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["PI35_5kx-4_bse", "PI35_2kx-4_BSE8", "PI100_2kx_bse"]
CONTAINMENT_THRESHOLD = .95


def validated_candidate_ids(masks, ids):
    if not isinstance(ids,list) or not ids or any(type(i) is not int or i not in masks for i in ids):
        raise ValueError('Select valid pores.')
    return sorted(set(ids))


def contained_candidates(mask, masks):
    """Measure coverage of each OLD mask, not IoU or the prompting box."""
    area = int(mask.sum())
    return [candidate_id for candidate_id, old in sorted(masks.items())
            if 0 < int(old.sum()) < area
            and int((mask & old).sum()) >= CONTAINMENT_THRESHOLD * int(old.sum())]


def replacement_preview(masks, ids, shape):
    rgba = np.zeros((*shape, 4), dtype=np.uint8)
    for candidate_id in ids:
        mask = masks[candidate_id]
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(rgba, contours, -1, (255, 90, 180, 255), 2)
        y, x = np.unravel_index(cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 3).argmax(), shape)
        cv2.putText(rgba, str(candidate_id), (int(x)-6, int(y)+4), cv2.FONT_HERSHEY_SIMPLEX, .4, (0,0,0,255), 3)
        cv2.putText(rgba, str(candidate_id), (int(x)-6, int(y)+4), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,90,180,255), 1)
    return png_url(Image.fromarray(rgba))


def png_url(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def mask_preview(mask):
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[mask] = [255, 166, 35, 115]
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(rgba, contours, -1, (255, 194, 60, 255), 2)
    return png_url(Image.fromarray(rgba))


def save_images(folder,gray,masks):
    """Keep human-viewable corrected images together, separate from measurements."""
    images=folder/"images"
    images.mkdir(exist_ok=True)
    original=Image.fromarray(gray).convert("RGB")
    original.save(images/"original.png")
    pixels=np.asarray(overlay(gray,[m for i,m in sorted(masks.items())])).copy()
    union=np.zeros(gray.shape,bool)
    for candidate_id,mask in sorted(masks.items()):
        union|=mask
        y,x=np.unravel_index(cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3).argmax(),mask.shape)
        cv2.putText(pixels,str(candidate_id),(int(x)-6,int(y)+4),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),3)
        cv2.putText(pixels,str(candidate_id),(int(x)-6,int(y)+4),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    result=Image.fromarray(pixels)
    result.save(images/"entrance_candidates_overlay.png")
    Image.fromarray(union.astype(np.uint8)*255).save(images/"union_mask.png")
    comparison=Image.new("RGB",(gray.shape[1]*2,gray.shape[0]))
    comparison.paste(original,(0,0))
    comparison.paste(result,(gray.shape[1],0))
    comparison.save(images/"comparison.png")


class Editor:
    def __init__(self, output_root=None, project_root=None, device='auto'):
        self.device = configure_device(device)
        self.output_root = Path(output_root or ROOT / "outputs" / "manual_edits")
        self.states = {}
        self.predictor = None
        self.encoded_dataset = None
        self.lock = threading.RLock()
        self.workflow = ProjectWorkflow(self, project_root or (self.output_root.parent/'projects'), save_images)
        self.large_search = LargePoreSearch(self, self.output_root.parent/'large_pore_search')

    def baseline(self, dataset):
        if dataset in DATASETS: return ROOT/'outputs'/f'{dataset}_first_pass'
        datasets = self.workflow.datasets()
        if dataset not in datasets: raise ValueError('Image not found.')
        return datasets[dataset][0]

    def datasets(self):
        projects = self.workflow.datasets()
        existing=[d for d in DATASETS if (self.baseline(d)/'report.json').is_file()]
        return dict(datasets=existing+list(projects),labels={**{d:d for d in existing},**{d:item[1] for d,item in projects.items()}},version='workspace-live-preprocessing-v5')

    def image_library(self):
        """Group exact source files, including legacy results, without loading masks."""
        groups={}
        def add(source,name,preview,project_id=None,datasets=()):
            if not source.is_file(): return
            digest=self.workflow.file_digest(source)
            entry=groups.setdefault(digest,dict(id=digest,name=name,preview_url=preview,
                                              project_id=project_id,analyses=[]))
            if project_id: entry['project_id']=project_id
            known={r['dataset'] for r in entry['analyses']}
            for dataset,label in datasets:
                if dataset in known: continue
                report=self.baseline(dataset)/'report.json'
                latest=self.output_root/dataset/'latest.json'
                revision=json.loads(latest.read_text())['revision'] if latest.is_file() else 0
                updated=max(report.stat().st_mtime_ns,latest.stat().st_mtime_ns if latest.is_file() else 0)
                entry['analyses'].append(dict(dataset=dataset,label=label,revision=revision,updated=updated))
        for dataset in DATASETS:
            baseline=self.baseline(dataset)
            if not (baseline/'report.json').is_file(): continue
            report=json.loads((baseline/'report.json').read_text(encoding='utf-8'))
            source=ROOT/report['image']
            add(source,source.name,f'/baseline-files/{dataset}/images/entrance_candidates_overlay.png',datasets=[(dataset,'Saved analysis')])
        with self.workflow.lock:
            projects=copy.deepcopy(list(self.workflow.projects.values()))
        for project in projects:
            source=self.workflow.root/project['id']/project['original_file']
            add(source,project['name'],f"/project-files/{project['id']}/input/normalized.png",project['id'],
                [(r['dataset'],f"Analysis {r['number']}") for r in project['runs']])
        for entry in groups.values():
            entry['analyses'].sort(key=lambda r:r['updated'],reverse=True)
            latest=entry['analyses'][0] if entry['analyses'] else None
            entry.update(analyzed=bool(latest),latest_dataset=latest['dataset'] if latest else None,
                         revision=latest['revision'] if latest else 0,analysis_count=len(entry['analyses']))
        return sorted(groups.values(),key=lambda e:(not e['analyzed'],e['name'].lower()))

    def open_image(self, image_id):
        entry=next((e for e in self.image_library() if e['id']==image_id),None)
        if entry is None: raise ValueError('Refresh the image list and select again.')
        if entry['analyzed']: return dict(dataset=entry['latest_dataset'],image=entry)
        return dict(project=self.workflow.inspect(entry['project_id']),image=entry)

    def receive_image(self, name, content):
        digest=hashlib.sha256(content).hexdigest()
        if any(e['id']==digest for e in self.image_library()):
            return dict(self.open_image(digest),existing=True)
        return dict(project=self.workflow.upload(name,content),existing=False)

    def state(self, dataset):
        if dataset not in self.states:
            baseline = self.baseline(dataset)
            report = json.loads((baseline / "report.json").read_text(encoding="utf-8"))
            gray = np.asarray(Image.open(ROOT / report["image"]).convert("L"))[:report["analysis_bottom_exclusive"]]
            state = dict(dataset=dataset, baseline=baseline, report=report, gray=gray, revision=0,
                         history=[], next_id=report["entrance_candidate_count"]+1, preview=None)
            state['sam_gray']=np.asarray(Image.open(report['sam_input_image']).convert('L')) if report.get('sam_input_image') else gray
            if state['sam_gray'].shape!=gray.shape: raise ValueError('Analysis and source image dimensions differ.')
            latest = self.output_root / dataset / "latest.json"
            revision = json.loads(latest.read_text())["revision"] if latest.exists() else 0
            state.update(self.snapshot(state, revision))
            self.states[dataset] = state
        return self.states[dataset]

    def revision_folder(self, dataset, revision):
        return self.output_root / dataset / f"revision_{revision:04d}"

    def snapshot(self, state, revision):
        folder = self.revision_folder(state["dataset"], revision) if revision else state["baseline"]
        with np.load(read_artifact(folder, "entrance_candidates.npz"), allow_pickle=False) as data:
            masks = {int(k.rsplit("_",1)[1]):data[k].copy() for k in data.files}
        if revision:
            meta = json.loads((folder / "edit.json").read_text(encoding="utf-8"))
        else:
            meta = dict(history=[], next_id=max(masks, default=0)+1,
                        annotations={str(i):dict(source="automatic",review_status="unreviewed") for i in masks})
        return dict(masks=masks, annotations=meta["annotations"], history=meta["history"],
                    next_id=meta["next_id"], revision=revision, preview=None)

    def response(self, state):
        ordered = sorted(state["masks"].items())
        table, stats, _, _, _ = measure_masks(ordered, state["gray"].shape, state["report"]["scale"]["um_per_pixel"])
        table['image_area_percent']=table['area_pixels']/(state['gray'].size)*100
        table['source']=table['candidate_id'].map(lambda i:state['annotations'].get(str(i),{}).get('source','automatic'))
        rgba = np.zeros((*state["gray"].shape,4), dtype=np.uint8)
        fill_rgba=np.zeros_like(rgba)
        edge_rgba=np.zeros_like(rgba)
        label_rgba=np.zeros_like(rgba)
        for candidate_id, mask in ordered:
            contours,_ = cv2.findContours(mask.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            color = (255,215,0,255)
            rgba[mask]=(255,215,0,102)
            fill_rgba[mask]=(255,255,255,255)
            cv2.drawContours(edge_rgba,contours,-1,(255,255,255,255),1)
            cv2.drawContours(rgba,contours,-1,color,1)
            y,x = np.unravel_index(cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3).argmax(),mask.shape)
            cv2.putText(rgba,str(candidate_id),(int(x)-6,int(y)+4),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,0,255),2)
            cv2.putText(rgba,str(candidate_id),(int(x)-6,int(y)+4),cv2.FONT_HERSHEY_SIMPLEX,.36,color,1)
            cv2.putText(label_rgba,str(candidate_id),(int(x)-6,int(y)+4),cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,0,255),2)
            cv2.putText(label_rgba,str(candidate_id),(int(x)-6,int(y)+4),cv2.FONT_HERSHEY_SIMPLEX,.36,(255,255,255,255),1)
        result_base = f"/files/{state['dataset']}/revision_{state['revision']:04d}" if state['revision'] else f"/baseline-files/{state['dataset']}"
        folder = self.revision_folder(state['dataset'],state['revision']) if state['revision'] else state['baseline']
        stored_report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        report_ready = current_report_exists(folder) and (not stored_report.get('report_deferred') or (folder/'report_ready.json').is_file())
        report_url = result_base + '/measurements/index.html' if report_ready else None
        return dict(dataset=state["dataset"],revision=state["revision"],width=state["gray"].shape[1],height=state["gray"].shape[0],
                    image=png_url(Image.fromarray(state["gray"])),overlay=png_url(Image.fromarray(rgba)),
                    fill_overlay=png_url(Image.fromarray(fill_rgba)),edge_overlay=png_url(Image.fromarray(edge_rgba)),label_overlay=png_url(Image.fromarray(label_rgba)),
                    stats=stats, candidates=table.astype(object).where(table.notna(),None).to_dict(orient="records"),can_undo=bool(state["history"]),report_url=report_url,
                    image_url=result_base+'/images/comparison.png' if report_ready else None,result_base=result_base,report_ready=report_ready,
                    export_default_directory=str(ROOT/'outputs/exports'),
                    selection_settings=state['report'].get('selection_settings',dict(min_contrast=8,min_area_pixels=100)),
                    saved_folder=str(self.revision_folder(state["dataset"],state["revision"])) if state["revision"] else None)

    def check_revision(self,state,payload):
        if payload.get("revision") != state["revision"]:
            raise ValueError("Results changed in another session. Reload the image.")

    def coords(self, values, shape):
        array = np.asarray(values,dtype=np.float32)
        if array.ndim != 2 or array.shape[1] != 2 or len(array)>2000 or not np.isfinite(array).all():
            raise ValueError("Invalid coordinates.")
        if (array<0).any() or (array[:,0]>=shape[1]).any() or (array[:,1]>=shape[0]).any():
            raise ValueError("Place points inside the image.")
        return array

    def preview(self,state,payload,polygon=False):
        self.check_revision(state,payload)
        effective_prompts=None
        if polygon:
            points=self.coords(payload.get("polygon",[]),state["gray"].shape)
            if len(points)<3: raise ValueError("A polygon needs at least three vertices.")
            mask=np.zeros(state["gray"].shape,np.uint8)
            cv2.fillPoly(mask,[np.rint(points).astype(np.int32)],1)
            masks=[mask.astype(bool)]
            scores=[None]
        else:
            points = payload.get("points",[])
            labels = payload.get("labels",[])
            coords=self.coords(points,state["gray"].shape) if points else None
            if len(labels)!=len(points) or any(v not in [0,1] for v in labels):
                raise ValueError("Invalid include/exclude points.")
            box=payload.get("box")
            selection=payload.get("shape")
            if selection is not None:
                if not isinstance(selection,dict) or selection.get("kind")!="ellipse" or box is not None:
                    raise ValueError("Draw one box or ellipse.")
                values=np.asarray([selection.get(k) for k in ['cx','cy','rx','ry']],dtype=np.float64)
                if not np.isfinite(values).all(): raise ValueError("Invalid ellipse coordinates.")
                cx,cy,rx,ry=values
                if rx<1.5 or ry<1.5: raise ValueError("Draw a larger ellipse.")
                box=[cx-rx,cy-ry,cx+rx,cy+ry]
                # SAM accepts boxes/points. Negative corner prompts distinguish
                # the oval selection from its enclosing rectangular box.
                corners=[[cx+sx*.9*rx,cy+sy*.9*ry] for sx in [-1,1] for sy in [-1,1]]
                positives=[p for p,label in zip(points,labels) if label==1]
                corners=[p for p in corners if not any(np.linalg.norm(np.asarray(p)-q)<5 for q in positives)]
                points=list(points)+corners
                labels=list(labels)+[0]*len(corners)
                coords=self.coords(points,state["gray"].shape)
            if box is not None:
                box=np.asarray(box,dtype=np.float32)
                if box.shape!=(4,): raise ValueError("Invalid box coordinates.")
                self.coords(box.reshape(2,2),state["gray"].shape)
                if box[2]-box[0]<3 or box[3]-box[1]<3: raise ValueError("Draw a larger box.")
            if box is None and (not labels or 1 not in labels):
                raise ValueError("Draw a box, ellipse or include point.")
            if selection is not None:
                effective_prompts=dict(box=box.tolist(),points=coords.tolist(),labels=labels)
            with inference_context(self.device):
                if self.predictor is None:
                    model=build_sam2("configs/sam2.1/sam2.1_hiera_s.yaml",str(ROOT/"checkpoints/sam2.1_hiera_small.pt"),device=self.device,apply_postprocessing=False)
                    self.predictor=SAM2ImagePredictor(model)
                if self.encoded_dataset!=state["dataset"]:
                    self.predictor.set_image(cv2.cvtColor(state.get('sam_gray',state['gray']),cv2.COLOR_GRAY2RGB))
                    self.encoded_dataset=state["dataset"]
                masks,scores,_=self.predictor.predict(point_coords=coords,point_labels=np.asarray(labels) if labels else None,
                                                     box=box,multimask_output=True)
            order=np.argsort(scores)[::-1]
            masks=[masks[i].astype(bool) for i in order]
            scores=[float(scores[i]) for i in order]
        if payload.get("fill_holes",True): masks=[ndi.binary_fill_holes(m) for m in masks]
        return self.preview_masks(state,payload,masks,scores,"manual_polygon" if polygon else "prompted_sam",effective_prompts)

    def preview_masks(self,state,payload,masks,scores,source,effective_prompts=None):
        self.check_revision(state,payload)
        valid=[(m,s) for m,s in zip(masks,scores) if m.any()]
        if not valid: raise ValueError("Empty result. Adjust the region or points.")
        token=secrets.token_urlsafe(18)
        state["preview"]=dict(token=token,masks=[m for m,s in valid],prompts=copy.deepcopy(payload),
                              source=source)
        if effective_prompts is not None:
            state["preview"]["prompts"]["effective_sam_prompts"]=effective_prompts
        union=np.zeros(state["gray"].shape,bool)
        for mask in state["masks"].values(): union|=mask
        choices=[]
        for mask,score in valid:
            contained_ids=contained_candidates(mask,state["masks"])
            intersecting_ids=[i for i,m in state['masks'].items() if (mask&m).any()]
            choices.append(dict(image=mask_preview(mask),score=score,
                                area_um2=float(mask.sum()*state["report"]["scale"]["um_per_pixel"]**2),
                                overlap_percent=float(100*(mask&union).sum()/mask.sum()),
                                contained_ids=contained_ids,
                                intersecting_ids=intersecting_ids,
                                replacement_image=replacement_preview(state["masks"],contained_ids,mask.shape)))
        return dict(token=token,choices=choices,containment_threshold=CONTAINMENT_THRESHOLD)

    def trim_preview_overlap(self,state,payload):
        self.check_revision(state,payload)
        preview=state['preview']
        if not preview or payload.get('token')!=preview['token']:
            raise ValueError('Generate a new preview.')
        choice=payload.get('choice',0)
        if type(choice) is not int or not 0<=choice<len(preview['masks']):
            raise ValueError('Select a valid preview.')
        target=payload.get('target_id')
        if target is not None and (type(target) is not int or target not in state['masks']):
            raise ValueError('Select a valid target pore.')
        mask=preview['masks'][choice]
        replaced=set(contained_candidates(mask,state['masks']))
        if target is not None: replaced.add(target)
        conflicts=[i for i,m in state['masks'].items() if i not in replaced and (mask&m).any()]
        if not conflicts: raise ValueError('No overlap to trim.')
        clipped=mask.copy()
        for i in conflicts: clipped &= ~state['masks'][i]
        if not clipped.any(): raise ValueError('Trimming removes the entire region. Select the existing pore as the replacement target.')
        # Disconnected regions are separate choices, never one pore spanning islands.
        components,count=ndi.label(clipped,structure=np.ones((3,3)))
        pieces=sorted((components==i for i in range(1,count+1)),key=lambda m:int(m.sum()),reverse=True)
        removed=int(mask.sum()-clipped.sum())
        prompts=copy.deepcopy(preview['prompts'])
        prompts['revision']=state['revision']
        prompts['overlap_trim']=dict(preserved_candidate_ids=conflicts,removed_pixels=removed,target_id=target)
        result=self.preview_masks(state,prompts,pieces,[1.]*len(pieces),preview['source'])
        result.update(trimmed_overlap_pixels=removed,split_count=count)
        return result

    def save(self,state,masks,annotations,history,next_id,action):
        # Old saved analyses may already share boundary pixels. Resolve only on a new save,
        # record every changed ID, and keep undo's exact historical snapshot intact.
        if action.get('type')!='undo':
            masks,cleanup=exclusive_existing(masks,annotations)
            action=dict(action,overlap_cleanup=cleanup)
            for i in cleanup['changed_ids']:
                if str(i) in annotations:
                    annotations[str(i)]=dict(annotations[str(i)],overlap_boundary_adjusted=True,review_status='boundary_adjusted_needs_review')
            for i in cleanup['removed_ids']:annotations.pop(str(i),None)
            assert overlap_pixels(masks.values())==0
        parent=self.output_root/state["dataset"]
        parent.mkdir(parents=True,exist_ok=True)
        existing=[int(p.name.split("_")[-1]) for p in parent.glob("revision_*") if p.is_dir()]
        revision=max(existing,default=0)+1
        folder=self.revision_folder(state["dataset"],revision)
        folder.mkdir()
        report=copy.deepcopy(state["report"])
        report.update(image=str((ROOT/report["image"]).resolve()),entrance_candidate_count=len(masks),
                      revision=revision,candidate_annotations=annotations,status="user edited candidates; not all objects reviewed",report_deferred=True,
                      overlay_relative_path="images/entrance_candidates_overlay.png",
                      candidates=[dict(candidate_id=i,area=int(m.sum())) for i,m in sorted(masks.items())])
        np.savez_compressed(folder/"entrance_candidates.npz",**{f"candidate_{i}":m for i,m in sorted(masks.items())})
        (folder/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
        meta=dict(revision=revision,parent_revision=state["revision"],history=history,next_id=next_id,annotations=annotations,action=action)
        (folder/"edit.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
        # Publish durable masks and history immediately; reports are explicitly requested later.
        pending=parent/"latest.pending.json"
        pending.write_text(json.dumps(dict(revision=revision)),encoding="utf-8")
        pending.replace(parent/"latest.json")
        state.update(masks=masks,annotations=annotations,history=history,next_id=next_id,revision=revision,preview=None)
        return self.response(state)

    def generate_report(self,state,payload):
        self.check_revision(state,payload)
        directory=payload.get('export_directory')
        if directory is not None:
            if not isinstance(directory,str) or not directory.strip() or not Path(directory.strip()).expanduser().is_absolute():
                raise ValueError('Enter the full export folder path.')
        folder = self.revision_folder(state['dataset'],state['revision']) if state['revision'] else state['baseline']
        stored = json.loads((folder/'report.json').read_text(encoding='utf-8'))
        ready = current_report_exists(folder) and (not stored.get('report_deferred') or (folder/'report_ready.json').is_file())
        if not ready:
            save_images(folder,state['gray'],state['masks'])
            export_folder(folder)
            pending = folder/'report_ready.pending.json'
            pending.write_text(json.dumps(dict(revision=state['revision'])),encoding='utf-8')
            pending.replace(folder/'report_ready.json')
        result=self.response(state)
        if directory is not None:
            from report_bundle import export_bundle
            result['exported_folder']=export_bundle(self,state,directory)
        return result

    def mutate(self,state,payload,action):
        self.check_revision(state,payload)
        masks=state["masks"].copy()
        annotations=copy.deepcopy(state["annotations"])
        history=state["history"]+[state["revision"]]
        next_id=state["next_id"]
        if action=="undo":
            if not state["history"]: raise ValueError("Nothing to undo.")
            restored=self.snapshot(state,state["history"][-1])
            return self.save(state,restored["masks"],restored["annotations"],state["history"][:-1],next_id,dict(type="undo"))
        if action=='resolve-overlaps':
            if not overlap_pixels(masks.values()):raise ValueError('No overlaps to resolve.')
            return self.save(state,masks,annotations,history,next_id,dict(type='resolve-overlaps'))
        if action=='cut':
            path=self.coords(payload.get('path', []), state['gray'].shape)
            width=payload.get('width', 3)
            if len(path)<2 or type(width) is not int or not 1<=width<=30:
                raise ValueError('Draw a cut with width from 1 to 30 px.')
            cutter=np.zeros(state['gray'].shape, np.uint8)
            cv2.polylines(cutter, [np.rint(path).astype(np.int32)], False, 1, 1)
            if width>1:
                cutter=cv2.dilate(cutter,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(width,width)))
            cut_pixels=cutter.astype(bool)
            changed=[]
            for candidate_id, old in list(masks.items()):
                if not (old & cut_pixels).any(): continue
                remaining=old & ~cut_pixels
                components, count=ndi.label(remaining, structure=np.ones((3,3)))
                pieces=sorted((components==i for i in range(1,count+1)), key=lambda m:int(m.sum()), reverse=True)
                del masks[candidate_id]
                original_annotation=annotations.pop(str(candidate_id), {})
                child_ids=[]
                for index, piece in enumerate(pieces):
                    child_id=candidate_id if index==0 else next_id
                    if index: next_id+=1
                    masks[child_id]=piece
                    annotations[str(child_id)]=dict(original_annotation, source='manual_cut', review_status='user_accepted', cut_from=candidate_id)
                    child_ids.append(child_id)
                changed.append(dict(candidate_id=candidate_id, resulting_ids=child_ids))
            if not changed: raise ValueError('The cut does not intersect a pore.')
            return self.save(state,masks,annotations,history,next_id,dict(type='cut', path=path.tolist(),width=width,changed=changed))
        if action=='delete':
            ids=validated_candidate_ids(masks,payload.get('target_ids',[payload.get('target_id')]))
            for candidate_id in ids:
                del masks[candidate_id]
                annotations.pop(str(candidate_id),None)
            detail=dict(type='delete',candidate_ids=ids)
            if len(ids)==1:detail['candidate_id']=ids[0]
            return self.save(state,masks,annotations,history,next_id,detail)
        target=payload.get("target_id")
        if target is not None and (type(target) is not int or target not in masks):
            raise ValueError("Select a valid target pore.")
        preview=state["preview"]
        if not preview or payload.get("token")!=preview["token"]: raise ValueError("Generate a new preview.")
        choice=payload.get("choice",0)
        if type(choice) is not int or not 0<=choice<len(preview["masks"]): raise ValueError("Select a valid preview.")
        mask=preview["masks"][choice]
        contained_ids=contained_candidates(mask,masks)
        replaced_ids=sorted(set(contained_ids + ([target] if target is not None else [])))
        conflicts=[i for i,old in masks.items() if i not in replaced_ids and (mask&old).any()]
        if conflicts:
            raise ValueError('Overlaps pores '+', '.join(map(str,conflicts))+'. Refine the boundary or select a replacement target.')
        if target is None:
            target=next_id
            next_id+=1
        # Exact or near duplicate additions usually mean the user intended a replacement.
        for old_id,old in masks.items():
            if old_id not in replaced_ids and (mask&old).sum()/(mask|old).sum()>.9:
                raise ValueError(f"Nearly identical to pore {old_id}. Select it as the replacement target.")
        for old_id in replaced_ids:
            del masks[old_id]
            annotations.pop(str(old_id),None)
        masks[target]=mask
        annotations[str(target)]=dict(source=preview["source"],review_status="user_accepted",prompts=preview["prompts"],
                                      replaced_candidate_ids=replaced_ids)
        detail=dict(type="apply",candidate_id=target,source=preview["source"],prompts=preview["prompts"],
                    replaced_candidate_ids=replaced_ids,contained_candidate_ids=contained_ids,
                    containment_threshold=CONTAINMENT_THRESHOLD)
        return self.save(state,masks,annotations,history,next_id,detail)


def make_handler(editor):
    class Handler(BaseHTTPRequestHandler):
        def send(self,code,content,kind="application/json; charset=utf-8"):
            if isinstance(content,dict): content=json.dumps(content,allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type",kind)
            self.send_header("Content-Length",str(len(content)))
            self.send_header("Cache-Control","no-store")
            self.end_headers()
            self.wfile.write(content)

        def valid_host(self):
            return self.headers.get("Host") in [f"127.0.0.1:{self.server.server_port}",f"localhost:{self.server.server_port}"]

        def do_GET(self):
            if not self.valid_host(): return self.send(403,dict(error="Local access only"))
            path=unquote(urlsplit(self.path).path)
            if path=="/":
                return self.send(200,(ROOT/"ui/editor.html").read_bytes(),"text/html; charset=utf-8")
            if path in ['/workspace.js','/workspace.css','/trials.js','/large-pores.js','/comparison.js','/pore-list.js','/pore-details.js']:
                return self.send(200,(ROOT/'ui'/path[1:]).read_bytes(),'text/javascript; charset=utf-8' if path.endswith('.js') else 'text/css; charset=utf-8')
            if path=="/api/datasets": return self.send(200,editor.datasets())
            if path=="/api/projects": return self.send(200,dict(projects=editor.workflow.list_projects()))
            if path=='/api/images': return self.send(200,dict(images=editor.image_library()))
            if path=='/api/trials':
                folders=sorted((ROOT/'outputs/generalization_trial').glob('*/report.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True)
                return self.send(200,dict(trials=[dict(name=p.parent.name,url=f'/trial-files/{p.parent.name}/index.html') for p in folders]))
            if path.startswith(("/files/","/project-files/","/baseline-files/","/trial-files/")):
                if path.startswith('/baseline-files/'):
                    parts=path[len('/baseline-files/'):].split('/',1)
                    if len(parts)!=2: return self.send(404,dict(error='File not found'))
                    try: root=editor.baseline(parts[0])
                    except ValueError: return self.send(404,dict(error='File not found'))
                    relative=parts[1]
                elif path.startswith('/project-files/'):
                    root=editor.workflow.root
                    relative=path[len('/project-files/'):]
                elif path.startswith('/trial-files/'):
                    root=ROOT/'outputs/generalization_trial'
                    relative=path[len('/trial-files/'):]
                else:
                    root=editor.output_root
                    relative=path[len('/files/'):]
                file=(root/relative).resolve()
                if not file.is_relative_to(root.resolve()) or not file.is_file():
                    return self.send(404,dict(error="File not found"))
                return self.send(200,file.read_bytes(),mimetypes.guess_type(file.name)[0] or "application/octet-stream")
            self.send(404,dict(error="Not found"))

        def do_POST(self):
            if not self.valid_host() or self.headers.get("X-Pore-Editor")!="1":
                return self.send(403,dict(error="Local editor requests only"))
            origin=self.headers.get("Origin")
            if origin and origin not in [f"http://127.0.0.1:{self.server.server_port}",f"http://localhost:{self.server.server_port}"]:
                return self.send(403,dict(error="Origin rejected"))
            try:
                length=int(self.headers.get("Content-Length","0"))
                limit=45_000_000 if self.path=='/api/upload' else 1_000_000
                if not 0<length<limit: raise ValueError("Input is too large.")
                payload=json.loads(self.rfile.read(length))
                if self.path=='/api/choose-export-folder':
                    import subprocess,sys
                    picker="import tkinter as tk; from tkinter import filedialog; r=tk.Tk(); r.withdraw(); r.attributes('-topmost', True); p=filedialog.askdirectory(title='PoreSAM - Select export folder',parent=r); print(p,flush=True); r.destroy()"
                    selected=subprocess.run([sys.executable,'-c',picker],capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'},creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    if selected.returncode:raise ValueError('Cannot open the folder picker. Enter the path manually.')
                    return self.send(200,dict(directory=selected.stdout.strip()))
                if self.path=='/api/upload':
                    result=editor.receive_image(payload.get('name',''),base64.b64decode(payload.get('content',''),validate=True))
                    return self.send(200,result)
                if self.path=='/api/open-image': return self.send(200,editor.open_image(payload.get('image_id')))
                if self.path=='/api/project': return self.send(200,editor.workflow.inspect(payload.get('project_id')))
                if self.path=='/api/preprocess-preview': return self.send(200,editor.workflow.preview_preprocessing(payload.get('project_id'),payload.get('config',{})))
                if self.path=='/api/analyze': return self.send(200,editor.workflow.start(payload.get('project_id'),payload.get('config',{})))
                if self.path=='/api/job': return self.send(200,editor.workflow.status(payload.get('job_id')))
                if self.path=='/api/large-job': return self.send(200,editor.large_search.status(payload.get('job_id')))
                if self.path=='/api/import-existing':
                    dataset=payload.get('dataset')
                    # Existing uploaded images retain their original source and run history.
                    with editor.workflow.lock:
                        project=next((copy.deepcopy(p) for p in editor.workflow.projects.values() if any(r['dataset']==dataset for r in p['runs'])),None)
                    if project:
                        result=editor.workflow.inspect(project['id'])
                        result['config']=next(r['config'] for r in project['runs'] if r['dataset']==dataset)
                        return self.send(200,result)
                    baseline=editor.baseline(payload.get('dataset'))
                    report=json.loads((baseline/'report.json').read_text(encoding='utf-8'))
                    source=ROOT/report['image']
                    return self.send(200,editor.workflow.upload(source.name,source.read_bytes(),seed=baseline))
                if editor.workflow.running:
                    raise ValueError('Wait for analysis to finish before editing.')
                with editor.lock:
                    state=editor.state(payload.get("dataset"))
                    if self.path=="/api/load": result=editor.response(state)
                    elif self.path=='/api/generate-report': result=editor.generate_report(state,payload)
                    elif self.path=='/api/details-plot':
                        editor.check_revision(state,payload)
                        from pore_details_export import export_plot
                        result=export_plot(state,payload)
                    elif self.path=='/api/pore-at-point':
                        editor.check_revision(state,payload)
                        x,y=np.floor(editor.coords([payload.get('point')],state['gray'].shape)[0]).astype(int)
                        candidate_id=next((i for i,mask in sorted(state['masks'].items()) if mask[y,x]),None)
                        result=dict(candidate_id=candidate_id)
                    elif self.path=='/api/pore-highlight':
                        editor.check_revision(state,payload)
                        ids=validated_candidate_ids(state['masks'],payload.get('candidate_ids',[payload.get('candidate_id')]))
                        mask=np.zeros(state['gray'].shape,bool)
                        for candidate_id in ids:mask|=state['masks'][candidate_id]
                        rgba=np.zeros((*mask.shape,4),np.uint8);rgba[mask]=(255,255,255,255)
                        result=dict(image=png_url(Image.fromarray(rgba)),candidate_ids=ids,candidate_id=ids[0] if len(ids)==1 else None)
                    elif self.path=='/api/large-search': result=editor.large_search.start(state,payload)
                    elif self.path=='/api/large-overview': result=editor.large_search.overview(state,payload)
                    elif self.path=='/api/large-pick': result=editor.large_search.pick(state,payload)
                    elif self.path in ["/api/predict","/api/polygon"]:
                        result=editor.preview(state,payload,self.path.endswith("polygon"))
                    elif self.path=='/api/trim-preview-overlap':
                        result=editor.trim_preview_overlap(state,payload)
                    elif self.path in ["/api/apply","/api/delete","/api/undo","/api/resolve-overlaps","/api/cut"]:
                        result=editor.mutate(state,payload,self.path.rsplit("/",1)[1])
                    else: return self.send(404,dict(error="Not found"))
                self.send(200,result)
            except (ValueError,KeyError,TypeError) as exc:
                self.send(400,dict(error=str(exc)))
            except Exception as exc:
                import traceback
                traceback.print_exc()
                self.send(500,dict(error=f"Request failed: {exc}"))
    return Handler


class LocalPoreServer(ThreadingHTTPServer):
    allow_reuse_address=False

    def server_bind(self):
        if os.name=='nt':self.socket.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
        super().server_bind()


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=8765)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto',help='auto: use CUDA if available, otherwise CPU')
    args=parser.parse_args()
    try:editor=Editor(device=args.device)
    except ValueError as exc:parser.error(str(exc))
    try:server=LocalPoreServer(("127.0.0.1",args.port),make_handler(editor))
    except OSError as exc:parser.error(f'Cannot open port {args.port}. Check for an existing PoreSAM server. ({exc})')
    print(f"Pore Editor: http://127.0.0.1:{args.port} | Device: {editor.device.upper()} | {Path(__file__).resolve()}",flush=True)
    server.serve_forever()
