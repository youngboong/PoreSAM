"""Separate immutable image and report exports."""
import base64
import colorsys
from datetime import datetime
from pathlib import Path
import re
import shutil
import uuid
import cv2
import numpy as np
from PIL import Image


def source_image(editor, state):
    for project in editor.workflow.projects.values():
        if any(run['dataset']==state['dataset'] for run in project['runs']):
            return editor.workflow.root/project['id']/'input/normalized.png', project['name']
    source=Path(state['report']['image'])
    if not source.is_absolute():source=Path(__file__).resolve().parents[1]/source
    return source, state['report'].get('source_name',source.name)


def export_paths(editor,state,directory,kind):
    if not isinstance(directory,str) or not directory.strip():raise ValueError('Choose a destination folder.')
    parent=Path(directory.strip()).expanduser()
    if not parent.is_absolute():raise ValueError('Enter the full destination folder path.')
    parent=parent.resolve()
    if parent.exists() and not parent.is_dir():raise ValueError('Choose a folder, not a file.')
    _,original_name=source_image(editor,state)
    label=re.sub(r'[^\w.-]+','_',Path(original_name).stem).strip('._')[:60] or 'image'
    name=f"{label}_{kind}_revision_{state['revision']:04d}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    destination=parent/name;staging=parent/('.'+name+'.partial')
    parent.mkdir(parents=True,exist_ok=True);staging.mkdir()
    return staging,destination


def export_images(editor,state,payload):
    color=payload.get('color','#ffd700');opacity=payload.get('opacity',40);labels=payload.get('labels',True)
    if not isinstance(color,str) or not re.fullmatch(r'#[0-9a-fA-F]{6}',color):raise ValueError('Select a valid pore color.')
    if type(opacity) not in (int,float) or not np.isfinite(opacity) or not 0<=opacity<=100:raise ValueError('Opacity must be between 0 and 100.')
    if type(labels) is not bool:raise ValueError('Select a valid label setting.')
    source,_=source_image(editor,state)
    with Image.open(source) as image:original=np.array(image.convert('RGB'))
    height,width=state['gray'].shape
    if original.shape[1]!=width or original.shape[0]<height:raise ValueError('Original image dimensions do not match the masks.')
    mono=original.copy();multi=original.copy();rgb=np.array([int(color[i:i+2],16) for i in (1,3,5)]);alpha=opacity/100
    for candidate_id,mask in sorted(state['masks'].items()):
        unique=np.array(colorsys.hsv_to_rgb((candidate_id*.61803398875)%1,.8,1))*255
        for pixels,tint in [(mono,rgb),(multi,unique)]:
            roi=pixels[:height];roi[mask]=np.rint(roi[mask]*(1-alpha)+tint*alpha).astype(np.uint8)
            if labels and mask.any():
                y,x=np.unravel_index(cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,3).argmax(),mask.shape)
                cv2.putText(roi,str(candidate_id),(int(x)-6,int(y)+4),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),3)
                cv2.putText(roi,str(candidate_id),(int(x)-6,int(y)+4),cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1)
    directory=payload.get('directory')
    if not isinstance(directory,str) or not directory.strip():raise ValueError('Choose a destination folder.')
    destination=Path(directory.strip()).expanduser()
    if not destination.is_absolute():raise ValueError('Enter the full destination folder path.')
    destination=destination.resolve();destination.mkdir(parents=True,exist_ok=True)
    raw_source=source
    _,original_name=source_image(editor,state)
    for project in editor.workflow.projects.values():
        if any(run['dataset']==state['dataset'] for run in project['runs']):
            raw_source=editor.workflow.root/project['id']/project['original_file'];break
    label=re.sub(r'[^\w.-]+','_',Path(original_name).stem).strip('._')[:60] or 'image'
    suffix=raw_source.suffix.lower() or '.png'
    counter=1
    while True:
        stem=label if counter==1 else f'{label}_{counter}'
        names=[stem+'_comparison.png',stem+'_segmentation.png',stem+'_pores_colored.png',stem+'_original'+suffix]
        if not any((destination/name).exists() for name in names):break
        counter+=1
    import tempfile
    with tempfile.TemporaryDirectory(prefix='.poresam-',dir=destination) as temp:
        staging=Path(temp)
        Image.fromarray(np.concatenate([original,mono],axis=1)).save(staging/names[0])
        Image.fromarray(mono).save(staging/names[1])
        Image.fromarray(multi).save(staging/names[2])
        shutil.copyfile(raw_source,staging/names[3])
        for name in names:(staging/name).rename(destination/name)
    return str(destination)


def export_bundle(editor,state,directory):
    folder=editor.revision_folder(state['dataset'],state['revision']) if state['revision'] else state['baseline']
    measurements=folder/'measurements'
    page=(measurements/'index.html').read_text(encoding='utf-8')
    # A standalone HTML report: embed its figures, with no external image folders.
    def embed(match):
        src=match.group(1)
        if src.startswith('data:'):return match.group(0)
        import html
        image=(measurements/html.unescape(src)).resolve()
        if not image.is_file():raise ValueError('Report image is missing.')
        return 'src="data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode()+'"'
    page=re.sub(r'src="([^"]+)"',embed,page)
    page=re.sub(r'<p><a href="candidates.csv">.*?</p>','<p><a href="report.pdf">PDF Report</a></p>',page,flags=re.S)
    staging,destination=export_paths(editor,state,directory,'report')
    shutil.copy2(measurements/'dashboard.pdf',staging/'report.pdf')
    (staging/'report.html').write_text(page,encoding='utf-8')
    staging.rename(destination)
    return str(destination)
