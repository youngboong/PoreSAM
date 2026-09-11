"""Export one immutable revision into a new folder without overwriting files."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import uuid
from datetime import datetime


def export_bundle(editor,state,directory):
    if not isinstance(directory,str) or not directory.strip():
        raise ValueError('저장 폴더를 지정해주세요.')
    parent=Path(directory.strip()).expanduser()
    if not parent.is_absolute():raise ValueError('저장 폴더는 전체 경로로 입력해주세요.')
    parent=parent.resolve()
    if parent.exists() and not parent.is_dir():raise ValueError('파일이 아닌 폴더를 지정해주세요.')
    source=Path(state['report']['image'])
    if not source.is_absolute():source=Path(__file__).resolve().parents[1]/source
    original_name=source.name
    for project in editor.workflow.projects.values():
        if any(run['dataset']==state['dataset'] for run in project['runs']):
            source=editor.workflow.root/project['id']/project['original_file']
            original_name=Path(project['name']).name
            break
    if not source.is_file():raise ValueError('원본 이미지 파일을 찾을 수 없습니다.')
    folder=editor.revision_folder(state['dataset'],state['revision']) if state['revision'] else state['baseline']
    if any(parent.is_relative_to((folder/name).resolve()) for name in ['images','measurements']):
        raise ValueError('현재 결과의 images 또는 measurements 내부 대신 별도 저장 폴더를 선택해주세요.')
    label=re.sub(r'[^\w.-]+','_',Path(original_name).stem).strip('._')[:60] or 'image'
    name=f"{label}_revision_{state['revision']:04d}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
    destination=parent/name
    staging=parent/('.'+name+'.partial')
    if not destination.resolve().is_relative_to(parent) or not staging.resolve().is_relative_to(parent):
        raise ValueError('저장 경로를 확인해주세요.')
    parent.mkdir(parents=True,exist_ok=True)
    staging.mkdir()
    (staging/'original').mkdir()
    # Keep original bytes (including TIFF bit depth and instrument metadata).
    shutil.copy2(source,staging/'original'/original_name)
    shutil.copytree(folder/'images',staging/'images')
    shutil.copytree(folder/'measurements',staging/'measurements')
    for filename in ['report.json','edit.json','entrance_candidates.npz']:
        if (folder/filename).is_file():shutil.copy2(folder/filename,staging/filename)
    if state['report'].get('sam_input_image'):
        sam_input=Path(state['report']['sam_input_image'])
        if sam_input.is_file():shutil.copy2(sam_input,staging/'images/analysis_input.png')
    manifest=dict(dataset=state['dataset'],revision=state['revision'],original_file='original/'+original_name,
                  original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),report='measurements/index.html')
    (staging/'export.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (staging/'index.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>PoreSAM</title><h1>PoreSAM</h1><p><a href="measurements/index.html">결과 보고서 열기</a></p><p>original: 원본 이미지 · images: 분석 이미지 · measurements: 보고서 및 측정값</p></html>',encoding='utf-8')
    staging.rename(destination)
    return str(destination)
