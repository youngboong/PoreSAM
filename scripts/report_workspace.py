"""Independent, persistent report drafts combining explicitly selected analyses."""
import copy
import hashlib
import json
import re
import uuid
import base64
import io
from pathlib import Path
from PIL import Image, ImageOps
import numpy as np
from datetime import datetime
import report_composer as composer


def folder(editor):
    path=editor.output_root/'report_workspaces';path.mkdir(parents=True,exist_ok=True);return path


def path(editor,identifier):
    if not isinstance(identifier,str) or not re.fullmatch(r'report_[a-f0-9]{32}',identifier):
        raise ValueError('Invalid report identifier.')
    return folder(editor)/(identifier+'.json')


def write(editor,report):
    report['updated']=datetime.now().isoformat(timespec='seconds')
    target=path(editor,report['report_id']);temporary=target.with_suffix('.pending')
    temporary.write_text(json.dumps(report,ensure_ascii=False),encoding='utf-8');temporary.replace(target)
    (folder(editor)/'active.json').write_text(json.dumps(report['report_id']),encoding='utf-8')


def asset_id(dataset,key):
    return 'source_'+hashlib.sha256(dataset.encode()).hexdigest()[:24]+'_'+key


def rebuild(editor,report,datasets):
    if not isinstance(datasets,list) or len(datasets)>30 or any(not isinstance(d,str) for d in datasets) or len(set(datasets))!=len(datasets):
        raise ValueError('Choose up to 30 distinct analyses.')
    known={a['dataset'] for image in editor.image_library() for a in image['analyses']}
    if any(d not in known for d in datasets):raise ValueError('An analysis is no longer available. Choose it again.')
    old_sources=report.get('sources',[]);old_summary=report.get('summary','');old_options=report.get('options')
    assets=copy.deepcopy(report.get('folder_assets',[]));sources=copy.deepcopy(report.get('folder_sources',[]));summaries=[]
    for dataset in datasets:
        state=editor.state(dataset);data=composer.content(editor,state)
        sources.append(dict(dataset=dataset,name=data['name'],revision=data['revision']))
        summaries.append(data['summary'])
        for asset in data['assets']:
            assets.append(dict(asset,id=asset_id(dataset,asset['id']),base_id=asset['id'],source_dataset=dataset,source_name=data['name']))
    specs=[p for p in report.get('plots',[]) if p['dataset'] in datasets]
    for spec in specs:
        prior=next((a for a in report.get('assets',[]) if a['id']==spec['id']),None)
        revision=editor.state(spec['dataset'])['revision']
        asset=prior if prior and prior.get('plot_revision')==revision and prior.get('plot_language')=='ko' and prior.get('plot_style')=='report-histogram-v4' else make_plot(editor,spec)
        assets.append(dict(asset,id=spec['id']))
    report['plots']=specs
    composer.fingerprint_assets(assets)
    changed=sources!=old_sources
    report.update(sources=sources,assets=assets,summary='\n\n'.join(summaries),revision=report.get('revision',0)+(1 if changed else 0))
    if old_options is None:
        options=composer.defaults(report)
        preset=editor.output_root/'report_conditions_default.json'
        if preset.is_file():options['conditions']=json.loads(preset.read_text(encoding='utf-8'))
    else:
        options=copy.deepcopy(old_options);allowed={a['id'] for a in assets}|{a['id'] for a in options['extra_images']}
        options['items']=[i for i in options['items'] if i['id'].startswith('figure_') or i['id'] in allowed]
        for item in options['items']:
            if 'image_ids' in item:
                item['image_ids']=[k for k in item['image_ids'] if k in allowed]
                item['rows']=[row for row in ([k for k in row if k in allowed] for row in item.get('rows',[[k] for k in item['image_ids']])) if row]
        if options['results']==old_summary:
            options['results']=report['summary'];options['summary_revision']=report['revision']
    old_datasets={s['dataset'] for s in old_sources}
    for source in sources:
        if source.get('kind')!='folder' and source['dataset'] not in old_datasets:
            key=asset_id(source['dataset'],'comparison')
            options['items'].append(dict(id='figure_'+uuid.uuid4().hex,selected=True,caption=source['name'],image_ids=[key],rows=[[key]],columns=1))
    report['options']=composer.validate(options,report)


def current(editor,report):
    for source in report['sources']:
        if source.get('kind')=='folder':continue
        if editor.state(source['dataset'])['revision']!=source['revision']:
            raise ValueError('An included analysis changed. Choose Load / Update Content before previewing or exporting.')


def import_folders(report,directories):
    if not isinstance(directories,list) or not directories or len(directories)>30 or any(not isinstance(d,str) for d in directories):
        raise ValueError('Choose 1 to 30 folders.')
    sources=copy.deepcopy(report.get('folder_sources',[]));assets=copy.deepcopy(report.get('folder_assets',[]))
    known={s['dataset'] for s in sources};size=sum(len(a['image']) for a in assets)
    for directory in directories:
        path=Path(directory).expanduser()
        if not path.is_absolute() or not path.is_dir():raise ValueError('Choose an existing folder: '+directory)
        path=path.resolve();key='folder_'+hashlib.sha256(str(path).casefold().encode()).hexdigest()[:24]
        if key in known:continue
        files=sorted((p for p in path.iterdir() if p.is_file() and p.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff','.bmp','.webp')),key=lambda p:p.name.casefold())
        if not files:raise ValueError('No supported images in '+path.name)
        if len(assets)+len(files)>120:raise ValueError('Use at most 120 folder images per report.')
        if len(sources)>=30:raise ValueError('Use at most 30 folders per report.')
        for file in files:
            with Image.open(file) as original:
                if original.width*original.height>40_000_000:raise ValueError('Image is too large: '+file.name)
                image=ImageOps.exif_transpose(original)
                if image.mode in ('I','F') or image.mode.startswith('I;16'):
                    pixels=np.asarray(image,dtype=float);finite=pixels[np.isfinite(pixels)]
                    lo,hi=(float(finite.min()),float(finite.max())) if finite.size else (0,0)
                    pixels=np.zeros_like(pixels) if hi<=lo else np.nan_to_num((pixels-lo)*255/(hi-lo),nan=0,posinf=255,neginf=0)
                    image=Image.fromarray(np.clip(pixels,0,255).astype('uint8'))
                image=image.convert('RGB');stream=io.BytesIO();image.save(stream,format='PNG')
            uri='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii');size+=len(uri)
            if size>40_000_000:raise ValueError('Folder images exceed 30 MB. Choose fewer images or smaller folders.')
            assets.append(dict(id=asset_id(key,file.name),kind='image',title=file.name,image=uri,source_dataset=key,source_name=path.name))
        sources.append(dict(dataset=key,name=path.name,kind='folder'));known.add(key)
    report.update(folder_sources=sources,folder_assets=assets)


def make_plot(editor,spec):
    from pore_details_export import KOREAN_LABELS as LABELS,export_plot
    state=editor.state(spec.get('dataset'))
    kind,x,y=spec.get('kind'),spec.get('x_key'),spec.get('y_key')
    if kind not in ('histogram','scatter') or x not in LABELS or (kind=='scatter' and y not in LABELS):raise ValueError('Choose valid plot axes.')
    exclude=spec.get('exclude_boundary',True)
    if type(exclude) is not bool:raise ValueError('Invalid boundary setting.')
    rows=editor.measurements(state)['candidates']
    ids=[r['candidate_id'] for r in rows if not exclude or not r['touches_image_edge']]
    result=export_plot(state,dict(kind=kind,x_key=x,y_key=y,bins=spec.get('bins','auto'),candidate_ids=ids),korean=True)
    from report_bundle import source_image
    _,name=source_image(editor,state)
    title=LABELS[x]+' 히스토그램' if kind=='histogram' else LABELS[y]+' vs '+LABELS[x]
    return dict(kind=kind,title=title,image=result['image'],source_dataset=state['dataset'],source_name=name,plot_revision=state['revision'],plot_language='ko',plot_style='report-histogram-v4',count=result['count'])


def handle(editor,payload):
    action=payload.get('action')
    if action=='plot_choices':
        from pore_details_export import KOREAN_LABELS as LABELS
        return dict(metrics=[dict(key=k,label=v) for k,v in LABELS.items()],sources=[dict(dataset=e['latest_dataset'],name=e['name']) for e in editor.image_library() if e['analyzed']])
    if action=='plot_preview':return make_plot(editor,payload.get('spec',{}))
    if action=='list':
        reports=[]
        for file in folder(editor).glob('report_*.json'):
            r=json.loads(file.read_text(encoding='utf-8'));reports.append(dict(id=r['report_id'],title=r['options']['title'],updated=r['updated'],count=len(r['sources'])))
        active=folder(editor)/'active.json'
        return dict(reports=sorted(reports,key=lambda r:r['updated'],reverse=True),active=json.loads(active.read_text()) if active.is_file() else None)
    if action=='create':
        identifier='report_'+uuid.uuid4().hex
        report=dict(report_id=identifier,dataset=identifier,revision=0,name='SEM report',assets=[],sources=[],summary='')
        rebuild(editor,report,payload.get('datasets',[]))
        # Import the current single-image draft without losing text or layout.
        if payload.get('import_options') is not None and len(report['sources'])==1:
            source=report['sources'][0]['dataset'];original=composer.validate(payload['import_options'],composer.content(editor,editor.state(source)))
            for item in original['items']:
                if not item['id'].startswith('figure_'):item['id']=asset_id(source,item['id'])
                if 'image_ids' in item:
                    rename=lambda k:k if k.startswith('extra_') else asset_id(source,k)
                    item['image_ids']=[rename(k) for k in item['image_ids']];item['rows']=[[rename(k) for k in row] for row in item['rows']]
            original['summary_revision']=report['revision'];report['options']=composer.validate(original,report)
        write(editor,report);return report
    report=json.loads(path(editor,payload.get('report_id')).read_text(encoding='utf-8'))
    if action=='add_plot':
        if len(report.get('plots',[]))>=40:raise ValueError('Use at most 40 custom plots per report.')
        report['options']=composer.validate(payload.get('options',report['options']),report)
        raw=payload.get('spec',{});spec={k:raw.get(k) for k in ('dataset','kind','x_key','y_key','bins','exclude_boundary')}
        asset=make_plot(editor,spec);spec['id']='plot_'+uuid.uuid4().hex
        old_items=copy.deepcopy(report['options']['items'])
        datasets=[s['dataset'] for s in report['sources'] if s.get('kind')!='folder']
        if spec['dataset'] not in datasets:datasets.append(spec['dataset'])
        report.setdefault('plots',[]).append(spec)
        report['assets'].append(dict(asset,id=spec['id']))
        rebuild(editor,report,datasets)
        report['options']['items']=old_items
        write(editor,report);return dict(report=report,asset_id=spec['id'])
    if action=='load':
        composer.fingerprint_assets(report['assets'])
        (folder(editor)/'active.json').write_text(json.dumps(report['report_id']),encoding='utf-8');return report
    if action=='import_folders':
        if payload.get('options') is not None:report['options']=composer.validate(payload['options'],report)
        import_folders(report,payload.get('directories'))
        rebuild(editor,report,[s['dataset'] for s in report['sources'] if s.get('kind')!='folder'])
        write(editor,report);return report
    if action=='remove_folder':
        if payload.get('options') is not None:report['options']=composer.validate(payload['options'],report)
        key=payload.get('source');report['folder_sources']=[s for s in report.get('folder_sources',[]) if s['dataset']!=key]
        report['folder_assets']=[a for a in report.get('folder_assets',[]) if a['source_dataset']!=key]
        rebuild(editor,report,[s['dataset'] for s in report['sources'] if s.get('kind')!='folder'])
        write(editor,report);return report
    if action=='refresh':
        if payload.get('options') is not None:report['options']=composer.validate(payload['options'],report)
        rebuild(editor,report,payload.get('datasets',[s['dataset'] for s in report['sources'] if s.get('kind')!='folder']))
        write(editor,report);return report
    if action=='conditions':
        options=composer.validate(dict(report['options'],conditions=payload.get('conditions')),report)
        target=editor.output_root/'report_conditions_default.json';temporary=target.with_suffix('.pending')
        temporary.write_text(json.dumps(options['conditions'],ensure_ascii=False),encoding='utf-8');temporary.replace(target);return dict(saved=True)
    if action not in ['save','preview','pdf_preview','export']:raise ValueError('Unknown report action.')
    options=composer.validate(payload.get('options',payload.get('report_options')),report)
    if action=='save':report['options']=options;write(editor,report);return dict(saved=True)
    current(editor,report)
    if action=='preview':return dict(html=composer.render_html(report,options))
    if action=='pdf_preview':return composer.preview_pdf(editor,report,options)
    report['options']=options;write(editor,report)
    return dict(exported_folder=composer.export_data(report,options,payload.get('export_directory'),options['title'] or 'SEM'))
