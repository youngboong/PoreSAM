"""Independent, persistent report drafts combining explicitly selected analyses."""
import copy
import hashlib
import json
import re
import uuid
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
    assets=[];sources=[];summaries=[]
    for dataset in datasets:
        state=editor.state(dataset);data=composer.content(editor,state)
        sources.append(dict(dataset=dataset,name=data['name'],revision=data['revision']))
        summaries.append(data['name']+'\n'+data['summary'])
        for asset in data['assets']:
            assets.append(dict(asset,id=asset_id(dataset,asset['id']),base_id=asset['id'],source_dataset=dataset,source_name=data['name']))
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
        if source['dataset'] not in old_datasets:
            key=asset_id(source['dataset'],'comparison')
            options['items'].append(dict(id='figure_'+uuid.uuid4().hex,selected=True,caption=source['name'],image_ids=[key],rows=[[key]],columns=1))
    report['options']=composer.validate(options,report)


def current(editor,report):
    for source in report['sources']:
        if editor.state(source['dataset'])['revision']!=source['revision']:
            raise ValueError('An included analysis changed. Choose Load / Update Content before previewing or exporting.')


def handle(editor,payload):
    action=payload.get('action')
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
    if action=='load':
        (folder(editor)/'active.json').write_text(json.dumps(report['report_id']),encoding='utf-8');return report
    if action=='refresh':
        if payload.get('options') is not None:report['options']=composer.validate(payload['options'],report)
        rebuild(editor,report,payload.get('datasets',[s['dataset'] for s in report['sources']]))
        write(editor,report);return report
    if action=='conditions':
        options=composer.validate(dict(report['options'],conditions=payload.get('conditions')),report)
        target=editor.output_root/'report_conditions_default.json';temporary=target.with_suffix('.pending')
        temporary.write_text(json.dumps(options['conditions'],ensure_ascii=False),encoding='utf-8');temporary.replace(target);return dict(saved=True)
    if action not in ['save','preview','export']:raise ValueError('Unknown report action.')
    options=composer.validate(payload.get('options',payload.get('report_options')),report)
    if action=='save':report['options']=options;write(editor,report);return dict(saved=True)
    current(editor,report)
    if action=='preview':return dict(html=composer.render_html(report,options))
    report['options']=options;write(editor,report)
    return dict(exported_folder=composer.export_data(report,options,payload.get('export_directory'),options['title'] or 'SEM'))
