"""Editable SEM report composition from measured pores, with local-only drafts."""
import base64
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import tempfile
import re
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

CONDITIONS = ['기기명', '제조사 및 모델명', '전자총', '가속전압', '분해능', '측정배율', '프로브 지름 / 전류', '작업거리', '검출기']
METRICS = {'length': ('length_um', '길이 (µm)'), 'width': ('width_um', '너비 (µm)'),
           'aspect': ('aspect_ratio', '종횡비'), 'diameter': ('equivalent_diameter_um', '등가원직경 (µm)'),
           'roundness': ('roundness', '원형도'), 'area_fraction': ('image_area_percent', '면적분율 (%)'),
           'area': ('area_um2', '면적 (µm²)')}
NOTE = '크기 통계와 히스토그램은 이미지 경계에 닿지 않는 pore만 포함한다. 길이·너비는 모멘트 등가 타원의 장축·단축이며 Feret 지름이 아니다. 면적분율은 2차원 투영면적 기준이다.'


def plotting():
    from app_paths import output_root
    cache = output_root()/'.matplotlib'; cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'Malgun Gothic', 'axes.unicode_minus': False})
    return plt


def data_url(image):
    stream = io.BytesIO(); image.save(stream, format='PNG')
    return 'data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode('ascii')


def values(rows, key):
    return np.array([r.get(key) for r in rows if not r['touches_image_edge'] and r.get(key) is not None and np.isfinite(r[key])], dtype=float)


def fingerprint_assets(assets):
    for asset in assets:
        if asset.get('image') and not asset.get('image_key'):
            with Image.open(io.BytesIO(base64.b64decode(asset['image'].split(',',1)[1]))) as im:
                im=im.convert('RGBA')
                asset['image_key']=hashlib.sha256(str(im.size).encode()+im.tobytes()).hexdigest()
    return assets


TABLE_METRICS=['diameter','length','width','aspect','roundness','area','area_fraction']
TABLE_COLUMN_LABELS=['Statistic','Equivalent diameter (µm)','Length (µm)','Width (µm)','Aspect ratio','Roundness','Area (µm²)','Area fraction (%)']
TABLE_EXTRA_COLUMNS=[
    ('angle_deg','각도 (°)','Angle (°)'),
    ('area_box_ratio','면적/박스','Area/Box'),
    ('brightness_max','최대 밝기','Brightness max'),
    ('brightness_mean','평균 밝기','Brightness mean'),
    ('brightness_min','최소 밝기','Brightness min'),
    ('brightness_std','밝기 표준편차','Brightness std dev'),
    ('centroid_x_um','중심 X (µm)','Center X (µm)'),
    ('centroid_y_um','중심 Y (µm)','Center Y (µm)'),
    ('convexity','볼록도','Convexity'),
    ('integral_density','적분 밝기','Integral density'),
    ('mass_center_x_um','밝기 가중 중심 X (µm)','Mass center X (µm)'),
    ('mass_center_y_um','밝기 가중 중심 Y (µm)','Mass center Y (µm)'),
    ('perimeter_um','둘레 (µm)','Perimeter (µm)'),
    ('rectangle_bottom_um','박스 아래 (µm)','Rectangle bottom (µm)'),
    ('rectangle_left_um','박스 왼쪽 (µm)','Rectangle left (µm)'),
    ('rectangle_right_um','박스 오른쪽 (µm)','Rectangle right (µm)'),
    ('rectangle_top_um','박스 위 (µm)','Rectangle top (µm)'),
    ('solidity','충실도','Solidity'),
    ('circularity','원형도','Circularity')
]
TABLE_COLUMNS=[(METRICS[key][0],METRICS[key][1],TABLE_COLUMN_LABELS[i+1]) for i,key in enumerate(TABLE_METRICS)]+TABLE_EXTRA_COLUMNS
TABLE_ROW_LABELS=['Minimum','Maximum','Median','Mean','Std. deviation']


def measurement_tables(stats,rows):
    raw=[]
    for label,func in [('최솟값',np.min),('최댓값',np.max),('중앙값',np.median),('평균값',np.mean),('표준편차',lambda v:np.std(v,ddof=1) if len(v)>1 else np.nan)]:
        row=[label]
        for key,_,_ in TABLE_COLUMNS:
            v=values(rows,key);x=float(func(v)) if len(v) else np.nan
            row.append(x if np.isfinite(x) else None)
        raw.append(row)
    headers=['통계']+[label for _,label,_ in TABLE_COLUMNS]
    statistics=dict(id='statistics',title='Pore 크기 통계표',kind='table',
                    table_data=dict(headers=headers,rows=raw),default_columns=[1,2,3,4],
                    column_keys=['statistic']+[key for key,_,_ in TABLE_COLUMNS],column_labels=['Statistic']+[label for _,_,label in TABLE_COLUMNS],row_labels=TABLE_ROW_LABELS,table_type='statistics')
    view=table_view(statistics,{})
    statistics.update(headers=view['headers'],rows=view['rows'])
    summary_rows=[['전체 pore 수',int(stats['candidate_count'])],['경계 pore 수',int(stats['edge_candidate_count'])],
                  ['크기 통계에 포함한 pore 수',int(stats['complete_candidate_count'])],
                  ['Pore 면적분율 (%)',stats['candidate_union_area_percent']],
                  ['분석 영역 너비 (µm)',stats['field_width_um']],['분석 영역 높이 (µm)',stats['field_height_um']]]
    summary=dict(id='analysis_summary',title='Pore 분석 요약표',kind='table',
                 table_data=dict(headers=['항목','값'],rows=summary_rows),default_columns=[1],integer_rows=[0,1,2],
                 table_widths=[.7,.3],column_labels=['Measurement','Value'],row_labels=['Total pores','Boundary pores','Pores used in size statistics','Pore area fraction (%)','Field width (µm)','Field height (µm)'],table_type='summary')
    view=table_view(summary,{})
    summary.update(headers=view['headers'],rows=view['rows'])
    for asset in [statistics,summary]:
        columns=list(range(1,len(asset['table_data']['headers'])))
        asset['table_formats']={str(precision):table_view(asset,dict(table_columns=columns,table_decimals=precision))['rows'] for precision in range(7)}
    return [statistics,summary]


def table_view(asset,item):
    catalog=asset.get('table_data',asset)
    columns=item.get('table_columns',asset.get('default_columns',list(range(1,len(catalog['headers'])))))
    rows=item.get('table_rows',list(range(len(catalog['rows']))));decimals=item.get('table_decimals',3)
    result=[]
    for index in rows:
        source=catalog['rows'][index];row=[source[0]]
        for column in columns:
            value=source[column]
            if value is None:formatted='—'
            elif isinstance(value,(int,float)) or table_numeric(value):
                formatted='—' if str(value)=='—' else str(int(float(value))) if index in asset.get('integer_rows',[]) else f'{float(value):.{decimals}f}'
            else:formatted=str(value)
            row.append(formatted)
        result.append(row)
    return dict(asset,headers=[catalog['headers'][0]]+[catalog['headers'][i] for i in columns],rows=result)


def content(editor, state):
    cached = state.get('report_content')
    from pore_details_export import REPORT_PLOT_DPI, REPORT_PLOT_STYLE
    if cached and cached['revision'] == state['revision'] and cached.get('plot_style') == REPORT_PLOT_STYLE: return cached
    measured = editor.measurements(state); stats = measured['stats']; rows = measured['candidates']
    from report_bundle import source_image
    _, name = source_image(editor, state)
    original = np.repeat(state['gray'][:, :, None], 3, axis=2)
    overlay = original.copy()
    for mask in state['masks'].values():
        overlay[mask] = np.rint(.6*overlay[mask]+.4*np.array([255,215,0])).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255,190,0), 1)
    assets = []
    for key, title, pixels in [('original','SEM 원본 (분석 영역)',original), ('segmentation','Pore 분할 결과',overlay),
                                ('comparison','SEM 원본 (왼쪽) / Pore 분할 (오른쪽)',np.concatenate([original,overlay],axis=1))]:
        assets.append(dict(id=key,title=title,kind='image',image=data_url(Image.fromarray(pixels))))
    plt = plotting()
    for key, (column, title) in METRICS.items():
        v = values(rows,column)
        fig, ax = plt.subplots(figsize=(6.4,4.4),layout='constrained')
        from pore_details_export import draw_report_histogram
        draw_report_histogram(ax,v,column,title)
        stream=io.BytesIO();fig.savefig(stream,format='png',dpi=REPORT_PLOT_DPI);plt.close(fig)
        assets.append(dict(id='hist_'+key,title=title+' 분포',kind='histogram',plot_style=REPORT_PLOT_STYLE,image='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()))
    assets.extend(measurement_tables(stats,rows))
    text=(f"분석 영역 {stats['field_width_um']:.2f} × {stats['field_height_um']:.2f} µm에서 현재 분할된 pore는 {stats['candidate_count']}개이며, "
          f"pore 면적분율은 {stats['candidate_union_area_percent']:.2f}%이다. "
          f"이미지 경계에 닿는 {stats['edge_candidate_count']}개를 제외한 {stats['complete_candidate_count']}개를 크기 통계에 사용하였다.")
    mean=stats['complete_equivalent_diameter_um_mean'];median=stats['complete_equivalent_diameter_um_median']
    if mean is not None: text+=f' 등가원직경의 평균은 {mean:.3f} µm, 중앙값은 {median:.3f} µm이다.'
    else: text+=' 경계에 닿지 않는 pore가 없어 크기 통계는 산출하지 않았다.'
    fingerprint_assets(assets)
    cached=dict(dataset=state['dataset'],revision=state['revision'],name=name,assets=assets,summary=text,stats=stats,plot_style=REPORT_PLOT_STYLE)
    state['report_content']=cached
    return cached


def default_file_name(name):
    stem=re.sub(r'[^\w.-]+','_',Path(name).stem).strip('._')[:80] or 'SEM'
    return stem+'_report'


def export_file_name(value):
    if not isinstance(value,str):raise ValueError('Enter a file name.')
    value=value.strip()
    if value.lower().endswith(('.pdf','.hwpx','.docx')):value=value.rsplit('.',1)[0]
    if not value or len(value)>120 or value.endswith('.') or any(ord(c)<32 or c in '<>:"/\\|?*' for c in value):
        raise ValueError('Enter a valid file name without folder paths or special characters: < > : " / \\ | ? *')
    if re.fullmatch(r'CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9]',value.split('.')[0],re.IGNORECASE):
        raise ValueError('This file name is reserved by Windows. Choose another name.')
    return value


def defaults(data):
    return dict(title='주사전자현미경(SEM)',sample='',conditions=[[k,''] for k in CONDITIONS],
                method='',results=data['summary'],summary_revision=data['revision'],
                items=[],extra_images=[],file_name=default_file_name(data.get('name','SEM')))


def validate(options, data):
    if not isinstance(options,dict): raise ValueError('Invalid report settings.')
    result={}
    for key,limit in [('title',200),('sample',300),('method',12000),('results',12000),('file_name',120)]:
        value=options.get(key,default_file_name(data.get('name','SEM')) if key=='file_name' else '')
        if not isinstance(value,str) or len(value)>limit: raise ValueError('Report text is too long.')
        result[key]=value
    conditions=options.get('conditions',[])
    if not isinstance(conditions,list) or len(conditions)>20: raise ValueError('Use at most 20 condition rows.')
    for row in conditions:
        if not isinstance(row,list) or len(row)!=2 or any(not isinstance(v,str) or len(v)>500 for v in row): raise ValueError('Invalid condition row.')
    result['conditions']=conditions
    extras=options.get('extra_images',[])
    if not isinstance(extras,list) or len(extras)>8: raise ValueError('Use at most 8 additional images.')
    normalized=[]
    for index,extra in enumerate(extras):
        uri=extra.get('image','') if isinstance(extra,dict) else ''
        if not isinstance(uri,str) or not uri.startswith(('data:image/png;base64,','data:image/jpeg;base64,')) or len(uri)>6_000_000: raise ValueError('Use a PNG or JPEG image up to 4 MB.')
        raw=base64.b64decode(uri.split(',',1)[1],validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width*image.height>20_000_000: raise ValueError('Additional image is too large.')
            image.load()
        normalized.append(dict(id=f'extra_{index}',title=f'Imported image {index+1}',kind='image',image=uri))
    result['extra_images']=normalized
    allowed={a['id'] for a in data['assets']+normalized};items=options.get('items',[])
    if not isinstance(items,list) or len(items)>100: raise ValueError('Invalid report content selection.')
    seen=set()
    for item in items:
        if not isinstance(item,dict) or (item.get('id') not in allowed and not (isinstance(item.get('id'),str) and re.fullmatch(r'figure_[a-zA-Z0-9_]+',item['id']))) or item['id'] in seen or type(item.get('selected')) is not bool or not isinstance(item.get('caption'),str) or len(item['caption'])>500: raise ValueError('Invalid report content selection.')
        seen.add(item['id'])
    result['items']=[{k:i[k] for k in ['id','selected','caption']} for i in items]
    tables={a['id']:a for a in data['assets'] if a['kind']=='table'}
    for source,item in zip(items,result['items']):
        if item['id'] not in tables:continue
        asset=tables[item['id']];catalog=asset.get('table_data',asset)
        for field,limit,minimum in [('table_columns',len(catalog['headers']),1),('table_rows',len(catalog['rows']),0)]:
            if field not in source:continue
            chosen=source[field]
            if not isinstance(chosen,list) or not chosen or any(type(v) is not int or not minimum<=v<limit for v in chosen) or len(set(chosen))!=len(chosen):raise ValueError('Choose at least one valid table column and row.')
            item[field]=chosen[:]
        if 'table_decimals' in source:
            decimals=source['table_decimals']
            if type(decimals) is not int or not 0<=decimals<=6:raise ValueError('Choose 0 to 6 decimal places.')
            item['table_decimals']=decimals
    image_ids={a['id'] for a in data['assets']+normalized if a['kind']!='table'}
    for source,item in zip(items,result['items']):
        if item['id'] not in image_ids and not item['id'].startswith('figure_'): continue
        panels=source.get('image_ids',[item['id']]);columns=source.get('columns',2)
        if not isinstance(panels,list) or not 0<=len(panels)<=8 or any(not isinstance(k,str) or k not in image_ids for k in panels) or len(set(panels))!=len(panels):
            raise ValueError('Choose 1 to 8 distinct images per figure.')
        if type(columns) is not int or columns not in [1,2,3]: raise ValueError('Choose 1, 2 or 3 figure columns.')
        rows=source.get('rows',[panels[i:i+columns] for i in range(0,len(panels),columns)])
        if not isinstance(rows,list) or any(not isinstance(row,list) or not row for row in rows) or [k for row in rows for k in row]!=panels:
            raise ValueError('Figure rows must contain each selected image exactly once, in order.')
        item.update(image_ids=panels,columns=columns,rows=rows)
    revision=options.get('summary_revision',data['revision'])
    if type(revision) is not int or revision<0: raise ValueError('Invalid summary revision.')
    result['summary_revision']=revision
    return result


def draft_path(editor,state):
    return editor.output_root/state['dataset']/'report_draft.json'


def load(editor,state):
    data=content(editor,state);path=draft_path(editor,state)
    options=validate(json.loads(path.read_text(encoding='utf-8')),data) if path.is_file() else defaults(data)
    if not path.is_file():
        preset=editor.output_root/'report_conditions_default.json'
        if preset.is_file():
            options['conditions']=validate(dict(options,conditions=json.loads(preset.read_text(encoding='utf-8'))),data)['conditions']
    if options['title']=='주사전자현미경(SEM) Pore 분석': options['title']='주사전자현미경(SEM)'
    return dict(data,options=options)


def save(editor,state,options):
    options=validate(options,content(editor,state));path=draft_path(editor,state);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending.json');pending.write_text(json.dumps(options,ensure_ascii=False),encoding='utf-8');pending.replace(path)
    return options


def save_conditions_default(editor,state,conditions):
    data=content(editor,state)
    checked=validate(dict(defaults(data),conditions=conditions),data)['conditions']
    path=editor.output_root/'report_conditions_default.json'
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix('.pending.json')
    pending.write_text(json.dumps(checked,ensure_ascii=False),encoding='utf-8');pending.replace(path)


def selected(data,options):
    assets={a['id']:a for a in data['assets']+options['extra_images']}
    result=[]
    for item in options['items']:
        if not item['selected']:continue
        panels=item.get('image_ids',[item['id']])
        if not panels:continue
        asset=assets[panels[0]] if item['id'].startswith('figure_') else assets[item['id']]
        if asset['kind']=='table' and any(k in item for k in ['table_columns','table_rows','table_decimals']):asset=table_view(asset,item)
        if asset['kind']!='table' and (len(panels)>1 or panels[0]!=asset['id']):
            layout=item.get('rows')
            if layout is None:
                columns=item.get('columns',2);layout=[panels[i:i+columns] for i in range(0,len(panels),columns)]
            gap=20;width=1800;plans=[];density=1.
            all_histograms=all(assets[key]['kind']=='histogram' for key in panels)
            for ids in layout:
                images=[];factors=[]
                for key in ids:
                    image=Image.open(io.BytesIO(base64.b64decode(assets[key]['image'].split(',',1)[1]))).convert('RGB')
                    images.append(image)
                    factors.append(.77 if assets[key]['kind']=='histogram' and not all_histograms else 1.)
                # Keep the page geometry, but choose pixels from the source
                # resolution instead of shrinking every figure to 1800 px.
                height=min(1200,width/sum(im.width/im.height for im in images))
                density=max(density,max(im.height/(height*f) for im,f in zip(images,factors)))
                plans.append((images,factors,height))
            base_height=sum(max(height*f for f in factors) for _,factors,height in plans)+(len(plans)-1)*gap
            # Bound memory for exceptionally large imported multi-panel figures.
            density=min(density,math.sqrt(80_000_000/(width*base_height)))
            pixel_width=round(width*density);gap=round(gap*density);rows=[]
            for images,factors,height in plans:
                row=[]
                for im,f in zip(images,factors):
                    size=(max(1,round(im.width/im.height*height*f*density)),max(1,round(height*f*density)))
                    row.append(im if im.size==size else im.resize(size,Image.Resampling.LANCZOS))
                rows.append(row)
            heights=[max(im.height for im in row) for row in rows]
            sheet_width=max(pixel_width,max(sum(im.width for im in row) for row in rows))
            sheet=Image.new('RGB',(sheet_width,sum(heights)+(len(rows)-1)*gap),'white');y=0
            for row,height in zip(rows,heights):
                x=(sheet_width-sum(im.width for im in row))//2
                for im in row:
                    sheet.paste(im,(x,y+(height-im.height)//2));x+=im.width
                y+=height+gap
            asset=dict(asset,image=data_url(sheet),kind='histogram' if all_histograms else 'image')
        result.append((item,dict(asset,panel_count=len(panels))))
    return result


def figure_scale(asset):
    if asset['kind']=='histogram':return .77
    if asset['kind']=='image' and asset.get('panel_count',1)==1 and asset.get('base_id',asset.get('id'))!='comparison':
        if asset.get('base_id',asset.get('id')) in ('original','segmentation'):return .75
        # Imported, prejoined panels have no composition metadata. Give wide
        # figures the same page width as panels joined inside the composer.
        name=Path(asset.get('title','')).stem.casefold()
        if name.endswith('_comparison'):return 1.
        with Image.open(io.BytesIO(base64.b64decode(asset['image'].split(',',1)[1]))) as image:
            if image.width>=2*image.height:return 1.
        return .75
    return 1.


def table_widths(count,has_headers):
    if count==1:return [1.]
    first=.18 if has_headers else .3
    return [first]+[(1-first)/(count-1)]*(count-1)


def table_header(value):
    return re.sub(r'\s+(\([^()]+\))$',r'\n\1',str(value))


def table_numeric(value):
    return str(value).strip()=='—' or bool(re.fullmatch(r'[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?',str(value).strip()))


def table_html(headers,rows,widths=None):
    esc=html.escape;count=len(headers or (rows[0] if rows else []))
    if not count:return ''
    columns='<colgroup>'+''.join(f'<col style="width:{v*100:g}%">' for v in (widths or table_widths(count,bool(headers))))+'</colgroup>'
    heading='<thead><tr>'+''.join('<th scope="col">'+esc(table_header(v)).replace('\n','<br>')+'</th>' for v in headers)+'</tr></thead>' if headers else ''
    body=''.join('<tr>'+''.join('<td'+(' class="numeric"' if headers and i and table_numeric(v) else '')+'>'+esc(str(v))+'</td>' for i,v in enumerate(row))+'</tr>' for row in rows)
    return '<table class="publication-table">'+columns+heading+'<tbody>'+body+'</tbody></table>'


def render_html(data,options):
    esc=html.escape
    p=lambda s:'<p class="paragraph">'+esc(s or '미입력')+'</p>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>
body{font:14px 'Malgun Gothic',sans-serif;color:#111;background:white;width:75%;max-width:760px;box-sizing:border-box;margin:36px auto;padding:0;line-height:1.8}h1{font-size:21px}h2{font-size:16px;margin:26px 0 10px;color:#174366}table{border-collapse:collapse;table-layout:fixed;width:100%;font-size:12px;border-top:1.5px solid #222;border-bottom:1.5px solid #222;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;text-align:left;border:0;overflow-wrap:anywhere;vertical-align:middle}thead{border-bottom:.8px solid #555}th{font-weight:600;text-align:center;line-height:1.5}th:first-child,td:first-child{text-align:left}td.numeric{text-align:right}.table-caption{text-align:left;font-weight:600;margin:0 0 10px}figure{margin:57.6px 0;break-inside:avoid}figure img{display:block;max-width:100%;max-height:650px;margin:auto}figcaption{text-align:center;font-size:12px;margin-top:8px}.paragraph{white-space:pre-wrap;overflow-wrap:anywhere}.note,.meta{font-size:11px;color:#555}@media print{body{margin:0;max-width:none}h2{break-after:avoid}}@page{size:A4;margin:20mm}</style>'''
    page+='<h1>'+esc(options['title'])+'</h1>'
    page+='<h2>가) 분석기기 및 조건</h2>'+table_html([],options['conditions'])
    page+='<h2>나) 분석 방법</h2>'+p(options['method'])+'<h2>다) 분석 결과</h2>'+p(options['results'])
    figures=tables=0
    for item,asset in selected(data,options):
        if asset['kind']=='table':
            tables+=1;page+=f'<figure><figcaption class="table-caption">표 {tables}. '+esc(item['caption'])+'</figcaption>'+table_html(asset['headers'],asset['rows'],asset.get('table_widths'))+'</figure>'
        else:
            figures+=1;scale=figure_scale(asset);page+=f'<figure><img style="max-width:{scale*100:g}%;max-height:{650*scale:g}px" src="'+asset['image']+'" alt="'+esc(item['caption'],quote=True)+f'"><figcaption>그림 {figures}. '+esc(item['caption'])+'</figcaption></figure>'
    page+='</html>'
    return page


def preview_pdf(editor,data,options):
    """Render the export PDF in memory; previews never write to the destination."""
    import secrets
    stream=io.BytesIO()
    render_pdf(stream,data,options)
    previews=getattr(editor,'report_pdf_previews',{})
    token=secrets.token_hex(16)
    previews[token]=stream.getvalue()
    while len(previews)>4:previews.pop(next(iter(previews)))
    editor.report_pdf_previews=previews
    url='/report-preview-pdf/'+token+'.pdf'
    show=getattr(editor,'show_pdf_preview',None)
    if show:show(url)
    return dict(url=url,opened=bool(show))


def render_pdf(path,data,options):
    """Paginated A4 export using the app's existing Matplotlib dependency."""
    from matplotlib.backends.backend_pdf import PdfPages
    plt=plotting()
    from matplotlib.font_manager import FontProperties
    from matplotlib.backends.backend_agg import RendererAgg
    renderer=RendererAgg(1,1,72)
    left,right=.125,.875
    page_width=8.27*72
    def lines(value,width=None,size=11):
        width=page_width*(right-left) if width is None else width
        font=FontProperties(family='Malgun Gothic',size=size)
        output=[]
        for paragraph in value.split('\n'):
            line=''
            for char in paragraph:
                candidate=line+char
                if line and renderer.get_text_width_height_descent(candidate,font,False)[0]>width:
                    output.append(line.rstrip());line=char.lstrip()
                else:line=candidate
            output.append(line)
        return output
    with PdfPages(path) as pdf:
        fig=None;y=0;number=0
        def new_page():
            nonlocal fig,y,number
            if fig is not None: pdf.savefig(fig);plt.close(fig)
            fig=plt.figure(figsize=(8.27,11.69),facecolor='white');number+=1;y=.92
            fig.text(.5,.035,str(number),ha='center',fontsize=9,color='#666')
        def text(value,size=11,color='#111',gap=.012,weight='normal'):
            nonlocal y
            for line in lines(value or '미입력',size=size):
                if y<.095:new_page()
                fig.text(left,y,line,va='top',fontsize=size,color=color,fontweight=weight);y-=size/842*1.75
            y-=gap
        def table(headers,rows,widths=None):
            nonlocal y
            if not headers and not rows:return
            count=len(headers or rows[0]);fractions=widths or table_widths(count,bool(headers))
            padding=6/page_width
            xs=[left+sum(fractions[:i])*(right-left) for i in range(count)]
            def rule(weight):
                fig.add_artist(plt.Line2D([left,right],[y,y],transform=fig.transFigure,color='#222',linewidth=weight))
            def wrapped_row(row,header=False):
                return [lines(table_header(v) if header else str(v),page_width*(right-left)*f-12,size=9.5) for v,f in zip(row,fractions)]
            def draw(row,header=False):
                nonlocal y
                wrapped=wrapped_row(row,header);height=max(map(len,wrapped))*.018+.022
                for i,(x,ls,f) in enumerate(zip(xs,wrapped,fractions)):
                    align='center' if header and i else 'right' if headers and i and table_numeric(row[i]) else 'left'
                    anchor=x+f*(right-left)/2 if align=='center' else x+f*(right-left)-padding if align=='right' else x+padding
                    for j,line in enumerate(ls):fig.text(anchor,y-.011-j*.018,line,ha=align,va='top',fontsize=9.5,fontweight='bold' if header else 'normal',color='#222')
                y-=height
            rule(1.)
            if headers:draw(headers,True);rule(.6)
            for row in rows:
                height=max(map(len,wrapped_row(row)))*.018+.022
                if y-height<.08:
                    rule(1.);new_page();rule(1.)
                    if headers:draw(headers,True);rule(.6)
                draw(row)
            rule(1.);y-=.025
        new_page();text(options['title'],17)
        text('가) 분석기기 및 조건',13,'#174366');table([],options['conditions'])
        text('나) 분석 방법',13,'#174366');text(options['method'])
        text('다) 분석 결과',13,'#174366');text(options['results'])
        figures=tables=0
        for index,(item,asset) in enumerate(selected(data,options)):
            if index:y-=.036
            if asset['kind']=='table':
                if y<.36:new_page()
                tables+=1;text(f'표 {tables}. '+item['caption'],10,weight='bold');table(asset['headers'],asset['rows'],asset.get('table_widths'))
            else:
                raw=base64.b64decode(asset['image'].split(',',1)[1]);im=Image.open(io.BytesIO(raw))
                scale=figure_scale(asset);width=(right-left)*scale
                height=min(.54,(right-left)*8.27/11.69*im.height/im.width)*scale
                caption_height=len(lines(item['caption'],size=10))*.024+.05
                if y-height-caption_height<.08:new_page()
                ax=fig.add_axes([.5-width/2,y-height,width,height]);ax.imshow(im,interpolation='none');ax.axis('off');y-=height+.015
                figures+=1;text(f'그림 {figures}. '+item['caption'],10)
        pdf.savefig(fig);plt.close(fig)


def export(editor,state,directory,options):
    data=content(editor,state);options=save(editor,state,options)
    from report_bundle import source_image
    _,name=source_image(editor,state)
    return export_data(data,options,directory,name)


def export_data(data,options,directory,name):
    from report_hwpx import export_hwpx
    from report_docx import export_docx
    if not isinstance(directory,str) or not directory.strip():raise ValueError('Choose a destination folder.')
    destination=Path(directory.strip()).expanduser()
    if not destination.is_absolute():raise ValueError('Enter the full destination folder path.')
    requested=options.get('file_name')
    if requested is not None:
        requested=export_file_name(requested)
    else:
        requested=f'{default_file_name(name)}_{datetime.now():%Y%m%d_%H%M%S_%f}'
    destination=destination.resolve();destination.mkdir(parents=True,exist_ok=True)
    base=requested;number=2
    while any((destination/f'{base}.{suffix}').exists() for suffix in ['pdf','hwpx','docx']):
        base=f'{requested} ({number})';number+=1
    # Stage privately; the chosen destination receives only the final reports.
    with tempfile.TemporaryDirectory(prefix='poresam_report_') as temporary:
        folder=Path(temporary)
        render_pdf(folder/'report.pdf',data,options)
        export_hwpx(folder/'report.hwpx',data,options)
        export_docx(folder/'report.docx',data,options)
        created=[]
        try:
            for suffix in ['pdf','hwpx','docx']:
                path=destination/f'{base}.{suffix}'
                with path.open('xb') as target:
                    created.append(path);target.write((folder/f'report.{suffix}').read_bytes())
        except Exception:
            for path in created:path.unlink(missing_ok=True)
            raise
    return str(destination)
