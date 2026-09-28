"""Editable SEM report composition from measured pores, with local-only drafts."""
import base64
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
METRICS = {'diameter': ('equivalent_diameter_um', '등가원직경 (µm)'),
           'length': ('length_um', '길이 (µm)'), 'width': ('width_um', '너비 (µm)'),
           'area': ('area_um2', '면적 (µm²)'), 'aspect': ('aspect_ratio', '종횡비')}
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
    return np.array([r[key] for r in rows if not r['touches_image_edge'] and r.get(key) is not None and np.isfinite(r[key])], dtype=float)


def content(editor, state):
    cached = state.get('report_content')
    if cached and cached['revision'] == state['revision']: return cached
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
        ax.tick_params(direction='in',top=True,right=True,width=1.2)
        for spine in ax.spines.values(): spine.set_linewidth(1.4)
        if len(v):
            bins = min(25,max(5,math.ceil(math.sqrt(len(v)))))
            lo, hi = min(0,float(v.min())),float(v.max())
            if hi<=lo: hi=lo+1
            ax.hist(v,bins=np.linspace(lo,hi,bins+1),facecolor='white',edgecolor='#e12626',hatch='\\\\\\',linewidth=1.1)
        else: ax.text(.5,.5,'경계에 닿지 않는 pore가 없습니다.',transform=ax.transAxes,ha='center')
        from matplotlib.ticker import MaxNLocator
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set(xlabel=title,ylabel='개수',ylim=(0,None))
        ax.text(.98,.96,f'n = {len(v)}',transform=ax.transAxes,ha='right',va='top',fontsize=9)
        stream=io.BytesIO();fig.savefig(stream,format='png',dpi=170);plt.close(fig)
        assets.append(dict(id='hist_'+key,title=title+' 분포',kind='histogram',image='data:image/png;base64,'+base64.b64encode(stream.getvalue()).decode()))
    table = []
    for label, func in [('최솟값',np.min),('최댓값',np.max),('중앙값',np.median),('평균값',np.mean),('표준편차',lambda v:np.std(v,ddof=1) if len(v)>1 else np.nan)]:
        row=[label]
        for key in ['diameter','length','width','aspect']:
            v=values(rows,METRICS[key][0]); x=float(func(v)) if len(v) else np.nan
            row.append(f'{x:.3f}' if np.isfinite(x) else '—')
        table.append(row)
    assets.append(dict(id='statistics',title='Pore 크기 통계표',kind='table',headers=['통계','등가원직경 (µm)','길이 (µm)','너비 (µm)','종횡비'],rows=table))
    text=(f"분석 영역 {stats['field_width_um']:.2f} × {stats['field_height_um']:.2f} µm에서 현재 분할된 pore는 {stats['candidate_count']}개이며, "
          f"pore 면적분율은 {stats['candidate_union_area_percent']:.2f}%이다. "
          f"이미지 경계에 닿는 {stats['edge_candidate_count']}개를 제외한 {stats['complete_candidate_count']}개를 크기 통계에 사용하였다.")
    mean=stats['complete_equivalent_diameter_um_mean'];median=stats['complete_equivalent_diameter_um_median']
    if mean is not None: text+=f' 등가원직경의 평균은 {mean:.3f} µm, 중앙값은 {median:.3f} µm이다.'
    else: text+=' 경계에 닿지 않는 pore가 없어 크기 통계는 산출하지 않았다.'
    text+=' 이 결과는 현재 분할 마스크의 2차원 측정값에 대한 요약이다.'
    cached=dict(dataset=state['dataset'],revision=state['revision'],name=name,assets=assets,summary=text,stats=stats)
    state['report_content']=cached
    return cached


def defaults(data):
    return dict(title='주사전자현미경(SEM)',sample='',conditions=[[k,''] for k in CONDITIONS],
                method='',results=data['summary'],summary_revision=data['revision'],
                items=[],extra_images=[])


def validate(options, data):
    if not isinstance(options,dict): raise ValueError('Invalid report settings.')
    result={}
    for key,limit in [('title',200),('sample',300),('method',12000),('results',12000)]:
        value=options.get(key,'')
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
        if asset['kind']!='table' and (len(panels)>1 or panels[0]!=asset['id']):
            layout=item.get('rows')
            if layout is None:
                columns=item.get('columns',2);layout=[panels[i:i+columns] for i in range(0,len(panels),columns)]
            gap=20;width=1800;rows=[]
            for ids in layout:
                cell_width=(width-gap*(len(ids)-1))//len(ids);row=[]
                for key in ids:
                    image=Image.open(io.BytesIO(base64.b64decode(assets[key]['image'].split(',',1)[1]))).convert('RGB')
                    scale=min(cell_width/image.width,1200/image.height)
                    row.append(image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.LANCZOS))
                rows.append(row)
            heights=[max(im.height for im in row) for row in rows]
            sheet=Image.new('RGB',(width,sum(heights)+(len(rows)-1)*gap),'white');y=0
            for row,height in zip(rows,heights):
                cell_width=(width-gap*(len(row)-1))//len(row)
                for col,im in enumerate(row):sheet.paste(im,(col*(cell_width+gap)+(cell_width-im.width)//2,y+(height-im.height)//2))
                y+=height+gap
            asset=dict(asset,image=data_url(sheet))
        result.append((item,asset))
    return result


def table_html(headers,rows):
    esc=html.escape
    return '<table>'+('<thead><tr>'+''.join('<th>'+esc(str(v))+'</th>' for v in headers)+'</tr></thead>' if headers else '')+'<tbody>'+''.join('<tr>'+''.join('<td>'+esc(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'


def render_html(data,options):
    esc=html.escape
    p=lambda s:'<p class="paragraph">'+esc(s or '미입력')+'</p>'
    page='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><style>
body{font:14px 'Malgun Gothic',sans-serif;color:#111;background:white;width:75%;max-width:760px;box-sizing:border-box;margin:36px auto;padding:0;line-height:1.8}h1{font-size:21px}h2{font-size:16px;margin:26px 0 10px;color:#174366}table{border-collapse:collapse;width:100%;font-size:12px;border-top:1px solid #111;border-bottom:1px solid #111}th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #ddd;overflow-wrap:anywhere}figure{margin:28px 0;break-inside:avoid}figure img{display:block;max-width:100%;max-height:650px;margin:auto}figcaption{text-align:center;font-size:12px;margin-top:8px}.paragraph{white-space:pre-wrap;overflow-wrap:anywhere}.note,.meta{font-size:11px;color:#555}@media print{body{margin:0;max-width:none}h2{break-after:avoid}}@page{size:A4;margin:20mm}</style>'''
    page+='<h1>'+esc(options['title'])+'</h1>'
    page+='<h2>가) 분석기기 및 조건</h2>'+table_html([],options['conditions'])
    page+='<h2>나) 분석 방법</h2>'+p(options['method'])+'<h2>다) 분석 결과</h2>'+p(options['results'])
    figures=tables=0
    for item,asset in selected(data,options):
        if asset['kind']=='table':
            tables+=1;page+=f'<figure><figcaption style="margin:0 0 8px">표 {tables}. '+esc(item['caption'])+'</figcaption>'+table_html(asset['headers'],asset['rows'])+'</figure>'
        else:
            figures+=1;page+='<figure><img src="'+asset['image']+'" alt="'+esc(item['caption'],quote=True)+f'"><figcaption>그림 {figures}. '+esc(item['caption'])+'</figcaption></figure>'
    page+='</html>'
    return page


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
        def text(value,size=11,color='#111',gap=.012):
            nonlocal y
            for line in lines(value or '미입력',size=size):
                if y<.095:new_page()
                fig.text(left,y,line,va='top',fontsize=size,color=color);y-=size/842*1.75
            y-=gap
        def table(headers,rows):
            nonlocal y
            allrows=([headers] if headers else [])+rows
            for row in allrows:
                fractions=[.3,.7] if len(row)==2 else [1/len(row)]*len(row)
                wrapped=[lines(str(v),page_width*(right-left)*f-8,size=9) for v,f in zip(row,fractions)]
                height=max(map(len,wrapped))*.019+.014
                if y-height<.08:new_page()
                xs=[left+sum(fractions[:i])*(right-left) for i in range(len(row))]
                for x,ls in zip(xs,wrapped):
                    for j,line in enumerate(ls):fig.text(x,y-j*.019,line,va='top',fontsize=9)
                fig.add_artist(plt.Line2D([left,right],[y+.004,y+.004],transform=fig.transFigure,color='#777',linewidth=.45))
                y-=height
            y-=.025
        new_page();text(options['title'],17)
        text('가) 분석기기 및 조건',13,'#174366');table([],options['conditions'])
        text('나) 분석 방법',13,'#174366');text(options['method'])
        text('다) 분석 결과',13,'#174366');text(options['results'])
        figures=tables=0
        for item,asset in selected(data,options):
            if asset['kind']=='table':
                if y<.36:new_page()
                tables+=1;text(f'표 {tables}. '+item['caption'],10);table(asset['headers'],asset['rows'])
            else:
                raw=base64.b64decode(asset['image'].split(',',1)[1]);im=Image.open(io.BytesIO(raw))
                height=min(.54,(right-left)*8.27/11.69*im.height/im.width)
                caption_height=len(lines(item['caption'],size=10))*.024+.05
                if y-height-caption_height<.08:new_page()
                ax=fig.add_axes([left,y-height,right-left,height]);ax.imshow(im);ax.axis('off');y-=height+.015
                figures+=1;text(f'그림 {figures}. '+item['caption'],10)
        pdf.savefig(fig);plt.close(fig)


def export(editor,state,directory,options):
    data=content(editor,state);options=save(editor,state,options)
    from report_bundle import source_image
    _,name=source_image(editor,state)
    return export_data(data,options,directory,name)


def export_data(data,options,directory,name):
    from report_hwpx import export_hwpx
    if not isinstance(directory,str) or not directory.strip():raise ValueError('Choose a destination folder.')
    destination=Path(directory.strip()).expanduser()
    if not destination.is_absolute():raise ValueError('Enter the full destination folder path.')
    destination=destination.resolve();destination.mkdir(parents=True,exist_ok=True)
    stem=re.sub(r'[^\w.-]+','_',Path(name).stem).strip('._')[:60] or 'SEM'
    base=f'{stem}_report_{datetime.now():%Y%m%d_%H%M%S_%f}'
    # Stage privately; the chosen destination receives only the two final reports.
    with tempfile.TemporaryDirectory(prefix='poresam_report_') as temporary:
        folder=Path(temporary)
        render_pdf(folder/'report.pdf',data,options)
        export_hwpx(folder/'report.hwpx',data,options)
        created=[]
        try:
            for suffix in ['pdf','hwpx']:
                path=destination/f'{base}.{suffix}'
                with path.open('xb') as target:
                    created.append(path);target.write((folder/f'report.{suffix}').read_bytes())
        except Exception:
            for path in created:path.unlink(missing_ok=True)
            raise
    return str(destination)
