"""Editable HWPX report export; no Hancom installation needed at runtime.

Package structure follows https://tech.hancom.com/hwpxformat/ . The bundled
template contains only formatting/object skeletons, never reference content.
"""
import base64
import copy
import io
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image

NS={'hp':'http://www.hancom.co.kr/hwpml/2011/paragraph',
    'hc':'http://www.hancom.co.kr/hwpml/2011/core',
    'hh':'http://www.hancom.co.kr/hwpml/2011/head',
    'hs':'http://www.hancom.co.kr/hwpml/2011/section',
    'opf':'http://www.idpf.org/2007/opf/',
    'ha':'http://www.hancom.co.kr/hwpml/2011/app'}
for prefix,uri in NS.items():ET.register_namespace(prefix,uri)


def tag(prefix,name):return '{'+NS[prefix]+'}'+name
def xml(node):return ET.tostring(node,encoding='utf-8',xml_declaration=True)


def export_hwpx(path,data,options):
    from pore_editor import ROOT
    from report_composer import selected
    with zipfile.ZipFile(ROOT/'ui/report-hwpx-template.zip') as source:
        files={n:source.read(n) for n in source.namelist()}
    templates=ET.fromstring(files.pop('templates.xml'))
    head=ET.fromstring(files.pop('header.xml'))
    charlist=head.find('.//hh:charProperties',NS)
    paralist=head.find('.//hh:paraProperties',NS)
    chars={};paras={}
    for key,height,bold in [('body',1100,False),('title',1700,True),('heading',1300,True),('table',950,False),('caption',1000,False)]:
        style=copy.deepcopy(charlist[0]);identifier=max(int(e.get('id')) for e in charlist)+1
        style.set('id',str(identifier));style.set('height',str(height))
        font=style.find('hh:fontRef',NS)
        for k in font.attrib:font.set(k,'0')
        if bold:ET.SubElement(style,tag('hh','bold'))
        charlist.append(style);chars[key]=str(identifier)
    charlist.set('itemCnt',str(len(charlist)))
    for key in ['body','heading','caption']:
        style=copy.deepcopy(paralist[0]);identifier=max(int(e.get('id')) for e in paralist)+1
        style.set('id',str(identifier));style.find('hh:align',NS).set('horizontal','CENTER' if key=='caption' else 'LEFT')
        style.find('hh:breakSetting',NS).set('keepWithNext','1' if key=='heading' else '0')
        paralist.append(style);paras[key]=str(identifier)
    paralist.set('itemCnt',str(len(paralist)))
    section=ET.Element(tag('hs','sec'));counter=100000;preview=[]
    def new_id():
        nonlocal counter
        counter+=1;return str(counter)
    def paragraph(text='',kind='body',parent=None):
        p=ET.SubElement(section if parent is None else parent,tag('hp','p'),dict(id=new_id(),paraPrIDRef=paras['heading' if kind in ['heading','title'] else 'caption' if kind=='caption' else 'body'],styleIDRef='0',pageBreak='0',columnBreak='0',merged='0'))
        run=ET.SubElement(p,tag('hp','run'),dict(charPrIDRef=chars[kind]))
        ET.SubElement(run,tag('hp','t')).text=text
        return p,run
    def prose(text,kind='body'):
        for line in (text or '미입력').splitlines():paragraph(line,kind)
        preview.append(text)
    title,run=paragraph(options['title'],'title')
    sec=copy.deepcopy(templates.find('hp:secPr',NS))
    page=sec.find('hp:pagePr',NS);page.set('landscape','WIDELY');page.set('width','59528');page.set('height','84186')
    page.find('hp:margin',NS).attrib.update(left='7441',right='7441',top='5669',bottom='5669',header='0',footer='0',gutter='0')
    run.insert(0,sec)
    ctrl=ET.Element(tag('hp','ctrl'));ET.SubElement(ctrl,tag('hp','colPr'),dict(id='',type='NEWSPAPER',layout='LEFT',colCount='1',sameSz='1',sameGap='0'));run.insert(1,ctrl)
    preview.append(options['title'])
    width=44646
    def table(headers,rows):
        allrows=([headers] if headers else [])+rows
        if not allrows:return
        cols=len(allrows[0]);p,run=paragraph()
        t=copy.deepcopy(templates.find('hp:tbl',NS));t.attrib.update(id=new_id(),rowCnt=str(len(allrows)),colCnt=str(cols),repeatHeader='1' if headers else '0')
        t.find('hp:sz',NS).attrib.update(width=str(width),height=str(2000*len(allrows)))
        t.find('hp:pos',NS).set('treatAsChar','1')
        for y,row in enumerate(allrows):
            tr=ET.SubElement(t,tag('hp','tr'))
            for x,value in enumerate(row):
                cell=copy.deepcopy(templates.find('hp:tc',NS));cell.set('header','1' if headers and y==0 else '0');cell.set('borderFillIDRef','5')
                cell.find('hp:cellAddr',NS).attrib.update(colAddr=str(x),rowAddr=str(y))
                cell.find('hp:cellSz',NS).attrib.update(width=str(width//cols+(width%cols if x==cols-1 else 0)),height='2000')
                sub=cell.find('hp:subList',NS)
                for line in str(value).split('\n'):paragraph(line,'table',sub)
                tr.append(cell)
            preview.append('\t'.join(map(str,row)))
        run.insert(0,t)
        paragraph()
    prose('가) 분석기기 및 조건','heading');table([],options['conditions'])
    prose('나) 분석 방법','heading');prose(options['method'])
    prose('다) 분석 결과','heading');prose(options['results'])
    binaries=[];figure_number=table_number=0
    for item,asset in selected(data,options):
        if asset['kind']=='table':
            table_number+=1;prose(f'표 {table_number}. '+item['caption'],'caption');table(asset['headers'],asset['rows'])
            continue
        figure_number+=1;raw=base64.b64decode(asset['image'].split(',',1)[1]);im=Image.open(io.BytesIO(raw)).convert('RGB')
        stream=io.BytesIO();im.save(stream,format='PNG');key=f'image{figure_number}';binaries.append((key,stream.getvalue()))
        scale=min(width/im.width,53000/im.height);w=max(1,round(im.width*scale));h=max(1,round(im.height*scale))
        p,run=paragraph();p.set('paraPrIDRef',paras['caption'])
        pic=copy.deepcopy(templates.find('hp:pic',NS));pic.attrib.update(id=new_id(),instid=new_id(),zOrder=str(figure_number),numberingType='NONE')
        for name in ['orgSz','curSz','sz']:pic.find('hp:'+name,NS).attrib.update(width=str(w),height=str(h))
        pic.find('hp:offset',NS).attrib.update(x='0',y='0')
        pic.find('hp:rotationInfo',NS).attrib.update(centerX=str(w//2),centerY=str(h//2))
        for matrix in pic.find('hp:renderingInfo',NS):matrix.attrib.update(e1='1',e2='0',e3='0',e4='0',e5='1',e6='0')
        pic.find('hc:img',NS).set('binaryItemIDRef',key)
        for node,coordinates in zip(pic.find('hp:imgRect',NS),[(0,0),(w,0),(w,h),(0,h)]):node.attrib.update(x=str(coordinates[0]),y=str(coordinates[1]))
        pic.find('hp:imgClip',NS).attrib.update(left='0',right=str(w),top='0',bottom=str(h))
        pic.find('hp:imgDim',NS).attrib.update(dimwidth=str(w),dimheight=str(h))
        pic.find('hp:pos',NS).attrib.update(treatAsChar='1',affectLSpacing='1',horzRelTo='PARA',horzAlign='LEFT',horzOffset='0',vertOffset='0')
        run.insert(0,pic);prose(f'그림 {figure_number}. '+item['caption'],'caption');paragraph()
    package=ET.Element(tag('opf','package'),dict(version='',id='',**{'unique-identifier':''}))
    metadata=ET.SubElement(package,tag('opf','metadata'));ET.SubElement(metadata,tag('opf','title')).text=options['title'];ET.SubElement(metadata,tag('opf','language')).text='ko'
    manifest=ET.SubElement(package,tag('opf','manifest'))
    for key,href,mime in [('header','Contents/header.xml','application/xml'),('section0','Contents/section0.xml','application/xml'),('settings','settings.xml','application/xml')]:ET.SubElement(manifest,tag('opf','item'),dict(id=key,href=href,**{'media-type':mime}))
    for key,raw in binaries:ET.SubElement(manifest,tag('opf','item'),dict(id=key,href=f'BinData/{key}.png',**{'media-type':'image/png','isEmbeded':'1'}))
    spine=ET.SubElement(package,tag('opf','spine'))
    for key in ['header','section0']:ET.SubElement(spine,tag('opf','itemref'),dict(idref=key,linear='yes'))
    settings=ET.Element(tag('ha','HWPApplicationSetting'));ET.SubElement(settings,tag('ha','CaretPosition'),dict(listIDRef='0',paraIDRef='0',pos='0'))
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as out:
        out.writestr('mimetype','application/hwp+zip',compress_type=zipfile.ZIP_STORED)
        for name,raw in files.items():out.writestr(name,raw)
        for name,node in [('Contents/header.xml',head),('Contents/section0.xml',section),('Contents/content.hpf',package),('settings.xml',settings)]:out.writestr(name,xml(node))
        out.writestr('Preview/PrvText.txt','\n'.join(preview).encode('utf-8'))
        for key,raw in binaries:out.writestr(f'BinData/{key}.png',raw)
