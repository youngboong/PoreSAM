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
    from report_composer import selected,figure_scale,table_widths,table_header,table_numeric
    with zipfile.ZipFile(ROOT/'ui/report-hwpx-template.zip') as source:
        files={n:source.read(n) for n in source.namelist()}
    templates=ET.fromstring(files.pop('templates.xml'))
    head=ET.fromstring(files.pop('header.xml'))
    charlist=head.find('.//hh:charProperties',NS)
    paralist=head.find('.//hh:paraProperties',NS)
    chars={};paras={}
    for key,height,bold in [('body',1100,False),('title',1700,True),('heading',1300,True),('table',950,False),('table_head',950,True),('caption',1000,False)]:
        style=copy.deepcopy(charlist[0]);identifier=max(int(e.get('id')) for e in charlist)+1
        style.set('id',str(identifier));style.set('height',str(height))
        font=style.find('hh:fontRef',NS)
        for k in font.attrib:font.set(k,'0')
        if bold:ET.SubElement(style,tag('hh','bold'))
        charlist.append(style);chars[key]=str(identifier)
    charlist.set('itemCnt',str(len(charlist)))
    for key in ['body','heading','caption','gap','table_left','table_right','table_center']:
        style=copy.deepcopy(paralist[0]);identifier=max(int(e.get('id')) for e in paralist)+1
        style.set('id',str(identifier));style.find('hh:align',NS).set('horizontal','CENTER' if key=='caption' else 'LEFT')
        style.find('hh:breakSetting',NS).set('keepWithNext','1' if key=='heading' else '0')
        if key.startswith('table_'):
            style.find('hh:align',NS).set('horizontal',key.split('_')[1].upper())
            for spacing in style.findall('.//hh:lineSpacing',NS):spacing.attrib.update(type='PERCENT',value='135')
        if key=='gap':
            for spacing in style.findall('.//hh:lineSpacing',NS):spacing.attrib.update(type='FIXED',value='2160')
        paralist.append(style);paras[key]=str(identifier)
    paralist.set('itemCnt',str(len(paralist)))
    borders=head.find('.//hh:borderFills',NS);border_ids={}
    for top,bottom in [(False,False),(True,False),(False,True),(True,True)]:
        border=copy.deepcopy(borders[0]);identifier=max(int(e.get('id')) for e in borders)+1
        border.set('id',str(identifier))
        for side in ['left','right','top','bottom']:
            visible=(side=='top' and top) or (side=='bottom' and bottom)
            border.find('hh:'+side+'Border',NS).attrib.update(type='SOLID' if visible else 'NONE',width='0.3 mm' if side=='top' else '0.2 mm',color='#222222')
        borders.append(border);border_ids[top,bottom]=str(identifier)
    borders.set('itemCnt',str(len(borders)))
    section=ET.Element(tag('hs','sec'));counter=100000;preview=[]
    def new_id():
        nonlocal counter
        counter+=1;return str(counter)
    def paragraph(text='',kind='body',parent=None):
        p=ET.SubElement(section if parent is None else parent,tag('hp','p'),dict(id=new_id(),paraPrIDRef=paras['heading' if kind in ['heading','title'] else 'caption' if kind=='caption' else 'gap' if kind=='gap' else 'body'],styleIDRef='0',pageBreak='0',columnBreak='0',merged='0'))
        run=ET.SubElement(p,tag('hp','run'),dict(charPrIDRef=chars['body' if kind=='gap' else kind]))
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
    def table(headers,rows,column_widths=None):
        allrows=([headers] if headers else [])+rows
        if not allrows:return
        cols=len(allrows[0]);p,run=paragraph();fractions=column_widths or table_widths(cols,bool(headers))
        widths=[round(width*f) for f in fractions];widths[-1]+=width-sum(widths)
        heights=[max(str(v).count('\n')+1 for v in (list(map(table_header,row)) if headers and i==0 else row))*1350+1100 for i,row in enumerate(allrows)]
        t=copy.deepcopy(templates.find('hp:tbl',NS));t.attrib.update(id=new_id(),rowCnt=str(len(allrows)),colCnt=str(cols),repeatHeader='1' if headers else '0',borderFillIDRef=border_ids[False,False])
        t.find('hp:sz',NS).attrib.update(width=str(width),height=str(sum(heights)))
        t.find('hp:pos',NS).set('treatAsChar','1')
        for y,row in enumerate(allrows):
            tr=ET.SubElement(t,tag('hp','tr'))
            for x,value in enumerate(row):
                cell=copy.deepcopy(templates.find('hp:tc',NS));cell.set('header','1' if headers and y==0 else '0');cell.set('borderFillIDRef',border_ids[y==0,(bool(headers) and y==0) or y==len(allrows)-1]);cell.set('hasMargin','1')
                cell.find('hp:cellMargin',NS).attrib.update(left='600',right='600',top='550',bottom='550')
                cell.find('hp:cellAddr',NS).attrib.update(colAddr=str(x),rowAddr=str(y))
                cell.find('hp:cellSz',NS).attrib.update(width=str(widths[x]),height=str(heights[y]))
                sub=cell.find('hp:subList',NS)
                is_header=bool(headers) and y==0
                align='center' if is_header and x else 'right' if headers and x and table_numeric(value) else 'left'
                for line in (table_header(value) if is_header else str(value)).split('\n'):
                    cell_p,_=paragraph(line,'table_head' if is_header else 'table',sub);cell_p.set('paraPrIDRef',paras['table_'+align])
                tr.append(cell)
            preview.append('\t'.join(map(str,row)))
        run.insert(0,t)
        paragraph()
    prose('가) 분석기기 및 조건','heading');table([],options['conditions'])
    prose('나) 분석 방법','heading');prose(options['method'])
    prose('다) 분석 결과','heading');prose(options['results'])
    binaries=[];figure_number=table_number=0
    for index,(item,asset) in enumerate(selected(data,options)):
        if index:paragraph(kind='gap')
        if asset['kind']=='table':
            table_number+=1;prose(f'표 {table_number}. '+item['caption'],'caption');table(asset['headers'],asset['rows'],asset.get('table_widths'))
            continue
        figure_number+=1;raw=base64.b64decode(asset['image'].split(',',1)[1]);im=Image.open(io.BytesIO(raw)).convert('RGB')
        stream=io.BytesIO();im.save(stream,format='PNG');key=f'image{figure_number}';binaries.append((key,stream.getvalue()))
        scale=min(width/im.width,53000/im.height)*figure_scale(asset);w=max(1,round(im.width*scale));h=max(1,round(im.height*scale))
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
