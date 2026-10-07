"""Editable Word reports using OOXML and the app's existing dependencies."""
import base64
import io
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image

NS = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
}
for prefix, uri in NS.items():
    ET.register_namespace(prefix, uri)
REL = 'http://schemas.openxmlformats.org/package/2006/relationships'
TYPES = 'http://schemas.openxmlformats.org/package/2006/content-types'
BODY_WIDTH = 8930  # A4 with the same 75% text width as the PDF and HWPX.


def tag(name):
    prefix, local = name.split(':')
    return '{'+NS[prefix]+'}'+local


def element(parent, qualified_name, **attributes):
    return ET.SubElement(parent, tag(qualified_name), {tag(k) if ':' in k else k: str(v) for k, v in attributes.items()})


def prop(parent, name, **attributes):
    return element(parent, 'w:'+name, **{'w:'+k: v for k, v in attributes.items()})


def xml(node):
    return ET.tostring(node, encoding='utf-8', xml_declaration=True)


def styles():
    root = ET.Element(tag('w:styles'))
    definitions = [('Normal', 22, False, '111111'), ('Title', 34, True, '111111'),
                   ('Heading1', 26, True, '174366'), ('Caption', 20, False, '111111'),
                   ('TableText', 19, False, '222222'), ('TableHeader', 19, True, '222222')]
    for name, size, bold, color in definitions:
        style = element(root, 'w:style', **{'w:type': 'paragraph', 'w:styleId': name})
        if name == 'Normal':
            style.set(tag('w:default'), '1')
        prop(style, 'name', val=name)
        if name != 'Normal':
            prop(style, 'basedOn', val='Normal')
        prop(style, 'next', val='Normal')
        paragraph = element(style, 'w:pPr')
        if name in ('Title', 'Heading1'):
            prop(paragraph, 'keepNext')
        prop(paragraph, 'keepLines')
        prop(paragraph, 'widowControl')
        prop(paragraph, 'spacing', before=220 if name == 'Heading1' else 0,
             after=0 if name.startswith('Table') else 140, line=270 if name.startswith('Table') else 360, lineRule='auto')
        if name == 'Heading1':
            prop(paragraph, 'outlineLvl', val=0)
        run = element(style, 'w:rPr')
        prop(run, 'rFonts', ascii='Malgun Gothic', hAnsi='Malgun Gothic', eastAsia='Malgun Gothic', cs='Malgun Gothic')
        if bold:
            prop(run, 'b')
        prop(run, 'color', val=color)
        prop(run, 'sz', val=size)
        prop(run, 'szCs', val=size)
        prop(run, 'lang', val='ko-KR', eastAsia='ko-KR')
    return root


def paragraph(parent, text='', style='Normal', align=None, keep=False, before=None, after=None):
    p = element(parent, 'w:p')
    properties = element(p, 'w:pPr')
    prop(properties, 'pStyle', val=style)
    if keep:
        prop(properties, 'keepNext')
    prop(properties, 'keepLines')
    if before is not None or after is not None:
        prop(properties, 'spacing', **dict((k, v) for k, v in [('before', before), ('after', after)] if v is not None))
    if align:
        prop(properties, 'jc', val=align)
    lines = str(text).split('\n')
    for index, line in enumerate(lines):
        run = element(p, 'w:r')
        if index:
            element(run, 'w:br')
        value = element(run, 'w:t')
        value.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
        # XML 1.0 rejects control characters even when Word text contains them.
        value.text = ''.join(c for c in line if ord(c) >= 32 or c == '\t')
    return p


def table(parent, headers, rows, fractions):
    from report_composer import table_header, table_numeric
    all_rows = ([headers] if headers else [])+rows
    if not all_rows:
        return
    widths = [round(BODY_WIDTH*f) for f in fractions]
    widths[-1] += BODY_WIDTH-sum(widths)
    result = element(parent, 'w:tbl')
    properties = element(result, 'w:tblPr')
    prop(properties, 'tblW', w=BODY_WIDTH, type='dxa')
    prop(properties, 'jc', val='center')
    borders = element(properties, 'w:tblBorders')
    for side in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
        prop(borders, side, val='single' if side in ('top', 'bottom') else 'nil', sz=8, color='222222')
    prop(properties, 'tblLayout', type='fixed')
    margins = element(properties, 'w:tblCellMar')
    for side in ('top', 'left', 'bottom', 'right'):
        prop(margins, side, w=110 if side in ('top', 'bottom') else 120, type='dxa')
    grid = element(result, 'w:tblGrid')
    for width in widths:
        prop(grid, 'gridCol', w=width)
    for index, values in enumerate(all_rows):
        header = bool(headers) and index == 0
        row = element(result, 'w:tr')
        row_properties = element(row, 'w:trPr')
        prop(row_properties, 'cantSplit')
        if header:
            prop(row_properties, 'tblHeader')
        for column, (value, width) in enumerate(zip(values, widths)):
            cell = element(row, 'w:tc')
            cell_properties = element(cell, 'w:tcPr')
            prop(cell_properties, 'tcW', w=width, type='dxa')
            if header:
                border = element(cell_properties, 'w:tcBorders')
                prop(border, 'bottom', val='single', sz=5, color='555555')
            prop(cell_properties, 'vAlign', val='center')
            align = 'center' if header and column else 'right' if headers and column and table_numeric(value) else 'left'
            paragraph(cell, table_header(value) if header else value, 'TableHeader' if header else 'TableText', align=align)


def picture(parent, relationship, number, image, scale):
    factor = min(BODY_WIDTH/image.width, 10600/image.height)*scale
    width, height = round(image.width*factor*635), round(image.height*factor*635)  # twips -> EMUs
    p = paragraph(parent, align='center', keep=True, before=720, after=100)
    drawing = element(element(p, 'w:r'), 'w:drawing')
    inline = element(drawing, 'wp:inline', distT=0, distB=0, distL=0, distR=0)
    element(inline, 'wp:extent', cx=width, cy=height)
    element(inline, 'wp:effectExtent', l=0, t=0, r=0, b=0)
    element(inline, 'wp:docPr', id=number, name='Figure '+str(number))
    locks = element(inline, 'wp:cNvGraphicFramePr')
    element(locks, 'a:graphicFrameLocks', noChangeAspect=1)
    graphic = element(element(inline, 'a:graphic'), 'a:graphicData', uri=NS['pic'])
    pic = element(graphic, 'pic:pic')
    nonvisual = element(pic, 'pic:nvPicPr')
    element(nonvisual, 'pic:cNvPr', id=0, name='image'+str(number)+'.png')
    element(nonvisual, 'pic:cNvPicPr')
    fill = element(pic, 'pic:blipFill')
    element(fill, 'a:blip', **{'r:embed': relationship})
    element(element(fill, 'a:stretch'), 'a:fillRect')
    shape = element(pic, 'pic:spPr')
    transform = element(shape, 'a:xfrm')
    element(transform, 'a:off', x=0, y=0)
    element(transform, 'a:ext', cx=width, cy=height)
    element(element(shape, 'a:prstGeom', prst='rect'), 'a:avLst')


def export_docx(path, data, options):
    from report_composer import selected, figure_scale, table_widths
    document = ET.Element(tag('w:document'))
    body = element(document, 'w:body')
    relationships = ET.Element('{'+REL+'}Relationships')
    def relationship(identifier, kind, target):
        ET.SubElement(relationships, '{'+REL+'}Relationship', dict(Id=identifier, Type=NS['r']+'/'+kind, Target=target))
    relationship('styles', 'styles', 'styles.xml')
    relationship('footer', 'footer', 'footer1.xml')
    paragraph(body, options['title'], 'Title')
    paragraph(body, '가) 분석기기 및 조건', 'Heading1')
    table(body, [], options['conditions'], table_widths(2, False))
    paragraph(body, '나) 분석 방법', 'Heading1')
    for line in (options['method'] or '미입력').splitlines():
        paragraph(body, line)
    paragraph(body, '다) 분석 결과', 'Heading1')
    for line in (options['results'] or '미입력').splitlines():
        paragraph(body, line)
    images = []
    figures = tables = 0
    for item, asset in selected(data, options):
        if asset['kind'] == 'table':
            tables += 1
            paragraph(body, f'표 {tables}. '+item['caption'], 'TableHeader', keep=True, before=720, after=140)
            table(body, asset['headers'], asset['rows'], asset.get('table_widths') or table_widths(len(asset['headers']), True))
        else:
            figures += 1
            raw = base64.b64decode(asset['image'].split(',', 1)[1])
            with Image.open(io.BytesIO(raw)) as image:
                stream = io.BytesIO()
                image.convert('RGB').save(stream, format='PNG')
                picture(body, 'image'+str(figures), figures, image, figure_scale(asset))
            images.append(stream.getvalue())
            relationship('image'+str(figures), 'image', f'media/image{figures}.png')
            paragraph(body, f'그림 {figures}. '+item['caption'], 'Caption', align='center', after=240)
    if list(body)[-1].tag == tag('w:tbl'):
        # Word requires a paragraph after the final table. Keep that paragraph
        # small so it cannot create an otherwise empty last page.
        end = paragraph(body, before=0, after=0)
        properties = end.find(tag('w:pPr'))
        properties.find(tag('w:spacing')).set(tag('w:line'), '20')
        properties.find(tag('w:spacing')).set(tag('w:lineRule'), 'exact')
        for owner in (properties, end.find(tag('w:r'))):
            formatting = ET.Element(tag('w:rPr'))
            owner.insert(0 if owner.tag == tag('w:r') else len(owner), formatting)
            prop(formatting, 'sz', val=2)
            prop(formatting, 'szCs', val=2)
    section = element(body, 'w:sectPr')
    element(section, 'w:footerReference', **{'w:type': 'default', 'r:id': 'footer'})
    prop(section, 'pgSz', w=11906, h=16838)
    prop(section, 'pgMar', top=1134, right=1488, bottom=1134, left=1488, header=0, footer=500, gutter=0)
    settings = ET.Element(tag('w:settings'))
    prop(settings, 'doNotAutoCompressPictures', val='true')
    relationship('settings', 'settings', 'settings.xml')
    footer = ET.Element(tag('w:ftr'))
    p = paragraph(footer, align='center')
    field = element(p, 'w:fldSimple', **{'w:instr': 'PAGE'})
    element(element(field, 'w:r'), 'w:t').text = '1'
    types = ET.Element('{'+TYPES+'}Types')
    for extension, mime in [('rels', 'application/vnd.openxmlformats-package.relationships+xml'), ('xml', 'application/xml'), ('png', 'image/png')]:
        ET.SubElement(types, '{'+TYPES+'}Default', dict(Extension=extension, ContentType=mime))
    for name, kind in [('document', 'document.main'), ('styles', 'styles'), ('footer1', 'footer'), ('settings', 'settings')]:
        ET.SubElement(types, '{'+TYPES+'}Override', dict(PartName='/word/'+name+'.xml', ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.'+kind+'+xml'))
    package_relationships = ET.Element('{'+REL+'}Relationships')
    ET.SubElement(package_relationships, '{'+REL+'}Relationship', dict(Id='document', Type=NS['r']+'/officeDocument', Target='word/document.xml'))
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, node in [('[Content_Types].xml', types), ('_rels/.rels', package_relationships), ('word/document.xml', document),
                           ('word/styles.xml', styles()), ('word/settings.xml', settings), ('word/footer1.xml', footer), ('word/_rels/document.xml.rels', relationships)]:
            archive.writestr(name, xml(node))
        for index, image in enumerate(images, 1):
            archive.writestr(f'word/media/image{index}.png', image)
