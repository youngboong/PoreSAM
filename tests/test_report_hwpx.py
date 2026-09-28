"""HWPX package integrity, editable content and embedded-image references."""
import io
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
import zipfile
import base64
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from report_hwpx import export_hwpx,NS


def verify_hwpx(path):
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert z.namelist()[0]=='mimetype' and z.getinfo('mimetype').compress_type==zipfile.ZIP_STORED
        assert z.read('mimetype')==b'application/hwp+zip'
        for name in z.namelist():
            if name.endswith(('.xml','.hpf','.rdf')):ET.fromstring(z.read(name))
        manifest=ET.fromstring(z.read('Contents/content.hpf'))
        files={e.get('id'):e.get('href') for e in manifest.findall('opf:manifest/opf:item',NS)}
        assert all(name in z.namelist() for name in files.values())
        header=ET.fromstring(z.read('Contents/header.xml'));section=ET.fromstring(z.read('Contents/section0.xml'))
        chars={e.get('id') for e in header.findall('.//hh:charPr',NS)};paras={e.get('id') for e in header.findall('.//hh:paraPr',NS)}
        assert all(e.get('charPrIDRef') in chars for e in section.findall('.//hp:run',NS))
        assert all(e.get('paraPrIDRef') in paras for e in section.findall('.//hp:p',NS))
        for e in section.findall('.//hc:img',NS):
            with Image.open(io.BytesIO(z.read(files[e.get('binaryItemIDRef')])) ) as image:image.verify()
        for table in section.findall('.//hp:tbl',NS):
            rows=table.findall('hp:tr',NS)
            assert len(rows)==int(table.get('rowCnt'))
            assert all(len(row.findall('hp:tc',NS))==int(table.get('colCnt')) for row in rows)
        text='\n'.join(e.text or '' for e in section.findall('.//hp:t',NS))
        assert 'Revision' not in text and 'Feret' not in text and 'TESCAN MAGNA' not in text
        return text,len(section.findall('.//hp:pic',NS)),len(section.findall('.//hp:tbl',NS))


class HwpxTest(unittest.TestCase):
    def test_editable_text_tables_and_multi_image(self):
        buf=io.BytesIO();Image.new('RGB',(240,100),'red').save(buf,format='PNG')
        image='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
        data=dict(name='not rendered',revision=29,assets=[dict(id='original',kind='image',image=image),dict(id='other',kind='image',image=image),dict(id='statistics',kind='table',headers=['Metric','Value'],rows=[['Mean','1.25']])])
        options=dict(title='주사전자현미경(SEM)',sample='MUST NOT APPEAR',method='직접 입력 & <분석 방법>\n둘째 문장',results='사용자 수정 결과',conditions=[['장비','직접 입력']],extra_images=[],items=[dict(id='original',selected=True,caption='두 이미지',image_ids=['other','original'],columns=2),dict(id='statistics',selected=True,caption='통계표')])
        with tempfile.TemporaryDirectory(dir=ROOT/'outputs') as folder:
            path=Path(folder)/'report.hwpx';export_hwpx(path,data,options)
            text,pics,tables=verify_hwpx(path)
            self.assertEqual((pics,tables),(1,2))
            self.assertIn('직접 입력 & <분석 방법>',text)
            self.assertIn('사용자 수정 결과',text)
            self.assertNotIn('MUST NOT APPEAR',text)
            self.assertIn('그림 1. 두 이미지',text)
            self.assertIn('표 1. 통계표',text)


if __name__=='__main__':unittest.main()
